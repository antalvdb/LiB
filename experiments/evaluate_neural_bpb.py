"""Tokenization-fair evaluation: neural Bits-per-Byte (BPB) + fertility.

Motivation
----------
The n-gram BPC metric in ``evaluate_bpc.py`` systematically *penalises*
supra-word tokens: a Kneser-Ney n-gram model sees each rare multi-word type
too few times to estimate it, so surprisal explodes through data sparsity, not
through any real deficiency of the tokenization.  This is why LiB (and SuperBPE)
look worse at 3-gram than at 2-gram — it is an artefact of the *estimator*, not
a property of the tokens.

A neural language model does not suffer this sparsity pathology, and
**Bits-per-Byte** — total cross-entropy over the test stream, normalised by the
raw UTF-8 byte count of the test text — is invariant to the tokenization's
segmentation granularity.  It is the standard tokenization-fair metric in the
compression / LM literature (used e.g. by SuperBPE).  A tokenizer that packs
more text into each token is not rewarded or punished for that alone; only the
*joint* compression achieved by (tokenizer + LM) is measured.

    BPB = ( Σ_i  -log2 P(token_i | context_i) )  /  ( total UTF-8 bytes of raw test text )

Because the LM output space is the tokenizer's *full* vocabulary, the real cost
of LiB's long tail of rare supra-word types shows up honestly: their embeddings
are under-trained, so the model predicts them poorly and BPB rises — a fair
neural penalty, not an n-gram artefact.

This script also reports the **efficiency** dimension where supra-word tokens
are expected to *win*:
  * tokens-per-byte (fertility; lower = more compressive),
  * bytes-per-token,
  * tokens-per-sentence,
which together determine sequence length and inference compute.

Design notes
------------
* Model output space = the tokenizer's full closed vocabulary, so no <unk>
  collapse hides the rare-type cost.  (An <unk> id exists only as a safety net.)
* BPB is computed with a strided sliding window so every scored token has at
  least ``block_size - stride`` tokens of left context, following the standard
  fixed-context perplexity recipe.
* Sentence starts are marked with <bos> and are *not* scored (they are context
  only); the number of sentences is identical across tokenizers, so this is a
  fair constant.

Requires
--------
    pip install torch     (already a transitive dep on most setups)

Usage
-----
    # English, 50k vocab, all tokenizers, default small model
    python evaluate_neural_bpb.py --lang en --vocab-size 50000

    # Faster smoke test
    python evaluate_neural_bpb.py --lang en --vocab-size 10000 --max-steps 1000

    # A couple of languages
    python evaluate_neural_bpb.py --lang en fi tr --vocab-size 50000

Warning: training one small Transformer per tokenizer is compute-heavy.  On a
laptop MPS/CPU use ``--max-steps`` in the low thousands for a smoke test; use a
GPU and ``--max-steps 20000+`` for numbers worth reporting.
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR   = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

sys.path.insert(0, str(REPO_ROOT))

# Reuse the exact tokenizer set + data conventions from the n-gram evaluator,
# so the two metrics are computed over identical models and corpora.
from experiments.evaluate_bpc import build_tokenizers, read_lines, LANGUAGES

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Vocabulary: LM output space = tokenizer's full closed vocabulary
# ---------------------------------------------------------------------------

PAD, BOS, UNK = "<pad>", "<bos>", "<unk>"


def build_vocab(wrapper, train_lines):
    """Map every token string the tokenizer can emit to a contiguous id.

    Starts from the tokenizer's declared vocabulary and unions in anything the
    tokenizer actually emits on the training text (covers the Char baseline,
    whose ``vocab()`` is empty by construction).
    """
    toks = set(wrapper.vocab())
    for line in train_lines:
        toks.update(wrapper.encode(line))
    itos = [PAD, BOS, UNK] + sorted(toks)
    stoi = {t: i for i, t in enumerate(itos)}
    return stoi, itos


def encode_stream(wrapper, lines, stoi):
    """Return (ids, scorable) where scorable[i] is True for real-token targets.

    <bos> is inserted at each sentence start and marked non-scorable (context
    only).  Unknown surface tokens fall back to <unk> (should not occur for the
    closed-vocab tokenizers, present for safety / the Char baseline).
    """
    bos, unk = stoi[BOS], stoi[UNK]
    ids, scorable = [], []
    for line in lines:
        if not line:
            continue
        ids.append(bos)
        scorable.append(False)
        for t in wrapper.encode(line):
            ids.append(stoi.get(t, unk))
            scorable.append(True)
    return ids, scorable


# ---------------------------------------------------------------------------
# A compact decoder-only Transformer (GPT-style)
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, d_model, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout,
                                          batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )
        mask = torch.triu(torch.ones(block_size, block_size), diagonal=1).bool()
        self.register_buffer("attn_mask", mask, persistent=False)

    def forward(self, x):
        T = x.size(1)
        h = self.ln1(x)
        m = self.attn_mask[:T, :T]
        a, _ = self.attn(h, h, h, attn_mask=m, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_head=4, n_layer=4,
                 block_size=256, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx):
        T = idx.size(1)
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def get_batch(ids_tensor, block_size, batch_size, device, generator):
    hi = ids_tensor.numel() - block_size - 1
    ix = torch.randint(0, hi, (batch_size,), generator=generator)
    x = torch.stack([ids_tensor[i:i + block_size] for i in ix])
    y = torch.stack([ids_tensor[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


def train_lm(model, train_ids, valid_ids, valid_scorable, args, device):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                            betas=(0.9, 0.95))
    gen = torch.Generator().manual_seed(args.seed)
    train_t = torch.tensor(train_ids, dtype=torch.long)

    best_val = float("inf")
    best_state = None
    patience_left = args.patience

    for step in range(1, args.max_steps + 1):
        model.train()
        x, y = get_batch(train_t, model.block_size, args.batch_size, device, gen)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.eval_every == 0 or step == args.max_steps:
            val_bpb, _ = score_stream(model, valid_ids, valid_scorable,
                                      raw_bytes=None, block_size=model.block_size,
                                      stride=args.stride, device=device)
            # raw_bytes=None -> returns mean bits per scored token (proxy for val)
            print(f"      step {step:>6}  train_ce={loss.item():.3f}  "
                  f"val_bits/tok={val_bpb:.4f}", flush=True)
            if val_bpb < best_val - 1e-4:
                best_val = val_bpb
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                patience_left = args.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print("      early stop", flush=True)
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# Scoring: strided sliding-window cross-entropy
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_stream(model, ids, scorable, raw_bytes, block_size, stride, device):
    """Sum -log2 P over scorable targets with a strided window (vectorised).

    Returns (metric, n_scored).  If ``raw_bytes`` is given, metric is
    bits-per-byte; otherwise it is mean bits per scored token (used for
    validation / early stopping).

    Within each window, position j (0-based) predicts the token at global
    index ``start + j + 1``.  A target is scored iff it is a real token
    (``scorable``) and has not already been scored by an earlier, overlapping
    window (``global index >= prev_end``).  The per-position log-probs are
    gathered in one ``gather`` rather than a Python loop, so cost is dominated
    by the forward passes.
    """
    model.eval()
    ids_t = torch.tensor(ids, dtype=torch.long)
    scor_t = torch.tensor(scorable, dtype=torch.bool)
    n = ids_t.numel()
    inv_ln2 = 1.0 / math.log(2.0)
    total_bits = 0.0
    n_scored = 0
    prev_end = 0

    start = 0
    while start < n - 1:
        end = min(start + block_size, n)
        L = end - start
        if L < 2:
            break
        window_x = ids_t[start:end].unsqueeze(0).to(device)
        logits = model(window_x)                       # (1, L, V)
        logp = torch.log_softmax(logits.float(), dim=-1)[0]  # (L, V)
        tgt = ids_t[start + 1:end].to(device)          # (L-1,)
        pos = torch.arange(L - 1, device=device)
        bits = -logp[pos, tgt] * inv_ln2               # (L-1,) bits per target

        gpos = torch.arange(start + 1, end)            # global indices (cpu)
        keep = (scor_t[start + 1:end] & (gpos >= prev_end)).to(device)
        total_bits += float(bits[keep].sum().item())
        n_scored += int(keep.sum().item())

        prev_end = end
        if end == n:
            break
        start += stride

    if n_scored == 0:
        return float("inf"), 0
    if raw_bytes is None:
        return total_bits / n_scored, n_scored
    return total_bits / raw_bytes, n_scored


# ---------------------------------------------------------------------------
# Fertility / efficiency
# ---------------------------------------------------------------------------

def fertility_stats(wrapper, test_lines):
    n_tokens = 0
    for line in test_lines:
        if line:
            n_tokens += len(wrapper.encode(line))
    raw_bytes = sum(len(l.encode("utf-8")) for l in test_lines if l)
    raw_chars = sum(len(l) for l in test_lines if l)
    n_sents = sum(1 for l in test_lines if l)
    return {
        "n_tokens": n_tokens,
        "raw_bytes": raw_bytes,
        "raw_chars": raw_chars,
        "tokens_per_byte": n_tokens / max(raw_bytes, 1),
        "bytes_per_token": raw_bytes / max(n_tokens, 1),
        "tokens_per_sent": n_tokens / max(n_sents, 1),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def fmt(x, d=4):
    return f"{x:.{d}f}"


def print_table(rows, title):
    print(f"\n{'='*78}\n  {title}\n{'='*78}")
    if not rows:
        print("  (no data)")
        return
    keys = [k for k in rows[0] if k != "name"]
    name_w = max(len(r["name"]) for r in rows) + 2
    col_w = max(max(len(k), max(len(str(r.get(k, "—"))) for r in rows))
                for k in keys) + 2
    print(f"  {'Tokenizer':<{name_w}}" + "".join(f"{k:>{col_w}}" for k in keys))
    print(f"  {'-'*name_w}" + "-" * (col_w * len(keys)))
    for r in rows:
        print(f"  {r['name']:<{name_w}}" +
              "".join(f"{str(r.get(k,'—')):>{col_w}}" for k in keys))


def evaluate_lang(lang, vocab_size, args, device):
    lang_data = DATA_DIR / lang
    if lang_data.exists():
        train_path, valid_path, test_path = (
            lang_data / "train.txt", lang_data / "valid.txt", lang_data / "test.txt")
    else:
        train_path, valid_path, test_path = (
            DATA_DIR / "train.txt", DATA_DIR / "valid.txt", DATA_DIR / "test.txt")

    if not train_path.exists() or not test_path.exists():
        print(f"  [{lang}] missing data — run prepare_corpus_multilingual.py")
        return []

    train_lines = read_lines(train_path)
    test_lines  = read_lines(test_path)
    valid_lines = read_lines(valid_path) if valid_path.exists() else test_lines
    if args.max_train_lines:
        train_lines = train_lines[:args.max_train_lines]
    # --max-valid-lines caps ONLY the validation set (the early-stopping signal),
    # which is scored repeatedly during training and dominates wall-clock. The
    # test set is left full so final BPB stays comparable across runs.
    if args.max_valid_lines:
        valid_lines = valid_lines[:args.max_valid_lines]
    if args.max_test_lines:
        test_lines = test_lines[:args.max_test_lines]

    tokenizers = build_tokenizers(lang, vocab_size, include_symbol=not args.no_symbol)
    if not tokenizers:
        return []

    rows = []
    for name, wrapper in tokenizers:
        print(f"\n    === {name} ===", flush=True)
        stoi, itos = build_vocab(wrapper, train_lines)
        train_ids, _        = encode_stream(wrapper, train_lines, stoi)
        valid_ids, valid_sc = encode_stream(wrapper, valid_lines, stoi)
        test_ids,  test_sc  = encode_stream(wrapper, test_lines,  stoi)
        print(f"      |V|={len(itos):,}  train_toks={len(train_ids):,}", flush=True)

        torch.manual_seed(args.seed)
        model = TinyGPT(len(itos), d_model=args.d_model, n_head=args.n_head,
                        n_layer=args.n_layer, block_size=args.block_size,
                        dropout=args.dropout)
        model = train_lm(model, train_ids, valid_ids, valid_sc, args, device)

        fert = fertility_stats(wrapper, test_lines)
        bpb, n_scored = score_stream(model, test_ids, test_sc,
                                     raw_bytes=fert["raw_bytes"],
                                     block_size=args.block_size,
                                     stride=args.stride, device=device)
        # BPC (bits per raw char) for continuity with the n-gram BPC table
        bpc_equiv = bpb * fert["raw_bytes"] / max(fert["raw_chars"], 1)

        rows.append({
            "name": name,
            "BPB": fmt(bpb),
            "BPC": fmt(bpc_equiv),
            "tok/byte": fmt(fert["tokens_per_byte"], 4),
            "byte/tok": fmt(fert["bytes_per_token"], 2),
            "tok/sent": fmt(fert["tokens_per_sent"], 1),
        })

    print_table(rows, f"{LANGUAGES.get(lang, lang)} — neural BPB — {vocab_size//1000}k vocab")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"neural_bpb_{lang}_{vocab_size}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"    saved {out}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", nargs="+", default=["en"],
                   choices=list(LANGUAGES.keys()), metavar="LANG")
    p.add_argument("--vocab-size", type=int, nargs="+", default=[50_000])
    p.add_argument("--no-symbol", action="store_true")
    # model
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    # training
    p.add_argument("--max-steps", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--stride", type=int, default=128,
                   help="eval sliding-window stride (<= block_size)")
    p.add_argument("--max-train-lines", type=int, default=0,
                   help="cap training sentences (0 = all)")
    p.add_argument("--max-valid-lines", type=int, default=0,
                   help="cap validation sentences used for early stopping "
                        "(0 = all). Speeds training without changing final BPB.")
    p.add_argument("--max-test-lines", type=int, default=0,
                   help="cap test sentences for final BPB (0 = all). Keep 0 for "
                        "comparable numbers; set only for smoke tests.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.stride > args.block_size:
        args.stride = args.block_size
    device = pick_device(args.device)
    print(f"device: {device}")

    for vocab_size in args.vocab_size:
        print(f"\n{'#'*60}\n  Vocab size: {vocab_size:,}\n{'#'*60}")
        for lang in args.lang:
            print(f"\n--- {lang.upper()}: {LANGUAGES.get(lang, lang)} ---")
            evaluate_lang(lang, vocab_size, args, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
