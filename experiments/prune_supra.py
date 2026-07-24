"""Frequency-utility pruning of LiB's supra-word vocabulary (Stage 1: fertility).

Go/no-go experiment for the hypothesis that LiB's MDL objective *over-admits*
supra-word tokens: a long tail of rare multi-word types that bloat the
vocabulary (and, we suspect, hurt neural BPB via under-trained embeddings)
while contributing almost nothing to sequence compression.

Mechanism
---------
The LiB Rust encoder is greedy longest-match over a trie built from the vocab
(``match_longest`` / ``match_two``, with ``skip_spaces = !use_supra_words``).
Removing a supra-word entry from the vocab therefore makes the encoder fall
back to shorter units — pruning at encode time, no retraining required.

Method
------
1. Encode the *training* corpus with full LiB; count how often each supra-word
   token actually fires.  (The vocab's stored integer fields are constant, so
   firing frequency must be measured, not read.)
2. Rank supra-word types by firing frequency.  Keep the top fraction ``k``;
   drop the rest by writing a pruned ``tokenizer.json``.
3. Measure fertility (tokens/byte, tokens/sentence) on the held-out *test* set
   for each ``k``, plus the resulting effective vocabulary size.

Decisive question (Stage 1)
---------------------------
Is supra-word compression concentrated in a small fraction of frequent types?
If keeping, say, the top 10% of supra types recovers ~all of the full-supra
compression, then the other ~90% (tens of thousands of types) are dead weight —
structural evidence of over-admission, and the setup for the Stage-2 BPB test
of whether removing them improves quality.

Validation
----------
Two extremes must reproduce known references, or the prune mechanism is wrong:
  * keep-all supra  (k=1.0)  -> fertility of full LiB
  * keep-no  supra  (k=0.0)  -> fertility of LiB with use_supra_words=False

Usage
-----
    python prune_supra.py --lang en --vocab-size 50000
"""

import argparse
import collections
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
OUT_DIR = Path(__file__).parent / "results"
sys.path.insert(0, str(REPO_ROOT))

from tokenizers import Tokenizer


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f if l.strip()]


def is_supra(tok_str):
    """A supra-word token spans a word boundary: it contains a space that is
    not the leading word-initial marker."""
    core = tok_str[1:] if tok_str[:1] == " " else tok_str
    return " " in core


def load_vocab(tok_json_path):
    with open(tok_json_path, encoding="utf-8") as f:
        d = json.load(f)
    return d, d["model"]["vocab"]


def write_pruned(base_dict, keep_supra_strings, out_path):
    """Write a tokenizer.json whose vocab keeps all non-supra entries plus the
    supra entries in ``keep_supra_strings``."""
    d = json.loads(json.dumps(base_dict))  # deep copy
    keep = set(keep_supra_strings)
    d["model"]["vocab"] = [
        e for e in d["model"]["vocab"]
        if (not is_supra(e[0])) or (e[0] in keep)
    ]
    d["model"]["use_supra_words"] = True
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)
    return out_path


def fertility(tok, test_lines):
    n_tokens = 0
    n_supra_fired = 0
    for line in test_lines:
        if not line:
            continue
        toks = tok.encode(line).tokens
        n_tokens += len(toks)
        n_supra_fired += sum(1 for t in toks if is_supra(t))
    raw_bytes = sum(len(l.encode("utf-8")) for l in test_lines if l)
    n_sents = sum(1 for l in test_lines if l)
    return {
        "tokens": n_tokens,
        "tok_per_byte": n_tokens / max(raw_bytes, 1),
        "tok_per_sent": n_tokens / max(n_sents, 1),
        "supra_fired_frac": n_supra_fired / max(n_tokens, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en")
    ap.add_argument("--vocab-size", type=int, default=50000)
    ap.add_argument("--keep-fracs", type=float, nargs="+",
                    default=[0.0, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.0],
                    help="fractions of supra types (ranked by firing freq) to keep")
    args = ap.parse_args()

    lib_dir = MODELS_DIR / args.lang / f"lib_{args.vocab_size}"
    tok_json = lib_dir / "tokenizer.json"
    if not tok_json.exists():
        sys.exit(f"missing {tok_json}")

    lang_data = DATA_DIR / args.lang
    train_lines = read_lines(lang_data / "train.txt")
    test_lines  = read_lines(lang_data / "test.txt")

    base_dict, vocab = load_vocab(tok_json)
    supra_strings = [e[0] for e in vocab if is_supra(e[0])]
    n_supra = len(supra_strings)
    n_total = len(vocab)
    print(f"[{args.lang} {args.vocab_size}] vocab={n_total:,}  "
          f"supra={n_supra:,} ({n_supra/n_total:.1%})")

    # --- 1. firing frequency of each supra token on the training corpus ---
    print("counting supra-word firing frequencies on train ...", flush=True)
    full = Tokenizer.from_file(str(tok_json))
    full.model.use_supra_words = True
    fire = collections.Counter()
    for i, line in enumerate(train_lines):
        if not line:
            continue
        for t in full.encode(line).tokens:
            if is_supra(t):
                fire[t] += 1
        if (i + 1) % 20000 == 0:
            print(f"  {i+1:,}/{len(train_lines):,}", flush=True)

    never_fired = [s for s in supra_strings if fire[s] == 0]
    print(f"supra types that NEVER fire on train: {len(never_fired):,} "
          f"({len(never_fired)/n_supra:.1%} of supra)")
    ranked = sorted(supra_strings, key=lambda s: fire[s], reverse=True)

    # firing-frequency concentration
    total_fires = sum(fire.values()) or 1
    for frac in (0.01, 0.05, 0.10, 0.25):
        k = int(frac * n_supra)
        share = sum(fire[s] for s in ranked[:k]) / total_fires
        print(f"  top {frac:>5.0%} of supra types carry {share:6.1%} of all supra firings")

    # --- 2/3. sweep keep-fractions, measure fertility on test ---
    scratch = Path("/private/tmp/claude-501/-Users-antalb-Experiments-LiB/"
                   "58ae9eb3-5429-4198-8037-fc094615d5ec/scratchpad")
    scratch.mkdir(parents=True, exist_ok=True)

    # references
    lib_off = Tokenizer.from_file(str(tok_json))
    lib_off.model.use_supra_words = False
    ref_off = fertility(lib_off, test_lines)
    ref_on = fertility(full, test_lines)
    print(f"\nREF full LiB (keep all): tok/byte={ref_on['tok_per_byte']:.4f}  "
          f"tok/sent={ref_on['tok_per_sent']:.1f}  supra_fired={ref_on['supra_fired_frac']:.1%}")
    print(f"REF LiB no-supra       : tok/byte={ref_off['tok_per_byte']:.4f}  "
          f"tok/sent={ref_off['tok_per_sent']:.1f}")

    rows = []
    for frac in sorted(set(args.keep_fracs)):
        k = int(round(frac * n_supra))
        keep = ranked[:k]
        pruned_path = scratch / f"lib_pruned_{args.lang}_{args.vocab_size}_{frac:.3f}.json"
        write_pruned(base_dict, keep, pruned_path)
        tok = Tokenizer.from_file(str(pruned_path))
        tok.model.use_supra_words = True
        f = fertility(tok, test_lines)
        eff_vocab = (n_total - n_supra) + k
        rows.append({
            "keep_frac": frac, "kept_supra": k, "eff_vocab": eff_vocab,
            "tok_per_byte": f["tok_per_byte"], "tok_per_sent": f["tok_per_sent"],
            "supra_fired": f["supra_fired_frac"],
        })
        print(f"  keep={frac:5.2f}  supra_kept={k:6,}  vocab={eff_vocab:6,}  "
              f"tok/byte={f['tok_per_byte']:.4f}  tok/sent={f['tok_per_sent']:5.1f}  "
              f"supra_fired={f['supra_fired_frac']:.1%}", flush=True)

    # validation of the prune mechanism at the extremes
    tol = 0.02
    top = next(r for r in rows if r["keep_frac"] == max(r2["keep_frac"] for r2 in rows))
    bot = next(r for r in rows if r["keep_frac"] == min(r2["keep_frac"] for r2 in rows))
    ok_top = abs(top["tok_per_byte"] - ref_on["tok_per_byte"]) / ref_on["tok_per_byte"] < tol
    ok_bot = abs(bot["tok_per_byte"] - ref_off["tok_per_byte"]) / ref_off["tok_per_byte"] < tol
    print(f"\nmechanism check: keep-all≈full-LiB? {ok_top}   "
          f"keep-none≈no-supra? {ok_bot}")

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"prune_supra_{args.lang}_{args.vocab_size}.json"
    with open(out, "w") as f:
        json.dump({"refs": {"full": ref_on, "no_supra": ref_off},
                   "never_fired": len(never_fired),
                   "n_supra": n_supra, "rows": rows}, f, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
