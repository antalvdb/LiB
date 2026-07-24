"""Frequency-utility pruning of LiB supra-words (Stage 2: neural BPB).

Go/no-go test. Stage 1 (prune_supra.py) showed LiB over-admits supra-words:
keeping the top ~10% by firing frequency recovers ~96% of the compression while
halving the vocabulary. Stage 2 asks the decisive quality question:

    Does the compact, denser pruned vocabulary IMPROVE neural bits-per-byte
    over the full LiB vocabulary?

Hypothesis: full LiB's BPB penalty is concentrated in a long tail of rare
supra-word types with under-trained embeddings. Removing them (frequency prune)
should lower BPB toward the subword baselines, at ~unchanged fertility.

Decision rule (stated up front):
  * POSITIVE  -> a pruned variant Pareto-improves on full LiB: BPB clearly lower
                 at ~equal fertility (and a much smaller vocab). LiB's supra
                 level is worth keeping and the MDL admission is fixable.
  * NEGATIVE  -> pruning does NOT lower BPB (rare supra were not the quality
                 problem; the over-fragmented subword base is). Per the agreed
                 stopping rule, the study halts.

All models train under identical settings; the test set is the full 10k lines,
so BPB is comparable to the English numbers in neural_bpb_en_50000.csv.

Usage
-----
    python prune_supra_bpb.py --keep-fracs 1.0 0.25 0.10 --max-steps 5000
"""

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
OUT_DIR = Path(__file__).parent / "results"
SCRATCH = Path("/private/tmp/claude-501/-Users-antalb-Experiments-LiB/"
               "58ae9eb3-5429-4198-8037-fc094615d5ec/scratchpad")
sys.path.insert(0, str(REPO_ROOT))

from tokenizers import Tokenizer
import torch

from experiments.prune_supra import is_supra, load_vocab, write_pruned, read_lines
from experiments.evaluate_neural_bpb import (
    pick_device, build_vocab, encode_stream, TinyGPT, train_lm, score_stream,
    fertility_stats, fmt, print_table,
)


class RawTokWrapper:
    """Wrap a tokenizer.json path; optionally force use_supra_words."""
    SPECIALS = {"[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"}

    def __init__(self, json_path, use_supra=None):
        self._tok = Tokenizer.from_file(str(json_path))
        if use_supra is not None:
            self._tok.model.use_supra_words = use_supra

    def vocab(self):
        return [t for t in self._tok.get_vocab() if t not in self.SPECIALS]

    def encode(self, text):
        return self._tok.encode(text).tokens


def rank_supra_by_firing(tok_json, train_lines):
    full = Tokenizer.from_file(str(tok_json))
    full.model.use_supra_words = True
    fire = collections.Counter()
    for line in train_lines:
        if not line:
            continue
        for t in full.encode(line).tokens:
            if is_supra(t):
                fire[t] += 1
    _, vocab = load_vocab(tok_json)
    supra = [e[0] for e in vocab if is_supra(e[0])]
    return sorted(supra, key=lambda s: fire[s], reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en")
    ap.add_argument("--vocab-size", type=int, default=50000)
    ap.add_argument("--keep-fracs", type=float, nargs="+", default=[1.0, 0.25, 0.10])
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--max-valid-lines", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-train-lines", type=int, default=0)
    args = ap.parse_args()
    # train_lm reads these via the args object:
    args.max_test_lines = 0

    device = pick_device(args.device)
    print(f"device: {device}")

    lib_json = MODELS_DIR / args.lang / f"lib_{args.vocab_size}" / "tokenizer.json"
    base_dict, vocab = load_vocab(lib_json)
    n_supra = sum(1 for e in vocab if is_supra(e[0]))

    lang_data = DATA_DIR / args.lang
    train_lines = read_lines(lang_data / "train.txt")
    valid_lines = read_lines(lang_data / "valid.txt")
    test_lines  = read_lines(lang_data / "test.txt")
    if args.max_train_lines:
        train_lines = train_lines[:args.max_train_lines]
    valid_for_es = valid_lines[:args.max_valid_lines] if args.max_valid_lines else valid_lines

    print("ranking supra by firing frequency on train ...", flush=True)
    ranked = rank_supra_by_firing(lib_json, train_lines)

    # Build the tokenizer set: pruned LiB variants (+ full at keep=1.0).
    tok_specs = []
    for frac in sorted(set(args.keep_fracs), reverse=True):
        k = int(round(frac * n_supra))
        pruned_path = SCRATCH / f"lib_pruned_{args.lang}_{args.vocab_size}_{frac:.3f}.json"
        write_pruned(base_dict, ranked[:k], pruned_path)
        label = "LiB (full)" if abs(frac - 1.0) < 1e-9 else f"LiB (keep {int(frac*100)}%)"
        tok_specs.append((label, RawTokWrapper(pruned_path, use_supra=True)))

    raw_bytes_test = sum(len(l.encode("utf-8")) for l in test_lines if l)
    raw_chars_test = sum(len(l) for l in test_lines if l)

    rows = []
    for label, wrapper in tok_specs:
        print(f"\n=== {label} ===", flush=True)
        stoi, itos = build_vocab(wrapper, train_lines)
        train_ids, _        = encode_stream(wrapper, train_lines, stoi)
        valid_ids, valid_sc = encode_stream(wrapper, valid_for_es, stoi)
        test_ids,  test_sc  = encode_stream(wrapper, test_lines, stoi)
        print(f"  |V|={len(itos):,}  train_toks={len(train_ids):,}", flush=True)

        torch.manual_seed(args.seed)
        model = TinyGPT(len(itos), d_model=args.d_model, n_head=args.n_head,
                        n_layer=args.n_layer, block_size=args.block_size,
                        dropout=args.dropout)
        model = train_lm(model, train_ids, valid_ids, valid_sc, args, device)

        fert = fertility_stats(wrapper, test_lines)
        bpb, _ = score_stream(model, test_ids, test_sc, raw_bytes=fert["raw_bytes"],
                              block_size=args.block_size, stride=args.stride, device=device)
        rows.append({
            "name": label, "|V|": f"{len(itos):,}",
            "BPB": fmt(bpb), "tok/byte": fmt(fert["tokens_per_byte"], 4),
            "tok/sent": fmt(fert["tokens_per_sent"], 1),
        })
        print_table(rows, f"{args.lang} — prune BPB (running)")

    print_table(rows, f"{args.lang.upper()} — supra-word prune: neural BPB vs fertility")
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"prune_bpb_{args.lang}_{args.vocab_size}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"saved {out}")
    print("\nReference (from neural_bpb_en_50000.csv): BPE BPB=1.805 tok/byte=0.200; "
          "SuperBPE BPB=1.815 tok/byte=0.176; LiB(no supra) BPB=2.044.")


if __name__ == "__main__":
    main()
