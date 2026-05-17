"""Run all evaluation experiments and print result tables.

Expects:
  experiments/data/train.txt        (from prepare_corpus.py)
  experiments/data/test.txt         (from prepare_corpus.py)
  experiments/models/bpe_50000/     (from train_competitors.py)
  experiments/models/wordpiece_50000/
  experiments/models/sp_unigram_50000/sp_unigram.model
  lib-tokenizer-trained/            (from train_tokenizer.py)

Usage:
    python evaluate_all.py [--vocab-size 50000] [--ngram 2 3] [--no-bpc]
"""

import argparse
import os
import sys

# Make sure the repo root is on the path for lib_tokenizers
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from experiments.compute_metrics import (
    HFTokenizerWrapper, SPUnigramWrapper, LiBWrapper, CharTokenizerWrapper,
    compute_dl, encode_corpus, train_ngram_lm, compute_bpc, avg_token_length,
)

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
LIB_DIR    = os.path.join(REPO_ROOT, "lib-tokenizer-trained")

TRAIN_PATH = os.path.join(DATA_DIR, "train.txt")
TEST_PATH  = os.path.join(DATA_DIR, "test.txt")


def read_lines(path: str) -> list[str]:
    with open(path) as f:
        return [l.rstrip("\n") for l in f if l.strip()]


def build_tokenizers(vocab_size: int) -> list[tuple[str, object]]:
    """Return (name, wrapper) pairs for each tokenizer to evaluate."""
    tok_list = []

    # Character baseline
    tok_list.append(("Symbol (char)", CharTokenizerWrapper()))

    # BPE
    bpe_path = os.path.join(MODELS_DIR, f"bpe_{vocab_size}", "tokenizer.json")
    if os.path.exists(bpe_path):
        tok_list.append((f"BPE-{vocab_size//1000}k", HFTokenizerWrapper(bpe_path)))
    else:
        print(f"[skip] BPE {vocab_size} not found at {bpe_path}")

    # WordPiece
    wp_path = os.path.join(MODELS_DIR, f"wordpiece_{vocab_size}", "tokenizer.json")
    if os.path.exists(wp_path):
        tok_list.append((f"WordPiece-{vocab_size//1000}k", HFTokenizerWrapper(wp_path)))
    else:
        print(f"[skip] WordPiece {vocab_size} not found at {wp_path}")

    # SentencePiece Unigram
    sp_path = os.path.join(MODELS_DIR, f"sp_unigram_{vocab_size}", "sp_unigram.model")
    if os.path.exists(sp_path):
        tok_list.append((f"SP-Unigram-{vocab_size//1000}k", SPUnigramWrapper(sp_path)))
    else:
        print(f"[skip] SP-Unigram {vocab_size} not found at {sp_path}")

    # LiB (supra-words disabled — ablation)
    if os.path.exists(LIB_DIR):
        tok_list.append((f"LiB-nosupra-{vocab_size//1000}k",
                         LiBWrapper(LIB_DIR, use_supra_words=False)))
        # LiB (supra-words enabled)
        tok_list.append((f"LiB-{vocab_size//1000}k",
                         LiBWrapper(LIB_DIR, use_supra_words=True)))
    else:
        print(f"[skip] LiB not found at {LIB_DIR}")

    return tok_list


def fmt(x: float, decimals: int = 1) -> str:
    return f"{x:.{decimals}f}"


def print_table(rows: list[dict], title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    if not rows:
        print("  (no data)")
        return

    keys = [k for k in rows[0] if k != "name"]
    col_w = max(len(k) for k in keys) + 2
    name_w = max(len(r["name"]) for r in rows) + 2

    header = f"  {'Tokenizer':<{name_w}}" + "".join(f"{k:>{col_w}}" for k in keys)
    print(header)
    print(f"  {'-'*name_w}" + "-" * (col_w * len(keys)))
    for r in rows:
        row_str = f"  {r['name']:<{name_w}}" + "".join(
            f"{str(r.get(k,'—')):>{col_w}}" for k in keys
        )
        print(row_str)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=50_000)
    parser.add_argument("--ngram", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--no-bpc", action="store_true",
                        help="Skip BPC computation (faster)")
    args = parser.parse_args()

    for path in (TRAIN_PATH, TEST_PATH):
        if not os.path.exists(path):
            sys.exit(f"Missing {path} — run prepare_corpus.py first")

    print("Reading corpora …")
    train_lines = read_lines(TRAIN_PATH)
    test_lines  = read_lines(TEST_PATH)
    raw_test_chars = sum(len(l) for l in test_lines)
    print(f"  train: {len(train_lines):,} lines  |  test: {len(test_lines):,} lines  "
          f"({raw_test_chars:,} chars)")

    tokenizers = build_tokenizers(args.vocab_size)

    # -----------------------------------------------------------------------
    # DL table
    # -----------------------------------------------------------------------
    print("\nComputing Description Lengths …")
    dl_rows = []
    tokenized_test_cache = {}   # name → list[list[str]] (for BPC reuse)

    for name, wrapper in tokenizers:
        print(f"  {name} …", flush=True)
        vocab = wrapper.vocab()
        test_tokens_flat = encode_corpus(wrapper, test_lines)
        tokenized_test_cache[name] = [
            wrapper.encode(l) for l in test_lines if l
        ]

        metrics = compute_dl(vocab, test_tokens_flat)
        avg_len = avg_token_length(test_tokens_flat)
        n_supra = sum(1 for t in vocab if " " in t and len(t) > 1)

        dl_rows.append({
            "name":        name,
            "vocab_size":  str(len(vocab)),
            "avg_tok_len": fmt(avg_len, 2),
            "supra_types": str(n_supra) if " " in "".join(vocab) else "0",
            "DL(lex) kb":  fmt(metrics["dl_lex_kb"]),
            "DL(corp) kb": fmt(metrics["dl_corpus_kb"]),
            "DL(total) kb":fmt(metrics["dl_total_kb"]),
        })

    print_table(dl_rows, f"Description Length — {args.vocab_size//1000}k vocab — test set")

    # -----------------------------------------------------------------------
    # BPC table
    # -----------------------------------------------------------------------
    if args.no_bpc:
        print("(BPC skipped via --no-bpc)")
        return

    print("Computing Bits-per-Character (trains n-gram LMs — may take a few minutes) …")
    bpc_rows = []
    for name, wrapper in tokenizers:
        print(f"  {name} …", flush=True)

        # Tokenize train set for LM training
        tokenized_train = [wrapper.encode(l) for l in train_lines if l]
        tokenized_test  = tokenized_test_cache[name]

        row = {"name": name}
        for n in args.ngram:
            print(f"    training {n}-gram LM …", flush=True)
            lm = train_ngram_lm(tokenized_train, n)
            bpc = compute_bpc(lm, tokenized_test, raw_test_chars, n)
            row[f"{n}-gram BPC"] = fmt(bpc, 3)

        bpc_rows.append(row)

    print_table(bpc_rows, f"Bits-per-Character — {args.vocab_size//1000}k vocab — test set")


if __name__ == "__main__":
    main()
