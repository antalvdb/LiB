"""Compute Bits-per-Character (BPC) for all tokenizers using KenLM.

Uses lmplz to build Kneser-Ney n-gram LMs (orders 2 and 3) on the
tokenised training corpus, then scores the tokenised test corpus.
KenLM is orders of magnitude faster than NLTK for this task.

Requires:
    pip install kenlm
    lmplz on PATH  (from KenLM: https://kheafield.com/code/kenlm/)

Usage
-----
    # English, all vocab sizes
    python evaluate_bpc.py

    # Specific language / vocab size
    python evaluate_bpc.py --lang en --vocab-size 50000

    # Multilingual, 50k only
    python evaluate_bpc.py --lang en de fi tr --vocab-size 50000

    # Skip Symbol (char) baseline — much faster
    python evaluate_bpc.py --no-symbol
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR   = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"

sys.path.insert(0, str(REPO_ROOT))

from experiments.compute_metrics import (
    HFTokenizerWrapper, SPUnigramWrapper, LiBWrapper, CharTokenizerWrapper,
    encode_corpus, train_ngram_lm_kenlm, compute_bpc_kenlm,
)

LANGUAGES = {
    "en": "English (analytic)",
    "de": "German (fusional/compound)",
    "es": "Spanish (fusional)",
    "fi": "Finnish (agglutinative)",
    "tr": "Turkish (agglutinative)",
    "ar": "Arabic (templatic)",
    "zh": "Chinese (logographic)",
}


def read_lines(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f if l.strip()]


def build_tokenizers(lang: str, vocab_size: int, include_symbol: bool):
    tok_list = []
    if include_symbol:
        tok_list.append(("Symbol", CharTokenizerWrapper()))

    model_dir = MODELS_DIR / lang

    configs = [
        ("BPE",            f"bpe_{vocab_size}",       "tokenizer.json",   "hf"),
        ("WordPiece",      f"wordpiece_{vocab_size}",  "tokenizer.json",   "hf"),
        ("SP-Unigram",     f"sp_unigram_{vocab_size}", "sp_unigram.model", "sp"),
        ("SuperBPE",       f"superbpe_{vocab_size}",   "tokenizer.json",   "hf"),
        ("LiB (no supra)", f"lib_{vocab_size}",        "tokenizer.json",   "lib_off"),
        ("LiB",            f"lib_{vocab_size}",        "tokenizer.json",   "lib_on"),
    ]

    for display, subdir, fname, kind in configs:
        path = model_dir / subdir / fname
        if not path.exists():
            print(f"    [skip] {display}: {path} not found")
            continue
        if kind == "hf":
            tok_list.append((display, HFTokenizerWrapper(str(path))))
        elif kind == "sp":
            tok_list.append((display, SPUnigramWrapper(str(path))))
        elif kind == "lib_off":
            tok_list.append((display, LiBWrapper(str(path.parent), use_supra_words=False)))
        elif kind == "lib_on":
            tok_list.append((display, LiBWrapper(str(path.parent), use_supra_words=True)))

    return tok_list


def fmt(x, d=3):
    return f"{x:.{d}f}"


def print_table(rows: list[dict], title: str):
    print(f"\n{'='*66}")
    print(f"  {title}")
    print(f"{'='*66}")
    if not rows:
        print("  (no data)")
        return
    keys = [k for k in rows[0] if k != "name"]
    name_w = max(len(r["name"]) for r in rows) + 2
    col_w  = max(max(len(k), max(len(str(r.get(k, "—"))) for r in rows))
                 for k in keys) + 2
    print(f"  {'Tokenizer':<{name_w}}" + "".join(f"{k:>{col_w}}" for k in keys))
    print(f"  {'-'*name_w}" + "-" * (col_w * len(keys)))
    for r in rows:
        print(f"  {r['name']:<{name_w}}" +
              "".join(f"{str(r.get(k,'—')):>{col_w}}" for k in keys))
    print()


def evaluate_bpc(lang: str, vocab_size: int, ngram_ns: list[int],
                 include_symbol: bool, tmp_dir: str):
    # Use per-language data if available, fall back to default English data
    lang_data = DATA_DIR / lang
    if lang_data.exists():
        train_path = lang_data / "train.txt"
        test_path  = lang_data / "test.txt"
    else:
        train_path = DATA_DIR / "train.txt"
        test_path  = DATA_DIR / "test.txt"

    if not train_path.exists() or not test_path.exists():
        print(f"  [{lang}] Missing data — run prepare_corpus_multilingual.py")
        return []

    train_lines = read_lines(train_path)
    test_lines  = read_lines(test_path)
    raw_test_chars = sum(len(l) for l in test_lines)

    tokenizers = build_tokenizers(lang, vocab_size, include_symbol)
    if not tokenizers:
        return []

    rows = []
    for name, wrapper in tokenizers:
        print(f"    {name} …", flush=True)
        tokenized_train = [wrapper.encode(l) for l in train_lines if l]
        tokenized_test  = [wrapper.encode(l) for l in test_lines  if l]

        row = {"name": name}
        for n in ngram_ns:
            print(f"      {n}-gram LM …", flush=True)
            lm = train_ngram_lm_kenlm(
                tokenized_train, n, tmp_dir,
                name=f"{lang}_{name.replace(' ', '_')}_{vocab_size}_{n}gram",
            )
            bpc = compute_bpc_kenlm(lm, tokenized_test, raw_test_chars)
            row[f"{n}-gram BPC"] = fmt(bpc)
        rows.append(row)

    lang_label = LANGUAGES.get(lang, lang)
    print_table(rows, f"{lang_label} — BPC — {vocab_size//1000}k vocab")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", nargs="+",
                        default=["en"],
                        choices=list(LANGUAGES.keys()) + ["en"],
                        metavar="LANG",
                        help="Language codes (default: en)")
    parser.add_argument("--vocab-size", type=int, nargs="+", default=[50_000])
    parser.add_argument("--ngram", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--no-symbol", action="store_true",
                        help="Skip Symbol (char) baseline — much faster")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="lib_bpc_") as tmp_dir:
        for vocab_size in args.vocab_size:
            print(f"\n{'#'*60}")
            print(f"  Vocab size: {vocab_size:,}")
            print(f"{'#'*60}")
            for lang in args.lang:
                print(f"\n--- {lang.upper()}: {LANGUAGES.get(lang, lang)} ---")
                evaluate_bpc(
                    lang, vocab_size, args.ngram,
                    include_symbol=not args.no_symbol,
                    tmp_dir=tmp_dir,
                )

    print("Done.")


if __name__ == "__main__":
    main()
