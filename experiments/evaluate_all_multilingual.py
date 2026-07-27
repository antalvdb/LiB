"""Evaluate all tokenizers across all languages.

Computes Description Length and (optionally) BPC for every
language × tokenizer × vocab_size combination, printing results
in per-language tables and a cross-language summary.

Expects:
    experiments/data/{lang}/train.txt
    experiments/data/{lang}/test.txt
    experiments/models/{lang}/{tok}_{vocab_size}/

Usage
-----
    # Full evaluation at 50k
    python evaluate_all_multilingual.py

    # Specific languages / vocab sizes
    python evaluate_all_multilingual.py --langs en de fi --vocab-size 50000

    # Skip slow BPC step
    python evaluate_all_multilingual.py --no-bpc

    # Only DL, all vocab sizes
    python evaluate_all_multilingual.py --vocab-size 10000 32000 50000 --no-bpc
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR   = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"

sys.path.insert(0, str(REPO_ROOT))

LANGUAGES = {
    "en": "English (analytic)",
    "de": "German (fusional/compound)",
    "es": "Spanish (fusional)",
    "fi": "Finnish (agglutinative)",
    "tr": "Turkish (agglutinative)",
    "ar": "Arabic (templatic)",
    "zh": "Chinese (logographic)",
}

from experiments.compute_metrics import (
    HFTokenizerWrapper, SPUnigramWrapper, LiBWrapper, CharTokenizerWrapper,
    compute_dl, encode_corpus, train_ngram_lm, compute_bpc, avg_token_length,
)


def read_lines(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f if l.strip()]


def count_supra(vocab: list[str]) -> int:
    # A supra-word token spans a word boundary: it has an *internal* space. The
    # leading space of a metaspace word-initial token (e.g. " cat") does NOT count.
    def is_supra(t: str) -> bool:
        core = t[1:] if t[:1] == " " else t
        return " " in core
    return sum(1 for t in vocab if is_supra(t))


def build_tokenizers(lang: str, vocab_size: int) -> list[tuple[str, object]]:
    tok_list = [("Symbol", CharTokenizerWrapper())]
    model_dir = MODELS_DIR / lang

    configs = [
        ("BPE",        f"bpe_{vocab_size}",        "tokenizer.json",   "hf"),
        ("WordPiece",  f"wordpiece_{vocab_size}",   "tokenizer.json",   "hf"),
        ("SP-Unigram", f"sp_unigram_{vocab_size}",  "sp_unigram.model", "sp"),
        ("LiB (no supra)", f"lib_{vocab_size}",     "tokenizer.json",   "lib_off"),
        ("LiB",        f"lib_{vocab_size}",         "tokenizer.json",   "lib_on"),
        ("LiB (no forget)", f"lib_{vocab_size}_noforget", "tokenizer.json", "lib_on"),
        ("LiB (cap)", f"lib_{vocab_size}_cap", "tokenizer.json", "lib_on"),
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


def fmt(x, d=1):
    return f"{x:.{d}f}"


def print_table(rows: list[dict], title: str):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")
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


def evaluate_language(lang: str, vocab_size: int, ngram_ns: list[int], skip_bpc: bool):
    train_path = DATA_DIR / lang / "train.txt"
    test_path  = DATA_DIR / lang / "test.txt"

    if not train_path.exists() or not test_path.exists():
        print(f"  Missing data for {lang} — run prepare_corpus_multilingual.py")
        return None, None

    train_lines = read_lines(train_path)
    test_lines  = read_lines(test_path)
    raw_test_chars = sum(len(l) for l in test_lines)

    tokenizers = build_tokenizers(lang, vocab_size)
    if not tokenizers:
        print(f"  No tokenizers found for {lang} at vocab_size={vocab_size}")
        return None, None

    # --- DL ---
    dl_rows = []
    tokenized_test_cache = {}

    for name, wrapper in tokenizers:
        vocab = wrapper.vocab()
        flat_tokens = encode_corpus(wrapper, test_lines)
        tokenized_test_cache[name] = [wrapper.encode(l) for l in test_lines if l]

        metrics  = compute_dl(vocab, flat_tokens)
        avg_len  = avg_token_length(flat_tokens)
        n_supra  = count_supra(vocab)
        supra_pct = f"{100*n_supra/max(len(vocab),1):.0f}%" if vocab else "—"

        dl_rows.append({
            "name":        name,
            "|V|":         str(len(vocab)) if vocab else "—",
            "avg_len":     fmt(avg_len, 2),
            "supra%":      supra_pct,
            "DL_lex(kb)":  fmt(metrics["dl_lex_kb"]),
            "DL_corp(kb)": fmt(metrics["dl_corpus_kb"]),
            "DL_tot(kb)":  fmt(metrics["dl_total_kb"]),
        })

    print_table(dl_rows, f"{LANGUAGES[lang]} — DL — {vocab_size//1000}k vocab")

    if skip_bpc:
        return dl_rows, None

    # --- BPC ---
    bpc_rows = []
    for name, wrapper in tokenizers:
        tokenized_train = [wrapper.encode(l) for l in train_lines if l]
        tokenized_test  = tokenized_test_cache[name]
        row = {"name": name}
        for n in ngram_ns:
            lm  = train_ngram_lm(tokenized_train, n)
            bpc = compute_bpc(lm, tokenized_test, raw_test_chars, n)
            row[f"{n}-gram BPC"] = fmt(bpc, 3)
        bpc_rows.append(row)

    print_table(bpc_rows, f"{LANGUAGES[lang]} — BPC — {vocab_size//1000}k vocab")
    return dl_rows, bpc_rows


def print_cross_language_summary(all_dl: dict, vocab_size: int):
    """Print a cross-language table of LiB DL_total vs best baseline."""
    print(f"\n{'#'*72}")
    print(f"  Cross-language summary — DL_total (kb) — {vocab_size//1000}k vocab")
    print(f"{'#'*72}")
    header_toks = ["Symbol", "BPE", "WordPiece", "SP-Unigram", "LiB"]
    lang_w = 28
    col_w  = 12
    print(f"  {'Language':<{lang_w}}" + "".join(f"{t:>{col_w}}" for t in header_toks))
    print(f"  {'-'*lang_w}" + "-" * (col_w * len(header_toks)))
    for lang, rows in all_dl.items():
        if not rows:
            continue
        val_map = {r["name"]: r.get("DL_tot(kb)", "—") for r in rows}
        print(f"  {LANGUAGES[lang]:<{lang_w}}" +
              "".join(f"{val_map.get(t,'—'):>{col_w}}" for t in header_toks))
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", default=list(LANGUAGES.keys()),
                        choices=list(LANGUAGES.keys()), metavar="LANG")
    parser.add_argument("--vocab-size", type=int, nargs="+", default=[50_000])
    parser.add_argument("--ngram", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--no-bpc", action="store_true",
                        help="Skip BPC computation (much faster)")
    args = parser.parse_args()

    for vocab_size in args.vocab_size:
        print(f"\n{'#'*72}")
        print(f"  VOCAB SIZE: {vocab_size:,}")
        print(f"{'#'*72}")

        all_dl = {}
        for lang in args.langs:
            print(f"\n--- {lang.upper()}: {LANGUAGES[lang]} ---")
            dl_rows, _ = evaluate_language(
                lang, vocab_size, args.ngram, args.no_bpc
            )
            all_dl[lang] = dl_rows or []

        print_cross_language_summary(all_dl, vocab_size)


if __name__ == "__main__":
    main()
