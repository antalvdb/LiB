"""Re-evaluate LiB without supra-word lookup after the no-supra fix.

Until tokenizers-fork 60aa7cbf, turning supra-word lookup off also blocked
every unit with a leading space, so "LiB (no supra)" lost all word-initial
units, not only the supra-word ones. This recomputes the CPU metrics of that
variant per language: DL (as in tab:dl_en), 2- and 3-gram KenLM BPC (as in
tab:bpc_en and tab:bpc_multi), and fertility in tokens per UTF-8 byte of the
test set (as in tab:fertility). Full LiB is evaluated alongside as a control:
its numbers must reproduce the paper's.

    python eval_nosupra.py --out results/nosupra_fixed.csv [--langs en zh]
"""

import argparse
import csv
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from compute_metrics import (LiBWrapper, avg_token_length, compute_bpc_kenlm,
                             compute_dl, encode_corpus, train_ngram_lm_kenlm)
from evaluate_all_multilingual import count_supra, read_lines

LANGUAGES = ["en", "de", "es", "fi", "tr", "ar", "zh"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", nargs="*", default=LANGUAGES)
    args = ap.parse_args()

    rows = []
    for lang in args.langs:
        train_lines = read_lines(HERE / "data" / lang / "train.txt")
        test_lines = read_lines(HERE / "data" / lang / "test.txt")
        raw_chars = sum(len(l) for l in test_lines)
        raw_bytes = sum(len(l.encode("utf-8")) for l in test_lines)
        for name, supra in [("LiB", True), ("LiB (no supra)", False)]:
            w = LiBWrapper(str(HERE / "models" / lang / "lib_50000"), use_supra_words=supra)
            vocab = w.vocab()
            flat = encode_corpus(w, test_lines)
            dl = compute_dl(vocab, flat)
            test_tok = [w.encode(l) for l in test_lines if l]
            train_tok = [w.encode(l) for l in train_lines if l]
            row = dict(lang=lang, tokenizer=name, V=len(vocab),
                       avg_len=f"{avg_token_length(flat):.2f}",
                       supra_pct=f"{100 * count_supra(vocab) / len(vocab):.0f}",
                       dl_lex_kb=f"{dl['dl_lex_kb']:.1f}",
                       dl_corp_kb=f"{dl['dl_corpus_kb']:.1f}",
                       dl_tot_kb=f"{dl['dl_total_kb']:.1f}",
                       tokens=sum(map(len, test_tok)),
                       tok_per_byte=f"{sum(map(len, test_tok)) / raw_bytes:.3f}")
            with tempfile.TemporaryDirectory() as tmp:
                for n in (2, 3):
                    lm = train_ngram_lm_kenlm(train_tok, n, tmp_dir=tmp, name=f"{lang}_{supra}_{n}")
                    row[f"bpc{n}"] = f"{compute_bpc_kenlm(lm, test_tok, raw_chars):.3f}"
            rows.append(row)
            print(row, flush=True)

    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


if __name__ == "__main__":
    main()
