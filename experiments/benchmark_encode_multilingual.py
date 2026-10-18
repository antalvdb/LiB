"""Encoding throughput per language for the four tokenizers in tab:speed.

Protocol (as in the paper): the first 1,000 test lines of each language, one
warm-up pass, then 3 timed passes, one line per encode() call. Baselines are
the 50k models; LiB is the emergent-size model in lib_50000, with supra-word
lookup on and off. Also reports model file size on disk.

    python benchmark_encode_multilingual.py --out results/encode_speed.csv \
        [--label new] [--only LiB] [--langs en zh]

To time an older build of the extension, put a directory containing a
`tokenizers` package with that build's .so first on PYTHONPATH.
"""

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from benchmark_speed import measure_encoding
from compute_metrics import HFTokenizerWrapper, LiBWrapper, SPUnigramWrapper

LANGUAGES = ["en", "de", "es", "fi", "tr", "ar", "zh"]
MODELS = HERE / "models"
DATA = HERE / "data"


def tokenizers_for(lang):
    m = MODELS / lang
    return {
        "BPE": (lambda: HFTokenizerWrapper(str(m / "bpe_50000" / "tokenizer.json")), m / "bpe_50000"),
        "WordPiece": (lambda: HFTokenizerWrapper(str(m / "wordpiece_50000" / "tokenizer.json")), m / "wordpiece_50000"),
        "SP-Unigram": (lambda: SPUnigramWrapper(str(m / "sp_unigram_50000" / "sp_unigram.model")), m / "sp_unigram_50000"),
        "LiB": (lambda: LiBWrapper(str(m / "lib_50000"), use_supra_words=True), m / "lib_50000"),
        "LiB (no supra)": (lambda: LiBWrapper(str(m / "lib_50000"), use_supra_words=False), m / "lib_50000"),
    }


def model_mb(path):
    files = [p for p in path.iterdir() if p.name in ("tokenizer.json", "sp_unigram.model")]
    return sum(p.stat().st_size for p in files) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--only", nargs="*", help="tokenizer names to time")
    ap.add_argument("--langs", nargs="*", default=LANGUAGES)
    ap.add_argument("--lines", type=int, default=1000)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    import tokenizers
    print(f"tokenizers extension: {tokenizers.__file__}", flush=True)

    rows = []
    for lang in args.langs:
        with open(DATA / lang / "test.txt") as f:
            lines = [l.rstrip("\n") for l in f if l.strip()][: args.lines]
        for name, (load, path) in tokenizers_for(lang).items():
            if args.only and name not in args.only:
                continue
            w = load()
            cps, tps = measure_encoding(w.encode, lines, args.reps)
            rows.append(dict(label=args.label, lang=lang, tokenizer=name,
                             mchars_per_s=f"{cps / 1e6:.3f}", mtokens_per_s=f"{tps / 1e6:.3f}",
                             model_mb=f"{model_mb(path):.2f}"))
            print(f"{lang} {name:15s} {cps / 1e6:6.2f} M chars/s  {model_mb(path):5.2f} MB", flush=True)

    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


if __name__ == "__main__":
    main()
