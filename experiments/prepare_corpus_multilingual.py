"""Prepare per-language corpora for multilingual tokenizer experiments.

Downloads Wikipedia via HuggingFace datasets (streaming) for each language,
extracts non-empty lines, and writes:

    experiments/data/{lang}/train.txt   (default 100k lines)
    experiments/data/{lang}/test.txt    (default 10k lines)

Languages
---------
Code  Language  Morphological type       Script
----  --------  -----------------------  ------
en    English   Analytic                 Latin (spaces)
de    German    Fusional + compounding   Latin (spaces)
es    Spanish   Fusional                 Latin (spaces)
fi    Finnish   Agglutinative            Latin (spaces)
tr    Turkish   Agglutinative            Latin (spaces)
ar    Arabic    Templatic/fusional       Arabic (spaces)
zh    Chinese   Isolating/logographic    Han (no word spaces)

Chinese note
------------
Chinese Wikipedia text contains no spaces between words. The script does NOT
add any segmentation; text is written character-by-character as found in the
source. Tokenizers that rely on Metaspace/whitespace pre-tokenization will
operate at the character level for Chinese, which is the standard approach
for byte-level or character-level models. LiB similarly operates over the raw
character stream.

Usage
-----
    # All languages
    python prepare_corpus_multilingual.py

    # Specific languages
    python prepare_corpus_multilingual.py --langs en de fi

    # Custom sizes
    python prepare_corpus_multilingual.py --train-size 50000 --test-size 5000

    # Reuse already-downloaded data (skip languages with existing files)
    python prepare_corpus_multilingual.py --skip-existing
"""

import argparse
import os
import re
import sys
import unicodedata
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

LANGUAGES = {
    "en": ("English",  "Analytic",         "20231101.en"),
    "de": ("German",   "Fusional/compound", "20231101.de"),
    "es": ("Spanish",  "Fusional",          "20231101.es"),
    "fi": ("Finnish",  "Agglutinative",     "20231101.fi"),
    "tr": ("Turkish",  "Agglutinative",     "20231101.tr"),
    "ar": ("Arabic",   "Templatic",         "20231101.ar"),
    "zh": ("Chinese",  "Isolating",         "20231101.zh"),
}

# Minimum number of characters a line must have to be kept.
MIN_LINE_CHARS = 20

# Wikipedia markup patterns to strip before writing lines.
_MARKUP_RE = re.compile(
    r"={2,}[^=]+=+|"          # section headers == Foo ==
    r"\[\[[^\]]*\]\]|"         # wikilinks [[...]]
    r"\{\{[^\}]*\}\}|"         # templates {{...}}
    r"<[^>]+>|"                # HTML tags
    r"https?://\S+"            # bare URLs
)


def _clean(text: str) -> str:
    """Strip Wikipedia markup and normalise whitespace."""
    text = _MARKUP_RE.sub(" ", text)
    # Collapse runs of whitespace (but preserve newlines for line splitting)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _is_mostly_ascii_control(line: str) -> bool:
    """Return True if the line is mostly non-printable or punctuation."""
    printable = sum(1 for c in line if unicodedata.category(c)[0] not in ("C", "Z"))
    return printable < len(line) * 0.5


def stream_lines(lang_code: str, wiki_config: str, train_size: int, test_size: int):
    """Yield cleaned lines from Wikipedia, up to train_size + test_size."""
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Install HuggingFace datasets:  pip install datasets")

    total_needed = train_size + test_size
    yielded = 0

    dataset = load_dataset(
        "wikimedia/wikipedia",
        wiki_config,
        split="train",
        streaming=True,
    )

    for article in dataset:
        text = _clean(article.get("text", ""))
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if len(line) < MIN_LINE_CHARS:
                continue
            if _is_mostly_ascii_control(line):
                continue
            yield line
            yielded += 1
            if yielded >= total_needed:
                return


def prepare_language(
    lang_code: str,
    train_size: int,
    test_size: int,
    skip_existing: bool,
):
    name, morph_type, wiki_config = LANGUAGES[lang_code]
    out_dir = DATA_DIR / lang_code
    train_path = out_dir / "train.txt"
    test_path  = out_dir / "test.txt"

    if skip_existing and train_path.exists() and test_path.exists():
        print(f"  [{lang_code}] {name}: already exists, skipping")
        return

    print(f"  [{lang_code}] {name} ({morph_type}) — streaming {wiki_config} …")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_count = test_count = 0
    with open(train_path, "w", encoding="utf-8") as f_train, \
         open(test_path,  "w", encoding="utf-8") as f_test:
        for line in stream_lines(lang_code, wiki_config, train_size, test_size):
            if train_count < train_size:
                f_train.write(line + "\n")
                train_count += 1
            elif test_count < test_size:
                f_test.write(line + "\n")
                test_count += 1
            else:
                break

    char_count = sum(len(l) for l in open(test_path, encoding="utf-8"))
    print(f"         train: {train_count:,} lines  |  "
          f"test: {test_count:,} lines ({char_count:,} chars)")

    if train_count < train_size:
        print(f"         WARNING: only got {train_count:,} train lines "
              f"(wanted {train_size:,}) — Wikipedia may be small for this language")
    if test_count < test_size:
        print(f"         WARNING: only got {test_count:,} test lines "
              f"(wanted {test_size:,})")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare multilingual Wikipedia corpora for tokenizer evaluation."
    )
    parser.add_argument(
        "--langs", nargs="+", default=list(LANGUAGES.keys()),
        choices=list(LANGUAGES.keys()),
        metavar="LANG",
        help=f"Language codes to prepare (default: all). Choices: {list(LANGUAGES.keys())}",
    )
    parser.add_argument("--train-size", type=int, default=100_000,
                        help="Training lines per language (default: 100000)")
    parser.add_argument("--test-size",  type=int, default=10_000,
                        help="Test lines per language (default: 10000)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip languages whose train.txt and test.txt already exist")
    args = parser.parse_args()

    print(f"Preparing {len(args.langs)} language(s): {', '.join(args.langs)}")
    print(f"Target: {args.train_size:,} train + {args.test_size:,} test lines each\n")

    for lang_code in args.langs:
        prepare_language(lang_code, args.train_size, args.test_size, args.skip_existing)

    print("\nDone. Data written to experiments/data/{lang}/")
    print("Next: python experiments/train_competitors_multilingual.py")
    print("      python experiments/evaluate_all_multilingual.py")


if __name__ == "__main__":
    main()
