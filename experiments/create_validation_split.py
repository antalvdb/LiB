"""Create train_sub.txt / valid.txt splits for hyperparameter search.

Reads experiments/data/{lang}/train.txt (100k lines), holds out the last
VALID_SIZE lines as valid.txt, and writes the remainder as train_sub.txt.
The original train.txt and test.txt are never modified.

Usage
-----
    # All languages
    python create_validation_split.py

    # Specific languages
    python create_validation_split.py --langs en de fi
"""

import argparse
import random
from pathlib import Path

DATA_DIR   = Path(__file__).parent / "data"
VALID_SIZE = 5_000
SEED       = 42

LANGUAGES = ["en", "de", "es", "fi", "tr", "ar", "zh"]


def split_language(lang: str, force: bool = False):
    train_path = DATA_DIR / lang / "train.txt"
    sub_path   = DATA_DIR / lang / "train_sub.txt"
    valid_path = DATA_DIR / lang / "valid.txt"

    if not train_path.exists():
        print(f"  [{lang}] train.txt not found — skipping")
        return

    if sub_path.exists() and valid_path.exists() and not force:
        print(f"  [{lang}] already split — skipping (use --force to redo)")
        return

    with open(train_path, encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]

    rng = random.Random(SEED)
    rng.shuffle(lines)

    valid_lines = lines[:VALID_SIZE]
    train_lines = lines[VALID_SIZE:]

    with open(sub_path, "w", encoding="utf-8") as f:
        for l in train_lines:
            f.write(l + "\n")

    with open(valid_path, "w", encoding="utf-8") as f:
        for l in valid_lines:
            f.write(l + "\n")

    print(f"  [{lang}] {len(train_lines):,} train_sub + {len(valid_lines):,} valid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", default=LANGUAGES, choices=LANGUAGES)
    parser.add_argument("--force", action="store_true",
                        help="Redo split even if files already exist")
    args = parser.parse_args()

    for lang in args.langs:
        split_language(lang, force=args.force)

    print("Done.")


if __name__ == "__main__":
    main()
