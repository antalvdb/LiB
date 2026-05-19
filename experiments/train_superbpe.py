"""Train SuperBPE tokenizers for all languages/vocab sizes.

Two-stage training (mirrors the SuperBPE paper):
  Stage 1: BPE with whitespace pretokenization → standard subword merges
  Stage 2: Resume without whitespace constraint → adds cross-word "superword" merges

Saves tokenizer.json to experiments/models/{lang}/superbpe_{vocab_size}/.

Must be run in the superbpe conda env:
    conda run -n superbpe python experiments/train_superbpe.py
    conda run -n superbpe python experiments/train_superbpe.py --langs en --vocab-sizes 50000
    conda run -n superbpe python experiments/train_superbpe.py --skip-existing
"""

import os
import json
import shutil
import time
import argparse
from pathlib import Path

# Custom tokenizers fork (superbpe conda env required)
from tokenizers import Tokenizer, pre_tokenizers, Regex
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel, Split
from tokenizers.trainers import BpeTrainer

DATA_DIR   = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
TMP_DIR    = Path(__file__).parent / "superbpe_tmp"

LANGUAGES   = ["en", "de", "es", "fi", "tr", "ar", "zh"]
VOCAB_SIZES = [10_000, 32_000, 50_000]

# Stage 1 regex: standard word-level splits (same as SuperBPE paper)
STAGE1_REGEX = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

# Stage 2 regex: minimal constraints — allows cross-whitespace merges
STAGE2_REGEX = r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)"


def _build_tokenizer(regex_string: str) -> tuple:
    tokenizer = Tokenizer(BPE())
    trainer = BpeTrainer(show_progress=True)
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        Split(pattern=Regex(regex_string), behavior="isolated", invert=False),
        ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
    ])
    return tokenizer, trainer


def train_stage1(text_file: str, out_dir: Path, vocab_size: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    orig = os.getcwd()
    os.chdir(out_dir)
    try:
        tokenizer, trainer = _build_tokenizer(STAGE1_REGEX)
        trainer.vocab_size = vocab_size
        tokenizer.train([text_file], trainer)
        tokenizer.model.save(".")   # writes merges.txt + vocab.json
        tokenizer.save("tokenizer.json")
    finally:
        os.chdir(orig)


def train_stage2(text_file: str, stage1_dir: Path, out_dir: Path, vocab_size: int,
                 num_inherit_merges: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    orig = os.getcwd()
    os.chdir(out_dir)
    try:
        # Copy the first N stage-1 merges; the custom BpeTrainer reads merges.txt
        # from CWD to initialise its merge table before continuing training.
        merges_src = stage1_dir / "merges.txt"
        with open(merges_src) as f:
            lines = f.readlines()
        header = [l for l in lines if l.startswith("#")]
        merges = [l for l in lines if not l.startswith("#") and l.strip()]
        with open("merges.txt", "w") as f:
            f.writelines(header)
            f.writelines(merges[:num_inherit_merges])

        tokenizer, trainer = _build_tokenizer(STAGE2_REGEX)
        trainer.vocab_size = vocab_size
        tokenizer.train([text_file], trainer)
        tokenizer.model.save(".")
        tokenizer.save("tokenizer.json")
    finally:
        os.chdir(orig)


def train_superbpe(lang: str, vocab_size: int, skip_existing: bool = False):
    train_path = DATA_DIR / lang / "train.txt"
    if not train_path.exists():
        print(f"  [{lang}] Missing {train_path}")
        return

    out_dir = MODELS_DIR / lang / f"superbpe_{vocab_size}"
    if skip_existing and (out_dir / "tokenizer.json").exists():
        print(f"  [{lang} superbpe {vocab_size:>6,}]  already exists, skipping")
        return

    stage1_dir = TMP_DIR / lang / str(vocab_size) / "stage1"
    stage2_dir = TMP_DIR / lang / str(vocab_size) / "stage2"

    print(f"  [{lang} superbpe {vocab_size:>6,}]  stage 1 ...", end="", flush=True)
    t0 = time.perf_counter()
    train_stage1(str(train_path), stage1_dir, vocab_size)
    t1 = time.perf_counter()
    print(f" {t1-t0:.0f}s  stage 2 ...", end="", flush=True)

    # Inherit half the vocab as subword merges, add the rest as superword merges
    num_inherit = vocab_size // 2
    train_stage2(str(train_path), stage1_dir, stage2_dir, vocab_size, num_inherit)
    t2 = time.perf_counter()
    print(f" {t2-t1:.0f}s  total {t2-t0:.0f}s")

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(stage2_dir / "tokenizer.json", out_dir / "tokenizer.json")
    print(f"    → {out_dir}/tokenizer.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", default=LANGUAGES,
                        choices=LANGUAGES, metavar="LANG")
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=VOCAB_SIZES)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    for lang in args.langs:
        print(f"\n{'='*60}\n  Language: {lang}\n{'='*60}")
        for vocab_size in args.vocab_sizes:
            train_superbpe(lang, vocab_size, skip_existing=args.skip_existing)

    print("\nAll SuperBPE training complete.")


if __name__ == "__main__":
    main()
