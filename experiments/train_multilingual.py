"""Train all tokenizers on all prepared language corpora.

Expects experiments/data/{lang}/train.txt for each language.
Saves models to experiments/models/{lang}/{tokenizer}_{vocab_size}/.

Usage
-----
    # All languages, all vocab sizes
    python train_multilingual.py

    # Specific languages / sizes
    python train_multilingual.py --langs de fi --vocab-sizes 50000

    # Skip already-trained models
    python train_multilingual.py --skip-existing

    # LiB only (competitors already trained)
    python train_multilingual.py --only lib
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR   = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"

sys.path.insert(0, str(REPO_ROOT))

LANGUAGES  = ["en", "de", "es", "fi", "tr", "ar", "zh"]
TOKENIZERS = ["bpe", "wordpiece", "sp_unigram", "lib"]


def fmt_time(s: float) -> str:
    return f"{s:.0f}s" if s < 60 else f"{s/60:.1f}min"


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def train_bpe(train_path: str, vocab_size: int, out_dir: Path):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Metaspace

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Metaspace(prepend_scheme="always")
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
        show_progress=False,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.train([train_path], trainer)
    tokenizer.save(str(out_dir / "tokenizer.json"))


def train_wordpiece(train_path: str, vocab_size: int, out_dir: Path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece
    from tokenizers.trainers import WordPieceTrainer
    from tokenizers.pre_tokenizers import Whitespace

    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
        show_progress=False,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.train([train_path], trainer)
    tokenizer.save(str(out_dir / "tokenizer.json"))


def train_sp_unigram(train_path: str, vocab_size: int, out_dir: Path):
    import sentencepiece as spm

    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(out_dir / "sp_unigram")
    spm.SentencePieceTrainer.train(
        input=train_path,
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=0.9995,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        input_sentence_size=200_000,   # cap to avoid OOM on large corpora
        shuffle_input_sentence=True,
    )


def train_lib(train_path: str, vocab_size: int, out_dir: Path):
    from lib_tokenizers import LiBTokenizerFast

    out_dir.mkdir(parents=True, exist_ok=True)
    tok = LiBTokenizerFast.train_new(
        train_path,
        vocab_size=vocab_size,
        num_epochs=10_000,
    )
    tok.save_pretrained(str(out_dir))


TRAIN_FNS = {
    "bpe":        train_bpe,
    "wordpiece":  train_wordpiece,
    "sp_unigram": train_sp_unigram,
    "lib":        train_lib,
}

MODEL_FILE = {
    "bpe":        "tokenizer.json",
    "wordpiece":  "tokenizer.json",
    "sp_unigram": "sp_unigram.model",
    "lib":        "tokenizer.json",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", default=LANGUAGES,
                        choices=LANGUAGES, metavar="LANG")
    parser.add_argument("--vocab-sizes", type=int, nargs="+",
                        default=[10_000, 32_000, 50_000])
    parser.add_argument("--only", nargs="+", default=TOKENIZERS,
                        choices=TOKENIZERS, metavar="TOK",
                        help="Train only these tokenizer types")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip if model file already exists")
    args = parser.parse_args()

    for lang in args.langs:
        train_path = DATA_DIR / lang / "train.txt"
        if not train_path.exists():
            print(f"[{lang}] Missing {train_path} — run prepare_corpus_multilingual.py first")
            continue

        print(f"\n{'='*60}")
        print(f"  Language: {lang}  ({train_path})")
        print(f"{'='*60}")

        for vocab_size in args.vocab_sizes:
            for tok_name in args.only:
                out_dir = MODELS_DIR / lang / f"{tok_name}_{vocab_size}"
                sentinel = out_dir / MODEL_FILE[tok_name]

                if args.skip_existing and sentinel.exists():
                    print(f"  [{tok_name:12s} {vocab_size:>6,}]  already exists, skipping")
                    continue

                print(f"  [{tok_name:12s} {vocab_size:>6,}]  training …", end="", flush=True)
                t0 = time.perf_counter()
                try:
                    TRAIN_FNS[tok_name](str(train_path), vocab_size, out_dir)
                    elapsed = time.perf_counter() - t0
                    print(f"  done in {fmt_time(elapsed)}")
                except Exception as exc:
                    elapsed = time.perf_counter() - t0
                    print(f"  FAILED after {fmt_time(elapsed)}: {exc}")

    print("\nAll training complete.")
    print("Next: python experiments/evaluate_all_multilingual.py")


if __name__ == "__main__":
    main()
