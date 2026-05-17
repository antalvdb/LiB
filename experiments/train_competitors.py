"""Train BPE, WordPiece, and SentencePiece Unigram tokenizers.

All are trained on experiments/data/train.txt at each requested vocab size
and saved under experiments/models/<name>_<vocabsize>/.

Usage:
    python train_competitors.py [--vocab-sizes 10000 32000 50000]
"""

import argparse
import os

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
TRAIN_PATH = os.path.join(DATA_DIR, "train.txt")

VOCAB_SIZES_DEFAULT = [10_000, 32_000, 50_000]


# ---------------------------------------------------------------------------
# BPE  (HuggingFace tokenizers, Metaspace pre-tokeniser)
# ---------------------------------------------------------------------------

def train_bpe(corpus_path: str, vocab_size: int, out_dir: str):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Metaspace

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Metaspace(prepend_scheme="always")
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
        show_progress=True,
    )
    tokenizer.train([corpus_path], trainer)
    os.makedirs(out_dir, exist_ok=True)
    tokenizer.save(os.path.join(out_dir, "tokenizer.json"))
    print(f"  BPE {vocab_size:,} → {out_dir}  (vocab={tokenizer.get_vocab_size()})")


# ---------------------------------------------------------------------------
# WordPiece  (HuggingFace tokenizers, BERT-style ## continuation)
# ---------------------------------------------------------------------------

def train_wordpiece(corpus_path: str, vocab_size: int, out_dir: str):
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece
    from tokenizers.trainers import WordPieceTrainer
    from tokenizers.pre_tokenizers import Whitespace

    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
        show_progress=True,
    )
    tokenizer.train([corpus_path], trainer)
    os.makedirs(out_dir, exist_ok=True)
    tokenizer.save(os.path.join(out_dir, "tokenizer.json"))
    print(f"  WordPiece {vocab_size:,} → {out_dir}  (vocab={tokenizer.get_vocab_size()})")


# ---------------------------------------------------------------------------
# SentencePiece Unigram
# ---------------------------------------------------------------------------

def train_sp_unigram(corpus_path: str, vocab_size: int, out_dir: str):
    import sentencepiece as spm

    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, "sp_unigram")
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=0.9995,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )
    print(f"  SP-Unigram {vocab_size:,} → {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=VOCAB_SIZES_DEFAULT)
    parser.add_argument("--models", nargs="+", default=["bpe", "wordpiece", "sp_unigram"],
                        choices=["bpe", "wordpiece", "sp_unigram"])
    args = parser.parse_args()

    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"{TRAIN_PATH} not found — run prepare_corpus.py first")

    for vocab_size in args.vocab_sizes:
        print(f"\n=== vocab_size={vocab_size:,} ===")
        for model_name in args.models:
            out_dir = os.path.join(MODELS_DIR, f"{model_name}_{vocab_size}")
            if model_name == "bpe":
                train_bpe(TRAIN_PATH, vocab_size, out_dir)
            elif model_name == "wordpiece":
                train_wordpiece(TRAIN_PATH, vocab_size, out_dir)
            elif model_name == "sp_unigram":
                train_sp_unigram(TRAIN_PATH, vocab_size, out_dir)


if __name__ == "__main__":
    main()
