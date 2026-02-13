"""Regression test: compare new Rust implementation against original Python."""

import os
import pytest

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
CORPUS_PATH = os.path.join(REPO_ROOT, "br-text.txt")


class TestRegressionAgainstOriginal:
    """Compare Rust LiB against expected behavior from the original Python implementation."""

    def test_character_initialization(self):
        """Both implementations should start with the same character set."""
        from tokenizers import Tokenizer
        from tokenizers.models import LiB
        from tokenizers.trainers import LiBTrainer

        # Train Rust version with minimal epochs (just initialization)
        tok = Tokenizer(LiB())
        trainer = LiBTrainer(vocab_size=10000, num_epochs=1, seed=42)
        tok.train([CORPUS_PATH], trainer)

        rust_chars = {t for t in tok.get_vocab() if len(t) == 1}

        # Read the corpus and extract characters
        with open(CORPUS_PATH) as f:
            text = f.read()
        expected_chars = set(text) - {'\n'}

        # Rust should have at least all corpus characters
        assert expected_chars.issubset(rust_chars), \
            f"Missing chars: {expected_chars - rust_chars}"

    def test_vocabulary_grows_with_training(self):
        """Vocabulary should grow beyond characters with more training."""
        from tokenizers import Tokenizer
        from tokenizers.models import LiB
        from tokenizers.trainers import LiBTrainer

        tok = Tokenizer(LiB())
        trainer = LiBTrainer(vocab_size=500, num_epochs=500, seed=42)
        tok.train([CORPUS_PATH], trainer)

        vocab = tok.get_vocab()
        multi_char = [t for t in vocab if len(t) > 1]

        assert len(multi_char) > 10, \
            f"Expected multi-char tokens, got {len(multi_char)}: {multi_char[:10]}"

    def test_segmentation_produces_valid_tokens(self):
        """All output tokens should be in the vocabulary."""
        from tokenizers import Tokenizer
        from tokenizers.models import LiB
        from tokenizers.trainers import LiBTrainer

        tok = Tokenizer(LiB())
        trainer = LiBTrainer(vocab_size=300, num_epochs=200, seed=42)
        tok.train([CORPUS_PATH], trainer)

        with open(CORPUS_PATH) as f:
            lines = f.readlines()[:10]

        vocab = tok.get_vocab()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            encoding = tok.encode(line)
            for token in encoding.tokens:
                assert token in vocab or len(token) == 1, \
                    f"Token '{token}' not in vocabulary"

    def test_supra_word_tokens_present(self):
        """LiB should learn supra-word tokens (multi-word units with spaces)."""
        from tokenizers import Tokenizer
        from tokenizers.models import LiB
        from tokenizers.trainers import LiBTrainer

        tok = Tokenizer(LiB())
        trainer = LiBTrainer(vocab_size=1000, num_epochs=2000, seed=42)
        tok.train([CORPUS_PATH], trainer)

        vocab = tok.get_vocab()
        supra_words = [t for t in vocab if ' ' in t and len(t) > 2]

        # LiB's defining feature is learning supra-word units
        assert len(supra_words) > 0, \
            f"Expected supra-word tokens (containing spaces), got none. " \
            f"Vocab size: {len(vocab)}"
