"""End-to-end integration tests for LiB tokenizer."""

import os
import tempfile
import pytest
from tokenizers import Tokenizer
from tokenizers.decoders import Fuse
from tokenizers.models import LiB
from tokenizers.trainers import LiBTrainer
from lib_tokenizers import LiBTokenizerFast


CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "br-text.txt")


class TestLiBModel:
    """Test the Rust LiB model directly via tokenizers."""

    def test_create_empty_model(self):
        tok = Tokenizer(LiB())
        assert tok.get_vocab_size() == 0

    def test_create_model_with_vocab(self):
        tok = Tokenizer(LiB(vocab={"hello": 0, "world": 1}))
        assert tok.get_vocab_size() == 2

    def test_train_small_corpus(self):
        tok = Tokenizer(LiB())
        trainer = LiBTrainer(vocab_size=100, num_epochs=50, seed=42)
        tok.train([CORPUS_PATH], trainer)
        assert tok.get_vocab_size() > 0

    def test_encode_decode_roundtrip(self):
        tok = Tokenizer(LiB())
        tok.decoder = Fuse()
        trainer = LiBTrainer(vocab_size=200, num_epochs=100, seed=42)
        tok.train([CORPUS_PATH], trainer)

        text = "the cat sat on the mat"
        encoding = tok.encode(text)
        decoded = tok.decode(encoding.ids)
        assert decoded == text

    def test_deterministic_training(self):
        def train_with_seed(seed):
            tok = Tokenizer(LiB())
            trainer = LiBTrainer(vocab_size=100, num_epochs=50, seed=seed)
            tok.train([CORPUS_PATH], trainer)
            return tok.get_vocab()

        v1 = train_with_seed(42)
        v2 = train_with_seed(42)
        assert v1 == v2

    def test_save_and_load_tokenizer_json(self):
        tok = Tokenizer(LiB())
        trainer = LiBTrainer(vocab_size=100, num_epochs=50, seed=42)
        tok.train([CORPUS_PATH], trainer)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tokenizer.json")
            tok.save(path)
            loaded = Tokenizer.from_file(path)
            assert loaded.get_vocab_size() == tok.get_vocab_size()

            text = "the cat"
            assert tok.encode(text).ids == loaded.encode(text).ids


class TestLiBTokenizerFast:
    """Test the HuggingFace wrapper."""

    @pytest.fixture
    def trained_tokenizer(self):
        return LiBTokenizerFast.train_new(
            CORPUS_PATH,
            vocab_size=200,
            num_epochs=100,
            seed=42,
        )

    def test_train_new(self, trained_tokenizer):
        assert trained_tokenizer.vocab_size > 0

    def test_encode(self, trained_tokenizer):
        tokens = trained_tokenizer("the cat sat on the mat")
        assert "input_ids" in tokens
        assert "attention_mask" in tokens
        assert len(tokens["input_ids"]) > 0

    def test_decode(self, trained_tokenizer):
        tokens = trained_tokenizer("hello world")
        decoded = trained_tokenizer.decode(tokens["input_ids"])
        assert isinstance(decoded, str)

    def test_save_and_load(self, trained_tokenizer):
        with tempfile.TemporaryDirectory() as tmpdir:
            trained_tokenizer.save_pretrained(tmpdir)
            loaded = LiBTokenizerFast.from_pretrained(tmpdir)
            assert loaded.vocab_size == trained_tokenizer.vocab_size

            # Same encoding after reload
            text = "the cat"
            orig = trained_tokenizer(text)["input_ids"]
            reloaded = loaded(text)["input_ids"]
            assert orig == reloaded

    def test_special_tokens(self, trained_tokenizer):
        assert trained_tokenizer.unk_token == "[UNK]"
        assert trained_tokenizer.pad_token == "[PAD]"
        assert trained_tokenizer.cls_token == "[CLS]"
        assert trained_tokenizer.sep_token == "[SEP]"

    def test_batch_encoding(self, trained_tokenizer):
        texts = ["hello", "world", "the cat sat"]
        batch = trained_tokenizer(texts, padding=True)
        assert len(batch["input_ids"]) == 3
        # All should be padded to same length
        lengths = [len(ids) for ids in batch["input_ids"]]
        assert len(set(lengths)) == 1

    def test_tokenizer_json_format(self, trained_tokenizer):
        with tempfile.TemporaryDirectory() as tmpdir:
            trained_tokenizer.save_pretrained(tmpdir)
            import json
            with open(os.path.join(tmpdir, "tokenizer.json")) as f:
                data = json.load(f)
            assert data["model"]["type"] == "LiB"
