"""LiB tokenizer with HuggingFace Transformers integration."""

from typing import Optional, List, Union
from tokenizers import Tokenizer, AddedToken
from tokenizers.models import LiB
from tokenizers.trainers import LiBTrainer
from transformers import PreTrainedTokenizerFast


class LiBTokenizerFast(PreTrainedTokenizerFast):
    """HuggingFace-compatible LiB tokenizer.

    LiB (Less is Better) is a cognitively-inspired tokenizer that builds
    a hierarchical vocabulary of subwords, words, and supra-words.

    Example:
        >>> tokenizer = LiBTokenizerFast.train_new("corpus.txt", vocab_size=32000)
        >>> tokens = tokenizer("The cat sat on the mat")
        >>> tokenizer.save_pretrained("./lib-tokenizer")
        >>> tokenizer = LiBTokenizerFast.from_pretrained("./lib-tokenizer")
    """

    vocab_files_names = {"tokenizer_file": "tokenizer.json"}
    model_type = "lib"

    def __init__(
        self,
        tokenizer_file: Optional[str] = None,
        tokenizer_object: Optional[Tokenizer] = None,
        unk_token: str = "[UNK]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        sep_token: str = "[SEP]",
        mask_token: str = "[MASK]",
        **kwargs,
    ):
        if tokenizer_object is None and tokenizer_file is None:
            tokenizer_object = Tokenizer(LiB())

        super().__init__(
            tokenizer_file=tokenizer_file,
            tokenizer_object=tokenizer_object,
            unk_token=unk_token,
            pad_token=pad_token,
            cls_token=cls_token,
            sep_token=sep_token,
            mask_token=mask_token,
            **kwargs,
        )

    @staticmethod
    def train_new(
        corpus_path: Union[str, List[str]],
        vocab_size: int = 30000,
        num_epochs: int = 10000,
        seed: Optional[int] = None,
        deterministic: bool = False,
        max_len: int = 12,
        memory_in: float = 0.25,
        memory_out: float = 0.0001,
        update_rate: float = 0.2,
        special_tokens: Optional[List[str]] = None,
        **kwargs,
    ) -> "LiBTokenizerFast":
        """Train a new LiB tokenizer from a text corpus.

        Args:
            corpus_path: Path to text file(s) for training.
            vocab_size: Target vocabulary size.
            num_epochs: Number of training epochs.
            seed: Random seed for reproducibility.
            deterministic: Use deterministic training mode.
            max_len: Maximum token length in characters.
            memory_in: Probability of memorizing a candidate (stochastic mode).
            memory_out: Fraction of low-priority units to prune per epoch.
            update_rate: How far units move on reward/punishment.
            special_tokens: Special tokens to add (default: UNK, PAD, CLS, SEP, MASK).

        Returns:
            A trained LiBTokenizerFast instance.
        """
        if special_tokens is None:
            special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]

        added_tokens = [AddedToken(t, special=True) for t in special_tokens]

        trainer_kwargs = dict(
            vocab_size=vocab_size,
            num_epochs=num_epochs,
            max_len=max_len,
            memory_in=memory_in,
            memory_out=memory_out,
            update_rate=update_rate,
            deterministic=deterministic,
            special_tokens=added_tokens,
        )
        if seed is not None:
            trainer_kwargs["seed"] = seed

        trainer = LiBTrainer(**trainer_kwargs)
        tokenizer = Tokenizer(LiB())

        if isinstance(corpus_path, str):
            corpus_path = [corpus_path]

        tokenizer.train(corpus_path, trainer)

        return LiBTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )
