"""Description Length and Bits-per-Character computation.

Description Length formula (Zhikov et al., 2013; Yang et al., 2020):

    DL(total) = DL(lexicon) + DL(corpus)

    DL(lexicon) = -Σ_s  Freq(s) · log₂ P(s)
    DL(corpus)  = -Σ_u  Freq(u) · log₂ P(u)

where s iterates over atomic symbols (characters) in the vocabulary and
u iterates over token types in the encoded corpus.

Bits-per-character (BPC):

    BPC = total_surprisal_bits / total_chars_in_raw_test_text

Surprisal is measured using a Kneser-Ney smoothed n-gram LM trained on the
tokenised training corpus, then evaluated on the tokenised test corpus.
"""

import math
import os
from collections import Counter
from typing import Iterable


# ---------------------------------------------------------------------------
# Tokenizer wrappers — unified encode() interface
# ---------------------------------------------------------------------------

class HFTokenizerWrapper:
    """Wraps a HuggingFace tokenizers.Tokenizer."""

    def __init__(self, tokenizer_path: str):
        from tokenizers import Tokenizer
        self._tok = Tokenizer.from_file(tokenizer_path)

    def vocab(self) -> list[str]:
        """All token strings in the vocabulary, excluding special tokens."""
        v = self._tok.get_vocab()
        specials = {"[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"}
        return [t for t in v if t not in specials]

    def encode(self, text: str) -> list[str]:
        return self._tok.encode(text).tokens


class SPUnigramWrapper:
    """Wraps a SentencePiece Unigram model."""

    def __init__(self, model_path: str):
        import sentencepiece as spm
        self._sp = spm.SentencePieceProcessor()
        self._sp.load(model_path)

    def vocab(self) -> list[str]:
        sp = self._sp
        specials = {"<pad>", "<unk>", "<s>", "</s>"}
        return [sp.id_to_piece(i) for i in range(sp.get_piece_size())
                if sp.id_to_piece(i) not in specials]

    def encode(self, text: str) -> list[str]:
        return self._sp.encode_as_pieces(text)


class LiBWrapper:
    """Wraps a saved LiBTokenizerFast directory."""

    def __init__(self, tokenizer_dir: str, use_supra_words: bool = True):
        from tokenizers import Tokenizer
        self._tok = Tokenizer.from_file(
            os.path.join(tokenizer_dir, "tokenizer.json")
        )
        self._tok.model.use_supra_words = use_supra_words

    def vocab(self) -> list[str]:
        specials = {"[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"}
        return [t for t in self._tok.get_vocab() if t not in specials]

    def encode(self, text: str) -> list[str]:
        return self._tok.encode(text).tokens


class CharTokenizerWrapper:
    """Baseline: one character per token."""

    def vocab(self) -> list[str]:
        return []  # not used for DL(lexicon) in the same way

    def encode(self, text: str) -> list[str]:
        return list(text)


# ---------------------------------------------------------------------------
# Description Length
# ---------------------------------------------------------------------------

def compute_dl(vocab_tokens: list[str], corpus_tokens: list[str]) -> dict:
    """
    Compute Description Length in kilobits.

    vocab_tokens : all token strings in the vocabulary
    corpus_tokens: flat list of token strings produced when encoding the corpus
    """
    # DL(lexicon): entropy of character distribution across the vocabulary
    char_counts: Counter = Counter()
    for tok in vocab_tokens:
        for ch in tok:
            char_counts[ch] += 1
    total_chars = sum(char_counts.values()) or 1
    dl_lex_bits = sum(
        -freq * math.log2(freq / total_chars)
        for freq in char_counts.values()
    )

    # DL(corpus): entropy of token-type distribution in the encoded corpus
    tok_counts: Counter = Counter(corpus_tokens)
    total_toks = sum(tok_counts.values()) or 1
    dl_corp_bits = sum(
        -freq * math.log2(freq / total_toks)
        for freq in tok_counts.values()
    )

    return {
        "dl_lex_kb":    dl_lex_bits  / 1000,
        "dl_corpus_kb": dl_corp_bits / 1000,
        "dl_total_kb":  (dl_lex_bits + dl_corp_bits) / 1000,
    }


def encode_corpus(wrapper, lines: Iterable[str]) -> list[str]:
    """Encode all lines and return a flat list of token strings."""
    tokens = []
    for line in lines:
        line = line.rstrip("\n")
        if line:
            tokens.extend(wrapper.encode(line))
    return tokens


# ---------------------------------------------------------------------------
# Bits-per-character  (n-gram LM)
# ---------------------------------------------------------------------------

def _sentence_ngrams(tokens: list[str], n: int):
    """Yield (context_tuple, word) pairs with <s>/<\s> padding."""
    padded = ["<s>"] * (n - 1) + tokens + ["</s>"]
    for i in range(n - 1, len(padded)):
        yield tuple(padded[i - n + 1 : i]), padded[i]


def train_ngram_lm(tokenized_sents: list[list[str]], n: int):
    """Train a Kneser-Ney interpolated n-gram LM with NLTK."""
    import nltk
    from nltk.lm import KneserNeyInterpolated
    from nltk.lm.preprocessing import padded_everygram_pipeline

    train_data, vocab = padded_everygram_pipeline(n, tokenized_sents)
    lm = KneserNeyInterpolated(n)
    lm.fit(train_data, vocab)
    return lm


def compute_bpc(
    lm,
    tokenized_test: list[list[str]],
    raw_test_chars: int,
    n: int,
) -> float:
    """
    BPC = total surprisal (bits) / total characters in raw test text.

    Uses NLTK logscore() which returns log₂(P(w|context)).
    """
    total_bits = 0.0
    for tokens in tokenized_test:
        for context, word in _sentence_ngrams(tokens, n):
            ls = lm.logscore(word, list(context))
            # logscore returns -inf for unseen events; clip to a floor
            if math.isfinite(ls):
                total_bits -= ls          # surprisal = -log₂(P)
            else:
                total_bits += 20.0        # ~1e-6 probability floor
    return total_bits / max(raw_test_chars, 1)


# ---------------------------------------------------------------------------
# Average token length in characters
# ---------------------------------------------------------------------------

def avg_token_length(corpus_tokens: list[str]) -> float:
    if not corpus_tokens:
        return 0.0
    return sum(len(t) for t in corpus_tokens) / len(corpus_tokens)
