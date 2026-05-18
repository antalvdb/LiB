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

def train_ngram_lm(tokenized_sents: list[list[str]], n: int):
    """Train a Kneser-Ney interpolated n-gram LM with NLTK (slow fallback)."""
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
    """BPC using NLTK LM (slow; prefer compute_bpc_kenlm)."""
    def _sentence_ngrams(tokens, n):
        padded = ["<s>"] * (n - 1) + tokens + ["</s>"]
        for i in range(n - 1, len(padded)):
            yield tuple(padded[i - n + 1 : i]), padded[i]

    total_bits = 0.0
    for tokens in tokenized_test:
        for context, word in _sentence_ngrams(tokens, n):
            ls = lm.logscore(word, list(context))
            if math.isfinite(ls):
                total_bits -= ls
            else:
                total_bits += 20.0
    return total_bits / max(raw_test_chars, 1)


# ---------------------------------------------------------------------------
# KenLM-based BPC  (fast — requires lmplz binary and kenlm Python package)
# ---------------------------------------------------------------------------

import shutil
import subprocess
import tempfile

LOG10_TO_LOG2 = math.log2(10)   # multiply log10 probs by this to get bits


def _write_tokenized(tokenized_sents: list[list[str]], path: str):
    """Write one sentence of space-joined tokens per line."""
    with open(path, "w", encoding="utf-8") as f:
        for sent in tokenized_sents:
            if sent:
                f.write(" ".join(sent) + "\n")


def train_ngram_lm_kenlm(
    tokenized_sents: list[list[str]],
    n: int,
    tmp_dir: str,
    name: str = "lm",
):
    """
    Build a Kneser-Ney n-gram LM with lmplz and return a kenlm.Model.

    Writes temporary files under tmp_dir.
    """
    import kenlm

    lmplz = shutil.which("lmplz")
    if lmplz is None:
        raise RuntimeError("lmplz not found on PATH — install KenLM")

    txt_path  = os.path.join(tmp_dir, f"{name}.txt")
    arpa_path = os.path.join(tmp_dir, f"{name}.arpa")

    _write_tokenized(tokenized_sents, txt_path)

    with open(txt_path, "rb") as fin, open(arpa_path, "w", encoding="utf-8") as fout:
        subprocess.run(
            [lmplz, "-o", str(n), "--discount_fallback"],
            stdin=fin, stdout=fout, stderr=subprocess.DEVNULL, check=True,
        )

    return kenlm.Model(arpa_path)


def compute_bpc_kenlm(
    model,                          # kenlm.Model
    tokenized_test: list[list[str]],
    raw_test_chars: int,
) -> float:
    """
    BPC = total surprisal (bits) / raw character count of test text.

    kenlm full_scores() yields (log10_prob, ngram_length, oov) per token,
    already conditioned on sentence-boundary context.
    """
    total_bits = 0.0
    for tokens in tokenized_test:
        if not tokens:
            continue
        sentence = " ".join(tokens)
        for log10_prob, _, _ in model.full_scores(sentence, bos=True, eos=True):
            total_bits -= log10_prob * LOG10_TO_LOG2   # surprisal in bits
    return total_bits / max(raw_test_chars, 1)


# ---------------------------------------------------------------------------
# Average token length in characters
# ---------------------------------------------------------------------------

def avg_token_length(corpus_tokens: list[str]) -> float:
    if not corpus_tokens:
        return 0.0
    return sum(len(t) for t in corpus_tokens) / len(corpus_tokens)
