"""Sweep the forgetting ratio omega (memory_out) to test whether stronger
forgetting lets the vocabulary self-regulate below the cap and clears the
dead-weight supra-word tail.

At omega=1e-4 (paper default) forgetting decrements only ~5 units/epoch, which is
negligible against admission, so the vocab pins at vocab_size and dead weight
persists. This sweep raises omega and reports, for each: final vocab size (does
it drop below the cap?), supra ratio, and the never-firing fraction.
"""

import collections
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "python"))
from tokenizers import Tokenizer
from lib_tokenizers.tokenizer import LiBTokenizerFast

DATA = REPO / "experiments" / "data" / "en" / "train.txt"
VOCAB_SIZE, NUM_EPOCHS, DOC_SIZE = 50000, 10000, 50
OMEGAS = [0.0001, 0.001, 0.005, 0.02]


def is_supra(t):
    core = t[1:] if t[:1] == " " else t
    return " " in core


def read_lines(p):
    with open(p, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f if l.strip()]


def main():
    train_lines = read_lines(DATA)
    print(f"{'omega':>8} {'vocab':>8} {'supra%':>7} {'never%':>7} {'train_s':>8}")
    for omega in OMEGAS:
        t0 = time.time()
        tok = LiBTokenizerFast.train_new(
            str(DATA), vocab_size=VOCAB_SIZE, num_epochs=NUM_EPOCHS,
            doc_size=DOC_SIZE, memory_out=omega, seed=0,
        )
        dt = time.time() - t0
        bt = tok.backend_tokenizer          # tokenizers.Tokenizer
        bt.model.use_supra_words = True
        vocab = list(bt.get_vocab().keys())
        supra = [t for t in vocab if is_supra(t)]
        n, ns = len(vocab), len(supra)
        fire = collections.Counter()
        for line in train_lines:
            for t in bt.encode(line).tokens:
                if is_supra(t):
                    fire[t] += 1
        never = sum(1 for s in supra if fire[s] == 0)
        print(f"{omega:>8.4f} {n:>8,} {ns/n:>6.1%} {never/max(ns,1):>6.1%} {dt:>7.0f}s",
              flush=True)
    print("Done.")


if __name__ == "__main__":
    main()
