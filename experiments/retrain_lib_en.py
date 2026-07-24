"""Retrain English LiB with the repaired observation-count forgetting, and
compare the resulting vocabulary against the old (broken) model.

The question: does forgetting now fill the vocabulary with *observed* units
instead of the dead-weight supra-word tail? We compare, on the same training
corpus:
  * supra-word ratio,
  * fraction of supra types that never fire,
  * firing concentration (share carried by the top 10% of supra types).
"""

import collections
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "python"))

from tokenizers import Tokenizer
from lib_tokenizers.tokenizer import LiBTokenizerFast

DATA = REPO / "experiments" / "data" / "en"
OLD = REPO / "experiments" / "models" / "en" / "lib_50000"
NEW = REPO / "lib-tokenizer-en-retrained"

VOCAB_SIZE = 50000
NUM_EPOCHS = 10000
DOC_SIZE = 50


def is_supra(tok):
    core = tok[1:] if tok[:1] == " " else tok
    return " " in core


def read_lines(p):
    with open(p, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f if l.strip()]


def analyse(tok_json_path, train_lines, label):
    d = json.load(open(tok_json_path, encoding="utf-8"))
    vocab = [e[0] for e in d["model"]["vocab"]]
    supra = [t for t in vocab if is_supra(t)]
    n, ns = len(vocab), len(supra)

    tk = Tokenizer.from_file(str(tok_json_path))
    tk.model.use_supra_words = True
    fire = collections.Counter()
    for line in train_lines:
        for t in tk.encode(line).tokens:
            if is_supra(t):
                fire[t] += 1
    never = sum(1 for s in supra if fire[s] == 0)
    ranked = sorted(supra, key=lambda s: fire[s], reverse=True)
    total = sum(fire.values()) or 1
    top10 = sum(fire[s] for s in ranked[: max(1, ns // 10)]) / total

    print(f"\n[{label}]")
    print(f"  vocab size          : {n:,}")
    print(f"  supra ratio         : {ns/n:.1%}  ({ns:,} types)")
    print(f"  supra never firing  : {never/max(ns,1):.1%}  ({never:,})")
    print(f"  top-10% supra share : {top10:.1%} of supra firings")
    return {"vocab": n, "supra": ns, "supra_ratio": ns / n,
            "never": never, "top10_share": top10}


def main():
    train_lines = read_lines(DATA / "train.txt")

    print(f"Training LiB (repaired): vocab={VOCAB_SIZE}, epochs={NUM_EPOCHS}, "
          f"doc_size={DOC_SIZE} ...", flush=True)
    t0 = time.time()
    tok = LiBTokenizerFast.train_new(
        str(DATA / "train.txt"), vocab_size=VOCAB_SIZE,
        num_epochs=NUM_EPOCHS, doc_size=DOC_SIZE, seed=0,
    )
    tok.save_pretrained(str(NEW))
    print(f"  trained + saved in {time.time()-t0:.0f}s -> {NEW}", flush=True)

    print("\nAnalysing firing on train corpus (both models) ...", flush=True)
    new = analyse(NEW / "tokenizer.json", train_lines, "REPAIRED (observation-count)")
    if (OLD / "tokenizer.json").exists():
        old = analyse(OLD / "tokenizer.json", train_lines, "OLD (broken prune)")
        print("\n=== change (repaired vs old) ===")
        print(f"  supra ratio  : {old['supra_ratio']:.1%} -> {new['supra_ratio']:.1%}")
        print(f"  never firing : {old['never']:,} -> {new['never']:,}")
        print(f"  top-10% share: {old['top10_share']:.1%} -> {new['top10_share']:.1%}")
    print("\nDone.")


if __name__ == "__main__":
    main()
