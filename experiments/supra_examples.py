"""Extract the most-used supra-word units per language, for the examples table.

Ranks each language's supra-word types (units whose core contains a space) by
how often they actually fire when the training corpus is re-encoded, so the
examples shown are the ones carrying real work rather than lexicon curiosities.

Usage
-----
    python experiments/supra_examples.py            # top 12 per language
    python experiments/supra_examples.py --top 20
"""

import argparse
import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "python"))

from tokenizers import Tokenizer

DATA = REPO / "experiments" / "data"
MODELS = REPO / "experiments" / "models"
LANGS = ["en", "de", "es", "fi", "tr", "ar", "zh"]


def is_supra(t):
    core = t[1:] if t[:1] == " " else t
    return " " in core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--json", type=str, default=None,
                    help="also write the full ranking to this JSON file")
    args = ap.parse_args()

    dump = {}
    for lang in LANGS:
        path = MODELS / lang / "lib_50000" / "tokenizer.json"
        corpus = DATA / lang / "train.txt"
        if not path.exists() or not corpus.exists():
            print(f"[{lang}] missing model or corpus", flush=True)
            continue
        bt = Tokenizer.from_file(str(path))
        try:
            bt.model.use_supra_words = True
        except Exception:
            pass

        vocab = list(bt.get_vocab().keys())
        supra = {t for t in vocab if is_supra(t)}

        fire = collections.Counter()
        with open(corpus, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                for t in bt.encode(line.rstrip("\n")).tokens:
                    if t in supra:
                        fire[t] += 1

        ranked = fire.most_common()
        total_fires = sum(fire.values())
        dump[lang] = [{"token": t, "count": c} for t, c in ranked]

        print(f"\n=== {lang}: {len(supra):,} supra types, "
              f"{total_fires:,} firings, {sum(1 for t in supra if fire[t]==0):,} never used")
        for t, c in ranked[:args.top]:
            print(f"  {c:>8,}  {t!r}")

    if args.json:
        Path(args.json).write_text(json.dumps(dump, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
