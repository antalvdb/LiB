"""Fill in the missing omega points of the forgetting sweep, under the FINAL
consensus configuration.

The omega sweep published in paper/forgetting_section.tex (Table tab:omega) was
produced before the forgetting mechanism was wired in the trainer and does not
reproduce (see experiments/results/omega_check_en.log). Two of its four points
are already covered by models trained with the fixed trainer:

    omega=0      -> models/en/lib_50000_noforget  (passive-forgetting ablation)
    omega=0.005  -> models/en/lib_50000           (headline model)

This script trains the two remaining points, omega=1e-4 and omega=2e-2, with
everything else held at the consensus configuration used for the headline
space-delimited models (regenerate_lib.py CONSENSUS, seed 42), then reports
lexicon size, supra-word ratio, and dead weight (never-used supra-word types)
for all four.

Usage
-----
    python experiments/omega_sweep_final.py            # train missing, report all
    python experiments/omega_sweep_final.py --report   # report only, no training
"""

import argparse
import collections
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "python"))

from lib_tokenizers.tokenizer import LiBTokenizerFast
from tokenizers import Tokenizer

DATA = REPO / "experiments" / "data" / "en" / "train.txt"
MODELS = REPO / "experiments" / "models" / "en"

# regenerate_lib.py CONSENSUS, minus memory_out (swept here).
CONSENSUS = dict(num_epochs=5000, doc_size=100, max_len=16, life=50,
                 memory_in=0.25, update_rate=0.4)
VOCAB_SIZE, SEED = 50000, 42

# (omega, model_dir, label) — dirs without a trainer entry already exist.
POINTS = [
    (0.0,    "lib_50000_noforget", "ablation"),
    (1e-4,   "lib_50000_omega1e4", None),
    (5e-3,   "lib_50000",          "headline"),
    (2e-2,   "lib_50000_omega2e2", None),
]
TO_TRAIN = {"lib_50000_omega1e4", "lib_50000_omega2e2"}


def is_supra(t):
    core = t[1:] if t[:1] == " " else t
    return " " in core


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f if l.strip()]


def train_point(omega, out_name):
    out = MODELS / out_name
    if (out / "tokenizer.json").exists():
        print(f"[omega={omega:g}] {out_name} exists, skipping training", flush=True)
        return
    print(f"[omega={omega:g}] training {CONSENSUS} ...", flush=True)
    t0 = time.time()
    tok = LiBTokenizerFast.train_new(str(DATA), vocab_size=VOCAB_SIZE, seed=SEED,
                                     memory_out=omega, **CONSENSUS)
    tok.save_pretrained(str(out))
    print(f"[omega={omega:g}] done in {time.time() - t0:.0f}s -> {out}", flush=True)


def measure(model_dir, lines):
    path = MODELS / model_dir / "tokenizer.json"
    if not path.exists():
        return None
    bt = Tokenizer.from_file(str(path))
    try:
        bt.model.use_supra_words = True
    except Exception:
        pass
    vocab = list(bt.get_vocab().keys())
    supra = [t for t in vocab if is_supra(t)]
    n, ns = len(vocab), len(supra)
    fire = collections.Counter()
    for line in lines:
        for t in bt.encode(line).tokens:
            if is_supra(t):
                fire[t] += 1
    never = sum(1 for s in supra if fire[s] == 0)
    return n, ns / n, never / max(ns, 1), never


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true",
                    help="Skip training; report on whatever models exist")
    args = ap.parse_args()

    if not args.report:
        for omega, name, _ in POINTS:
            if name in TO_TRAIN:
                train_point(omega, name)

    lines = read_lines(DATA)
    print(f"\nEnglish, consensus config (alpha=0.25, Delta=0.4, life=50, "
          f"d=100, max_len=16, 5000 epochs), 50k cap\n")
    print(f"{'omega':>8} {'|L|':>8} {'supra%':>7} {'never%':>7} {'never_n':>8}  note")
    for omega, name, label in POINTS:
        got = measure(name, lines)
        if got is None:
            print(f"{omega:>8g} {'MISSING':>8}  ({name})")
            continue
        n, supra_r, never_r, never_n = got
        print(f"{omega:>8g} {n:>8,} {supra_r:>6.1%} {never_r:>6.1%} "
              f"{never_n:>8,}  {label or ''}")


if __name__ == "__main__":
    main()
