"""Random hyperparameter search for LiB tokenizer, optimizing 3-gram BPC.

For each trial:
  1. Sample a random configuration from the search space.
  2. Train LiB on experiments/data/{lang}/train_sub.txt.
  3. Evaluate 3-gram BPC on experiments/data/{lang}/valid.txt via KenLM.
  4. Append the result to a CSV.

Run create_validation_split.py first to generate train_sub.txt / valid.txt.

Usage
-----
    # 30 trials, English, 50k vocab
    python hparam_search.py --lang en

    # 50 trials, Turkish, 50k vocab
    python hparam_search.py --lang tr --n-trials 50

    # All languages, 20 trials each (runs sequentially)
    python hparam_search.py --lang en de fi tr es ar zh --n-trials 20

    # Resume: skips configs already recorded in the CSV
    python hparam_search.py --lang en --resume
"""

import argparse
import csv
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR   = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

sys.path.insert(0, str(REPO_ROOT))

from experiments.compute_metrics import train_ngram_lm_kenlm, compute_bpc_kenlm

LANGUAGES = ["en", "de", "es", "fi", "tr", "ar", "zh"]

# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

SEARCH_SPACE = {
    "num_epochs":  [10_000, 25_000, 50_000, 100_000, 200_000],
    "max_len":     [8, 10, 12, 16, 20, 25],
    "life":        [5, 10, 20, 50, 100],
    "memory_in":   [0.05, 0.1, 0.2, 0.25, 0.4, 0.5, 0.75],
    "memory_out":  [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
    "update_rate": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
}

CSV_FIELDS = [
    "trial", "lang", "vocab_size",
    "num_epochs", "max_len", "life",
    "memory_in", "memory_out", "update_rate",
    "vocab_actual", "bpc_3gram", "train_time_s",
]

DEFAULTS = {
    "num_epochs":  10_000,
    "max_len":     12,
    "life":        10,
    "memory_in":   0.25,
    "memory_out":  1e-4,
    "update_rate": 0.2,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_lines(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f if l.strip()]


def sample_config(rng: random.Random) -> dict:
    return {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}


def load_existing_configs(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def config_key(cfg: dict) -> tuple:
    return tuple(float(cfg[k]) for k in sorted(SEARCH_SPACE))


def train_lib(train_path: Path, vocab_size: int, cfg: dict, tmp_dir: str) -> tuple[str, int]:
    """Train LiB and return (tokenizer_dir, actual_vocab_size)."""
    from lib_tokenizers import LiBTokenizerFast

    out_dir = Path(tmp_dir) / "lib_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = LiBTokenizerFast.train_new(
        str(train_path),
        vocab_size=vocab_size,
        num_epochs=cfg["num_epochs"],
        max_len=cfg["max_len"],
        life=cfg["life"],
        memory_in=cfg["memory_in"],
        memory_out=cfg["memory_out"],
        update_rate=cfg["update_rate"],
        seed=42,
    )
    tok.save_pretrained(str(out_dir))
    vocab_actual = tok.vocab_size
    return str(out_dir), vocab_actual


def evaluate_bpc(tokenizer_dir: str, train_lines: list[str],
                 valid_lines: list[str], raw_valid_chars: int,
                 tmp_dir: str, tag: str) -> float:
    from experiments.compute_metrics import LiBWrapper

    wrapper = LiBWrapper(tokenizer_dir, use_supra_words=True)
    tokenized_train = [wrapper.encode(l) for l in train_lines if l]
    tokenized_valid = [wrapper.encode(l) for l in valid_lines if l]

    lm = train_ngram_lm_kenlm(tokenized_train, n=3, tmp_dir=tmp_dir, name=tag)
    return compute_bpc_kenlm(lm, tokenized_valid, raw_valid_chars)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_search(lang: str, vocab_size: int, n_trials: int, resume: bool,
               seed: int, verbose: bool):
    train_path = DATA_DIR / lang / "train_sub.txt"
    valid_path = DATA_DIR / lang / "valid.txt"

    if not train_path.exists() or not valid_path.exists():
        print(f"[{lang}] Missing train_sub.txt or valid.txt — run create_validation_split.py first")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"hparam_{lang}_{vocab_size}.csv"

    existing = load_existing_configs(csv_path)
    seen_keys = {config_key({k: row[k] for k in SEARCH_SPACE}) for row in existing}
    next_trial = len(existing) + 1

    train_lines = read_lines(train_path)
    valid_lines = read_lines(valid_path)
    raw_valid_chars = sum(len(l) for l in valid_lines)

    rng = random.Random(seed)

    write_header = not csv_path.exists()
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()

    # Always include the default config as trial 0 if not already done
    configs_to_run = []
    if not any(
        all(float(row[k]) == float(DEFAULTS[k]) for k in SEARCH_SPACE)
        for row in existing
    ):
        configs_to_run.append(DEFAULTS.copy())

    trials_done = 0
    while trials_done < n_trials:
        cfg = configs_to_run.pop(0) if configs_to_run else sample_config(rng)
        key = config_key(cfg)

        if resume and key in seen_keys:
            continue

        seen_keys.add(key)

        tag = f"{lang}_{vocab_size}_t{next_trial}"
        print(
            f"  Trial {next_trial:3d} | epochs={cfg['num_epochs']:>7,} "
            f"max_len={cfg['max_len']:>2} life={cfg['life']:>3} "
            f"mem_in={cfg['memory_in']:.3f} mem_out={cfg['memory_out']:.5f} "
            f"lr={cfg['update_rate']:.3f}",
            end="", flush=True,
        )

        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="lib_hps_") as tmp_dir:
            try:
                tok_dir, vocab_actual = train_lib(train_path, vocab_size, cfg, tmp_dir)
                bpc = evaluate_bpc(tok_dir, train_lines, valid_lines, raw_valid_chars, tmp_dir, tag)
            except Exception as e:
                print(f"  ERROR: {e}")
                trials_done += 1
                next_trial += 1
                continue

        elapsed = time.perf_counter() - t0
        print(f"  →  BPC={bpc:.4f}  vocab={vocab_actual:,}  ({elapsed:.0f}s)")

        row = {
            "trial": next_trial,
            "lang": lang,
            "vocab_size": vocab_size,
            **cfg,
            "vocab_actual": vocab_actual,
            "bpc_3gram": round(bpc, 6),
            "train_time_s": round(elapsed, 1),
        }
        writer.writerow(row)
        csv_file.flush()

        trials_done += 1
        next_trial += 1

    csv_file.close()

    # Print best found so far
    all_rows = load_existing_configs(csv_path)
    if all_rows:
        best = min(all_rows, key=lambda r: float(r["bpc_3gram"]))
        print(f"\n  Best so far: BPC={best['bpc_3gram']}  "
              f"epochs={best['num_epochs']} max_len={best['max_len']} "
              f"life={best['life']} mem_in={best['memory_in']} "
              f"mem_out={best['memory_out']} lr={best['update_rate']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", nargs="+", default=["en"],
                        choices=LANGUAGES, metavar="LANG")
    parser.add_argument("--vocab-size", type=int, default=50_000)
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Number of random configs to try per language")
    parser.add_argument("--resume", action="store_true",
                        help="Skip configs already in the CSV")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for config sampling")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    for lang in args.lang:
        print(f"\n{'='*72}")
        print(f"  Hyperparameter search: {lang}  vocab={args.vocab_size:,}  "
              f"n_trials={args.n_trials}")
        print(f"{'='*72}")
        run_search(
            lang=lang,
            vocab_size=args.vocab_size,
            n_trials=args.n_trials,
            resume=args.resume,
            seed=args.seed,
            verbose=args.verbose,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
