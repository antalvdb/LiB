"""Stream FineWeb-Edu and save train.txt (100k lines) and test.txt (10k lines).

The two splits come from the same dataset shard but are non-overlapping.
Run once before any other experiment script.

Usage:
    python prepare_corpus.py [--train-size 100000] [--test-size 10000]
"""

import argparse
import os
from datasets import load_dataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.txt")
TEST_PATH  = os.path.join(DATA_DIR, "test.txt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=100_000)
    parser.add_argument("--test-size",  type=int, default=10_000)
    args = parser.parse_args()

    total_needed = args.train_size + args.test_size
    print(f"Streaming {total_needed:,} sentences from HuggingFaceFW/fineweb-edu …")

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    os.makedirs(DATA_DIR, exist_ok=True)
    train_count = test_count = 0

    with open(TRAIN_PATH, "w") as f_train, open(TEST_PATH, "w") as f_test:
        for example in dataset:
            for line in example["text"].splitlines():
                line = line.strip()
                if not line:
                    continue
                if train_count < args.train_size:
                    f_train.write(line + "\n")
                    train_count += 1
                elif test_count < args.test_size:
                    f_test.write(line + "\n")
                    test_count += 1
                else:
                    break
            if train_count >= args.train_size and test_count >= args.test_size:
                break

    print(f"Saved {train_count:,} lines → {TRAIN_PATH}")
    print(f"Saved {test_count:,}  lines → {TEST_PATH}")


if __name__ == "__main__":
    main()
