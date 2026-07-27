"""Benchmark training speed, encoding throughput, and memory for all tokenizers.

Measures
--------
Training
  - Wall-clock time to train from train.txt at each vocab size
  - Peak RSS memory increase during training (MB)

Encoding
  - Throughput: characters per second and tokens per second on test.txt
  - Tested with N_REPS repetitions to stabilise the estimate

Model footprint
  - Size of the saved model files on disk (MB)

Usage
-----
    python benchmark_speed.py [--vocab-sizes 10000 32000 50000] [--no-train]
                              [--encode-reps 3]

Notes
-----
Memory numbers are process-RSS deltas, which include Python/OS overhead and
are approximate.  For a cleaner reading run each tokenizer in its own process
(use --single <name>) and compare baseline RSS.

Training is re-run from scratch even if the model already exists, so you get
a clean timing.  Pass --no-train to skip training and report only encoding
speed for already-saved models.
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import psutil

REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR   = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
LIB_DIR    = REPO_ROOT / "lib-tokenizer-trained"
TRAIN_PATH = DATA_DIR / "train.txt"
TEST_PATH  = DATA_DIR / "test.txt"
TRAIN_STR  = str(TRAIN_PATH)

sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def peak_rss_mb(fn):
    """Call fn(), return (result, peak_delta_mb)."""
    gc.collect()
    baseline = rss_mb()
    peak = baseline
    result = None

    import threading

    def poll():
        nonlocal peak
        while not done:
            peak = max(peak, rss_mb())
            time.sleep(0.05)

    done = False
    t = threading.Thread(target=poll, daemon=True)
    t.start()
    try:
        result = fn()
    finally:
        done = True
        t.join()

    return result, max(peak - baseline, 0.0)


def read_lines(path) -> list[str]:
    with open(path) as f:
        return [l.rstrip("\n") for l in f if l.strip()]


def dir_size_mb(path) -> float:
    total = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
    return total / 1024 / 1024


# ---------------------------------------------------------------------------
# Training wrappers  (return the wrapper object + timing + memory)
# ---------------------------------------------------------------------------

def train_bpe(vocab_size: int, out_dir: Path):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Metaspace

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Metaspace(prepend_scheme="always")
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
        show_progress=False,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    def _train():
        tokenizer.train([TRAIN_STR], trainer)
        tokenizer.save(str(out_dir / "tokenizer.json"))

    t0 = time.perf_counter()
    _, mem_mb = peak_rss_mb(_train)
    elapsed = time.perf_counter() - t0
    return tokenizer, elapsed, mem_mb


def train_wordpiece(vocab_size: int, out_dir: Path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece
    from tokenizers.trainers import WordPieceTrainer
    from tokenizers.pre_tokenizers import Whitespace

    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
        show_progress=False,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    def _train():
        tokenizer.train([TRAIN_STR], trainer)
        tokenizer.save(str(out_dir / "tokenizer.json"))

    t0 = time.perf_counter()
    _, mem_mb = peak_rss_mb(_train)
    elapsed = time.perf_counter() - t0
    return tokenizer, elapsed, mem_mb


def train_sp_unigram(vocab_size: int, out_dir: Path):
    import sentencepiece as spm
    from experiments.compute_metrics import SPUnigramWrapper

    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(out_dir / "sp_unigram")

    def _train():
        spm.SentencePieceTrainer.train(
            input=TRAIN_STR,
            model_prefix=prefix,
            vocab_size=vocab_size,
            model_type="unigram",
            character_coverage=0.9995,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        )

    t0 = time.perf_counter()
    _, mem_mb = peak_rss_mb(_train)
    elapsed = time.perf_counter() - t0
    wrapper = SPUnigramWrapper(prefix + ".model")
    return wrapper, elapsed, mem_mb


def train_lib(vocab_size: int, out_dir: Path):
    from lib_tokenizers import LiBTokenizerFast

    out_dir.mkdir(parents=True, exist_ok=True)

    def _train():
        tok = LiBTokenizerFast.train_new(
            TRAIN_STR,
            vocab_size=vocab_size,
            num_epochs=10_000,
        )
        tok.save_pretrained(str(out_dir))
        return tok

    t0 = time.perf_counter()
    (tok, mem_mb) = peak_rss_mb(_train)  # returns (tok, peak_mb)
    elapsed = time.perf_counter() - t0
    # Wrap it
    from experiments.compute_metrics import LiBWrapper
    wrapper = LiBWrapper(str(out_dir))
    return wrapper, elapsed, mem_mb


# ---------------------------------------------------------------------------
# Encoding throughput
# ---------------------------------------------------------------------------

def measure_encoding(encode_fn, lines: list[str], n_reps: int = 3):
    """
    Returns (chars_per_sec, tokens_per_sec).
    n_reps full passes over lines; first pass is a warm-up (not counted).
    """
    total_chars = sum(len(l) for l in lines)

    # Warm-up
    for l in lines:
        encode_fn(l)

    gc.collect()
    t0 = time.perf_counter()
    total_tokens = 0
    for _ in range(n_reps):
        for l in lines:
            total_tokens += len(encode_fn(l))
    elapsed = time.perf_counter() - t0

    chars_sec  = (total_chars  * n_reps) / elapsed
    tokens_sec = total_tokens / elapsed
    return chars_sec, tokens_sec


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_time(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    return f"{s/60:.1f}min"


def fmt_throughput(chars_sec: float) -> str:
    if chars_sec > 1e6:
        return f"{chars_sec/1e6:.2f} MC/s"
    return f"{chars_sec/1e3:.1f} kC/s"


def fmt_mb(mb: float) -> str:
    return f"{mb:.0f} MB"


def print_table(rows: list[dict], title: str):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    if not rows:
        return
    keys = [k for k in rows[0] if k != "Tokenizer"]
    nw = max(len(r["Tokenizer"]) for r in rows) + 2
    cw = max(max(len(k), max(len(str(r.get(k, "—"))) for r in rows)) for k in keys) + 2
    header = f"  {'Tokenizer':<{nw}}" + "".join(f"{k:>{cw}}" for k in keys)
    print(header)
    print(f"  {'-' * (nw + cw * len(keys))}")
    for r in rows:
        print(f"  {r['Tokenizer']:<{nw}}" +
              "".join(f"{str(r.get(k,'—')):>{cw}}" for k in keys))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=[50_000])
    parser.add_argument("--no-train", action="store_true",
                        help="Skip training; only measure encoding on existing models")
    parser.add_argument("--encode-reps", type=int, default=3,
                        help="Number of repetitions for encoding benchmark")
    parser.add_argument("--lang", default=None,
                        help="Benchmark per-language models/data (models/<lang>/*, "
                             "data/<lang>/*) instead of the legacy top-level dirs. "
                             "Use this to time the regenerated (fixed) LiB models.")
    parser.add_argument("--encode-lines", type=int, default=1000,
                        help="Number of test lines to use for encoding benchmark")
    args = parser.parse_args()

    if args.lang:
        global MODELS_DIR, TRAIN_PATH, TEST_PATH, TRAIN_STR
        MODELS_DIR = MODELS_DIR / args.lang
        TRAIN_PATH = DATA_DIR / args.lang / "train.txt"
        TEST_PATH = DATA_DIR / args.lang / "test.txt"
        TRAIN_STR = str(TRAIN_PATH)

    for p in (TRAIN_PATH, TEST_PATH):
        if not p.exists():
            sys.exit(f"Missing {p} — run prepare_corpus.py first")

    test_lines = read_lines(TEST_PATH)[: args.encode_lines]
    print(f"Encoding benchmark: {len(test_lines)} lines × {args.encode_reps} reps")

    for vocab_size in args.vocab_sizes:
        print(f"\n{'#'*60}")
        print(f"  Vocab size: {vocab_size:,}")
        print(f"{'#'*60}")

        train_rows  = []
        encode_rows = []

        configs = [
            ("BPE",        "bpe",        train_bpe,        "tokenizer.json"),
            ("WordPiece",  "wordpiece",  train_wordpiece,  "tokenizer.json"),
            ("SP-Unigram", "sp_unigram", train_sp_unigram, "sp_unigram.model"),
        ]

        for display_name, key, train_fn, model_file in configs:
            out_dir = MODELS_DIR / f"{key}_{vocab_size}"
            print(f"\n  [{display_name}]")

            # ---- Training ----
            if not args.no_train:
                print("    training …", flush=True)
                wrapper, t_sec, mem_mb = train_fn(vocab_size, out_dir)
                print(f"    done in {fmt_time(t_sec)}, peak +{fmt_mb(mem_mb)}")
            else:
                # Load existing model
                if key == "sp_unigram":
                    from experiments.compute_metrics import SPUnigramWrapper
                    mp = out_dir / model_file
                    if not mp.exists():
                        print(f"    model not found at {mp}, skipping")
                        continue
                    wrapper = SPUnigramWrapper(str(mp))
                else:
                    from experiments.compute_metrics import HFTokenizerWrapper
                    mp = out_dir / model_file
                    if not mp.exists():
                        print(f"    model not found at {mp}, skipping")
                        continue
                    wrapper = HFTokenizerWrapper(str(mp))
                t_sec = mem_mb = None

            disk_mb = dir_size_mb(out_dir) if out_dir.exists() else 0.0

            # ---- Encoding ----
            print("    benchmarking encoding …", flush=True)
            chars_sec, toks_sec = measure_encoding(
                wrapper.encode, test_lines, args.encode_reps
            )

            row_t = {"Tokenizer": display_name}
            if t_sec is not None:
                row_t["Train time"] = fmt_time(t_sec)
                row_t["Train mem (peak +MB)"] = fmt_mb(mem_mb)
            row_t["Model size (MB)"] = f"{disk_mb:.1f}"
            train_rows.append(row_t)

            encode_rows.append({
                "Tokenizer":   display_name,
                "Throughput":  fmt_throughput(chars_sec),
                "k tokens/s":  f"{toks_sec/1000:.1f}",
            })

        # ---- LiB ----
        print(f"\n  [LiB]")
        lib_out = MODELS_DIR / f"lib_{vocab_size}"

        if not args.no_train:
            print("    training (this takes several minutes) …", flush=True)
            lib_wrapper, t_sec, mem_mb = train_lib(vocab_size, lib_out)
            print(f"    done in {fmt_time(t_sec)}, peak +{fmt_mb(mem_mb)}")
        else:
            from experiments.compute_metrics import LiBWrapper
            # Fall back to the pre-trained model if per-size model doesn't exist
            src = lib_out if lib_out.exists() else LIB_DIR
            if not src.exists():
                print(f"    no LiB model found, skipping")
                lib_wrapper = None
            else:
                lib_wrapper = LiBWrapper(str(src))
                t_sec = mem_mb = None

        if lib_wrapper is not None:
            disk_mb = dir_size_mb(lib_out if lib_out.exists() else LIB_DIR)

            print("    benchmarking encoding (supra ON) …", flush=True)
            chars_sec, toks_sec = measure_encoding(
                lib_wrapper.encode, test_lines, args.encode_reps
            )

            row_t = {"Tokenizer": "LiB"}
            if t_sec is not None:
                row_t["Train time"] = fmt_time(t_sec)
                row_t["Train mem (peak +MB)"] = fmt_mb(mem_mb)
            row_t["Model size (MB)"] = f"{disk_mb:.1f}"
            train_rows.append(row_t)

            encode_rows.append({
                "Tokenizer":  "LiB (supra on)",
                "Throughput": fmt_throughput(chars_sec),
                "k tokens/s": f"{toks_sec/1000:.1f}",
            })

            # LiB supra OFF
            from experiments.compute_metrics import LiBWrapper
            src = lib_out if lib_out.exists() else LIB_DIR
            lib_off = LiBWrapper(str(src), use_supra_words=False)
            print("    benchmarking encoding (supra OFF) …", flush=True)
            chars_sec, toks_sec = measure_encoding(
                lib_off.encode, test_lines, args.encode_reps
            )
            encode_rows.append({
                "Tokenizer":  "LiB (supra off)",
                "Throughput": fmt_throughput(chars_sec),
                "k tokens/s": f"{toks_sec/1000:.1f}",
            })

        print_table(train_rows,  f"Training — vocab {vocab_size:,}")
        print_table(encode_rows, f"Encoding throughput — vocab {vocab_size:,} — "
                                 f"{len(test_lines)} lines × {args.encode_reps} reps")


if __name__ == "__main__":
    main()
