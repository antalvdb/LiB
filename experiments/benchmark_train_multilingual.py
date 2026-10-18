"""Training wall-clock time per language, LiB against the baselines.

LiB is trained with the configurations of regenerate_lib.py (seed 42), and
its ordered vocabulary is compared with the saved model in lib_50000, so the
run doubles as a check that the build reproduces the paper's vocabularies.
Baselines are trained at 50k with the settings of train_multilingual.py,
once with their default thread pools and once on a single thread (LiB's
trainer is single-threaded).

    python benchmark_train_multilingual.py --out results/train_speed.csv \
        [--langs en zh] [--only LiB]

Models are written to a temporary directory and discarded.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
LANGUAGES = ["en", "de", "es", "fi", "tr", "ar", "zh"]

# One tokenizer is trained per subprocess, so a thread setting cannot leak
# between runs and each timing starts from a fresh interpreter.
CHILD = r"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, {here!r}); sys.path.insert(0, {repo!r}); sys.path.insert(0, {python!r})
name, train, out, threads = sys.argv[1], sys.argv[2], Path(sys.argv[3]), int(sys.argv[4])
t0 = time.perf_counter()
if name == "LiB":
    from regenerate_lib import CONFIGS
    from lib_tokenizers.tokenizer import LiBTokenizerFast
    lang = sys.argv[5]
    tok = LiBTokenizerFast.train_new(train, vocab_size=50000, seed=42, **CONFIGS[lang])
    tok.save_pretrained(str(out))
elif name == "SP-Unigram":
    import sentencepiece as spm
    out.mkdir(parents=True, exist_ok=True)
    kw = dict(num_threads=threads) if threads else dict()
    spm.SentencePieceTrainer.train(
        input=train, model_prefix=str(out / "sp_unigram"), vocab_size=50000,
        model_type="unigram", character_coverage=1.0,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        input_sentence_size=200_000, shuffle_input_sentence=True, **kw)
else:
    import train_multilingual as tm
    fn = dict(BPE=tm.train_bpe, WordPiece=tm.train_wordpiece)[name]
    fn(train, 50000, out)
print(json.dumps(dict(seconds=time.perf_counter() - t0)))
"""


def run(name, lang, threads, tmp):
    env = dict(os.environ)
    if threads:
        env["RAYON_RS_NUM_THREADS"] = str(threads)
    out = Path(tmp) / f"{lang}_{name}_{threads}"
    code = CHILD.format(here=str(HERE), repo=str(REPO), python=str(REPO / "python"))
    res = subprocess.run(
        [sys.executable, "-c", code, name, str(HERE / "data" / lang / "train.txt"),
         str(out), str(threads), lang],
        env=env, capture_output=True, text=True, check=True)
    return json.loads(res.stdout.strip().splitlines()[-1])["seconds"], out


def vocab_order(path):
    model = json.load(open(path / "tokenizer.json"))["model"]
    vocab = model["vocab"]
    if isinstance(vocab, dict):
        return [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    return [v[0] if isinstance(v, list) else v for v in vocab]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", nargs="*", default=LANGUAGES)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()

    # (tokenizer, threads); 0 = library default
    plan = [("LiB", 0), ("BPE", 0), ("BPE", 1), ("WordPiece", 0), ("WordPiece", 1),
            ("SP-Unigram", 0), ("SP-Unigram", 1)]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for lang in args.langs:
            for name, threads in plan:
                if args.only and name not in args.only:
                    continue
                secs, out = run(name, lang, threads, tmp)
                row = dict(lang=lang, tokenizer=name,
                           threads=threads or "default", seconds=f"{secs:.1f}")
                if name == "LiB":
                    new = vocab_order(out)
                    old = vocab_order(HERE / "models" / lang / "lib_50000")
                    row["vocab_size"] = len(new)
                    row["identical_to_paper"] = new == old
                rows.append(row)
                print(row, flush=True)

    fields = ["lang", "tokenizer", "threads", "seconds", "vocab_size", "identical_to_paper"]
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)


if __name__ == "__main__":
    main()
