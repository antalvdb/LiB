"""Joint multilingual tokenizers: train on all seven languages at once, then
cross-test on each language.

Emulates a joint multilingual vocabulary by concatenating the seven
per-language training sets (7 x 100k sentences). Every tokenizer is trained on
the concatenation; every metric is reported on the concatenated test set and
on each language's own test set, so the joint tokenizer can be compared with
the monolingual one trained for that language.

Design
------
* Baselines (BPE, WordPiece, SP-Unigram, SuperBPE) at the standard 50k budget,
  and BPE/WordPiece/SP-Unigram again at LiB's emergent joint size (matched).
* LiB with a 350k cap (7 x 50k), so the cap does not bind and the joint
  vocabulary size is emergent. One configuration must serve all languages; we
  use the space-delimited consensus configuration (Chinese's own searched
  configuration differs, which is a caveat for the Chinese numbers).
* LiB documents stay single-language. In its normal (random) mode the trainer
  draws each document as ``doc_size`` sentences sampled at random from the
  whole corpus; on a concatenated corpus that would mix ~14 sentences from
  each language into one document and weaken the twice-within-a-document
  admission rule. Instead we write a block file in which each run of
  ``doc_size`` lines is drawn (with replacement, as in random mode) from one
  language, with the language of each block chosen uniformly, and train in
  deterministic mode, which reads documents as consecutive blocks.
* Exposure is matched to the monolingual runs: 7 x 5,000 document-epochs, so
  each language is seen for ~5,000 documents, as in its monolingual run.
* Neural BPB: one small Transformer per tokenizer, trained on the joint
  training stream and scored separately on each language's test set. Same
  architecture and step budget as the monolingual sweep (5,000 steps), so a
  joint LM sees each language for roughly 1/7 of the steps a monolingual LM
  did: compare tokenizers with each other within the joint setting, not joint
  BPB with monolingual BPB.

Stages (run in order; each is idempotent and skips finished outputs)
------
    python experiments/joint_multilingual.py prepare
    python experiments/joint_multilingual.py baselines          # 50k BPE/WP/SP-U
    conda run -n superbpe python experiments/train_superbpe.py ...   (see run_joint.sh)
    python experiments/joint_multilingual.py lib                # LiB, 350k cap
    python experiments/joint_multilingual.py matched            # at LiB's size
    python experiments/joint_multilingual.py eval               # DL/BPC/fertility/allocation
    python experiments/joint_multilingual.py bpb [--only NAME ...]
"""

import argparse
import csv
import json
import random
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "python"))

DATA = REPO / "experiments" / "data"
MODELS = REPO / "experiments" / "models"
RESULTS = REPO / "experiments" / "results"
JOINT = "joint"
JDATA = DATA / JOINT
JMODELS = MODELS / JOINT

LANGS = ["en", "de", "es", "fi", "tr", "ar", "zh"]
SEED = 42

LIB_CAP = 350_000
LIB_CFG = dict(num_epochs=7 * 5000, doc_size=100, max_len=16, life=50,
               memory_in=0.25, memory_out=0.005, update_rate=0.4)
VALID_PER_LANG = 1000


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f if l.strip()]


def write_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for l in lines:
            f.write(l + "\n")


def is_supra(t):
    core = t[1:] if t[:1] in (" ", "▁") else t
    return " " in core or "▁" in core


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------

def prepare(args):
    rng = random.Random(SEED)
    train, valid, test = [], [], []
    per_lang_train = {}
    for lang in LANGS:
        tr = read_lines(DATA / lang / "train.txt")
        per_lang_train[lang] = tr
        train += tr
        valid += read_lines(DATA / lang / "valid.txt")[:VALID_PER_LANG]
        test += read_lines(DATA / lang / "test.txt")
    rng.shuffle(train)   # baselines are order-insensitive, SP subsamples
    write_lines(JDATA / "train.txt", train)
    write_lines(JDATA / "valid.txt", valid)
    write_lines(JDATA / "test.txt", test)
    print(f"joint train={len(train):,} valid={len(valid):,} test={len(test):,}")

    blocks = JDATA / "lib_blocks.txt"
    n_docs, d = LIB_CFG["num_epochs"], LIB_CFG["doc_size"]
    doc_langs = Counter()
    with open(blocks, "w", encoding="utf-8") as f:
        for _ in range(n_docs):
            lang = rng.choice(LANGS)
            doc_langs[lang] += 1
            src = per_lang_train[lang]
            for _ in range(d):
                f.write(src[rng.randrange(len(src))] + "\n")
    print(f"LiB block file: {n_docs:,} single-language documents x {d} "
          f"sentences -> {blocks}; per-language documents: {dict(doc_langs)}")


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def _train_baseline(kind, size):
    from experiments.train_multilingual import (
        train_bpe, train_wordpiece)
    out = JMODELS / f"{kind}_{size}"
    sentinel = out / ("sp_unigram.model" if kind == "sp_unigram" else "tokenizer.json")
    if sentinel.exists():
        print(f"  [{kind} {size:,}] exists, skipping"); return
    t0 = time.time()
    train_path = str(JDATA / "train.txt")
    if kind == "bpe":
        train_bpe(train_path, size, out)
    elif kind == "wordpiece":
        train_wordpiece(train_path, size, out)
    elif kind == "sp_unigram":
        import sentencepiece as spm
        out.mkdir(parents=True, exist_ok=True)
        spm.SentencePieceTrainer.train(
            input=train_path, model_prefix=str(out / "sp_unigram"),
            vocab_size=size, model_type="unigram", character_coverage=1.0,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            input_sentence_size=700_000, shuffle_input_sentence=True,
        )
    print(f"  [{kind} {size:,}] done in {time.time()-t0:.0f}s", flush=True)


def baselines(args):
    for kind in ("bpe", "wordpiece", "sp_unigram"):
        _train_baseline(kind, 50_000)


def lib_size():
    from tokenizers import Tokenizer
    t = Tokenizer.from_file(str(JMODELS / f"lib_{LIB_CAP}" / "tokenizer.json"))
    specials = {"[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"}
    return sum(1 for v in t.get_vocab() if v not in specials)


def lib(args):
    from lib_tokenizers.tokenizer import LiBTokenizerFast
    out = JMODELS / f"lib_{LIB_CAP}"
    if (out / "tokenizer.json").exists():
        print(f"  [lib] exists ({lib_size():,} units), skipping"); return
    print(f"  [lib] training, cap {LIB_CAP:,}, deterministic block order: {LIB_CFG}",
          flush=True)
    t0 = time.time()
    tok = LiBTokenizerFast.train_new(str(JDATA / "lib_blocks.txt"),
                                     vocab_size=LIB_CAP, seed=SEED,
                                     deterministic=True, **LIB_CFG)
    tok.save_pretrained(str(out))
    v = list(tok.backend_tokenizer.get_vocab())
    print(f"  [lib] done in {(time.time()-t0)/60:.0f} min -> {out} | vocab={len(v):,} "
          f"supra={sum(map(is_supra, v))/len(v):.1%}", flush=True)


def matched(args):
    size = lib_size()
    print(f"  LiB joint size: {size:,}")
    for kind in ("bpe", "wordpiece", "sp_unigram"):
        _train_baseline(kind, size)


# ---------------------------------------------------------------------------
# tokenizer set
# ---------------------------------------------------------------------------

def joint_tokenizers(include_matched=True):
    from experiments.compute_metrics import (
        HFTokenizerWrapper, SPUnigramWrapper, LiBWrapper)
    cfgs = [
        ("BPE", "bpe_50000", "hf"), ("WordPiece", "wordpiece_50000", "hf"),
        ("SP-Unigram", "sp_unigram_50000", "sp"), ("SuperBPE", "superbpe_50000", "hf"),
        ("LiB (no supra)", f"lib_{LIB_CAP}", "lib_off"), ("LiB", f"lib_{LIB_CAP}", "lib_on"),
    ]
    if include_matched and (JMODELS / f"lib_{LIB_CAP}" / "tokenizer.json").exists():
        m = lib_size()
        cfgs += [("BPE @m", f"bpe_{m}", "hf"), ("WordPiece @m", f"wordpiece_{m}", "hf"),
                 ("SP-Unigram @m", f"sp_unigram_{m}", "sp"), ("SuperBPE @m", f"superbpe_{m}", "hf")]
    out = []
    for name, sub, kind in cfgs:
        d = JMODELS / sub
        f = d / ("sp_unigram.model" if kind == "sp" else "tokenizer.json")
        if not f.exists():
            print(f"    [skip] {name}: {f} not found"); continue
        if kind == "hf":
            out.append((name, HFTokenizerWrapper(str(f))))
        elif kind == "sp":
            out.append((name, SPUnigramWrapper(str(f))))
        else:
            out.append((name, LiBWrapper(str(d), use_supra_words=(kind == "lib_on"))))
    return out


# ---------------------------------------------------------------------------
# eval: DL, 3-gram BPC, fertility, per-language vocabulary allocation
# ---------------------------------------------------------------------------

def evaluate(args):
    from experiments.compute_metrics import (
        compute_dl, train_ngram_lm_kenlm, compute_bpc_kenlm)
    train_lines = read_lines(JDATA / "train.txt")
    tests = {lang: read_lines(DATA / lang / "test.txt") for lang in LANGS}
    tests["all"] = read_lines(JDATA / "test.txt")

    rows = []
    toks = joint_tokenizers()
    if args.only:
        toks = [(n, w) for n, w in toks if n in args.only]
    for name, w in toks:
        print(f"\n=== {name} ===", flush=True)
        vocab = w.vocab()
        enc = {k: [w.encode(l) for l in ls] for k, ls in tests.items()}
        used = {k: set(t for s in e for t in s) for k, e in enc.items() if k != "all"}
        vocab_set = set(vocab)

        # 3-gram KenLM trained once on the joint training stream, scored per language
        with tempfile.TemporaryDirectory() as tmp:
            lm = train_ngram_lm_kenlm([w.encode(l) for l in train_lines], 3,
                                      tmp_dir=tmp, name=name.replace(" ", "_").replace("@", "m"))
            for k in list(LANGS) + ["all"]:
                raw_chars = sum(len(l) for l in tests[k])
                raw_bytes = sum(len(l.encode("utf-8")) for l in tests[k])
                flat = [t for s in enc[k] for t in s]
                dl = compute_dl(vocab, flat)
                row = {
                    "tokenizer": name, "test": k, "V": len(vocab),
                    "supra_pct": round(100 * sum(map(is_supra, vocab)) / max(len(vocab), 1), 1),
                    "dl_lex_kb": round(dl["dl_lex_kb"], 1),
                    "dl_corp_kb": round(dl["dl_corpus_kb"], 1),
                    "dl_tot_kb": round(dl["dl_total_kb"], 1),
                    "bpc3": round(compute_bpc_kenlm(lm, enc[k], raw_chars), 4),
                    "tok_per_byte": round(len(flat) / raw_bytes, 4),
                }
                if k != "all":
                    u = used[k] & vocab_set
                    others = set().union(*(used[o] for o in LANGS if o != k))
                    # DL_lex of only the units this language actually uses
                    row["types_used"] = len(u)
                    row["types_exclusive"] = len(u - others)
                    row["dl_lex_used_kb"] = round(compute_dl(sorted(u), [])["dl_lex_kb"], 1)
                    row["supra_used_pct"] = round(100 * sum(map(is_supra, u)) / max(len(u), 1), 1)
                rows.append(row)
                print(f"  {k:4s} bpc3={row['bpc3']:.4f} tok/B={row['tok_per_byte']:.4f} "
                      f"DLcorp={row['dl_corp_kb']} used={row.get('types_used','-')}", flush=True)

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "joint_eval.csv"
    if args.only and out.exists():
        # keep the other tokenizers' rows and replace only those just scored
        rows = [r for r in csv.DictReader(open(out)) if r["tokenizer"] not in args.only] + rows
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k)
                  if k in rows[0] else 99)
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader(); wr.writerows(rows)
    print(f"\nsaved {out}")


# ---------------------------------------------------------------------------
# neural BPB: one LM per tokenizer on the joint stream, scored per language
# ---------------------------------------------------------------------------

def bpb(args):
    import torch
    from experiments import evaluate_neural_bpb as nb
    device = nb.pick_device("auto")
    hp = argparse.Namespace(lr=3e-4, max_steps=args.max_steps, batch_size=32,
                            eval_every=500, patience=6, stride=128, seed=0)
    train_lines = read_lines(JDATA / "train.txt")
    valid_lines = read_lines(JDATA / "valid.txt")
    tests = {lang: read_lines(DATA / lang / "test.txt") for lang in LANGS}

    out = RESULTS / "joint_neural_bpb.csv"
    done = set()
    if out.exists():
        done = {r["tokenizer"] for r in csv.DictReader(open(out))}
    toks = joint_tokenizers(include_matched=True)
    if args.only:
        toks = [(n, w) for n, w in toks if n in args.only]
    print(f"device: {device}; steps {args.max_steps}", flush=True)
    for name, w in toks:
        if name in done:
            print(f"  [{name}] done already, skipping"); continue
        print(f"\n=== {name} ===", flush=True)
        stoi, itos = nb.build_vocab(w, train_lines)
        train_ids, _ = nb.encode_stream(w, train_lines, stoi)
        valid_ids, valid_sc = nb.encode_stream(w, valid_lines, stoi)
        print(f"  |V|={len(itos):,} train_toks={len(train_ids):,}", flush=True)
        torch.manual_seed(0)
        model = nb.TinyGPT(len(itos), d_model=256, n_head=4, n_layer=4,
                           block_size=256, dropout=0.1)
        t0 = time.time()
        model = nb.train_lm(model, train_ids, valid_ids, valid_sc, hp, device)
        row = {"tokenizer": name, "train_min": round((time.time() - t0) / 60, 1)}
        for lang in LANGS:
            ids, sc = nb.encode_stream(w, tests[lang], stoi)
            fert = nb.fertility_stats(w, tests[lang])
            v, _ = nb.score_stream(model, ids, sc, raw_bytes=fert["raw_bytes"],
                                   block_size=256, stride=128, device=device)
            row[f"bpb_{lang}"] = round(v, 4)
            row[f"tpb_{lang}"] = round(fert["tokens_per_byte"], 4)
            print(f"  {lang} BPB={v:.4f} tok/B={fert['tokens_per_byte']:.4f}", flush=True)
        new = not out.exists()
        with open(out, "a", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(row))
            if new:
                wr.writeheader()
            wr.writerow(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prepare", "baselines", "lib", "matched", "eval", "bpb"])
    p.add_argument("--only", nargs="+", default=None)
    p.add_argument("--max-steps", type=int, default=5000)
    a = p.parse_args()
    {"prepare": prepare, "baselines": baselines, "lib": lib, "matched": matched,
     "eval": evaluate, "bpb": bpb}[a.stage](a)


if __name__ == "__main__":
    main()
