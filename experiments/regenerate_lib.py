"""Regenerate all LiB models with the tuned (reference-faithful) trainer.

(b): retrain LiB per language on the full training set with the
hyperparameters chosen by hparam_search.py, saving to
experiments/models/<lang>/lib_50000/ (superseding the old broken-mechanism
models). en/fi/tr/zh use their own searched best configs; de/es/ar use the
en/fi/tr consensus config (they are space-delimited, like en/fi/tr).
An extra self-regulating English variant (α=0.25) is saved separately.
"""

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "python"))
from lib_tokenizers.tokenizer import LiBTokenizerFast

DATA = REPO / "experiments" / "data"
MODELS = REPO / "experiments" / "models"

# The en/fi/tr search agreed on doc=100, max_len=16, life=50, ω=0.005, Δ=0.4.
# We use α=0.25 (not the BPC-best 0.5): it self-regulates the vocabulary instead
# of saturating the 50k cap (α=0.5 pinned English at the cap), costs only ~0.01
# bits of 3-gram BPC (within the search's noise), trains far faster, and should
# score better under neural BPB. This is the config family for space-delimited
# languages; Chinese uses its own searched best.
CONSENSUS = dict(num_epochs=5000, doc_size=100, max_len=16, life=50,
                 memory_in=0.25, memory_out=0.005, update_rate=0.4)
ZH = dict(num_epochs=5000, doc_size=50, max_len=20, life=10,
          memory_in=0.25, memory_out=0.01, update_rate=0.4)

CONFIGS = {
    "en": CONSENSUS, "de": CONSENSUS, "es": CONSENSUS,
    "fi": CONSENSUS, "tr": CONSENSUS, "ar": CONSENSUS,
    "zh": ZH,
}

# Which corpus each config trains on (the *_selfreg variant reuses en data).
CORPUS = {k: (k.split("_")[0]) for k in CONFIGS}


def is_supra(t):
    core = t[1:] if t[:1] == " " else t
    return " " in core


def main():
    order = sys.argv[1:] or list(CONFIGS.keys())
    for name in order:
        cfg = CONFIGS[name]
        lang = CORPUS[name]
        train = DATA / lang / "train.txt"
        if not train.exists():
            print(f"[{name}] missing {train}", flush=True)
            continue
        out = MODELS / lang / ("lib_50000" if "_" not in name else f"lib_50000_{name.split('_')[1]}")
        print(f"[{name}] training on {lang}: {cfg} ...", flush=True)
        t0 = time.time()
        tok = LiBTokenizerFast.train_new(str(train), vocab_size=50000, seed=42, **cfg)
        tok.save_pretrained(str(out))
        bt = tok.backend_tokenizer; bt.model.use_supra_words = True
        v = list(bt.get_vocab().keys()); s = [x for x in v if is_supra(x)]
        print(f"[{name}] done in {time.time()-t0:.0f}s -> {out} | "
              f"vocab={len(v):,} supra={len(s)/max(len(v),1):.1%}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
