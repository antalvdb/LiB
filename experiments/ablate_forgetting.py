"""Passive-forgetting ablation: retrain LiB per language with ω=0.

Same final configs as regenerate_lib.py, except memory_out=0 (ω=0), which
disables the probation/life pipeline (passive forgetting): the bottom-ω tail
slice is empty, so no unit ever enters probation or expires. Active
forgetting (ordinal reward re-ranking, group_move) still operates — good
chunks move headward, bad chunks tailward, and a bad chunk demoted past |L|
is still deleted — so some regulation remains: empirically the lexicon
inflates ~40% over the full mechanism but only Chinese (life=10) approaches
the 50k cap. This isolates exactly the passive mechanism. Saves to
experiments/models/<lang>/lib_50000_noforget/.
"""

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "python"))
from lib_tokenizers.tokenizer import LiBTokenizerFast

DATA = REPO / "experiments" / "data"
MODELS = REPO / "experiments" / "models"

# Final configs from regenerate_lib.py with memory_out zeroed. All other
# knobs (α, life, doc_size, max_len, Δ, epochs) unchanged so passive
# forgetting is the only difference.
CONSENSUS = dict(num_epochs=5000, doc_size=100, max_len=16, life=50,
                 memory_in=0.25, memory_out=0.0, update_rate=0.4)
ZH = dict(num_epochs=5000, doc_size=50, max_len=20, life=10,
          memory_in=0.25, memory_out=0.0, update_rate=0.4)

CONFIGS = {
    "en": CONSENSUS, "de": CONSENSUS, "es": CONSENSUS,
    "fi": CONSENSUS, "tr": CONSENSUS, "ar": CONSENSUS,
    "zh": ZH,
}


def is_supra(t):
    core = t[1:] if t[:1] == " " else t
    return " " in core


def main():
    order = sys.argv[1:] or list(CONFIGS.keys())
    for lang in order:
        cfg = CONFIGS[lang]
        train = DATA / lang / "train.txt"
        if not train.exists():
            print(f"[{lang}] missing {train}", flush=True)
            continue
        out = MODELS / lang / "lib_50000_noforget"
        print(f"[{lang}] training (no forgetting): {cfg} ...", flush=True)
        t0 = time.time()
        tok = LiBTokenizerFast.train_new(str(train), vocab_size=50000, seed=42, **cfg)
        tok.save_pretrained(str(out))
        bt = tok.backend_tokenizer; bt.model.use_supra_words = True
        v = list(bt.get_vocab().keys()); s = [x for x in v if is_supra(x)]
        print(f"[{lang}] done in {time.time()-t0:.0f}s -> {out} | "
              f"vocab={len(v):,} supra={len(s)/max(len(v),1):.1%}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
