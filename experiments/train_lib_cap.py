"""Cap-filling LiB (budget-matched arm b): English, final config but α=0.5.

The ω=0 ablation bounds what weaker decay alone can do: even with passive
forgetting off, admission tops out below the cap (en 41.7k). Filling the 50k
budget "as the other tokenizers do" therefore requires raising the admission
rate: α=0.5 pins English at the cap under otherwise-final knobs (hparam
search), with decay untouched (ω=0.005) — a single-knob change, honestly
labeled as an admission-rate change, not a decay change. Saves to
experiments/models/en/lib_50000_cap/.
"""

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "python"))
from lib_tokenizers.tokenizer import LiBTokenizerFast

CFG = dict(num_epochs=5000, doc_size=100, max_len=16, life=50,
           memory_in=0.5, memory_out=0.005, update_rate=0.4)


def is_supra(t):
    core = t[1:] if t[:1] == " " else t
    return " " in core


def main():
    train = REPO / "experiments" / "data" / "en" / "train.txt"
    out = REPO / "experiments" / "models" / "en" / "lib_50000_cap"
    print(f"[en] training (cap-filling, α=0.5): {CFG} ...", flush=True)
    t0 = time.time()
    tok = LiBTokenizerFast.train_new(str(train), vocab_size=50000, seed=42, **CFG)
    tok.save_pretrained(str(out))
    bt = tok.backend_tokenizer; bt.model.use_supra_words = True
    v = list(bt.get_vocab().keys()); s = [x for x in v if is_supra(x)]
    print(f"[en] done in {time.time()-t0:.0f}s -> {out} | "
          f"vocab={len(v):,} supra={len(s)/max(len(v),1):.1%}", flush=True)


if __name__ == "__main__":
    main()
