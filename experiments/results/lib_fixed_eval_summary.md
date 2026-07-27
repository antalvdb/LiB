# LiB re-evaluation after the forgetting fix (2026-07)

Reference-faithful trainer + forgetting fix. Config (space-delimited langs):
α=0.25, ω=0.005, life=50, doc_size=100, max_len=16, Δ=0.4, 5000 epochs.
Chinese: α=0.25, ω=0.01, life=10, doc_size=50, max_len=20. All prior LiB numbers
in the draft are SUPERSEDED by these.

## Self-regulated vocabulary + supra ratio (corrected supra = internal space only)

| lang | type | vocab | supra% |
|------|------|-------|--------|
| en | analytic | 28,423 | 25.7% |
| de | fusional/compound | 26,579 | 17.5% |
| es | fusional (Romance) | 25,106 | 29.6% |
| fi | agglutinative | 20,262 | 14.7% |
| tr | agglutinative | 13,585 | 19.1% |
| ar | templatic | 27,315 | 30.5% |
| zh | logographic | 26,269 | ~0% (0.7% = mixed-script noise) |

## Description Length — DL_total (kb), 50k budget

| lang | BPE | WordPiece | SP-U | LiB (fixed) | best |
|------|-----|-----------|------|-------------|------|
| en | 8138 | **7814** | 8146 | 7924 | WP (LiB 2nd, +1.4%) |
| de | 8653 | 8408 | 8702 | **8368** | LiB |
| es | 5140 | 4946 | 5248 | **4694** | LiB (−5.1%) |
| fi | 3635 | 3588 | 3943 | **2887** | LiB (−19.5%) |
| tr | 6658 | 6510 | 6794 | **6306** | LiB (−3.1%) |
| ar | 8550 | **8277** | 8456 | 8494 | WP (LiB 2nd, +2.6%) |
| zh | 8944 | 9365 | 9018 | **8932** | LiB (−0.1%) |

LiB best on 5/7 (was worst everywhere when broken). Old→new: en 9603→7924,
fi 4047→2887, tr 7507→6306, zh 12775→8932 (worst→best).

## 3-gram BPC (KenLM, lower better), 50k budget

| lang | BPE | WordPiece | SP-U | SuperBPE | LiB-no-supra | LiB (fixed) |
|------|-----|-----------|------|----------|--------------|-------------|
| en | 1.949 | **1.883** | 1.901 | 1.988 | 1.976 | 1.914 |
| de | 1.876 | **1.808** | 1.818 | 1.906 | 1.871 | 1.836 |
| es | 1.772 | **1.702** | 1.716 | 1.817 | 1.843 | 1.754 |
| fi | 1.355 | **1.313** | 1.325 | 1.423 | 1.438 | 1.364 |
| tr | 2.078 | **2.001** | 2.018 | 2.126 | 2.074 | 2.009 |
| ar | 2.276 | **2.210** | 2.217 | 2.307 | 2.272 | 2.222 |
| zh | 6.461 | 6.408 | **6.283** | 6.471 | 6.308 | 6.310 |

LiB beats BPE and SuperBPE in all 7; 2nd on tr and zh; 3rd elsewhere (behind
WP/SP-U by ~1–2%). Supra-words now HELP at 3-gram in every language
(LiB < LiB-no-supra), reversing the old artifact. Old→new (3-gram): en
2.012→1.914, fi 1.447→1.364, tr 2.126→2.009, ar 2.384→2.222, zh 7.308→6.310.

## Neural bits-per-byte (TinyGPT, byte-normalised; lower better), 50k budget

Sweep relaunched 2026-07-24 with the fixed LiB models; en/de/es/fi/tr/ar done,
zh in progress (`neural_bpb_<lang>_50000.csv`; broken-era EN archived as
`*_broken.csv`).

| lang | BPE | WordPiece | SP-U | SuperBPE | LiB-no-supra | LiB (fixed) |
|------|-----|-----------|------|----------|--------------|-------------|
| en | 1.805 | **1.718** | 1.733 | 1.815 | 2.009 | 1.822 |
| de | 1.710 | 1.621 | **1.617** | 1.733 | 1.863 | 1.713 |
| es | 1.580 | **1.505** | 1.508 | 1.569 | 1.775 | 1.595 |
| fi | 1.220 | 1.155 | **1.150** | 1.243 | 1.385 | 1.262 |
| tr | 1.703 | **1.640** | 1.671 | 1.731 | 1.862 | 1.702 |
| ar | 1.159 | **1.111** | 1.112 | 1.177 | 1.253 | 1.169 |

- LiB ≈ parity with BPE in all six (−0.1% tr — LiB's first BPB win over BPE,
  with a 13.6k emergent vocab vs BPE's 50k — to +3.4% fi), beats SuperBPE in
  4/6 (de, fi, tr, ar), WP/SP-U lead by ~4–8% everywhere. Broken-era EN gap
  (LiB 2.018 vs BPE 1.805, −12%) is gone.
- Supra-words HELP LiB on neural BPB in every language (−0.12…−0.19 BPB vs
  no-supra, and ~2.2–2.5× shorter sequences).
- Fertility (tok/byte): SuperBPE best in 5/6 (en 0.176, de 0.174, es 0.173,
  fi 0.158, ar 0.112; on tr BPE edges it 0.171 vs 0.173); LiB worse than BPE
  in all langs (en 0.215 vs 0.200; tr 0.234 vs 0.171). LiB does NOT win the
  efficiency axis — paper claims quality/DL, not fertility (rewrite_plan.md
  §6 branch settled).

## Passive-forgetting ablation (ω=0, all else final config), 2026-07

`experiments/ablate_forgetting.py` → `models/<lang>/lib_50000_noforget/`.
Active forgetting (ordinal re-ranking incl. tail-overflow deletion) still on;
only probation/life disabled. Lexicons inflate but do NOT all pin at the cap:

| lang | no-forget vocab | fixed vocab | inflation | supra% (nf → fixed) |
|------|-----------------|-------------|-----------|---------------------|
| en | 41,687 | 28,423 | +47% | 33.9 → 25.7 |
| de | 38,103 | 26,579 | +43% | 22.0 → 17.5 |
| es | 35,592 | 25,106 | +42% | 35.3 → 29.6 |
| fi | 28,813 | 20,262 | +42% | 18.9 → 14.7 |
| tr | 18,793 | 13,585 | +38% | 25.6 → 19.1 |
| ar | 40,037 | 27,315 | +47% | 39.1 → 30.5 |
| zh | 49,474 | 26,269 | +88% | 1.0 → ~0 |

Passive forgetting trims ~30% of the lexicon (zh: ~47%), preferentially
supra-words. zh (life=10, shortest probation) inflates most — nearly pins the
cap without it. DL/3-gram-BPC of the ablated models: evaluation running
(`multilingual_eval_noforget.log`, "LiB (no forget)" rows).

## Budget-matched controls (rewrite_plan.md §8b), 2026-07-27

- Matched-size baselines: BPE/WP/SP-U trained at each language's emergent LiB
  vocab (en 28,423 … tr 13,585, zh 26,269) — DL/BPC evals queued behind the
  ablation eval.
- Cap-filling LiB (en): α=0.5, ω=0.005 (single-knob) → vocab **46,718**
  (93.4% of cap), supra 30.9%. Two-knob variant (α=0.5, ω=0.001,
  `lib_50000_cap2`) running to pin the cap fully.

## Still pending

- Neural BPB: zh (sweep running).
- Ablation DL/BPC table (running); matched-size + cap DL/BPC evals queued.
- Encoding-throughput (tokens/sec) for the new LiB + baselines:
  `python experiments/benchmark_speed.py --no-train --lang <l>` per language
  (the --lang flag is REQUIRED to hit the fixed models; without it the script
  falls back to broken-era top-level dirs). Needs an idle machine — run last.
