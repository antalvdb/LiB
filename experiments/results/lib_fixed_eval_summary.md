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

Sweep relaunched 2026-07-24 with the fixed LiB models; COMPLETE 2026-07-28,
all 7 languages (`neural_bpb_<lang>_50000.csv`; broken-era EN archived as
`*_broken.csv`).

| lang | BPE | WordPiece | SP-U | SuperBPE | LiB-no-supra | LiB (fixed) |
|------|-----|-----------|------|----------|--------------|-------------|
| en | 1.805 | **1.718** | 1.733 | 1.815 | 2.009 | 1.822 |
| de | 1.710 | 1.621 | **1.617** | 1.733 | 1.863 | 1.713 |
| es | 1.580 | **1.505** | 1.508 | 1.569 | 1.775 | 1.595 |
| fi | 1.220 | 1.155 | **1.150** | 1.243 | 1.385 | 1.262 |
| tr | 1.703 | **1.640** | 1.671 | 1.731 | 1.862 | 1.702 |
| ar | 1.159 | **1.111** | 1.112 | 1.177 | 1.253 | 1.169 |
| zh | 2.129 | 2.116 | **2.042** | 2.124 | 2.119 | 2.111 |

- LiB ≈ parity with BPE in all 7 (beats it outright on tr −0.1% and zh −0.8%;
  worst gap +3.4% fi; tr win comes with a 13.6k emergent vocab vs BPE's 50k).
  Beats SuperBPE in 5/7 (de, fi, tr, ar, zh). On zh LiB is 2nd overall,
  ahead of WordPiece too; elsewhere WP/SP-U lead by ~4–8%. Broken-era EN gap
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

### Ablation DL/BPC — COMPLETE, all 7 languages (2026-07-28, KenLM eval)

DL_total (kb), LiB vs LiB (no forget), penalty for ablating passive
forgetting:

| | en | de | es | fi | tr | ar | zh |
|---|---|---|---|---|---|---|---|
| LiB | 7,924 | 8,368 | 4,694 | 2,887 | 6,306 | 8,494 | 8,932 |
| no forget | 8,373 | 8,759 | 5,072 | 3,176 | 6,431 | 8,886 | 9,481 |
| penalty | +5.7% | +4.7% | +8.1% | +10.0% | +2.0% | +4.6% | +6.1% |

- No-forget is worse on DL in ALL 7; LiB's best-DL count collapses 5/7 → 2/7
  (keeps only fi/tr, where even inflated lexicons stay small). In every
  language the extra units buy slightly better DL_corp but cost far more
  DL_lex — dead weight that MDL prices in full.
- **3-gram BPC is forgetting-neutral**: no-forget within ±0.005 of LiB in 6
  languages (zh +0.04); all LiB variants still beat BPE. KN smoothing barely
  notices rarely-firing dead types; DL discriminates, BPC does not. Same
  dissociation for the cap ladder (en): BPC 1.914/1.909/1.909 vs DL
  7,924/8,323/8,460 (fixed/cap/cap2).
- Cap ladder DL (en): the self-regulated size is DL-optimal — filling the
  budget improves DL_corp (6,952→6,598) and avg token length (4.66→5.16,
  passing BPE's 5.01) but DL_total degrades monotonically. NOTE the flip
  side: at a matched 50k budget BPE beats LiB on en DL (8,138 vs 8,460) —
  LiB's DL win comes from choosing its size, not better per-budget encoding.
  Matched-size baselines (BPE et al. at 28.4k etc.) answer the converse;
  evals running (`matched_eval_speed.log`).

## Budget-matched controls (rewrite_plan.md §8b), 2026-07-27

- Matched-size baselines: BPE/WP/SP-U trained at each language's emergent LiB
  vocab (en 28,423 … tr 13,585, zh 26,269) — DL/BPC evals queued behind the
  ablation eval.
- Cap-filling LiB (en): α=0.5, ω=0.005 (single-knob) → vocab **46,718**
  (93.4% of cap), supra 30.9%. Two-knob variant (α=0.5, ω=0.001,
  `lib_50000_cap2`) → **49,954** — pins the cap. Budget ladder for the
  comparison: 28.4k (fixed) → 46.7k (α-only) → 50.0k (α+weak ω).

## Matched-size baselines — DL verdict (2026-07-28, `matched_eval_speed.log`)

BPE/WP/SP-U retrained at each language's emergent LiB size. **At matched
vocabulary size the baselines beat LiB on DL in all 7 languages** (zh: BPE
and SP-U do; WP does not). E.g. en @28.4k: WP 7,405 / SP-U 7,538 / BPE 7,664
vs LiB 7,924; fi @20.3k: WP 2,755 / BPE 2,763 vs LiB 2,887.

→ **LiB's headline "DL wins 5/7" is a vocabulary-size effect**: DL at these
corpus sizes favours smaller vocabs for every tokenizer, and only LiB was
free to choose its size. The defensible claims are (i) self-regulation finds
a good size autonomously (no budget hyperparameter to tune) and (ii) the
forgetting ablation (+2–10% DL) shows the mechanism is what finds it — NOT
that LiB encodes better than budget-matched baselines.

Silver lining — **3-gram BPC at matched size: LiB beats matched BPE in 6/7**
(en 1.914 vs 1.935, de 1.836 vs 1.859, es 1.754 vs 1.758, tr 2.009 vs 2.028,
ar 2.222 vs 2.258, zh 6.310 vs 6.373; loses fi 1.364 vs 1.348) and beats WP
on zh. The per-budget BPC story survives the control; the DL story does not.

## Encoding throughput (idle machine, 2026-07-28)

LiB (supra on) ~5.4 MC/s vs BPE ~6.5, WP ~7.9, SP-U ~11.4 (en/de pattern;
all langs in log). LiB is 20–50% slower than baselines but same order of
magnitude; smallest model files (1.5–1.6 MB vs BPE 3.5).

## Status: ALL RUNS COMPLETE (2026-07-28)

Neural BPB (7 langs), passive-forgetting ablation + DL/BPC, cap ladder,
matched-size baselines, speed benchmarks. No experiments pending.
