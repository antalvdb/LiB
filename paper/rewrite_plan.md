# Full rewrite plan (post-forgetting-fix, 2026-07-24)

Supersedes `reframe_outline.md`. That outline was written against the *broken*
LiB (no forgetting): its premise — "LiB never wins on intrinsic metrics, so the
metrics must be biased against supra-words" — is now inverted by the fixed
results in `experiments/results/lib_fixed_eval_summary.md`. Everything in the
paper is up for change (AvdB, 2026-07-24).

---

## 1. What the fix changed under the paper

Reference-faithful trainer + working forgetting (memory strength, reinforcement,
tail decay). All prior LiB numbers superseded. New headline facts:

- **DL: LiB is best on 5/7 languages** (de, es, fi, tr, zh; 2nd on en, ar;
  fi −19.5% vs best baseline). The prior draft's LiB numbers came from code
  with the forgetting mechanism unwired — they were never legitimate results
  and are simply superseded, not "improved upon."
- **3-gram BPC: LiB beats BPE and SuperBPE in all 7 languages**; 2nd on tr/zh,
  ~1–2% behind WordPiece/SP-U elsewhere.
- **Supra-words now HELP at 3-gram in every language** (LiB < LiB-no-supra).
  The old 2-gram-helps/3-gram-hurts reversal is gone for LiB.
- **Vocabulary self-regulates** below the 50k cap in every language
  (13.6k tr … 28.4k en): size is an emergent equilibrium of admission vs
  forgetting, governed by ω.
- **Supra ratios collapsed** from the old 65–81% to 15–31%
  (ar 30.5, es 29.6, en 25.7, tr 19.1, de 17.5, fi 14.7, zh ≈0).

## 2. Which existing narratives die

1. **"English bias → LiB is language-agnostic" (current paper.tex spine).**
   Dead as an apology — LiB no longer needs excusing. English-bias critique can
   survive as one *motivation* in the intro, nothing more.
2. **"Intrinsic metrics are rigged against supra-words" (reframe_outline.md).**
   Dead for LiB. The 2/3-gram reversal was a symptom of the broken lexicon's
   dead supra-word tail, not pure estimator sparsity — with forgetting active,
   supra-words help even a 3-gram model. The sparsity argument survives only as
   a *secondary* explanation of why SuperBPE (frequency-selected, budget-filling)
   still loses n-gram BPC. Do NOT build the paper on it.
3. **Old typology claim ("76–81% supra for analytic/templatic").** Dead numbers.
   But the new ratios tell a *cleaner* story — see §4, contribution 4.
4. **`intro_reframed.tex` [EN] empirical claims** (77% supra, LiB less
   compressive than BPE, neural BPB gap, prune results). All computed with the
   broken tokenizer (|V| pinned at 50k). The *prose architecture* of that intro
   is reusable; every number and the "supra-words don't improve BPB" verdict
   must await the re-run.

## 3. Proposed new thesis

> **Forgetting is the load-bearing mechanism.** In a cognitively grounded
> memorise-and-forget MDL tokenizer, forgetting is not a detail but the
> component that makes the objective work: with it, vocabulary size becomes an
> emergent equilibrium rather than a hyperparameter, the lexicon sheds dead
> weight, description length is won outright in 5 of 7 typologically diverse
> languages, and the surviving supra-word units genuinely help; ablate it, and
> the lexicon silts up with dead types until every metric collapses.

This is a *mechanism* paper with a strong empirical payoff, not a metric-
critique paper. It also keeps the cognitive grounding (human passive/active
forgetting; eye-fixation work) as support, which travels to either an *CL or a
CogSci venue.

## 4. Contributions (proposed)

1. **The forgetting mechanism, made explicit and shown to be load-bearing.**
   Ablation across 7 languages: removing forgetting silts the lexicon with dead
   supra-word types and degrades every metric (e.g. zh DL 8,932 → 12,775
   ablated). The ω-sweep (forgetting_section.tex Table): dead weight falls
   monotonically; sharp admission/forgetting phase transition; vocab size
   becomes emergent.
2. **State-of-competitive intrinsic results.** Best DL on 5/7 (state the
   own-objective caveat plainly: DL is LiB's optimization target, so this is
   "wins its own game" + "competitive on the neutral metric", not "best
   tokenizer"); 3-gram BPC beats BPE and SuperBPE everywhere, within 1–2% of
   WordPiece/SP-U.
3. **Supra-word tokens help once properly forgotten.** LiB < LiB-no-supra at
   3-gram in all 7 languages. Contrast with SuperBPE (frequency admission, no
   forgetting, fills budget) which loses BPC everywhere → the *selection and
   retention rule* is what makes supra-words pay, not their mere presence.
4. **Supra ratio as typological/script diagnostic (new numbers).** Among
   space-delimited languages the ratio inversely tracks word-internal
   morphological complexity: analytic/Romance/templatic (en/es/ar 26–31%) >
   compounding/agglutinative (de/tr/fi 15–19%); Chinese ≈0% because the script
   has no spaces to cross (frame on script, per terminology note; 0.7% is
   mixed-script noise). Redundancy lives above the word where morphology is
   poor, inside it where morphology is rich — and LiB finds it in either place
   with the same objective.
5. **Tokenization-fair evaluation** (neural BPB + fertility) — PENDING the
   sweep. Role in the paper depends on outcome (see §6).
6. **HF-compatible Rust implementation** (unchanged).

## 5. Section plan (old → new)

| § | paper.tex now | Rewrite |
|---|---------------|---------|
| Abstract | English-bias framing | New: forgetting thesis (§3) + headline numbers |
| 1 Intro | English bias is the thesis | Rebuild from intro_reframed.tex *structure* (whitespace-floor opening still works) but land on the forgetting thesis; English bias demoted to one motivation |
| 2 English Bias in Tokenization | standalone section | Fold into intro + related work; cut |
| 3 The LiB Tokenizer | admission-only §3.2 | Insert forgetting_section.tex block (A) as §3.2; algorithm now = memorise + forget |
| 4 Setup | n-gram only | Add neural-LM setup, fertility metrics, per-language hyperparameters (consensus config + zh config from regenerate_lib.py) |
| 5 English Results | DL/BPC as verdict | Recast: forgetting ablation (the load-bearing demonstration), ω sweep (block B), then the DL/BPC results |
| NEW | — | Neural BPB + fertility (pending numbers) |
| 6 Multilingual | DL gaps as "LiB degrades least" | Now "LiB wins 5/7"; supra-ratio typology with new numbers; Turkish saturation → generalize: *every* language self-regulates (13.6k–28.4k) |
| 7 Discussion | — | Own-objective caveat for DL; SuperBPE contrast (retention rule, not supra-words, is the differentiator); limitations |
| 8 Conclusion | — | Rebuild on the trade-off/forgetting result |

Also: title needs replacing (current title is bias-framed).

## 6. Decision points pending the BPB sweep

The sweep (fixed LiB, all 7 langs) determines how contribution 5 is framed:

- **If fixed LiB is competitive on neural BPB** (plausible — the dead tail that
  starved embeddings is gone and |V| is smaller): BPB becomes confirming
  evidence; the paper claims wins on DL + fertility + parity on BPB/BPC.
- **If a BPB gap remains**: honest trade-off framing (supra-words buy
  compression at a small quality cost) — i.e. the intro_reframed.tex §"trade-off"
  paragraph survives, now with clean numbers.
- Fertility of fixed LiB is unknown (old: 0.2247 tok/byte ≈ BPE). With 25.7%
  supra instead of 77% dead-tail supra, tok/byte could go either way; if LiB
  still doesn't beat BPE on fertility, do NOT claim the efficiency axis — leave
  that to SuperBPE and claim quality/DL instead.
- Old broken-LiB CSVs (neural_bpb_en_*.csv, prune_*) archived as
  `*_broken.*`; not paper material (the bugged runs are not a valid
  no-forgetting ablation either — the ablation should be run deliberately
  with the final config if we report it).

## 7. How to present the fix (not as errata)

Follow forgetting_section.tex's framing: the paper *describes the mechanism as
designed* (memorise + forget, faithful to Yang et al.'s model) and reports its
behaviour; the reproducibility note states that reinforcement and decay are both
active in the released implementation. No "we fixed a bug in our port" narrative
in the main text.

**Framing rules (AvdB, 2026-07-24):**
- Ablation, not errata: the demonstration that forgetting matters is a
  deliberate LiB-no-forgetting ablation, run with the final config.
- Never present the bugged-code results as a former legitimate position of
  LiB — no "turned from worst to best" / redemption arc, in the paper or in
  its story. The old numbers originated from broken code and experiments;
  they are superseded, full stop.

- ω-sweep table in forgetting_section.tex is marked [EN] with the *old* run
  config (10k epochs, doc=50, α=0.5?); regenerate with the final consensus
  config (5000 epochs, doc=100, α=0.25) for consistency, or relabel.
- ~~forgetting_section.tex block (A) mischaracterized the algorithm~~ —
  REWRITTEN 2026-07-24 to match trainer.rs/trie.rs: (i) *active* forgetting =
  Yang's ordinal reward re-ranking (good chunk headward, bad chunk tailward
  by ⌊Θ·Δ⌋+1, deletion on tail overflow; group_move §2.3.2), driven by the
  branch-rollout evaluation (§2.3.1: unknowns → chunk count → ordinal sum);
  (ii) *passive* forgetting = bottom-ω probation with a `life` counter,
  decremented per document, cancelled when the unit is used without being
  judged bad, deletion on expiry; seed units start on watch; admission =
  α-sampled adjacent-chunk concatenations, twice-within-a-document rule, new
  units enter at the tail. Two-route framing (comparative vs absolute)
  strengthens the cognitive story.

## 8. Remaining experimental queue

1. Neural BPB sweep, 7 langs, fixed LiB (waiting for machine to settle; MPS).
2. `benchmark_speed.py --no-train` on idle machine (throughput table).
3. Regenerate ω-sweep table under final config (CPU, cheap) — for §5.
4. **LiB passive-forgetting ablation** (required for contribution 1):
   `experiments/ablate_forgetting.py` retrains per language with ω=0 —
   everything else at the final config — then DL/BPC rows via
   evaluate_all_multilingual.py ("LiB (no forget)" row wired in). ω=0
   disables exactly the probation/life pipeline (the repaired mechanism);
   active forgetting (ordinal reward re-ranking with tail-overflow deletion,
   trie.rs group_move) still operates and retains partial regulation:
   empirically the lexicon inflates ~40% (zh +88%, near the cap) rather than
   pinning. Frame in the paper as ablating *passive* forgetting, per
   Yang's active/passive decomposition. The bugged-era numbers cannot stand
   in for this — they differ from the final setup in more than just
   forgetting. (Optional second arm, needs a code flag: also disable
   group_move deletion to ablate active forgetting.)

## 8b. Budget-matched controls (added 2026-07-27, AvdB request)

Answer "is LiB's DL win just a smaller vocabulary?" from both sides:

- **Matched-size baselines** (rigorous arm): BPE/WP/SP-U retrained at each
  language's emergent LiB size (en 28,423 … tr 13,585, zh 26,269); DL/BPC
  compared at exactly matched budgets. `matched_budget.log`; evaluate per
  language with `evaluate_all_multilingual.py --langs <l> --vocab-size <n>`.
- **Cap-filling LiB** (demonstration arm, en only): final config but α=0.5
  (`train_lib_cap.py` → `lib_50000_cap`; "LiB (cap)" eval row). Honest label:
  admission-rate change, NOT a decay change — the ω=0 ablation bounds what
  weaker decay alone can reach (en 41.7k < cap), so cap-filling requires
  raising supply. Decay stays at final ω=0.005.

## 9. Venue

Mechanism + cognition + multilingual results → ACL/EMNLP/TACL main remains the
target; the forgetting/memory framing also gives a genuine CogSci/CMCL fallback.
Decide after BPB numbers land.
