# Reframed outline: *When do supra-word tokens help?*

Working reframe of the LiB paper around the research question the project
actually cares about, motivated by the critique that the current draft argues
*for* supra-word tokens using metrics (Description Length, n-gram BPC) that are
structurally biased *against* them.

---

## 1. The core problem with the current framing

The present draft is an **"English bias → LiB is a language-agnostic
tokenizer"** paper whose every headline number is a *loss*: LiB trails
WordPiece on DL and n-gram BPC in all seven languages, and trails **SuperBPE**
(the competing supra-word method) on 3-gram BPC. A method that never wins, on
metrics hostile to its central feature, is a hard sell.

Two facts make this fixable rather than fatal:

1. **The metrics are rigged against supra-words, and we can show it.**
   - `DL_corp = N_tokens × H(token distribution)`; every once-seen supra-word
     type costs ~`log2 N` bits. DL punishes large, long-tailed vocabularies
     *by construction* — i.e. exactly LiB's strategy.
   - n-gram BPC punishes rare tokens through **estimator sparsity**, not through
     any real property of the tokens. The tell is already in our data: supra-
     words help at 2-gram and hurt at 3-gram, for **both** LiB *and* SuperBPE.
     That is a property of the *n-gram estimator*, not of supra-word tokens.

2. **The positive case for supra-words lives on a different axis** —
   downstream task accuracy and **sequence-length / inference efficiency**
   (SuperBPE: +4% over 30 tasks, −27% compute). We currently run **zero**
   neural or efficiency-forward experiments, so we never measure supra-words on
   the axis where they win.

**Reframe:** stop trying to prove "LiB is a better tokenizer." Answer the
question we actually care about — *when, and at what linguistic level, do
supra-word tokens help, and why do standard intrinsic metrics fail to see it?*
LiB and SuperBPE become two instances of supra-word tokenizers; the
contribution is a principled, cross-linguistic account of the trade-off.

---

## 2. Proposed thesis (one sentence)

> Supra-word tokens trade *predictability under a fixed-order n-gram model* for
> *sequence compression and downstream efficiency*; standard intrinsic metrics
> (DL, n-gram BPC) see only the cost, an MDL objective (LiB) gives a principled
> way to decide *how much* supra-word structure to admit, and the pay-off is
> language-dependent.

---

## 3. Contributions (reframed)

1. **A demonstration that intrinsic tokenization metrics systematically
   mis-rank supra-word tokenizers.** The 2-gram-helps / 3-gram-hurts reversal is
   shown for LiB *and* SuperBPE across seven languages and explained as an
   n-gram sparsity artefact — confirmed by a neural bits-per-byte evaluation
   that removes the pathology.
2. **A tokenization-fair evaluation** (neural BPB + fertility/compute) placing
   supra-word tokenizers on the axis where they are designed to help.
3. **LiB as a *principled* route to supra-word vocabularies**: the MDL
   objective jointly discovers sub- and supra-word units with a stopping
   criterion tied to information gain (Turkish saturation at 29k), versus
   SuperBPE's frequency heuristic and fixed budget.
4. **The supra-word ratio as a typological diagnostic**, and its interaction
   with the analytic–synthetic continuum.
5. A production HuggingFace-compatible Rust implementation (unchanged).

*Demote* the Blasi-style English-bias argument from spine to **secondary
motivation** — keep it in the intro as framing, not as the claim the results
have to support.

---

## 4. Section-by-section plan

| § | Current | Reframed |
|---|---------|----------|
| Intro | English bias is the thesis | Supra-word question is the thesis; English bias is one motivation among several (efficiency, cognition) |
| 2 | "English Bias in Tokenization" | Shorten; fold into intro + related work |
| 3 | LiB objective & impl | Keep; add explicit note that MDL admission ≠ frequency admission (sets up §7 analysis) |
| 4 | Experimental setup | Add neural LM setup + fertility metrics |
| 5 | English results (DL, BPC) | Reframe DL/BPC as *diagnostic of metric bias*, not verdict |
| **NEW** | — | **Neural BPB + efficiency** — the fair comparison |
| **NEW** | — | **Why n-gram BPC mis-ranks supra-words** — the 2/3-gram reversal, LiB + SuperBPE |
| 6 | Multilingual DL/BPC | Keep; recast gaps as "cost side of the trade-off," pair with efficiency "benefit side" |
| 7 | Discussion | Add: does MDL admit the *right* supra-words? (avg-length parity analysis) |
| 8 | Conclusion | Recast around the trade-off + the metric finding |

---

## 5. The load-bearing new experiment

**Neural bits-per-byte** (`experiments/evaluate_neural_bpb.py`): one small
Transformer LM per tokenizer, cross-entropy over the test stream normalised by
raw UTF-8 bytes. Byte-normalisation makes it tokenization-invariant and
cross-lingually comparable; a neural model removes the n-gram sparsity penalty.
Paired with **fertility** (tokens/byte, bytes/token, tokens/sentence) to expose
the efficiency benefit.

**Prediction to test:** relative to n-gram BPC, neural BPB *narrows or closes*
the LiB/SuperBPE gap to BPE/WordPiece, and supra-word tokenizers win decisively
on fertility. If BPB still shows a gap, that is the honest finding — and it
localises the cost to under-trained rare-type embeddings, which motivates §7.

---

## 6. The uncomfortable internal finding to confront (not hide)

Realized average token length: LiB-with-supra **4.74** vs BPE **5.13**,
WordPiece **4.16**. LiB's supra-words lift it from its no-supra 2.50 only to
*parity* with BPE — **not past it**. SuperBPE, which picks supra-words by
frequency, actually shortens sequences (−27% tokens). This suggests **LiB may be
building the right *level* with the wrong *admission criterion*** — MDL admits a
long tail of collocations that inflate entropy without firing often enough to
shorten real sequences.

Turn this into a contribution: analyse MDL-utility vs frequency-utility for
supra-word admission; report tokens/byte per tokenizer; if a frequency/utility
prune closes the fertility gap, that is a genuine improvement, not an
embarrassment.

---

## 7. Positioning vs SuperBPE (must be explicit and early)

SuperBPE (2025) already: produces supra-word tokens, wins downstream, is
published. LiB's honest differentiators:

- **Principled stopping** (MDL) → Turkish saturation; SuperBPE has none.
- **Joint** sub+supra from one objective; SuperBPE bolts on a 2nd BPE stage.
- **Cognitive grounding** (eye fixations) — *support*, not lead, for an NLP venue.

Do **not** claim the supra-word level is "neglected" — as of SuperBPE it is not.
Claim instead: *principled* rather than *heuristic* supra-word discovery.

---

## 8. Draft reframed abstract

> Tokenizers overwhelmingly refuse to place tokens above the word: whitespace is
> a hard floor for BPE, WordPiece, and Unigram. Recent work (SuperBPE) shows
> that crossing that floor improves downstream models, yet standard intrinsic
> metrics — Description Length and n-gram bits-per-character — rank supra-word
> tokenizers *worse*, not better. We resolve this contradiction. Across seven
> typologically diverse languages we show that the intrinsic penalty against
> supra-word tokens is an **artefact of n-gram estimator sparsity**: the same
> supra-word tokens that help a 2-gram model hurt a 3-gram model, for both LiB
> and SuperBPE, and the penalty disappears under a neural bits-per-byte
> evaluation that also exposes their large sequence-compression benefit. We
> further present **LiB**, an MDL tokenizer that discovers sub- and supra-word
> units jointly and, unlike frequency-heuristic methods, stops when no unit
> reduces description length — saturating Turkish at 29k rather than filling an
> arbitrary 50k budget. The fraction of supra-word tokens LiB discovers tracks
> the analytic–synthetic continuum (2% for Chinese, 65–69% for agglutinative,
> 76–81% for analytic and templatic languages), giving a data-driven typological
> diagnostic. Our results argue that the supra-word level deserves a principled
> place in tokenization, and that evaluating it requires metrics that do not
> structurally penalise it.

---

## 9. Venue implication

- Reframed spine (metric + trade-off + MDL) → *ACL/EMNLP/TACL* main tokenization
  audience; the neural BPB experiment is expected there.
- Current cognitive-plausibility framing alone → *CogSci / CMCL* venue, where
  the eye-fixation grounding leads and downstream numbers matter less. Pick the
  venue **before** finalising which argument leads.

---

## 10. Concrete to-do (in priority order)

1. Run `evaluate_neural_bpb.py` on en + one agglutinative (fi/tr) + zh at 50k;
   confirm the BPB gap narrows vs n-gram BPC and fertility favours supra-words.
2. Add a fertility/compute table to the paper (cheap; data already computable).
3. Recast §5–6: DL/BPC as *cost side + metric-bias evidence*, BPB/fertility as
   *benefit side*.
4. Rewrite intro + abstract around the trade-off thesis; demote English bias.
5. (Optional, high value) frequency/utility prune of LiB supra-words; report
   whether it closes the fertility gap.
6. Make SuperBPE a first-class comparison throughout, not a footnote.
