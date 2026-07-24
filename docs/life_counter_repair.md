# LiB forgetting-mechanism repair — design note

## What was broken

The Rust port of LiB (`tokenizers-fork/.../models/lib/`) exposed the exact
hyperparameters of Yang et al. (2020), "Less is Better" (CogALex) — α=`memory_in`,
Δ=`update_rate`, ω=`memory_out`, τ₀-family=`life` — but never implemented the
paper's **forgetting** mechanism (§2.3.2). Instead the loop called
`prune(memory_out)`, which **immediately deletes** the tail ω|L| entries every
epoch — the strategy the paper explicitly rejects. The `life` and `frequency`
fields were serialized but never updated (constant `10`/`0` in every saved
model), so the observation-count logic that is supposed to keep useful units
alive did not exist.

Because immediate deletion only ever touches the lowest-priority tail, and the
port also never re-ranked existing units, rarely re-encountered supra-word units
froze mid-lexicon and survived. Empirically (English, 50k): 58% of the vocabulary
is supra-word units, 7.5% never fire, and the bottom ~90% carry ~3% of all supra
firings. **Every prior LiB number reflects this broken state.**

## The repair — reference-faithful active + passive forgetting

> An earlier iteration of this repair used a single observation-count per unit
> (`observe`/`rerank_and_forget`); it was replaced by the design below, which
> follows Yang et al. (2020) §2.3.1–2.3.2 directly. All five 2020 knobs are
> live.

The lexicon is a priority-ordered list; a unit's ordinal position Θ (0 = head)
is its memory. Each epoch processes one **document** = `doc_size` sentences
(new parameter; the paper's epoch unit), accumulating one reward list and one
per-document candidate set (twice-rule), then applies one batched update:

- **Admission** (`trainer.rs::read_sentence`): adjacent-chunk concatenations
  are α-sampled as candidates; a candidate sighted twice within the same
  document is admitted (below the cap) and appended at the tail — lowest
  priority, immediately at risk.
- **Active forgetting = ordinal reward re-ranking** (`trie.rs::group_move`,
  §2.3.2): during segmentation the longest match c₁ is evaluated against the
  second-longest c₂ by rolling both branches out greedily until they reconverge
  (§2.3.1: fewer unknowns, then fewer chunks, then lower ordinal sum wins).
  Good chunks move headward by ⌊Θ·Δ⌋+1, bad chunks tailward by the same step;
  a bad chunk demoted past |L| is deleted.
- **Passive forgetting = probation** (`trainer.rs::to_dropout`,
  `trie.rs::group_remove`): once per document every watched unit's remaining
  life is decremented and expired units are forgotten; then the bottom ω·|L|
  tail is placed on watch with life τ₀. Seed units start on watch. Probation
  is cancelled when a unit is used without being judged bad.
- Atomic units (single chars, `<0x..>` byte-fallback) and special tokens are
  never deleted by either route (lossless reconstruction preserved).

A unit that keeps winning comparisons converges multiplicatively on the safe
head; a stale unit is demoted out of it, drifts into the watched tail, and
expires after τ₀ unused documents. The vocabulary self-regulates: admission
and forgetting reach a balance rather than pinning at the `vocab_size` cap.

### Parameter meanings after the repair

| Symbol | Field | Role now |
|---|---|---|
| α | `memory_in` | candidate sampling probability (unchanged) |
| Δ | `update_rate` | ordinal re-ranking rate (active forgetting): step ⌊Θ·Δ⌋+1 |
| ω | `memory_out` | forgetting ratio: fraction of the tail placed on probation per document |
| τ₀ | `life` | probation period in documents (passive forgetting) |
| — | `doc_size` | sentences per document/epoch (new; default 50) |

Status: compiles with no warnings; full LiB suite passes, incl. the end-to-end
training tests. `prune`/`batch_update` are retained (still unit-tested) but no
longer called by the training loop — `group_move`/`group_remove` replace them.

## Follow-ups before retraining — all done (2026-07)

1. ~~Python binding~~: `doc_size` exposed in `bindings/python/src/trainers.rs`
   and `LiBTokenizerFast.train_new`; extension rebuilt.
2. ~~`num_epochs` semantics~~: an epoch is one document of `doc_size` sentences;
   retuned (final: 5000 epochs × doc_size 100, zh doc_size 50).
3. ~~Re-run `hparam_search.py`~~: done for en/fi/tr/zh at 50k
   (`results/hparam_obs_*.csv`); consensus config in
   `experiments/regenerate_lib.py`.
4. Fate of `prune`/`batch_update`: still open — remove or keep as unit-tested
   alternatives.
5. ~~Re-run all evaluations~~: done — `results/lib_fixed_eval_summary.md`.
   **All prior LiB numbers are superseded.** The frequency-prune experiment
   (`prune_supra*.py`) was the diagnostic that motivated this repair, not a
   proposed method.
