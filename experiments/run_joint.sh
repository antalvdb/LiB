#!/bin/bash
# Joint multilingual experiment driver (see joint_multilingual.py).
# Lane A (CPU): LiB at a 350k cap. Lane B: 50k baselines, SuperBPE, then their
# neural LMs on MPS. Then matched baselines, full eval, and LiB's neural LMs.
set -u
cd "$(dirname "$0")/.."
R=experiments/results
J="python experiments/joint_multilingual.py"

( $J lib > $R/joint_lib.log 2>&1; echo "LANE A EXIT $?" >> $R/joint_lib.log ) &
A=$!

(
  $J baselines
  conda run -n superbpe --no-capture-output python -c "import sys; sys.path.insert(0,'experiments'); from train_superbpe import train_superbpe; train_superbpe('joint', 50000, skip_existing=True)"
  $J bpb --only BPE WordPiece SP-Unigram SuperBPE
  echo "LANE B EXIT $?"
) > $R/joint_baselines.log 2>&1 &
B=$!

wait $A $B
(
  $J matched && $J eval
  $J bpb --only "LiB (no supra)" LiB
  echo "JOINT ALL DONE"
) > $R/joint_final.log 2>&1
