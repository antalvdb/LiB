#!/bin/bash
# SuperBPE at LiB's emergent vocabulary size, monolingual and joint.
# Trains the tokenizers (CPU), then runs n-gram BPC/DL (CPU lane) and
# neural BPB (MPS lane) in parallel. Monolingual BPB uses the settings of
# the original sweep (--max-steps 5000 --eval-every 1000).
set -u
cd "$(dirname "$0")/.."
R=experiments/results
J="python experiments/joint_multilingual.py"
declare -a SIZES=(en:28423 de:26579 es:25106 fi:20262 tr:13585 ar:27315 zh:26269)
JOINT=116322

for ls in joint:$JOINT "${SIZES[@]}"; do
  conda run -n superbpe --no-capture-output python -c \
    "import sys; sys.path.insert(0,'experiments'); from train_superbpe import train_superbpe; train_superbpe('${ls%%:*}', ${ls##*:}, skip_existing=True)"
done > $R/superbpe_matched_train.log 2>&1
echo "TRAIN EXIT $?" >> $R/superbpe_matched_train.log

(
  $J eval --only "SuperBPE @m"
  for ls in "${SIZES[@]}"; do
    python experiments/evaluate_bpc.py --lang ${ls%%:*} --vocab-size ${ls##*:} --no-symbol
  done
  echo "CPU LANE DONE"
) > $R/superbpe_matched_bpc.log 2>&1 &
A=$!

(
  $J bpb --only "SuperBPE @m"
  for ls in "${SIZES[@]}"; do
    python experiments/evaluate_neural_bpb.py --lang ${ls%%:*} --vocab-size ${ls##*:} \
      --only SuperBPE --no-symbol --max-steps 5000 --eval-every 1000
  done
  echo "MPS LANE DONE"
) > $R/superbpe_matched_bpb.log 2>&1 &
B=$!

wait $A $B
echo "SUPERBPE MATCHED ALL DONE"
