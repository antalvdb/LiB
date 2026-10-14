#!/bin/bash
# Monolingual neural BPB for BPE, WordPiece and SP-Unigram at LiB's emergent
# size, with the settings of the original sweep. evaluate_neural_bpb.py
# rewrites neural_bpb_{lang}_{size}.csv with only the rows it ran, so the
# committed SuperBPE row is merged back afterwards.
set -u
cd "$(dirname "$0")/.."
R=experiments/results
for ls in en:28423 de:26579 es:25106 fi:20262 tr:13585 ar:27315 zh:26269; do
  l=${ls%%:*}; s=${ls##*:}; f=$R/neural_bpb_${l}_${s}.csv
  sbpe=$(git show HEAD:$f | grep '^SuperBPE,')
  python experiments/evaluate_neural_bpb.py --lang $l --vocab-size $s \
    --only BPE WordPiece SP-Unigram --no-symbol --max-steps 5000 --eval-every 1000
  echo "$sbpe" >> $f
done
echo "MATCHED MONO BPB DONE"
