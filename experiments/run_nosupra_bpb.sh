#!/bin/bash
# Neural BPB for LiB without supra-word lookup, after the no-supra fix in
# tokenizers-fork 60aa7cbf, with the settings of the original sweep.
# evaluate_neural_bpb.py rewrites neural_bpb_{lang}_50000.csv with only the
# rows it ran, so the new no-supra row is merged into the committed file.
# Then the joint model: its old no-supra row is dropped so bpb() reruns it.
set -u
cd "$(dirname "$0")/.."
R=experiments/results
for l in en de es fi tr ar zh; do
  f=$R/neural_bpb_${l}_50000.csv
  cp $f $f.orig
  python experiments/evaluate_neural_bpb.py --lang $l --vocab-size 50000 \
    --only "LiB (no supra)" --no-symbol --max-steps 5000 --eval-every 1000
  new=$(grep '^LiB (no supra),' $f)
  python - "$f.orig" "$new" > $f <<'E'
import sys
orig, new = sys.argv[1], sys.argv[2]
for line in open(orig):
    print(new if line.startswith("LiB (no supra),") else line.rstrip("\n"))
E
  rm $f.orig
done
echo "MONO NOSUPRA BPB DONE"
j=$R/joint_neural_bpb.csv
grep -v '^LiB (no supra),' $j > $j.tmp && mv $j.tmp $j
python experiments/joint_multilingual.py bpb --only "LiB (no supra)"
echo "JOINT NOSUPRA BPB DONE"
