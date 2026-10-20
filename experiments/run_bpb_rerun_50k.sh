#!/bin/bash
# Rerun the monolingual 50k neural BPB table, one tokenizer column per process.
# The 2026-07-28 sweep, which trained every tokenizer in one process, gives
# BPB 0.01-0.07 higher for BPE, WordPiece, SuperBPE and LiB than every later
# run with the same seed, data and code (zh LiB 2.1113 -> 2.0557), so its rows
# cannot be compared with the size-matched and no-supra runs. The cause is not
# known. It is not the macOS 27 update: SP-Unigram, rerun per process on
# 2026-08-01 under macOS 26, reproduces to four decimals. LiB goes first (zh
# LiB was rerun already), then the baselines. Rows were collected in
# neural_bpb_{lang}_50000_macos27.csv and then merged into
# neural_bpb_{lang}_50000.csv together with the LiB (no supra) row.
set -u
cd "$(dirname "$0")/.."
R=experiments/results

run() {  # lang, tokenizer names...
  local l=$1; shift
  local f=$R/neural_bpb_${l}_50000.csv out=$R/neural_bpb_${l}_50000_macos27.csv
  cp $f $f.keep
  python experiments/evaluate_neural_bpb.py --lang $l --vocab-size 50000 \
    --only "$@" --no-symbol --max-steps 5000 --eval-every 1000
  [ -f $out ] || head -1 $f > $out
  tail -n +2 $f >> $out
  mv $f.keep $f
}

for l in en de es fi tr ar; do run $l LiB; done
echo "LIB DONE"
for l in en de es fi tr ar zh; do run $l BPE WordPiece SP-Unigram SuperBPE; done
echo "BASELINES DONE"
