#!/bin/bash
: "${DATA_ROOT:=<DATA_ROOT>}"
: "${PYTHON:=python}"
: "${PYTHON_PADDLE:=python}"
: "${REPO_ROOT:=.}"
# 实验15补：EfficientNet LOO 全列补重复到 r3（家族连带格需三轮均值才能引）
set -u
VPY=${PYTHON:-python}
OUT=${DATA_ROOT}/runs/bb01
MARK=$OUT/status.txt
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$MARK"; }

log "BB_SEQ3_START (EfficientNet LOO r2/r3 for all 7 gens)"

run_one(){
  local EXC=$1 R=$2
  local TAG="efficientnet_b0_holdout_${EXC}_r${R}"
  if [ -f "$OUT/cfg_${TAG}.json" ]; then log "SKIP $TAG"; return 0; fi
  log "=== $TAG START ==="
  $VPY -u code/pytorch_chain/bb_main.py --backbone efficientnet_b0 --exclude "$EXC" --run "$R" --epochs 3 \
    > "$OUT/log_${TAG}.txt" 2>&1
  if grep -q TRAIN_DONE "$OUT/log_${TAG}.txt"; then
    log "=== $TAG OK ==="
  else
    log "!! $TAG FAILED"
    return 1
  fi
}

for R in 2 3; do
  for G in ADM BigGAN GLIDE Midjourney SD15 VQDM Wukong; do
    run_one "$G" "$R"
  done
done

log "BB_SEQ3_ALL_DONE"
