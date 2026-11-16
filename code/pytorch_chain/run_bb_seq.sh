#!/bin/bash
: "${DATA_ROOT:=<DATA_ROOT>}"
: "${PYTHON:=python}"
: "${PYTHON_PADDLE:=python}"
: "${REPO_ROOT:=.}"
# 实验15：EfficientNet-B0 骨干扩展——基线x3 + LOO 7列x1 + MJ额外x2
set -u
VPY=${PYTHON:-python}
OUT=${DATA_ROOT}/runs/bb01
MARK=$OUT/status.txt
mkdir -p "$OUT"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$MARK"; }

log "BB_SEQ_START (efficientnet_b0: baseline x3 + LOO 7 + MJ x2)"

run_one(){
  local EXC=$1 R=$2
  local TAG
  if [ "$EXC" = "FULL" ]; then TAG="efficientnet_b0_full_r${R}"; else TAG="efficientnet_b0_holdout_${EXC}_r${R}"; fi
  if [ -f "$OUT/cfg_${TAG}.json" ]; then log "SKIP $TAG"; return 0; fi
  log "=== $TAG START ==="
  if [ "$EXC" = "FULL" ]; then
    $VPY -u code/pytorch_chain/bb_main.py --backbone efficientnet_b0 --run "$R" --epochs 3 \
      > "$OUT/log_${TAG}.txt" 2>&1
  else
    $VPY -u code/pytorch_chain/bb_main.py --backbone efficientnet_b0 --exclude "$EXC" --run "$R" --epochs 3 \
      > "$OUT/log_${TAG}.txt" 2>&1
  fi
  if grep -q TRAIN_DONE "$OUT/log_${TAG}.txt"; then
    log "=== $TAG OK ==="
  else
    log "!! $TAG FAILED"
    return 1
  fi
}

for R in 1 2 3; do run_one FULL "$R"; done
for G in ADM BigGAN GLIDE Midjourney SD15 VQDM Wukong; do run_one "$G" 1; done
for R in 2 3; do run_one Midjourney "$R"; done

log "BB_SEQ_ALL_DONE"
