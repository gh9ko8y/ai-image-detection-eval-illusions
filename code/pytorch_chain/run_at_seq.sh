#!/bin/bash
: "${DATA_ROOT:=<DATA_ROOT>}"
: "${PYTHON:=python}"
: "${PYTHON_PADDLE:=python}"
: "${REPO_ROOT:=.}"
# 实验17：AT×泛化——PGD-AT 基线x3 + 关键 LOO 列（MJ/ADM/VQDM 各x1）
set -u
VPY=${PYTHON:-python}
OUT=${DATA_ROOT}/runs/at01
MARK=$OUT/status.txt
mkdir -p "$OUT"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$MARK"; }

log "AT_SEQ_START (PGD-AT resnet18: baseline x3 + MJ/ADM/VQDM loo)"

run_one(){
  local EXC=$1 R=$2
  local TAG
  if [ "$EXC" = "FULL" ]; then TAG="at_resnet18_full_r${R}"; else TAG="at_resnet18_holdout_${EXC}_r${R}"; fi
  if [ -f "$OUT/cfg_${TAG}.json" ]; then log "SKIP $TAG"; return 0; fi
  log "=== $TAG START ==="
  if [ "$EXC" = "FULL" ]; then
    $VPY -u code/pytorch_chain/at_main.py --run "$R" --epochs 3 > "$OUT/log_${TAG}.txt" 2>&1
  else
    $VPY -u code/pytorch_chain/at_main.py --exclude "$EXC" --run "$R" --epochs 3 > "$OUT/log_${TAG}.txt" 2>&1
  fi
  if grep -q TRAIN_DONE "$OUT/log_${TAG}.txt"; then
    log "=== $TAG OK ==="
  else
    log "!! $TAG FAILED"
    return 1
  fi
}

for R in 1 2 3; do run_one FULL "$R"; done
for G in Midjourney ADM VQDM; do run_one "$G" 1; done

log "AT_SEQ_ALL_DONE"
