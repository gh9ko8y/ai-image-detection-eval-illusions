#!/bin/bash
: "${DATA_ROOT:=<DATA_ROOT>}"
: "${PYTHON:=python}"
: "${PYTHON_PADDLE:=python}"
: "${REPO_ROOT:=.}"
# 实验16：CLIP ViT-B/16 骨干扩展——基线补 r2/r3 + LOO 7列 + MJ额外x2
# （r1 已由冒烟测试以正式口径完成：3 epoch？否——冒烟是 1 epoch，故 r1 也重跑）
set -u
export HF_ENDPOINT=https://hf-mirror.com
VPY=${PYTHON:-python}
OUT=${DATA_ROOT}/runs/bb02
MARK=$OUT/status.txt
mkdir -p "$OUT"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$MARK"; }

log "BB2_SEQ_START (clip_vitb16: baseline x3 + LOO 7 + MJ x2)"

run_one(){
  local EXC=$1 R=$2
  local TAG
  if [ "$EXC" = "FULL" ]; then TAG="clip_vitb16_full_r${R}"; else TAG="clip_vitb16_holdout_${EXC}_r${R}"; fi
  if [ -f "$OUT/cfg_${TAG}.json" ]; then log "SKIP $TAG"; return 0; fi
  log "=== $TAG START ==="
  if [ "$EXC" = "FULL" ]; then
    $VPY -u code/pytorch_chain/bb02_main.py --run "$R" --epochs 3 > "$OUT/log_${TAG}.txt" 2>&1
  else
    $VPY -u code/pytorch_chain/bb02_main.py --exclude "$EXC" --run "$R" --epochs 3 > "$OUT/log_${TAG}.txt" 2>&1
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

log "BB2_SEQ_ALL_DONE"
