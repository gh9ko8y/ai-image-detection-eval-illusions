#!/bin/bash
: "${DATA_ROOT:=<DATA_ROOT>}"
: "${PYTHON:=python}"
: "${PYTHON_PADDLE:=python}"
: "${REPO_ROOT:=.}"
# LOO 矩阵补重复（实验 13 第二批）：
#   全 7 列跑 r2 + GLIDE/VQDM/SD15/Wukong 跑 r3（家族连带关键列，凑满三次）
# 输出与第一批同目录（runs/loo_01），CSV 追加，run 列区分；已完成列自动 SKIP。
# 用法:
#   nohup bash code/pytorch_chain/run_loo_seq2.sh > ${DATA_ROOT}/runs/loo_seq2.log 2>&1 &
set -u
VPY=${PYTHON:-python}
PREP=${DATA_ROOT}/data_npy
OUT=${DATA_ROOT}/runs/loo_01
MARK=$OUT/status.txt
mkdir -p "$OUT"

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$MARK"; }

log "LOO2_SEQ_START （补重复：全列 r2 + GLIDE/VQDM/SD15/Wukong r3）"

run_one(){
  local G=$1 R=$2
  local TAG=${G}_r${R}
  local CKPT=$OUT/detector_${TAG}.pth
  if [ -f "$OUT/eval_${TAG}.json" ]; then log "SKIP $TAG （评估已存在）"; return 0; fi
  log "=== $TAG TRAIN START ==="
  if [ ! -f "$CKPT" ]; then
    $VPY -u code/pytorch_chain/train_detector.py --prep $PREP --exclude "$G" --out $OUT/detector_${TAG} \
         --epochs 3 --bs 64 --lr 1e-4 --seed 42 > $OUT/log_${TAG}.txt 2>&1
    local rc=$?
    if [ $rc -ne 0 ] || ! grep -q TRAIN_DONE $OUT/log_${TAG}.txt; then
      log "!! $TAG TRAIN FAILED rc=$rc"; return 1
    fi
  else
    log "$TAG 已有 ckpt，跳过训练"
  fi
  log "=== $TAG EVAL ==="
  $VPY -u code/pytorch_chain/loo_eval.py --prep $PREP --ckpt "$CKPT" --held-out "$G" --run "$R" \
       --outdir $OUT >> $OUT/log_${TAG}.txt 2>&1
  if grep -q EVAL_DONE $OUT/log_${TAG}.txt; then
    log "=== $TAG OK ==="
  else
    log "!! $TAG EVAL FAILED"
    return 1
  fi
}

for G in ADM BigGAN GLIDE Midjourney SD15 VQDM Wukong; do
  run_one "$G" 2
done
for G in GLIDE VQDM SD15 Wukong; do
  run_one "$G" 3
done

log "LOO2_SEQ_ALL_DONE"
