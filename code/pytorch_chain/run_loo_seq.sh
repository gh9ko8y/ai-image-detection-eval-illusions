#!/bin/bash
: "${DATA_ROOT:=<DATA_ROOT>}"
: "${PYTHON:=python}"
: "${PYTHON_PADDLE:=python}"
: "${REPO_ROOT:=.}"
# LOO 矩阵顺序连跑：7 个生成器逐一留出重训 + 结构化评估；Midjourney 额外重复 2 次。
# 训练完全复用 train_detector.py（与轮次 1~3 同配：Adam 1e-4 / bs 64 / 3 epoch / seed 42）。
# 用法:
#   nohup bash code/pytorch_chain/run_loo_seq.sh > ${DATA_ROOT}/runs/loo_seq.log 2>&1 &
set -u
VPY=${PYTHON:-python}
PREP=${DATA_ROOT}/data_npy
OUT=${DATA_ROOT}/runs/loo_01
MARK=$OUT/status.txt
mkdir -p "$OUT"

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$MARK"; }

log "LOO_SEQ_START"

# 数据指纹 + 环境快照（一次性；跨实验可比性的根基）
if [ ! -f "$OUT/data_checksum.txt" ]; then
  md5sum $PREP/X_train.npy $PREP/X_validation.npy $PREP/y_train.npy $PREP/y_validation.npy \
    > "$OUT/data_checksum.txt" 2>&1
  $VPY - <<'PYEOF' > "$OUT/env.json"
import json, torch, torchvision, numpy, pandas
print(json.dumps({'torch': torch.__version__, 'torchvision': torchvision.__version__,
                  'numpy': numpy.__version__, 'pandas': pandas.__version__,
                  'gpu': torch.cuda.get_device_name(0)}, indent=1))
PYEOF
fi

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
  run_one "$G" 1
done
# 头条单元格（留出 MJ）额外重复 2 次：同 seed 依赖运行非确定性，与轮次 1~3 方法一致
for R in 2 3; do
  run_one Midjourney "$R"
done

log "LOO_SEQ_ALL_DONE"
