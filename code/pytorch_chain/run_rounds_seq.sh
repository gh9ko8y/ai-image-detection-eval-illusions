#!/bin/bash
: "${DATA_ROOT:=<DATA_ROOT>}"
: "${PYTHON:=python}"
: "${PYTHON_PADDLE:=python}"
: "${REPO_ROOT:=.}"
# 顺序连跑多个实验轮次（利用 GPU 空闲时间，一轮接一轮，不并行以免抢显存）
#
# 用法:
#   bash code/pytorch_chain/run_rounds_seq.sh 2 3 4          # 跑 round_02/03/04
#   bash code/pytorch_chain/run_rounds_seq.sh --after 1 2 3  # 先等 round_01 跑完，再跑 2/3
#
# 设计要点：
#   - 每轮全量重跑，不复用上一轮任何结果（多轮的意义就在于独立重复）
#   - 每轮用独立日志文件，便于单轮排查
#   - 不并行：PyTorch 训练 + FFT 攻击会吃满显存，两个进程同时跑必 OOM
#   - 前一轮失败不静默继续：记 FAILED 并通过标记文件暴露出来

set -u
VPY=${PYTHON:-python}
RUNS=${DATA_ROOT}/runs
MARK=$RUNS/rounds_seq.status

WAIT_FOR=""
if [ "${1:-}" = "--after" ]; then
  WAIT_FOR="$2"
  shift 2
fi

ROUNDS="$*"
if [ -z "$ROUNDS" ]; then
  echo "用法: bash $0 [--after N] <轮次1> <轮次2> ..."
  exit 1
fi

echo "START $(date '+%F %T')  rounds=[$ROUNDS] wait_for=[${WAIT_FOR:-none}]" | tee "$MARK"

# 等指定轮次结束
if [ -n "$WAIT_FOR" ]; then
  RID=$(printf "round_%02d" "$WAIT_FOR")
  echo "等待 $RID 完成 ..." | tee -a "$MARK"
  while pgrep -f "run_round.py $WAIT_FOR" >/dev/null 2>&1; do
    sleep 30
  done
  if grep -q "ROUND_$(printf '%02d' "$WAIT_FOR")_DONE" "$RUNS/$RID.log" 2>/dev/null; then
    echo "$RID 已完成，开始后续轮次" | tee -a "$MARK"
  else
    echo "!! $RID 未见完成标记，仍继续（会记录在状态文件里）" | tee -a "$MARK"
  fi
fi

for r in $ROUNDS; do
  RID=$(printf "round_%02d" "$r")
  if [ -d "$RUNS/$RID" ]; then
    echo "SKIP $RID （目录已存在，多轮原则要求不得复用）" | tee -a "$MARK"
    continue
  fi
  echo "=== $RID START $(date '+%F %T') ===" | tee -a "$MARK"
  $VPY -u code/pytorch_chain/run_round.py "$r" > "$RUNS/$RID.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && grep -q "ROUND_$(printf '%02d' "$r")_DONE" "$RUNS/$RID.log"; then
    echo "=== $RID OK $(date '+%F %T') ===" | tee -a "$MARK"
  else
    echo "=== $RID FAILED rc=$rc $(date '+%F %T') ===" | tee -a "$MARK"
  fi
done

echo "ALL_DONE $(date '+%F %T')" | tee -a "$MARK"
