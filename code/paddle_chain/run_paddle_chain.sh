#!/bin/bash
: "${DATA_ROOT:=<DATA_ROOT>}"
: "${PYTHON:=python}"
: "${PYTHON_PADDLE:=python}"
: "${REPO_ROOT:=.}"
# Paddle 链（实验 1~10 从零复现）——等待 PyTorch 复现链 P8 完成后自动接续
set -u
PPY=${PYTHON_PADDLE:-python}
BASE=${DATA_ROOT}
MARK=$BASE/repro_status.txt
PLOG=$BASE/repro_logs

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$MARK"; }

log "PAD_CHAIN_WAITING (等 p8.done)"
while [ ! -f "$PLOG/p8.done" ]; do sleep 60; done
log "PAD_CHAIN_START (paddle 2.6.2, 实验链 1~10)"

run_exp(){
  local NAME=$1 SCRIPT=$2
  if grep -q "PAD_${NAME}_DONE" "$PLOG/paddle_all.log" 2>/dev/null; then
    log "SKIP pad_$NAME"; return 0
  fi
  log "=== pad_$NAME START ==="
  $PPY -u "${REPO_ROOT}/code/paddle_chain/$SCRIPT" >> "$PLOG/paddle_all.log" 2>&1
  if grep -q "PAD_${NAME}_DONE" "$PLOG/paddle_all.log"; then
    log "=== pad_$NAME OK ==="
  else
    log "!! pad_$NAME FAILED（详见 repro_logs/paddle_all.log 尾部）"
    return 1
  fi
}

run_exp EXP1 pad_exp1_mnist.py      || exit 1
run_exp EXP2_EXP9 pad_exp2_train.py || exit 1
run_exp EXP3 pad_exp3_attacks.py    || exit 1
run_exp EXP4 pad_exp4_at.py         || exit 1
run_exp EXP8 pad_exp5678_defenses.py || exit 1   # 一个脚本按序打 5/6/7/8 四个 DONE 标记
run_exp EXP10 pad_exp10_noMJ.py     || exit 1

log "PAD_CHAIN_ALL_DONE（实验 1~10 全部完成）"
