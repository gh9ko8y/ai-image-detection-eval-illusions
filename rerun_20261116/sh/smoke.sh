: "${DATA_ROOT:=<DATA_ROOT>}"
: "${PYTHON:=python}"
: "${PYTHON_PADDLE:=python}"
: "${REPO_ROOT:=.}"
R=${DATA_ROOT}/rerun_20261116; PY=${PYTHON:-python}; PP=${PYTHON_PADDLE:-python}
cd $R; mkdir -p smoke R2 R3 R4 R6
cp sh/up/lowpass_round_v2.py code/pytorch_chain/; cp sh/up/univfd_stage3_v2.py code/pipelines/; cp sh/up/pad_exp5678_defenses_v2.py code/paddle_chain/
echo == R7 status; tail -2 logs/R7_run1.log logs/R7_run2.log
echo == R3 smoke
cd ${REPO_ROOT:-.} && timeout 600 $PY ${REPO_ROOT}/code/pytorch_chain/lowpass_round_v2.py --ckpt ${DATA_ROOT}/runs/round_01/detector.pth --outdir $R/smoke/R3 --n 300 --eps 0.05 2>&1 | tail -5
echo == R2 smoke
HF_HUB_OFFLINE=1 timeout 900 $PY ${REPO_ROOT}/code/pipelines/univfd_stage3_v2.py --n-fake-per-gen 5 --n-real 35 --restarts 1 --out $R/smoke/R2.json 2>&1 | grep -v "^batch" | tail -12
echo == R4 smoke
cd ${REPO_ROOT:-.} && timeout 900 $PP ${REPO_ROOT}/code/paddle_chain/pad_exp5678_defenses_v2.py --n 100 --eps 0.05 --out $R/smoke/R4.json 2>&1 | tail -12
