: "${DATA_ROOT:=<DATA_ROOT>}"
: "${PYTHON:=python}"
: "${PYTHON_PADDLE:=python}"
: "${REPO_ROOT:=.}"
R=${DATA_ROOT}/rerun_20261116; PY=${PYTHON:-python}; cd ${REPO_ROOT:-.}
for r in 1 2 3; do
$PY ${REPO_ROOT}/code/pytorch_chain/lowpass_round_v2.py --ckpt ${DATA_ROOT}/runs/round_0$r/detector.pth --outdir $R/R3/r$r --eps 0.02 0.05 0.10 > $R/logs/R3_r$r.log 2>&1; echo EXIT $? >> $R/logs/R3_r$r.log
done
