: "${DATA_ROOT:=<DATA_ROOT>}"
: "${PYTHON:=python}"
: "${PYTHON_PADDLE:=python}"
: "${REPO_ROOT:=.}"
R=${DATA_ROOT}/rerun_20261116; PY=${PYTHON:-python}; cd ${REPO_ROOT:-.}
$PY ${REPO_ROOT}/code/pytorch_chain/lowpass_round_v2.py --ckpt ${DATA_ROOT}/runs/round_01/detector.pth --outdir $R/R11/lowpass_clamp_r1 --eps 0.02 0.05 --clamp > $R/logs/R11_lowpass_clamp_r1.log 2>&1; echo EXIT $? >> $R/logs/R11_lowpass_clamp_r1.log
