: "${DATA_ROOT:=<DATA_ROOT>}"
: "${PYTHON:=python}"
: "${PYTHON_PADDLE:=python}"
: "${REPO_ROOT:=.}"
R=${DATA_ROOT}/rerun_20261116; PY=${PYTHON:-python}; cd ${REPO_ROOT:-.}
$PY ${REPO_ROOT}/code/paddle_chain/pad_defenses_torch_port.py --eps 0.02 0.05 0.10 --out $R/R4/defenses_torch_port.json > $R/logs/R4.log 2>&1; echo EXIT $? >> $R/logs/R4.log
