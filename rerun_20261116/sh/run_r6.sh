: "${DATA_ROOT:=<DATA_ROOT>}"
: "${PYTHON:=python}"
: "${PYTHON_PADDLE:=python}"
: "${REPO_ROOT:=.}"
R=${DATA_ROOT}/rerun_20261116; PY=${PYTHON:-python}; cd ${REPO_ROOT:-.}; export HF_HUB_OFFLINE=1
$PY ${REPO_ROOT}/code/pipelines/genimage_test_zero_shot_v4.py --out $R/R6/result_v4.json > $R/logs/R6.log 2>&1; echo EXIT $? >> $R/logs/R6.log
