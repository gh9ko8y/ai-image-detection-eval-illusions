: "${DATA_ROOT:=<DATA_ROOT>}"
: "${PYTHON:=python}"
: "${PYTHON_PADDLE:=python}"
: "${REPO_ROOT:=.}"
R=${DATA_ROOT}/rerun_20261116; PY=${PYTHON:-python}; cd ${REPO_ROOT:-.}
for r in 1 2; do
$PY ${REPO_ROOT}/code/analysis/rethreshold_calibration.py --prep ${DATA_ROOT}/data_npy --full ${DATA_ROOT}/runs/round_0$r/detector.pth --loo-dir ${DATA_ROOT}/runs/loo_01 --run $r --out $R/R7/result_run$r.json > $R/logs/R7_run$r.log 2>&1
echo EXIT $? >> $R/logs/R7_run$r.log
done
