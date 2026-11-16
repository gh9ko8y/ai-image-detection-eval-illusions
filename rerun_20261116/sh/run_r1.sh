: "${DATA_ROOT:=<DATA_ROOT>}"
: "${PYTHON:=python}"
: "${PYTHON_PADDLE:=python}"
: "${REPO_ROOT:=.}"
R=${DATA_ROOT}/rerun_20261116; PY=${PYTHON:-python}; cd $R
$PY code/pytorch_chain/spec_resid_fixed.py --outdir $R/R1_smoketest --n-perm 50 > logs/R1_smoketest.log 2>&1 && \
$PY code/pytorch_chain/spec_resid_fixed.py --outdir $R/R1 --n-perm 5000 > logs/R1.log 2>&1
echo EXIT $? >> logs/R1.log
