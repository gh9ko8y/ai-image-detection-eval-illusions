: "${DATA_ROOT:=<DATA_ROOT>}"
: "${PYTHON:=python}"
: "${PYTHON_PADDLE:=python}"
: "${REPO_ROOT:=.}"
R=${DATA_ROOT}/rerun_20261116; PY=${PYTHON:-python}; cd ${REPO_ROOT:-.}; export HF_HUB_OFFLINE=1
$PY ${REPO_ROOT}/code/pipelines/univfd_stage3_v2.py --eps 0.05 --steps 10 --restarts 3 --n-real 700 --out $R/R2/eval_v2_eps005_r3_real700.json > $R/logs/R2_a.log 2>&1; echo EXIT $? >> $R/logs/R2_a.log
$PY ${REPO_ROOT}/code/pipelines/univfd_stage3_v2.py --eps 0.05 --steps 10 --restarts 1 --n-real 100 --out $R/R2/eval_v2_eps005_r1_real100.json > $R/logs/R2_b.log 2>&1; echo EXIT $? >> $R/logs/R2_b.log
$PY ${REPO_ROOT}/code/pipelines/univfd_stage3_v2.py --eps 0.02 --steps 10 --restarts 1 --n-real 700 --out $R/R2/eval_v2_eps002_r1_real700.json > $R/logs/R2_c.log 2>&1; echo EXIT $? >> $R/logs/R2_c.log
