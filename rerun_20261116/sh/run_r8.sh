: "${DATA_ROOT:=<DATA_ROOT>}"
: "${PYTHON:=python}"
: "${PYTHON_PADDLE:=python}"
: "${REPO_ROOT:=.}"
R=${DATA_ROOT}/rerun_20261116; PY=${PYTHON:-python}; cd ${REPO_ROOT:-.}
for s in 0 1 2; do $PY ${REPO_ROOT}/code/pytorch_chain/train_detector_v2.py --seed $s --out $R/R8/full_seed$s > $R/logs/R8_full_seed$s.log 2>&1; echo EXIT $? >> $R/logs/R8_full_seed$s.log; done
for t in a b; do $PY ${REPO_ROOT}/code/pytorch_chain/train_detector_v2.py --seed 42 --deterministic --out $R/R8/full_det42_$t > $R/logs/R8_full_det42_$t.log 2>&1; echo EXIT $? >> $R/logs/R8_full_det42_$t.log; done
for s in 0 1 2; do $PY ${REPO_ROOT}/code/pytorch_chain/train_detector_v2.py --seed $s --exclude Midjourney --out $R/R8/loo_Midjourney_seed$s > $R/logs/R8_looMJ_seed$s.log 2>&1; echo EXIT $? >> $R/logs/R8_looMJ_seed$s.log; done
