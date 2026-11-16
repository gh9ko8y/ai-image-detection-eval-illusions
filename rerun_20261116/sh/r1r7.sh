: "${DATA_ROOT:=<DATA_ROOT>}"
: "${PYTHON:=python}"
: "${PYTHON_PADDLE:=python}"
: "${REPO_ROOT:=.}"
R=${DATA_ROOT}/rerun_20261116; PY=${PYTHON:-python}
cd $R; mkdir -p R1 R7 logs
${PYTHON:-python} -c "import numpy as np;y=np.load('${DATA_ROOT}/data_npy/y_validation.npy').reshape(-1);print('y uniq',np.unique(y,return_counts=True))"
ls ~/.cache/torch/hub/checkpoints 2>/dev/null
cat > $R/sh/run_r1.sh <<'S'
R=${DATA_ROOT}/rerun_20261116; PY=${PYTHON:-python}; cd $R
$PY code/pytorch_chain/spec_resid_fixed.py --outdir $R/R1_smoketest --n-perm 50 > logs/R1_smoketest.log 2>&1 && \
$PY code/pytorch_chain/spec_resid_fixed.py --outdir $R/R1 --n-perm 5000 > logs/R1.log 2>&1
echo EXIT $? >> logs/R1.log
S
cat > $R/sh/run_r7.sh <<'S'
R=${DATA_ROOT}/rerun_20261116; PY=${PYTHON:-python}; cd ${REPO_ROOT:-.}
for r in 1 2; do
$PY ${REPO_ROOT}/code/analysis/rethreshold_calibration.py --prep ${DATA_ROOT}/data_npy --full ${DATA_ROOT}/runs/round_0$r/detector.pth --loo-dir ${DATA_ROOT}/runs/loo_01 --run $r --out $R/R7/result_run$r.json > $R/logs/R7_run$r.log 2>&1
echo EXIT $? >> $R/logs/R7_run$r.log
done
S
nohup bash $R/sh/run_r1.sh > /dev/null 2>&1 &
nohup bash $R/sh/run_r7.sh > /dev/null 2>&1 &
sleep 60; tail -5 logs/R1_smoketest.log; tail -15 logs/R7_run1.log
