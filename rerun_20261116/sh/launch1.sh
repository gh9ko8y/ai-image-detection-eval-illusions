: "${DATA_ROOT:=<DATA_ROOT>}"
: "${PYTHON:=python}"
: "${PYTHON_PADDLE:=python}"
: "${REPO_ROOT:=.}"
R=${DATA_ROOT}/rerun_20261116; cd $R
for f in logs/R7_run1.log logs/R7_run2.log; do echo == $f; tail -n 3 $f; done
cat > sh/run_r3.sh <<'S'
R=${DATA_ROOT}/rerun_20261116; PY=${PYTHON:-python}; cd ${REPO_ROOT:-.}
for r in 1 2 3; do
$PY ${REPO_ROOT}/code/pytorch_chain/lowpass_round_v2.py --ckpt ${DATA_ROOT}/runs/round_0$r/detector.pth --outdir $R/R3/r$r --eps 0.02 0.05 0.10 > $R/logs/R3_r$r.log 2>&1; echo EXIT $? >> $R/logs/R3_r$r.log
done
S
cat > sh/run_r2.sh <<'S'
R=${DATA_ROOT}/rerun_20261116; PY=${PYTHON:-python}; cd ${REPO_ROOT:-.}; export HF_HUB_OFFLINE=1
$PY ${REPO_ROOT}/code/pipelines/univfd_stage3_v2.py --eps 0.05 --steps 10 --restarts 3 --n-real 700 --out $R/R2/eval_v2_eps005_r3_real700.json > $R/logs/R2_a.log 2>&1; echo EXIT $? >> $R/logs/R2_a.log
$PY ${REPO_ROOT}/code/pipelines/univfd_stage3_v2.py --eps 0.05 --steps 10 --restarts 1 --n-real 100 --out $R/R2/eval_v2_eps005_r1_real100.json > $R/logs/R2_b.log 2>&1; echo EXIT $? >> $R/logs/R2_b.log
$PY ${REPO_ROOT}/code/pipelines/univfd_stage3_v2.py --eps 0.02 --steps 10 --restarts 1 --n-real 700 --out $R/R2/eval_v2_eps002_r1_real700.json > $R/logs/R2_c.log 2>&1; echo EXIT $? >> $R/logs/R2_c.log
S
cat > sh/run_r6.sh <<'S'
R=${DATA_ROOT}/rerun_20261116; PY=${PYTHON:-python}; cd ${REPO_ROOT:-.}; export HF_HUB_OFFLINE=1
$PY ${REPO_ROOT}/code/pipelines/genimage_test_zero_shot_v4.py --out $R/R6/result_v4.json > $R/logs/R6.log 2>&1; echo EXIT $? >> $R/logs/R6.log
S
nohup bash sh/run_r3.sh >/dev/null 2>&1 &
nohup bash sh/run_r2.sh >/dev/null 2>&1 &
nohup bash sh/run_r6.sh >/dev/null 2>&1 &
echo == convert_ckpt.py; cat code/paddle_chain/convert_ckpt.py
echo == r2 smoke head; head -c 1500 smoke/R2.json
sleep 45; tail -n 3 logs/R6.log logs/R3_r1.log logs/R2_a.log; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
