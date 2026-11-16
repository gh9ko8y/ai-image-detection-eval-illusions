: "${DATA_ROOT:=<DATA_ROOT>}"
: "${PYTHON:=python}"
: "${PYTHON_PADDLE:=python}"
: "${REPO_ROOT:=.}"
R=${DATA_ROOT}/rerun_20261116; PY=${PYTHON:-python}; cd ${REPO_ROOT:-.}; mkdir -p $R/R4
cp $R/sh/up/pad_defenses_torch_port.py ${REPO_ROOT}/code/paddle_chain/
CUDA_VISIBLE_DEVICES="" ${PYTHON_PADDLE:-python} ${REPO_ROOT}/code/paddle_chain/pad_defenses_torch_port.py convert 2>&1 | grep -v "^W1116" | tail -3
timeout 900 $PY ${REPO_ROOT}/code/paddle_chain/pad_defenses_torch_port.py --n 300 --eps 0.02 --out $R/smoke/R4_port.json 2>&1 | tail -12
cat > $R/sh/run_r4.sh <<'S'
R=${DATA_ROOT}/rerun_20261116; PY=${PYTHON:-python}; cd ${REPO_ROOT:-.}
$PY ${REPO_ROOT}/code/paddle_chain/pad_defenses_torch_port.py --eps 0.02 0.05 0.10 --out $R/R4/defenses_torch_port.json > $R/logs/R4.log 2>&1; echo EXIT $? >> $R/logs/R4.log
S
nohup bash $R/sh/run_r4.sh >/dev/null 2>&1 &
cd $R; for f in logs/R3_r1.log logs/R3_r2.log logs/R2_a.log logs/R2_b.log logs/R6.log; do echo == $f; grep -v "^batch" $f | tail -n 4 | cut -c1-600; done
