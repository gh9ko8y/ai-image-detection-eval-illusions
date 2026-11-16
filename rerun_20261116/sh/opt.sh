: "${DATA_ROOT:=<DATA_ROOT>}"
: "${PYTHON:=python}"
: "${PYTHON_PADDLE:=python}"
: "${REPO_ROOT:=.}"
R=${DATA_ROOT}/rerun_20261116; PY=${PYTHON:-python}; cd $R; mkdir -p R8 R11
cat > sh/run_r8.sh <<'S'
R=${DATA_ROOT}/rerun_20261116; PY=${PYTHON:-python}; cd ${REPO_ROOT:-.}
for s in 0 1 2; do $PY ${REPO_ROOT}/code/pytorch_chain/train_detector_v2.py --seed $s --out $R/R8/full_seed$s > $R/logs/R8_full_seed$s.log 2>&1; echo EXIT $? >> $R/logs/R8_full_seed$s.log; done
for t in a b; do $PY ${REPO_ROOT}/code/pytorch_chain/train_detector_v2.py --seed 42 --deterministic --out $R/R8/full_det42_$t > $R/logs/R8_full_det42_$t.log 2>&1; echo EXIT $? >> $R/logs/R8_full_det42_$t.log; done
for s in 0 1 2; do $PY ${REPO_ROOT}/code/pytorch_chain/train_detector_v2.py --seed $s --exclude Midjourney --out $R/R8/loo_Midjourney_seed$s > $R/logs/R8_looMJ_seed$s.log 2>&1; echo EXIT $? >> $R/logs/R8_looMJ_seed$s.log; done
S
cat > sh/run_r11.sh <<'S'
R=${DATA_ROOT}/rerun_20261116; PY=${PYTHON:-python}; cd ${REPO_ROOT:-.}
$PY ${REPO_ROOT}/code/pytorch_chain/lowpass_round_v2.py --ckpt ${DATA_ROOT}/runs/round_01/detector.pth --outdir $R/R11/lowpass_clamp_r1 --eps 0.02 0.05 --clamp > $R/logs/R11_lowpass_clamp_r1.log 2>&1; echo EXIT $? >> $R/logs/R11_lowpass_clamp_r1.log
S
nohup bash sh/run_r8.sh >/dev/null 2>&1 &
nohup bash sh/run_r11.sh >/dev/null 2>&1 &
sleep 90; tail -n 3 logs/R8_full_seed0.log logs/R11_lowpass_clamp_r1.log
cd ${DATA_ROOT}; for r in 2 3; do for f in full holdout_SD15 holdout_Wukong; do echo runs/bb01/$f r$r; grep -E "^(SD15|Wukong) " runs/bb01/log_efficientnet_b0_${f}_r$r.txt | tail -2; done; done
