: "${DATA_ROOT:=<DATA_ROOT>}"
: "${PYTHON:=python}"
: "${PYTHON_PADDLE:=python}"
: "${REPO_ROOT:=.}"
cd ${DATA_ROOT}/rerun_20261116; date
ps aux | grep -E "train_detector_v2|pad_defenses|univfd|lowpass_round" | grep -v grep | wc -l
awk '/评估 epoch 3/{p=1} p&&/^(整体|Midjourney|Real)/{print}' logs/R8_looMJ_seed2.log; tail -n 2 logs/R8_looMJ_seed2.log
tar czf rr_part3.tgz --exclude='*.npz' --exclude='*.pth' --exclude='*.npy' --exclude='rr_part*.tgz' R1 R2 R3 R4 R5 R6 R7 R8 R11 logs sh code
ls -la rr_part3.tgz
