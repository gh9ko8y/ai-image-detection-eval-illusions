: "${DATA_ROOT:=<DATA_ROOT>}"
: "${PYTHON:=python}"
: "${PYTHON_PADDLE:=python}"
: "${REPO_ROOT:=.}"
cd ${DATA_ROOT}/rerun_20261116; date
ps aux | grep -E "pad_defenses_torch_port|train_detector_v2|univfd|lowpass_round" | grep -v grep | wc -l
tail -n 6 logs/R4.log; for f in logs/R8_*.log; do echo "== $f"; grep -E "EXIT|overall|Overall" $f | tail -n 3; done
tar czf ${DATA_ROOT}/rerun_20261116/rr_part2.tgz --exclude='*.npz' --exclude='*.pth' --exclude='*.npy' --exclude='rr_part*.tgz' R1 R2 R3 R4 R5 R6 R7 R8 R11 logs sh code 2>&1 | tail -n 3
ls -la rr_part2.tgz
