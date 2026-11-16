: "${DATA_ROOT:=<DATA_ROOT>}"
: "${PYTHON:=python}"
: "${PYTHON_PADDLE:=python}"
: "${REPO_ROOT:=.}"
cd ${DATA_ROOT}
echo == gi; ls -la genimage_test genimage_parquet; ls genimage_test/* | head -30; ls genimage_parquet/data
echo == cache; ls ~/.cache; ls ~/.cache/huggingface/hub 2>/dev/null | head; ls ~/.cache/clip 2>/dev/null
echo == univfd work; ls ${WORK_DIR:-.}/univfd_fix.work ${WORK_DIR:-.}/univfd_v2.work ${WORK_DIR:-.}/genimage_eval_v3.work
grep -n "open_clip\|pretrained\|HF_\|cache_dir" ${WORK_DIR:-.}/univfd_fix.work/*.py code/pytorch_chain/bb_clip_models.py 2>/dev/null | head -20
ls runs/univfd/heads
echo == R1; tail -3 rerun_20261116/logs/R1.log; echo == R7; tail -4 rerun_20261116/logs/R7_run2.log
nvidia-smi --query-gpu=memory.used --format=csv
