: "${DATA_ROOT:=<DATA_ROOT>}"
: "${PYTHON:=python}"
: "${PYTHON_PADDLE:=python}"
: "${REPO_ROOT:=.}"
cd ${DATA_ROOT}
echo == grep 91.03; grep -rl --include=*.{log,txt,json,csv,md} "91\.03" runs repro_logs repro_master.log ${WORK_DIR:-.}/*.log 2>/dev/null | head; grep -rh --include=*.{log,txt,json,csv,md} "91\.03" runs repro_logs repro_master.log 2>/dev/null | head -5 | cut -c1-300
echo == round summaries; for r in 1 2 3; do cat runs/round_0$r/summary.json | head -c 800; echo; done
echo == at01; cat runs/at01/status.txt | tail -20; grep -h -i "overall\|整体\|clean" runs/at01/log_at_resnet18_full_r*.txt | tail -12
echo == bb01 effnet eval; ls runs/bb01 | grep -v "\.pth" | grep -v cfg_ | head; grep -h "SD15\|Wukong" runs/bb01/log_efficientnet_b0_*_{full,holdout_SD15,holdout_Wukong}_r*.txt 2>/dev/null | head -40
echo == repro_logs; ls repro_logs | head -50; tail -5 repro_status.txt
