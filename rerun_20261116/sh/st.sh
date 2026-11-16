: "${DATA_ROOT:=<DATA_ROOT>}"
: "${PYTHON:=python}"
: "${PYTHON_PADDLE:=python}"
: "${REPO_ROOT:=.}"
R=${DATA_ROOT}/rerun_20261116; cd $R
for f in logs/R3_r1.log logs/R3_r2.log logs/R3_r3.log logs/R2_a.log logs/R2_b.log logs/R2_c.log logs/R4.log; do echo == $f; [ -f $f ] && grep -v "^batch" $f | tail -n 6 | cut -c1-400; [ -f $f ] && grep -c "^batch" $f; done
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; ps aux | grep -c "[p]ython.*rerun_20261116"
