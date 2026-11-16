: "${DATA_ROOT:=<DATA_ROOT>}"
: "${PYTHON:=python}"
: "${PYTHON_PADDLE:=python}"
: "${REPO_ROOT:=.}"
cd ${DATA_ROOT}/rerun_20261116; date
for f in logs/R2_b.log logs/R2_c.log logs/R4.log logs/R11_lowpass_clamp_r1.log; do echo "== $f"; tail -n 4 $f; done
for f in logs/R8_*.log; do echo "== $f"; grep -E "overall|^(ADM|BigGAN|GLIDE|Midjourney|SD15|VQDM|Wukong|Real) |EXIT" $f | tail -n 10; done
ls -la R2 R4 R8 R11
ps aux | grep -E "python" | grep -v grep | awk '{print $2,$10,$12,$13,$14,$15,$16}'
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
