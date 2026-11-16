: "${DATA_ROOT:=<DATA_ROOT>}"
: "${PYTHON:=python}"
: "${PYTHON_PADDLE:=python}"
: "${REPO_ROOT:=.}"
cd ${DATA_ROOT}/rerun_20261116
grep -E "^(clean|clean_filtered|whitebox_pgd|lowpass_transfer|lowpass_adaptive) \{|n_eval|eps" logs/R2_a.log | head -20
python3 -c "
import json;d=json.load(open('R2/eval_v2_eps005_r3_real700.json'));print(d['n_eval'],d['n_fake'],d['n_real'],d['config'])
for k,v in d['loo_clean_matrix'].items(): print(k, {a:round(b,1) for a,b in v.items()})
"
