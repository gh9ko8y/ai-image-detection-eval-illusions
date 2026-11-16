: "${DATA_ROOT:=<DATA_ROOT>}"
: "${PYTHON:=python}"
: "${PYTHON_PADDLE:=python}"
: "${REPO_ROOT:=.}"
cd ${REPO_ROOT:-.}
md5sum spec_resid.py spec_fp.py lowpass_round.py lowpass_final.py lowpass_whitebox.py lowpass_small_eps.py lowpass_adaptive_torch.py train_detector.py pad_common.py pad_exp5678_defenses.py bb_models.py bb_clip_models.py at_main.py loo_eval.py eval_auroc.py
ls univfd_fix.work univfd_v2.work genimage_eval_v3.work sdxl_work_mod spec_perm_holm.work
echo == R5
mkdir -p ${DATA_ROOT}/rerun_20261116/R5
cd ${DATA_ROOT}/rerun_20261116
${PYTHON:-python} code/analysis/check_val_order.py ${DATA_ROOT}/data_npy/validation_meta_named.csv | tee R5/check_val_order.txt
${PYTHON:-python} - <<'P' | tee -a R5/check_val_order.txt
import pandas as pd
m=pd.read_csv('${DATA_ROOT}/data_npy/validation_meta_named.csv')
g=m.gen_name.values
import itertools
runs=[(k,len(list(v))) for k,v in itertools.groupby(g)]
print('n_runs',len(runs),'first runs',runs[:12])
P
echo == spec_resid.py; cat code/pytorch_chain/spec_resid_fixed.py
