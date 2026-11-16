: "${DATA_ROOT:=<DATA_ROOT>}"
: "${PYTHON:=python}"
: "${PYTHON_PADDLE:=python}"
: "${REPO_ROOT:=.}"
cd ${DATA_ROOT}
for r in round_01 loo_01 univfd paddle_01 genimage_test_eval sdxl_zeroshot spec_resid spec_perm spec_01 bb01 bb02 at01 auroc_audit; do echo "== runs/$r"; ls runs/$r | head -30; done
echo == genimage_test; ls genimage_test | head; ls genimage_parquet; ls genimage_parquet/data | head -40
echo == hf_models; ls hf_models
ls venv/bin | grep python; venv/bin/python -c "import torch,sklearn,scipy,numpy;print(torch.__version__,torch.cuda.is_available(),numpy.__version__)"
venv_paddle/bin/python -c "import paddle;print(paddle.__version__)" 2>&1 | tail -1
venv/bin/python - <<'P'
import pandas as pd
m=pd.read_csv('data_npy/validation_meta_named.csv'); print(m.shape); print(m.gen_name.value_counts().to_dict())
P
