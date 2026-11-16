# Evaluation Illusions in AI-Generated Image Detection

**Paper title:** Evaluation Illusions in AI-Generated Image Detection: Threat-Model Framing, Aggregated Metrics, and Run-to-Run Variance

This repository is the reproducibility package for the paper. It contains training, attack, evaluation, and figure-generation scripts; the original quantitative outputs packaged with the artifact; and the logs and result tables from the 16 November 2026 (UTC+8) reruns (R1–R7, plus partial optional runs).

Trained model weights are not included in this repository. They can be reproduced from the scripts and public data, or requested from the authors. A public release of the checkpoints is planned.

## Contents

| Path | Description |
|---|---|
| `code/pytorch_chain/` | PyTorch training, attacks, LOO evaluation, spectral fingerprint, and low-pass sweeps |
| `code/paddle_chain/` | PaddlePaddle 2.6.2 chain (MNIST calibration, detector training, attacks, PGD-AT, four defenses, data prep) and the PyTorch port of the four-defense evaluation |
| `code/pipelines/` | UnivFD, SDXL, and GenImage test-set probes |
| `code/analysis/` | Threshold-free AUROC audit and the R5/R7 calibration scripts |
| `code/figures/` | Paper figure generation |
| `data/results_json_csv/` | Original (pre-rerun) quantitative outputs |
| `data/figures/` | Figures as packaged in the original artifact |
| `rerun_20261116/` | Result tables, logs, sanitized launch scripts, updated figures, and notes from the 2026-11-16 reruns |

Corrected scripts added for those reruns live **alongside** the originals under `code/` (for example `spec_resid_fixed.py`, `univfd_stage3_v2.py`, `lowpass_round_v2.py`, `pad_defenses_torch_port.py`). Original filenames were not overwritten.

## Environment

- One CUDA GPU (≥12 GB recommended; some UnivFD / SDXL settings need more).
- PyTorch 2.x, torchvision, NumPy, pandas, Pillow, scikit-learn, `open_clip_torch`.
- For the original PaddlePaddle chain: PaddlePaddle 2.6.2 (see [Known pitfalls](#known-pitfalls)).
- The 2026-11-16 reruns were executed on an NVIDIA RTX PRO 6000 Blackwell (compute capability sm_120). PaddlePaddle 2.6.2 does not support that GPU, so the four-defense evaluation (R4) was repeated with a PyTorch port of the Paddle detector. Clean accuracy after the port was 90.79%, matching the original Paddle number recorded for that checkpoint.

The original full-repro script `code/pytorch_chain/run_full_repro.sh` (header dated 2026-10-16) pins `torch==2.14.1`, `torchvision==0.29.1`, `numpy==2.5.4`, `pandas==3.0.6`, and also installs `pyarrow` and `open_clip_torch`. It sets `HF_ENDPOINT=https://hf-mirror.com`.

## Paths

Hard-coded server paths in the original scripts have been replaced. Python scripts read the environment variable `DATA_ROOT` (they fall back to the placeholder `<DATA_ROOT>` if it is unset). Shell scripts use `DATA_ROOT`, `PYTHON`, `PYTHON_PADDLE`, and `REPO_ROOT`.

```bash
export DATA_ROOT=/path/to/your/data    # data_npy/, later runs/, optional parquet
export REPO_ROOT=/path/to/this/repo    # default: current directory
export PYTHON=python                   # optional
```

Expected layout under `$DATA_ROOT`:

- `data_npy/X_train.npy`, `X_validation.npy`, `y_train.npy`, `y_validation.npy`, plus metadata CSVs such as `validation_meta_named.csv`
- `runs/` for checkpoints and intermediate outputs created by the training scripts
- `genimage_parquet/data/` for the GenImage test-set parquet shards used by R6
- `hf_models/sdxl-base/` only if you run the optional SDXL generation script

## Data preparation

**Tiny-GenImage**, a subset of GenImage (Zhu et al., NeurIPS 2023), is loaded as the Hugging Face dataset `TheKernel01/Tiny-GenImage`. Preparation script: `code/paddle_chain/prep_dataset.py`.

```bash
export DATA_ROOT=/path/to/your/data
HF_ENDPOINT=https://hf-mirror.com python code/paddle_chain/prep_dataset.py --full
```

If you already have parquet under `$DATA_ROOT/data_raw`, use `code/paddle_chain/prep_from_parquet.py` instead.

- Training set: 28,000 images = 14,000 real (ImageNet) + 2,000 each from ADM, BigGAN, GLIDE, Midjourney, SD15, VQDM, Wukong.
- Validation: 7,000 images = 3,500 real + 500 per generator.
- Images are stored as uint8 NHWC NumPy arrays at 224×224 after PIL BICUBIC resize.
- Normalization: ImageNet statistics for ResNet / EfficientNet; CLIP statistics for CLIP.

MD5 fingerprints (must match before any run that claims to reproduce the paper numbers):

```
X_train.npy       060071725e58b4b053c821f766064c25
X_validation.npy  dedd300482068bddaa1f1083332f34fc
y_train.npy       c4301f1fa4784a258f472e214dbe3625
y_validation.npy  051169b80f28e2472e702ab1a813f832
```

**Second dataset (R6):** the GenImage official test set. The original artifact describes it as the jzousz mirror of `Chengpong/genimage_test` on Hugging Face. The evaluation scripts read local parquet shards under `$DATA_ROOT/genimage_parquet/data/` (one shard per listed generator, 3,000 fakes each). Evaluation samples 300 fakes per generator against 300 real photographs from Tiny-GenImage validation. The optional `--real-parquet` argument in `genimage_test_zero_shot_v4.py` (GenImage-native nature reals) was **not** used on 2026-11-16: the local parquet tree contained only generator images.

Do not commit datasets, `.npy`, or `.parquet` files; they are listed in `.gitignore`.

## How to run

### Minimal package

Prerequisites: a CUDA GPU, PyTorch 2.x, and the four dataset files matching the MD5s above.

```bash
export DATA_ROOT=/path/to/data
export REPO_ROOT=$(pwd)

python code/pytorch_chain/train_detector.py --prep $DATA_ROOT/data_npy --out $DATA_ROOT/runs/det --epochs 3
python code/analysis/eval_auroc.py
python code/paddle_chain/pad_exp3_attacks.py
python code/pytorch_chain/lowpass_round_v2.py --ckpt $DATA_ROOT/runs/det.pth --outdir out/lowpass --eps 0.02 0.05
python code/pytorch_chain/bb02_main.py --exclude Midjourney --run 1
```

The original artifact's pass criteria (for the phenomena discussed in the paper) were: PGD at ε=0.05 below 1%; a transfer-vs-adaptive framing ratio above 50×; LOO diagonal below 60% with GLIDE above 85%.

A full multi-run reproduction follows the sequence scripts in `code/pytorch_chain/` (`run_full_repro.sh`, `run_rounds_seq.sh`, `run_loo_seq.sh`, and related). Those scripts now honor `DATA_ROOT` / `PYTHON` / `REPO_ROOT`.

Protocol used throughout: Adam, learning rate 1e-4 (CLIP backbone 1e-5), batch size 64, 3 epochs, seed 42 unless a script says otherwise. Backbones: ResNet-18, EfficientNet-B0, CLIP ViT-B/16 (OpenAI weights, `force_quick_gelu=True`). PGD: 10 steps, step size ε/4, L∞, unless a script documents a different step-size rule. Metrics: accuracy, per-generator AUROC, TPR at FPR=1%.

### Corrected scripts used on 16 November 2026

These are the scripts that produced the files under `rerun_20261116/`. They assume `$DATA_ROOT` already contains `data_npy/` and trained checkpoints under `runs/` (for example `runs/round_0{1,2,3}/detector.pth`, `runs/loo_01/`, and UnivFD features/heads under `runs/univfd/`).

**R1** — residual spectrum with the fftshift fix (CPU, about 5–15 minutes, 5,000 permutations):

```bash
python code/pytorch_chain/spec_resid_fixed.py --n-perm 5000 --outdir out/R1
```

**R2** — UnivFD with fp32-gradient PGD, random starts, restarts, and a true low-pass adaptive attack:

```bash
python code/pipelines/univfd_stage3_v2.py --eps 0.05 --steps 10 --restarts 3 --n-real 700 --out out/R2/eval.json
```

**R3** — same PyTorch detector, same ε, transfer vs adaptive low-pass, with per-class filtered accuracy:

```bash
python code/pytorch_chain/lowpass_round_v2.py \
  --ckpt $DATA_ROOT/runs/round_01/detector.pth --outdir out/R3/r1 --eps 0.02 0.05 0.10
```

**R4** — four defenses on a PyTorch port of the Paddle exp2 detector:

```bash
# The convert subcommand imports code/paddle_chain/convert_ckpt.py, which is not in this repo (see below).
python code/paddle_chain/pad_defenses_torch_port.py --out out/R4/defenses_torch_port.json
```

**R5** — class composition of the validation set and of the old first-2,000 JPEG subset:

```bash
python code/analysis/check_val_order.py $DATA_ROOT/data_npy/validation_meta_named.csv
```

**R6** — GenImage test-set zero-shot, recording fake and real accuracies separately:

```bash
python code/pipelines/genimage_test_zero_shot_v4.py --out out/R6/result_v4.json
```

**R7** — real-only threshold recalibration and ECE:

```bash
python code/analysis/rethreshold_calibration.py --prep $DATA_ROOT/data_npy \
  --full $DATA_ROOT/runs/round_01/detector.pth \
  --loo-dir $DATA_ROOT/runs/loo_01 --run 1 --out out/R7/result.json
```

Sanitized copies of the launch scripts used on 2026-11-16 are in `rerun_20261116/sh/`. Outputs are in `rerun_20261116/R1`–`R7` (and optional `R8`, `R11`); logs are in `rerun_20261116/logs/`. A narrative of what was and was not run is in `rerun_20261116/rerun_notes.md`.

## 2026-11-16 reruns: what changed

The notes below are taken from the corrected scripts and from the JSON/CSV/logs in `rerun_20261116/`. Original pre-rerun tables remain in `data/results_json_csv/`.

**R1 (required).** `spec_resid_fixed.py` applies `np.fft.fftshift` before the radial profile, subtracts the per-image mean, and drops the true DC bin. It also reports 150-vs-150 sample-matched correlations and two-sided permutation p-values with Holm correction. The run used `--n-perm 5000` and seed 42 on CPU. After the fix, residual-spectrum correlations are no longer uniformly ~0.97–0.99: values in `rerun_20261116/R1/resid_corr.csv` range from −0.283 (ADM–VQDM) to 0.994 (SD15–Wukong). The permutation-null mean is about 0.997; 18 of 21 generator pairs had an observed correlation below all 5,000 permutation values. Full p-values are in `rerun_20261116/R1/perm_holm_table.csv`. The updated spectral figure is `rerun_20261116/figures/fig2_spectral_corr.png`.

**R2 (required).** `univfd_stage3_v2.py` uses fp32 gradients, random PGD starts, optional restarts, and a low-pass adaptive attack that is not identical to transfer. Three recorded configurations, all with `clamp=false`:

| Config | n_real + n_fake | Clean overall | White-box PGD | LP transfer | LP adaptive |
|---|---|---|---|---|---|
| ε=0.05, 3 restarts | 700+700 | 93.29% | 0.0% | 46.5% | 0.0% |
| ε=0.05, 1 restart | 100+700 | 94.21% | 0.0% | 49.57% | 0.0% |
| ε=0.02, 1 restart | 700+700 | 93.29% | 0.0% | 54.0% | 0.0% |

For comparison, the original `data/results_json_csv/univfd_eval.json` (`n_eval=800`) reported white-box PGD overall 52.71%, with transfer and adaptive both 57.21%.

**R3 (required).** `lowpass_round_v2.py` evaluates transfer vs adaptive on the same PyTorch detector and the same ε, and reports per-class clean-filtered accuracy (real vs each generator). Three rounds, `n_eval=7000`, ratio 0.15, no clamp. Mean overall accuracy:

- clean bare 89.56% (range 2.90)
- clean filtered 51.11% (range 0.44); filtered real accuracy is about 3–5%, filtered fake accuracy about 98–99%
- ε=0.02: transfer 49.60%, adaptive 12.14%
- ε=0.05: transfer 48.20%, adaptive 0.42%
- ε=0.10: transfer 46.89%, adaptive 0.05%

**R4 (required).** Paddle 2.6.2 could not be run on the Blackwell GPU (the forward pass produced 0% or NaN). `pad_defenses_torch_port.py` ports the Paddle exp2 ResNet-18 weights to torchvision and repeats the four-defense protocol (randomization, bit-reduce 0.2, JPEG q50, low-pass 0.15) with PGD-10, α=2.5ε/10, no random start, and EOT σ=0.02, k=10. Clean bare accuracy after the port: 90.79% (`n=7000`), matching the original Paddle number for that checkpoint. JPEG q50 on the full 7,000-image set: no-attack 74.13%; at ε=0.05, transfer 46.24%, naive white-box 74.13%, adaptive (BPDA) 3.06%. The first-2,000 JPEG subset (see R5) is also reported in the same JSON.

The convert subcommand imports `convert_ckpt.py` (the Paddle-to-torchvision key mapping). That helper lived on the experiment server and is **not** in the original artifact or in this repository. The converted `.npz` weights are also not included. The evaluation JSON from the completed port is in `rerun_20261116/R4/`. To reconvert Paddle weights, request `convert_ckpt.py` and the `.pdparams` / `.npz` files from the authors.

**R5 (required).** From `validation_meta_named.csv`: all 7,000 images = 3,500 Real + 500 per generator. The first 2,000 (the old JPEG subset) = 1,000 Real + 143 per generator except Wukong 142 (50% real).

**R6 (required).** `genimage_test_zero_shot_v4.py` saves per-run fake and real accuracies as well as balanced accuracy (the original script stored only the balanced number, which had been mislabelled as fake-accuracy in an earlier draft). Settings: 300 fakes per generator, 300 reals from Tiny-GenImage validation, three in-house backbones × 3 runs, plus UnivFD. Balanced accuracy across generators: ResNet 48.78–50.06%, EfficientNet 49.28–50.28%, CLIP 49.78–50.56%, UnivFD 49.83–51.00%. AUROCs are likewise near chance (see `rerun_20261116/R6/result_v4.json`). The GenImage-native real-image control was not run.

**R7 (recommended).** Real-only threshold recalibration and ECE for the full detector and each LOO detector, runs 1 and 2. Results: `rerun_20261116/R7/result_run1.json` and `result_run2.json`. The script fits the real-only threshold on the same real validation images it then evaluates; a held-out real calibration split was not rerun.

**Optional work, not part of the required set**

- **R8 (partial):** full-training seeds 0/1/2, two deterministic seed-42 runs, and LOO-Midjourney seeds 0/1/2. The full 7-generator × 3-seed LOO grid was not run. Configs and logs are under `rerun_20261116/R8/` and `rerun_20261116/logs/R8_*.log`.
- **R9** (SDXL regeneration) was not run. Scripts: `sdxl_generate_500_v2.py`, `sdxl_eval_nine_models_v2.py`.
- **R10** (AutoAttack) was not run. Script: `autoattack_resnet.py`.
- **R11:** low-pass round 1 with `--clamp`. At ε=0.05, transfer 48.39%, adaptive 0.29%, close to the unclamped R3 run-1 numbers.

Updated figures after the rerun (especially fig. 2) are in `rerun_20261116/figures/`. `data/figures/` keeps the original artifact renders.

## Known pitfalls

These are the pitfalls documented in the original artifact; they remain relevant.

1. PaddlePaddle 2.6.2 boolean-tensor `mean()` always returns 1.0. Accuracy comparisons are done in NumPy.
2. `[B, 1]` labels can broadcast against `[B]` predictions into a `[B, B]` matrix. Labels are reshaped to `[-1]` before comparison.
3. torchvision rejects `weights=` and `num_classes=` together. Build the default head first, then replace `fc`.
4. CLIP OpenAI weights require `force_quick_gelu=True`.
5. sklearn binary `LogisticRegression.coef_` has shape `[1, n]`; the pixel pipeline builds the two-output head as `[-w, +w]`.
6. `diffusers` `from_single_file` in offline mode requires `config=` pointing at the local model directory and `local_files_only=True`.

## Model weights

Checkpoints (`.pth`, `.pdparams`), converted weight archives (`.npz`), and feature caches are excluded from this repository (see `.gitignore`). They can be reproduced by running the training scripts on the MD5-matched Tiny-GenImage arrays, or obtained from the authors on request. A public release of the weights is planned.

## License

MIT License. Copyright (c) 2026 The authors.
