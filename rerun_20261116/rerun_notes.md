# 重跑实验清单（TIFS 修改版配套）

> 生成：2026-11-16（UTC+8）。原始脚本一律不改；修正版都在 `revised/code_patches/` 下，文件头注明了修了什么。
> 服务器约定沿用 artifact：脚本放 `<REPO_ROOT>/`，数据 `<DATA_ROOT>/data_npy`，PyTorch 环境 `python`，Paddle 环境 `python`。
> 跑完把输出 JSON/CSV 发回来，我把论文里的 `\rerun{...}` 红字替换成实测值（**不要手填**）。

## 0. 准备（一次）

```bash
# 本地把 revised/code_patches 整个目录传到服务器 <REPO_ROOT>/code_patches 后：
cd <REPO_ROOT>
cp code_patches/pytorch_chain/*.py code_patches/pipelines/*.py code_patches/analysis/*.py <REPO_ROOT>/
cp code_patches/paddle_chain/*.py  <REPO_ROOT>/   # 如果 Paddle 脚本放在别的目录，与 pad_common.py 放一起
PY=python
PYP=python
```

## 1. 总表

| # | 审稿项 | 重跑什么 | 脚本（修正版） | 资源 | 预计耗时 | 必须/可选 | 回填到论文 |
|---|---|---|---|---|---|---|---|
| R1 | M5, M6, M15(Fig.4), S16 | 残差谱 + split-half 参考 + 置换/Holm（修 fftshift，同样本量参考，双侧检验） | `pytorch_chain/spec_resid_fixed.py`；画图 `figures/redraw_fig2_v2.py` | CPU，内存 ≥16 GB | 10–20 分钟 | **必须** | Sec. VI 频谱段、Fig. 4 及图注、Stage 3、讨论/结论里的频谱数字 |
| R2 | M8, S3 | UnivFD：fp32 梯度 PGD（随机起点、多重启）+ 真正的 LP-adaptive + 干净图滤波 + 真图 700 张 | `pipelines/univfd_stage3_v2.py` | GPU（≥16 GB） | 1–2 小时 | **必须** | Sec. VIII、Table VII 的 PGD / LP 行 |
| R3 | M3, M7, S1, S2 | 低通：同一 PyTorch 检测器、同一 ε 下 transfer vs adaptive；滤波后真/假分侧准确率 | `pytorch_chain/lowpass_round_v2.py` | GPU（显存 ≥12 GB，样本整体存 GPU） | 约 1 小时（3 个 round × 2 个 ε） | **必须** | 摘要、引言、Sec. V（L335 待确认句）、Table II 低通行、Fig. 1(a) |
| R4 | M18, S2, M7 | Paddle 四种防御补全 Transfer 列；JPEG 用全量 7000 张；每个 ε 单独成行 | `paddle_chain/pad_exp5678_defenses_v2.py` | GPU | 2–3 小时（JPEG/BPDA 最慢） | **必须**（至少 JPEG 全量） | Table II 全表及脚注 |
| R5 | M18 | 查验证集前 2000 张类别构成 | `analysis/check_val_order.py` | CPU | 几秒 | **必须**（先跑，决定 R4 是否必须） | Table II 脚注 |
| R6 | M9, S12 | GenImage：三骨干分别保存 fake/real 准确率；可选用 GenImage 自带真图 | `pipelines/genimage_test_zero_shot_v4.py` | GPU | 30–60 分钟 | **必须** | Sec. X、Table IX |
| R7 | S16 | 校准漂移证据：ECE + "只用真图重定阈值" 实验 | `analysis/rethreshold_calibration.py` | GPU | 5 分钟（8 个模型推理） | 强烈建议 | Sec. VI 校准段、Stage 2、讨论 |
| R8 | M13 | 换 seed（0/1/2）与确定性模式，拆开 "seed 方差" 与 "GPU 非确定性" | `pytorch_chain/train_detector_v2.py` | GPU | 全量：约 7 分钟/次；LOO 7 格 × 3 seed ≈ 2.5 小时 | 可选（审稿人很可能要） | 复现性段落、Stage 4 |
| R9 | M17 | SDXL 重新生成：1000 类 × 10 模板均匀抽 500 张 | `pipelines/sdxl_generate_500_v2.py` + `sdxl_eval_nine_models_v2.py` | GPU（SDXL ≥16 GB） | 生成约 1 小时 + 评测 10 分钟 | 可选（不跑则正文如实披露单模板/前 500 类，已改） | Sec. VII、Table VI |
| R10 | S4 | AutoAttack（standard, L∞）测 ResNet-18 | `pytorch_chain/autoattack_resnet.py` | GPU | 1000 张约 20–40 分钟 | 可选 | Related work / Sec. V 的 AutoAttack 句 |
| R11 | S1 | clamp 到合法像素域后复测关键格 | R2/R3 加 `--clamp` | GPU | 各 +1 小时 | 可选 | Threat model 段说明 |

**必须项合计 ≈ 5–7 小时 GPU + 20 分钟 CPU**（R1–R6，可在一台 4080S 上过夜跑完）。

## 2. 逐项命令

### R5 先看 JPEG 子集构成（秒级）
```bash
$PY check_val_order.py <DATA_ROOT>/data_npy/validation_meta_named.csv | tee runs_check_val_order.txt
```
如果前 2000 张不是 7 类 + 真图大致按比例，Table II 的 JPEG 行（78.00/78.00/22.35）不可比，R4 必须跑。

### R1 频谱（CPU）
```bash
$PY spec_resid_fixed.py --prep <DATA_ROOT>/data_npy \
    --outdir <DATA_ROOT>/runs/spec_resid_fixed --n-perm 5000 2>&1 | tee spec_fixed.log
$PY code_patches/figures/redraw_fig2_v2.py <DATA_ROOT>/runs/spec_resid_fixed <DATA_ROOT>/runs/spec_resid_fixed/figs
```
回传：`resid_corr.csv`、`resid_corr_matched150.csv`、`null_stats.csv`、`perm_holm_result.json`、`perm_holm_table.csv`、`figs/fig2_spectral_corr.png`（替换 `revised/figs/fig2_spectral_corr.png`）。
脚本启动时会自检（频率 k 的余弦必须落在第 k 个 bin），不通过直接报错退出。

### R2 UnivFD（GPU）
```bash
$PY univfd_stage3_v2.py --eps 0.05 --steps 10 --restarts 3 --n-real 700 \
    --out <DATA_ROOT>/runs/univfd/eval_v2.json 2>&1 | tee univfd_v2.log
# 可选：合法像素域
$PY univfd_stage3_v2.py --eps 0.05 --steps 10 --restarts 3 --n-real 700 --clamp \
    --out <DATA_ROOT>/runs/univfd/eval_v2_clamp.json
```
看到 `UNIVFD_EVAL_V2_DONE` 即完成。注意：CLIP 归一化 std≈0.27，ε=0.05 ≈ 3.4/255，和 ResNet 的 ε=0.05（≈2.9/255）不是同一像素预算，论文里已写明。

### R3 低通 transfer vs adaptive（GPU）
```bash
for r in 1 2 3; do
  $PY lowpass_round_v2.py --ckpt <DATA_ROOT>/runs/round_0$r/detector.pth \
      --outdir <DATA_ROOT>/runs/lowpass_v2/r$r --eps 0.02 0.05 2>&1 | tee lowpass_v2_r$r.log
done
```
每个输出 `lowpass_v2.json` 包含：`clean_bare`、`clean_filtered`（**真图/各生成器分侧，回答 L335**）、每个 ε 的 `bare_attacked / transfer / adaptive`。

### R4 Paddle 四种防御（GPU）
```bash
cd <REPO_ROOT>   # pad_common.py 所在目录
$PYP pad_exp5678_defenses_v2.py --eps 0.02 0.05 --jpeg-subset full 2>&1 | tee pad5678_v2.log
# 时间紧可先跑分层 2000 张：--jpeg-subset stratified
```
输出 `<DATA_ROOT>/runs/paddle_01/exp5678_v2.json`。日志开头会打印旧 JPEG 子集（前 2000 张）的类别构成。

### R6 GenImage（GPU）
```bash
$PY genimage_test_zero_shot_v4.py --out <DATA_ROOT>/runs/genimage_test_eval/result_v4.json 2>&1 | tee genimage_v4.log
# 可选（S12 混杂对照）：用 GenImage 自带 nature 真图
$PY genimage_test_zero_shot_v4.py --real-parquet <GenImage nature 真图 parquet 路径> \
    --out <DATA_ROOT>/runs/genimage_test_eval/result_v4_genreal.json
```

### R7 重定阈值 / ECE（GPU，强烈建议）
```bash
$PY rethreshold_calibration.py --prep <DATA_ROOT>/data_npy \
    --full <DATA_ROOT>/runs/round_01/detector.pth \
    --loo-dir <DATA_ROOT>/runs/loo_01 --run 1 \
    --out <DATA_ROOT>/runs/rethreshold/result.json
```
判读：若非对角格 Δacc 在重定阈值后回到 ±2 点内、对角格仍崩，就能直接支撑 "calibration drift vs. true inseparability"。

### R8 换 seed（可选）
```bash
for s in 0 1 2; do
  $PY train_detector_v2.py --seed $s --out runs/seed/full_s$s
  for g in ADM BigGAN GLIDE Midjourney SD15 VQDM Wukong; do
    $PY train_detector_v2.py --seed $s --exclude $g --out runs/seed/loo_${g}_s$s
  done
done
# 确定性对照：同 seed 跑两次，应逐位一致
$PY train_detector_v2.py --seed 42 --deterministic --out runs/seed/det42_a
$PY train_detector_v2.py --seed 42 --deterministic --out runs/seed/det42_b
```

### R9 SDXL 重生成（可选）
```bash
$PY sdxl_generate_500_v2.py            # → runs/sdxl_zeroshot/gen_v2/ 与 prompts_v2.json
$PY sdxl_eval_nine_models_v2.py <DATA_ROOT>/runs/sdxl_zeroshot/gen_v2 <DATA_ROOT>/runs/sdxl_zeroshot/result_v2.json
$PY code_patches/figures/redraw_fig4_v3.py <DATA_ROOT>/runs/sdxl_zeroshot/result_v2.json <DATA_ROOT>/runs/sdxl_zeroshot/figs
```

### R10 AutoAttack（可选）
```bash
$PY -m pip install git+https://github.com/fra31/auto-attack
$PY autoattack_resnet.py --ckpt <DATA_ROOT>/runs/round_01/detector.pth --n 1000 --eps255 2.9 --out runs/autoattack_r1.json
```

## 3. 回传清单
`spec_resid_fixed/`（全部）、`univfd/eval_v2*.json`、`lowpass_v2/r{1,2,3}/lowpass_v2.json`、`paddle_01/exp5678_v2.json`、`check_val_order` 输出、`genimage_test_eval/result_v4*.json`、（可选）`rethreshold/result.json`、`seed/` 下的 eval 输出、`sdxl_zeroshot/result_v2.json`、`autoattack_r1.json`，以及所有 `.log`。

## 4. 另外需要你从服务器找的旧数据（论文里无法溯源的数字，见 修改说明.md "需用户提供"）
- Table I caption 的 91.03%、±0.17；AT 84.7（range 0.13）；EffNet −8.1/−11.5；原始谱相关 0.974–0.999 → 对应日志或 JSON。
- Table II 各 transfer 格的框架与 ε（尤其 47.4 是 ε=0.02/0.05/0.10 中哪个）。
- `loo_01/eval_*.json`、`lowpass_round` 的 `summary.json`、Paddle 各实验日志、AT 日志 → 建议一并放进 artifact。

---

## 5. 执行状态（2026-11-16，UTC+8；服务器 RTX PRO 6000 Blackwell）

| 项 | 状态 | 输出 |
|---|---|---|
| R5 | ✅ 前 2000 张为 1000 Real + 每个生成器约 143 张，按类分层 | R5/check_val_order.txt |
| R1 | ✅ CPU | R1/*.csv, perm_holm_result.json |
| R7 | ✅ LOO run1/run2 | R7/result_run{1,2}.json |
| R3 | ✅ 三轮 | R3/r{1,2,3}/lowpass_v2.json |
| R6 | ✅ | R6/result_v4.json |
| R2 | ✅ 三个配置：(a) ε0.05、3 次重启、1400 张；(b) ε0.05、1 次重启、800 张；(c) ε0.02、1 次重启、1400 张 | R2/eval_v2_*.json |
| R4 | ✅ **改用 torch 移植版**。Paddle 2.6.2 GPU 版不支持 sm_120（Blackwell），GPU 上全部输出 0% 或 NaN，所以新写了 `paddle_chain/pad_defenses_torch_port.py`：在 venv_paddle 里用 CPU 把 exp2 权重转成 npz，再用 torch 评估。移植后 clean 90.79，与原始 Paddle 一致 | R4/defenses_torch_port.json |
| R8 | ✅ 可选，部分完成：全训练 seed 0/1/2，seed42 确定性模式跑两次，LOO-MJ seed 0/1/2。完整 21 格 × 3 seed 没有跑 | logs/R8_*.log |
| R11 | ✅ 可选：低通 round1 加 `--clamp` | R11/lowpass_clamp_r1/ |
| R9 SDXL 重生成 | ⏭ 未跑（可选） | — |
| R10 AutoAttack | ⏭ 未跑（可选；需要在服务器 venv 里安装新包） | — |
| R6 GenImage 自带真图对照 | ❌ 无法执行：genimage_parquet 里只有生成器图，没有 nature 真图 | — |

本轮新增或修改的脚本：`pytorch_chain/lowpass_round_v2.py`（y 加 reshape）、`pipelines/univfd_stage3_v2.py`（`--n-fake-per-gen`、HF 离线）、`paddle_chain/pad_defenses_torch_port.py`（新增）、`figures/redraw_fig2_v3.py`（每对取两类阈值中较大的那个）、`figures/redraw_fig1_v4.py`（新增）。
