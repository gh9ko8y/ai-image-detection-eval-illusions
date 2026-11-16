# -*- coding: utf-8 -*-
"""
低通滤波：迁移攻击 vs 白盒自适应攻击（PyTorch 版，AutoDL RTX 4080 SUPER）
==========================================================================
与 Paddle 版逐行等价，只换框架。每个 keep_ratio 同时给三条线：
  1. 干净（仅滤波）     —— 防御代价
  2. 迁移攻击后（滤波） —— 旧口径：攻击者不知道滤波存在
  3. 白盒攻击后（滤波） —— 攻击者知道滤波、梯度穿过滤波（FFT 可微）

mask 沿用原设计：h=int(H*r), w=int(W*r) 四角方块
  实际保留比例 = 4*h*w/(H*W) ≈ 2*r^2（H=W 时），不是 r
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision

# ===================== 配置 =====================
# 固定路径布局（fetch3.py 产出），不再依赖当前工作目录
DATA_DIR = os.path.join(os.environ.get('DATA_ROOT', '<DATA_ROOT>'), 'data_npy')
RUNS_DIR = os.path.join(os.environ.get('DATA_ROOT', '<DATA_ROOT>'), 'runs')
DATA_X = os.path.join(DATA_DIR, 'X_validation.npy')
DATA_Y = os.path.join(DATA_DIR, 'y_validation.npy')
CKPT_T = os.path.join(RUNS_DIR, 'detector_resnet18_torch.pth')
OUT_CSV = os.path.join(RUNS_DIR, 'lowpass_adaptive.csv')
OUT_PNG = os.path.join(RUNS_DIR, 'fig7_lowpass_adaptive.png')

N_SUB = 0            # 0 = 用全部 7000 张（4080 跑得动；复现历史可设 2000）
SEED = 42
EPS_LIST = [0.02, 0.05]
STEPS = 10
BATCH = 100
# 扫描区间：实留 ≤8.7%（r ≤0.15）时干净准确率已掉到 ~51%（二分类=乱猜），该区间无意义。
# 真正要回答：在检测器仍可用（干净 ≥70%）的区间里，自适应攻击能不能打穿？
KEEP_RATIOS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

os.makedirs(RUNS_DIR, exist_ok=True)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(SEED)
np.random.seed(SEED)
print('device:', DEV, '|', torch.cuda.get_device_name(0) if DEV == 'cuda' else 'cpu')

# ===================== 数据与模型 =====================
model = torchvision.models.resnet18(weights=None, num_classes=2)
sd = torch.load(CKPT_T, map_location='cpu')
if isinstance(sd, dict) and 'state_dict' in sd:
    sd = sd['state_dict']
missing, unexpected = model.load_state_dict(sd, strict=False)
print('权重加载: missing=%d unexpected=%d' % (len(missing), len(unexpected)))
if missing:
    print('  missing keys 前 5 个:', missing[:5])
if unexpected:
    print('  unexpected keys 前 5 个:', unexpected[:5])
assert not missing, '有关键权重没加载上，检查 checkpoint 是否完整'
model = model.to(DEV).eval()

X_val = np.load(DATA_X)
y_val = np.load(DATA_Y).reshape(-1)
MEAN = np.array([0.485, 0.456, 0.406], dtype='float32').reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype='float32').reshape(1, 3, 1, 1)

if N_SUB and N_SUB < len(X_val):
    idx = np.random.RandomState(SEED).permutation(len(X_val))[:N_SUB]
else:
    idx = np.arange(len(X_val))
print('评测子集: %d 张 (全集 %d)' % (len(idx), len(X_val)))

Xs = X_val[idx].astype('float32') / 255.0
Xs = Xs.transpose(0, 3, 1, 2)
Xs = (Xs - MEAN) / STD
Xt = torch.from_numpy(np.ascontiguousarray(Xs)).to(DEV)
y_np = y_val[idx].astype('int64')
yt = torch.from_numpy(y_np).to(DEV)

# ===================== 低通滤波 =====================
def low_pass_filter(x, keep_ratio):
    """x: (B,C,H,W)。FFT -> 四角低频 mask -> IFFT。可微。"""
    B, C, H, W = x.shape
    f = torch.fft.fft2(x, dim=(-2, -1))
    h = max(2, int(H * keep_ratio))
    w = max(2, int(W * keep_ratio))
    mask = torch.zeros((H, W), dtype=f.dtype, device=x.device)
    mask[:h, :w] = 1
    mask[:h, W - w:] = 1
    mask[H - h:, :w] = 1
    mask[H - h:, W - w:] = 1
    f = f * mask.reshape(1, 1, H, W)
    return torch.fft.ifft2(f, dim=(-2, -1)).real


def retained_fraction(keep_ratio, H=224, W=224):
    """实际保留的频域系数比例（论文里要报这个数）"""
    h = min(max(2, int(H * keep_ratio)), H)
    w = min(max(2, int(W * keep_ratio)), W)
    m = np.zeros((H, W), dtype='float32')
    m[:h, :w] = 1
    m[:h, W - w:] = 1
    m[H - h:, :w] = 1
    m[H - h:, W - w:] = 1
    return float(m.sum()) / (H * W)

# ===================== 评测与攻击 =====================
@torch.no_grad()
def acc(x, defense=None, keep_ratio=None, batch=BATCH):
    correct = 0
    for i in range(0, x.shape[0], batch):
        xb = x[i:i + batch]
        if defense == 'lowpass':
            xb = low_pass_filter(xb, keep_ratio)
        correct += int((model(xb).argmax(1).cpu().numpy() == y_np[i:i + batch]).sum())
    return 100.0 * correct / x.shape[0]


def pgd(x, eps, steps=STEPS, batch=BATCH, defense=None, keep_ratio=None):
    """PGD L_inf。defense='lowpass' 时梯度穿过滤波 —— 白盒自适应攻击。"""
    alpha = eps / steps
    advs = []
    for i in range(0, x.shape[0], batch):
        xb = x[i:i + batch]
        yb = yt[i:i + batch]
        x_adv = xb.clone()
        x_adv = x_adv + (torch.rand_like(x_adv) * 2 - 1) * eps          # 随机起点
        x_adv = torch.max(torch.min(x_adv, xb + eps), xb - eps)
        for _ in range(steps):
            x_adv = x_adv.detach().requires_grad_(True)
            out = model(low_pass_filter(x_adv, keep_ratio) if defense == 'lowpass' else x_adv)
            loss = nn.functional.cross_entropy(out, yb)
            g = torch.autograd.grad(loss, x_adv, allow_unused=True)[0]
            if g is None:
                g = torch.zeros_like(x_adv)
            x_adv = x_adv.detach() + alpha * g.sign()
            x_adv = torch.max(torch.min(x_adv, xb + eps), xb - eps)
        advs.append(x_adv.detach())
    return torch.cat(advs)

# ===================== 自检锚点 =====================
print('=' * 70)
clean_bare = acc(Xt)
print('[锚点1] 无防御干净准确率 = %.2f%%    （应 ≈90.09%%）' % clean_bare)

adv_bare = {e: pgd(Xt, e) for e in EPS_LIST}
for e in EPS_LIST:
    print('[锚点2] 无防御 PGD eps=%.2f 攻击后 = %.2f%%  （应 ≈0%%）' % (e, acc(adv_bare[e])))
assert acc(adv_bare[0.05]) < 5.0, '锚点2 失败：攻击没跑对，先停下检查'
assert clean_bare > 80.0, '锚点1 失败：干净准确率远低于 90%%，权重转换或预处理有问题，先别继续'
print('锚点全部通过，开始主实验')
print('=' * 70)

# ===================== 主实验 =====================
rows = []
for r in KEEP_RATIOS:
    frac = retained_fraction(r)
    d = {'keep_ratio': r, 'retained': frac, 'clean_bare': clean_bare}
    d['clean_filt'] = acc(Xt, 'lowpass', r)
    for e in EPS_LIST:
        d['transfer_%.2f' % e] = acc(low_pass_filter(adv_bare[e], r))       # 迁移口径
        d['whitebox_%.2f' % e] = acc(pgd(Xt, e, defense='lowpass', keep_ratio=r))
    rows.append(d)
    print('keep=%.2f (实留 %.1f%%)  干净滤波 %5.2f%%  |  eps=0.05: 迁移 %5.2f%% -> 白盒 %5.2f%%'
          % (r, frac * 100, d['clean_filt'], d['transfer_0.05'], d['whitebox_0.05']), flush=True)

# ===================== 汇总 =====================
print('\n' + '=' * 70)
head = ['%-8s' % 'keep', '%-9s' % '实留%', '%-10s' % '干净滤波%']
for e in EPS_LIST:
    head += ['%-12s' % ('迁移.%.2f%%' % e), '%-12s' % ('白盒.%.2f%%' % e)]
print(' '.join(head))
for d in rows:
    cells = ['%-8.2f' % d['keep_ratio'], '%-9.2f' % (d['retained'] * 100), '%-10.2f' % d['clean_filt']]
    for e in EPS_LIST:
        cells += ['%-12.2f' % d['transfer_%.2f' % e], '%-12.2f' % d['whitebox_%.2f' % e]]
    print(' '.join(cells))

hdr = ['keep_ratio', 'retained_frac', 'clean_bare', 'clean_filtered']
for e in EPS_LIST:
    hdr += ['transfer_eps%.2f' % e, 'whitebox_eps%.2f' % e]
lines = [','.join(hdr)]
bare_row = ['bare', '1.0000', '%.2f' % clean_bare, '']
for e in EPS_LIST:
    bare_row += ['%.2f' % acc(adv_bare[e]), '']
lines.append(','.join(bare_row))
for d in rows:
    row = ['%.2f' % d['keep_ratio'], '%.4f' % d['retained'], '%.2f' % d['clean_bare'], '%.2f' % d['clean_filt']]
    for e in EPS_LIST:
        row += ['%.2f' % d['transfer_%.2f' % e], '%.2f' % d['whitebox_%.2f' % e]]
    lines.append(','.join(row))
csv_text = '\n'.join(lines)
print('\n===== CSV =====')
print(csv_text)
with open(OUT_CSV, 'w', encoding='utf-8') as f:
    f.write(csv_text + '\n')

# ===================== 图 =====================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False

xs = [d['keep_ratio'] for d in rows]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
ax[0].plot(xs, [d['clean_filt'] for d in rows], 'o-', color='#2b6cb0', label='filtered (clean)')
ax[0].axhline(clean_bare, ls='--', color='gray', label='bare detector %.2f%%' % clean_bare)
ax[0].set_xlabel('keep_ratio (nominal)'); ax[0].set_ylabel('accuracy (%)')
ax[0].set_title('Cost of the defense'); ax[0].legend(); ax[0].grid(alpha=.3)

styles = [('-', 'o'), ('--', 's'), (':', '^'), ('-.', 'D')]
for k, e in enumerate(EPS_LIST):
    ls, mk = styles[k % len(styles)]
    ax[1].plot(xs, [d['transfer_%.2f' % e] for d in rows], mk + ls, color='#c53030',
               label='transfer eps=%.2f' % e)
    ax[1].plot(xs, [d['whitebox_%.2f' % e] for d in rows], mk + ls, color='#2f855a',
               label='white-box adaptive eps=%.2f' % e)
ax[1].set_xlabel('keep_ratio (nominal)'); ax[1].set_ylabel('accuracy under attack (%)')
ax[1].set_title('Transfer vs adaptive attack'); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
print('\n已保存: %s / %s' % (OUT_CSV, OUT_PNG))
print('DONE')
