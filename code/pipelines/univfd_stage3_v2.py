### UNIVFD_STAGE3_V2 ### TIFS 审稿 M8 修正版（原文件 univfd_stage3_fixed.py 保持不变）
"""Stage3 v2：UnivFD 像素级攻防探针。
相对 univfd_stage3_fixed.py 的修正：
  [M8-a] LP-adaptive 真正把 lowpass_corner 放进 PGD 循环（旧版两个口径用同一个 adv，
         低通不在攻击回路里 → "LP-transfer = LP-adaptive" 只是同一对象算两次）。
  [M8-b] 攻击梯度在 fp32 下计算（旧版 autocast fp16 取梯度，可能削弱攻击）；
         默认随机起点 + 可选多次重启（取对每张图最有效的那次）。
  [M8-c] 评估子集改为 100/生成器 × 7 + 真图 700（真假各 700，平衡）；旧版只有 100 张真图。
  [M8-d] 新增 clean_filtered（干净图只过低通、不攻击），把"滤波本身"与"攻击+滤波"分开。
  [S1]   可选 --clamp：对抗样本裁剪回合法像素域 [0,1]（在归一化空间按通道换算）。
输出：runs/univfd/eval_v2.json（字段含 fake_acc / real_acc / overall_acc / auroc / tpr1 与配置）
用法：python univfd_stage3_v2.py [--eps 0.05] [--steps 10] [--restarts 1] [--n-real 700] [--clamp]
"""

import argparse
import json
import os
import pickle
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

BASE = os.environ.get('DATA_ROOT', '<DATA_ROOT>')
FEATDIR = BASE + '/runs/univfd'
DEV = 'cuda'
GENS = ['ADM', 'BigGAN', 'GLIDE', 'Midjourney', 'SD15', 'VQDM', 'Wukong']
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073],
                     dtype='float32').reshape(1, 3, 1, 1)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711],
                    dtype='float32').reshape(1, 3, 1, 1)


def to_norm(x_u8):
    x = x_u8.astype('float32') / 255.0
    x = x.transpose(0, 3, 1, 2)
    return torch.from_numpy(np.ascontiguousarray((x - CLIP_MEAN) / CLIP_STD))


def load_visual():
    import open_clip
    m, _, _ = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='openai', force_quick_gelu=True)
    return m.visual.to(DEV).eval()


class UnivFDPixel(torch.nn.Module):
    def __init__(self, visual, clf):
        super().__init__()
        self.visual = visual
        w = torch.tensor(clf.coef_, dtype=torch.float32)  # [1,768]
        b = torch.tensor(clf.intercept_, dtype=torch.float32)  # [1]
        W2 = torch.cat([-w, w], dim=0)  # [2,768]
        b2 = torch.cat([-b, b], dim=0)  # [2]
        self.head_w = torch.nn.Parameter(W2)
        self.head_b = torch.nn.Parameter(b2)

    fp16 = True   # 评估可用 fp16；攻击时由 pgd() 临时切到 fp32

    def forward(self, x):
        if self.fp16:
            with torch.autocast('cuda', dtype=torch.float16):
                f = self.visual(x).float()
        else:
            f = self.visual(x.float()).float()
        return f @ self.head_w.T + self.head_b


def lowpass_corner(x_t, r=0.15):
    B, C, H, W = x_t.shape
    f = torch.fft.fft2(x_t, dim=(-2, -1))
    h = max(2, int(H * r))
    w = max(2, int(W * r))
    m = torch.zeros((H, W), dtype=f.dtype, device=x_t.device)
    m[:h, :w] = 1
    m[:h, W - w:] = 1
    m[H - h:, :w] = 1
    m[H - h:, W - w:] = 1
    return torch.fft.ifft2(f * m.view(1, 1, H, W), dim=(-2, -1)).real


LO = torch.tensor((0.0 - CLIP_MEAN) / CLIP_STD).view(1, 3, 1, 1)
HI = torch.tensor((1.0 - CLIP_MEAN) / CLIP_STD).view(1, 3, 1, 1)


def pgd(model, xb, yb, eps, alpha, steps, defense=None, rand_init=True,
        restarts=1, clamp=False):
    """L_inf PGD（归一化空间）。defense 不为 None 时把防御放进前向 = 自适应攻击。
    梯度在 fp32 下计算；多次重启时保留使损失最大的样本。"""
    model.fp16 = False
    lo, hi = LO.to(xb.device), HI.to(xb.device)
    best = xb.clone().detach()
    best_loss = torch.full((len(xb),), -1e9, device=xb.device)
    for _r in range(restarts):
        adv = xb.clone().detach()
        if rand_init:
            adv = adv + (torch.rand_like(adv) * 2 - 1) * eps
        for _ in range(steps):
            adv = adv.detach().requires_grad_(True)
            inp = defense(adv) if defense is not None else adv
            loss = F.cross_entropy(model(inp), yb)
            gr = torch.autograd.grad(loss, adv)[0]
            with torch.no_grad():
                adv = torch.max(torch.min(adv + alpha * gr.sign(), xb + eps), xb - eps)
                if clamp:
                    adv = torch.max(torch.min(adv, hi), lo)
        with torch.no_grad():
            inp = defense(adv) if defense is not None else adv
            per = F.cross_entropy(model(inp), yb, reduction='none')
            better = per > best_loss
            best[better] = adv[better]
            best_loss = torch.where(better, per, best_loss)
    model.fp16 = True
    return best.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eps', type=float, default=0.05)
    ap.add_argument('--steps', type=int, default=10)
    ap.add_argument('--restarts', type=int, default=1)
    ap.add_argument('--n-real', type=int, default=700)
    ap.add_argument('--n-fake-per-gen', type=int, default=100)
    ap.add_argument('--clamp', action='store_true')
    ap.add_argument('--out', default=FEATDIR + '/eval_v2.json')
    args = ap.parse_args()
    alpha = 2.5 * args.eps / args.steps
    X = np.load(BASE + '/data_npy/X_validation.npy', mmap_mode='r')
    meta_v = pd.read_csv(BASE + '/data_npy/validation_meta_named.csv')
    gva = meta_v['gen_name'].values
    rng = np.random.RandomState(42)
    idx = np.concatenate(
        [rng.choice(np.where(gva == g)[0], args.n_fake_per_gen, replace=False) for g in GENS]
        + [rng.choice(np.where(gva == 'Real')[0], args.n_real, replace=False)])
    gsub = gva[idx]
    Xsub = np.asarray(X[idx])

    with open(FEATDIR + '/heads/full.pkl', 'rb') as fh:
        clf = pickle.load(fh)
    visual = load_visual()
    model = UnivFDPixel(visual, clf).to(DEV).eval()
    # 识别性自检：一个 batch 前向形状
    with torch.no_grad():
        probe = model(to_norm(Xsub[:4]).to(DEV))
    print('logit shape', tuple(probe.shape), flush=True)
    assert probe.shape == (4, 2), 'head shape wrong'

    def batch_scores(x_pixels, apply_lp=False):
        ss = []
        with torch.no_grad():
            for i in range(0, len(x_pixels), 50):
                xb = to_norm(x_pixels[i:i + 50]).to(DEV)
                if apply_lp:
                    xb = lowpass_corner(xb)
                p = F.softmax(model(xb), 1)[:, 1]
                ss.append(p.float().cpu().numpy())
        return np.concatenate(ss)

    names = ['clean', 'clean_filtered', 'whitebox_pgd', 'lowpass_transfer', 'lowpass_adaptive']
    S = {n: ([], []) for n in names}
    B = 25
    lp = lambda t: lowpass_corner(t)
    for i in range(0, len(Xsub), B):
        xb = to_norm(Xsub[i:i + B]).to(DEV)
        yb = torch.tensor(
            [0 if gsub[j] == 'Real' else 1 for j in range(i, i + len(xb))],
            device=DEV, dtype=torch.long)
        kw = dict(rand_init=True, restarts=args.restarts, clamp=args.clamp)
        adv_bare = pgd(model, xb, yb, args.eps, alpha, args.steps, defense=None, **kw)
        adv_lp = pgd(model, xb, yb, args.eps, alpha, args.steps, defense=lp, **kw)  # 低通在回路内
        with torch.no_grad():
            outs = {
                'clean': model(xb),
                'clean_filtered': model(lp(xb)),
                'whitebox_pgd': model(adv_bare),
                'lowpass_transfer': model(lp(adv_bare)),   # 攻击裸模型，再滤波
                'lowpass_adaptive': model(lp(adv_lp)),     # 攻击"滤波+模型"整体
            }
        cm = (yb == 0).cpu().numpy()
        for n in names:
            p = F.softmax(outs[n], 1)[:, 1].float().cpu().numpy()
            S[n][0].append(p[~cm]); S[n][1].append(p[cm])
        print('batch', i, flush=True)

    def agg(fp, rp):
        return np.concatenate(fp), np.concatenate(rp)

    out = {'n_eval': int(len(Xsub)), 'n_fake': int((gsub != 'Real').sum()),
           'n_real': int((gsub == 'Real').sum()),
           'config': {'eps_normalized': args.eps, 'eps_pixel_approx_255': args.eps * 0.27 * 255,
                      'steps': args.steps, 'alpha': alpha, 'rand_init': True,
                      'restarts': args.restarts, 'clamp': args.clamp, 'grad_dtype': 'fp32'}}
    for name in names:
        fp, rp = S[name]
        sf, sr = agg(fp, rp)
        y = np.r_[np.ones(len(sf)), np.zeros(len(sr))]
        sc = np.r_[sf, sr]
        out[name] = {
            'fake_acc': float((sf >= 0.5).mean() * 100),
            'real_acc': float((sr < 0.5).mean() * 100),
            'overall_acc': float(((sf >= 0.5).mean() * 100
                                  + (sr < 0.5).mean() * 100) / 2),
            'auroc': float(roc_auc_score(y, sc) * 100),
            'tpr1': float((sf >= np.quantile(sr, 0.99)).mean() * 100)}
        print(name, out[name], flush=True)

    # LOO 干净矩阵（特征级，快速）
    Fva = np.load(FEATDIR + '/feat_validation.npy')
    loo = {}
    for tag in ['full'] + ['loo_' + g for g in GENS]:
        with open(FEATDIR + '/heads/%s.pkl' % tag, 'rb') as fh:
            c2 = pickle.load(fh)
        s = c2.predict_proba(Fva)[:, 1]
        s_real = s[gva == 'Real']
        thr = np.quantile(s_real, 0.99)
        e = {'overall_acc': float(((s[gva != 'Real'] >= 0.5).mean() * 100
                                   + (s_real < 0.5).mean() * 100) / 2),
             'real_acc': float((s_real < 0.5).mean() * 100)}
        for gen in GENS:
            pos = s[gva == gen]
            e[gen] = float((pos >= 0.5).mean() * 100)
            e[gen + '_auroc'] = float(roc_auc_score(
                np.r_[np.ones(len(pos)), np.zeros(len(s_real))],
                np.r_[pos, s_real]) * 100)
            e[gen + '_tpr1'] = float((pos >= thr).mean()) * 100
        loo[tag] = e
        print(tag, 'loo-done', flush=True)
    out['loo_clean_matrix'] = loo
    with open(args.out, 'w') as fh:
        json.dump(out, fh, indent=1)
    print('UNIVFD_EVAL_V2_DONE', flush=True)


main()
