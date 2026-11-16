#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
[v2 · TIFS 审稿 M7/M3/S1/S2 修正版；原 lowpass_round.py 保持不变]
新增：(1) 同一检测器、同一 ε 下同时测 transfer（攻击裸模型→再滤波）与 adaptive（滤波在 PGD 回路内），
          消除"迁移 47.4 来自 Paddle、自适应 0.27 来自 PyTorch"的跨链不对等；
      (2) 滤波后干净图的逐类准确率（真图 / 各生成器），回答论文 L335 "to be confirmed"；
      (3) --eps 可给多个值；--clamp 把对抗样本裁回合法像素域；结果写 JSON。
用法：python lowpass_round_v2.py --ckpt runs/round_01/detector.pth --outdir runs/lowpass_v2/r1 --eps 0.02 0.05 [--clamp]
原说明：
轮次实验库：低通滤波自适应攻击 + 逐生成器评估
================================================
从 lowpass_whitebox.py 提炼成可复用模块，供 run_round.py 每轮调用。

两条硬规则（踩过坑，别改）：
  1. 任何含 PGD 的函数都不能加 @torch.no_grad()，否则 autograd.grad 报
     "does not require grad and does not have a grad_fn"
  2. 不要把全量数据一次性放 GPU —— 7000*3*224*224*4B=4.2GB，FFT 复数中间量再翻几倍，
     32GB 显存会爆。一律分批读盘。
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision

BASE = os.environ.get('DATA_ROOT', '<DATA_ROOT>')
DATA = os.path.join(BASE, 'data_npy')
DEV = 'cuda'
MEAN = np.array([0.485, 0.456, 0.406], dtype='float32').reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype='float32').reshape(1, 3, 1, 1)
BS = 100
STEPS = 10
EPS = 0.05
RATIOS = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
HIST = {'ADM': 96.60, 'BigGAN': 98.80, 'GLIDE': 98.60, 'Midjourney': 64.80,
        'SD15': 83.20, 'VQDM': 78.20, 'Wukong': 78.00, 'Real': 94.71}


def retained_fraction(r, H=224, W=224):
    h = max(2, int(H * r)); w = max(2, int(W * r))
    m = np.zeros((H, W), dtype='float32')
    m[:h, :w] = 1; m[:h, W - w:] = 1; m[H - h:, :w] = 1; m[H - h:, W - w:] = 1
    return float(m.sum()) / (H * W)


def lowpass(x, r):
    """四角低频 mask 低通滤波（可微）。注意 ifft2 自带 1/N，不要再除。"""
    B, C, H, W = x.shape
    f = torch.fft.fft2(x, dim=(-2, -1))
    h = max(2, int(H * r)); w = max(2, int(W * r))
    m = torch.zeros((H, W), dtype=f.dtype, device=x.device)
    m[:h, :w] = 1; m[:h, W - w:] = 1; m[H - h:, :w] = 1; m[H - h:, W - w:] = 1
    return torch.fft.ifft2(f * m.view(1, 1, H, W), dim=(-2, -1)).real


class Runner:
    def __init__(self, ckpt, seed=42, n=None):
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model = torchvision.models.resnet18(weights=None, num_classes=2)
        sd = torch.load(ckpt, map_location='cpu')
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        miss, unexp = self.model.load_state_dict(sd, strict=False)
        if miss:
            raise RuntimeError('权重缺失 %d 个 key，checkpoint 不完整: %s' % (len(miss), miss[:5]))
        self.model = self.model.to(DEV).eval()
        self.X = np.load(os.path.join(DATA, 'X_validation.npy'), mmap_mode='r')
        self.y = np.load(os.path.join(DATA, 'y_validation.npy')).reshape(-1).astype('int64')
        meta = pd.read_csv(os.path.join(DATA, 'validation_meta_named.csv'))
        self.gen = meta['gen_name'].values
        self.n = n or len(self.y)

    def batch(self, i, bs=BS):
        X = np.asarray(self.X[i:i + bs])
        Xs = ((X.astype('float32') / 255.0).transpose(0, 3, 1, 2) - MEAN) / STD
        return torch.from_numpy(np.ascontiguousarray(Xs)).to(DEV)

    @torch.no_grad()
    def predict(self, r=None, n=None):
        n = n or self.n
        out = []
        for i in range(0, n, BS):
            xb = self.batch(i)
            if r is not None:
                xb = lowpass(xb, r)
            out.append(self.model(xb).argmax(1).cpu().numpy())
        return np.concatenate(out)

    clamp = False
    LO = torch.tensor((0.0 - MEAN) / STD).view(1, 3, 1, 1)
    HI = torch.tensor((1.0 - MEAN) / STD).view(1, 3, 1, 1)

    def pgd(self, r=None, eps=EPS, steps=STEPS, n=None):
        """白盒自适应攻击。r=None 表示无防御基线（其结果再过滤波即 transfer 口径）。"""
        n = n or self.n
        alpha = eps / steps
        out = []
        for i in range(0, n, BS):
            nb = min(BS, n - i)
            xb = self.batch(i, nb)
            yb = torch.from_numpy(self.y[i:i + nb]).to(DEV)
            torch.manual_seed(self.seed + i)          # 每批固定初值，整轮可复现
            xa = xb + (torch.rand_like(xb) * 2 - 1) * eps
            xa = torch.max(torch.min(xa, xb + eps), xb - eps)
            for _ in range(steps):
                xa = xa.detach().requires_grad_(True)
                loss = nn.functional.cross_entropy(
                    self.model(lowpass(xa, r) if r else xa), yb)
                g = torch.autograd.grad(loss, xa, allow_unused=True)[0]
                if g is None:
                    g = torch.zeros_like(xa)
                xa = xa.detach() + alpha * g.sign()
                xa = torch.max(torch.min(xa, xb + eps), xb - eps)
                if self.clamp:
                    xa = torch.max(torch.min(xa, self.HI.to(DEV)), self.LO.to(DEV))
            out.append(xa.detach())
        return torch.cat(out)

    @torch.no_grad()
    def acc_of(self, x, r=None):
        c = 0
        for i in range(0, len(x), BS):
            xb = x[i:i + BS]
            c += int((self.model(lowpass(xb, r) if r else xb).argmax(1) ==
                      torch.from_numpy(self.y[i:i + len(xb)]).to(DEV)).sum())
        return 100.0 * c / len(x)


def per_class_acc(pred, y, gen):
    out = {}
    for g in ['ADM', 'BigGAN', 'GLIDE', 'Midjourney', 'SD15', 'VQDM', 'Wukong', 'Real']:
        m = gen == g
        if m.sum():
            out[g] = float((pred[m] == y[m]).mean() * 100)
    fake = gen != 'Real'
    out['fake_all'] = float((pred[fake] == y[fake]).mean() * 100)
    out['overall'] = float((pred == y).mean() * 100)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n', type=int, default=0, help='0=全量')
    ap.add_argument('--eps', type=float, nargs='+', default=[0.02, 0.05])
    ap.add_argument('--ratio', type=float, default=0.15)
    ap.add_argument('--clamp', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    t0 = time.time()
    run = Runner(a.ckpt, seed=a.seed, n=a.n or None)
    run.clamp = a.clamp
    n = run.n
    y, gen = run.y[:n], run.gen[:n]
    res = {'ckpt': a.ckpt, 'n_eval': n, 'ratio': a.ratio, 'steps': STEPS,
           'alpha': 'eps/steps, uniform random start', 'clamp': a.clamp,
           'retained_frac': retained_fraction(a.ratio)}
    res['clean_bare'] = per_class_acc(run.predict(), y, gen)
    res['clean_filtered'] = per_class_acc(run.predict(a.ratio), y, gen)   # L335 所需的真/假分侧
    print('clean bare', res['clean_bare'])
    print('clean filtered', res['clean_filtered'], flush=True)
    for eps in a.eps:
        adv_bare = run.pgd(r=None, eps=eps)
        adv_lp = run.pgd(r=a.ratio, eps=eps)
        with torch.no_grad():
            def preds(x, r):
                out = []
                for i in range(0, len(x), BS):
                    xb = x[i:i + BS]
                    out.append(run.model(lowpass(xb, r) if r else xb).argmax(1).cpu().numpy())
                return np.concatenate(out)
            res['eps=%.3f' % eps] = {
                'bare_attacked': per_class_acc(preds(adv_bare, None), y, gen),
                'transfer': per_class_acc(preds(adv_bare, a.ratio), y, gen),
                'adaptive': per_class_acc(preds(adv_lp, a.ratio), y, gen),
            }
        print('eps=%.3f' % eps, {k: v['overall'] for k, v in res['eps=%.3f' % eps].items()},
              '%.0fs' % (time.time() - t0), flush=True)
        json.dump(res, open(os.path.join(a.outdir, 'lowpass_v2.json'), 'w'), indent=1)
    print('LOWPASS_V2_DONE')


if __name__ == '__main__':
    main()
