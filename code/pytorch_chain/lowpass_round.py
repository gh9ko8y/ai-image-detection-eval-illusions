#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
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
        self.y = np.load(os.path.join(DATA, 'y_validation.npy')).astype('int64')
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

    def pgd(self, r=None, eps=EPS, steps=STEPS, n=None):
        """白盒自适应攻击。r=None 表示无防御基线。"""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n', type=int, default=0, help='0=全量')
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    t0 = time.time()
    run = Runner(a.ckpt, seed=a.seed, n=a.n or None)
    n = run.n
    print('检测器: %s' % a.ckpt)
    print('评测集: %d 张 | eps=%.2f | steps=%d | seed=%d' % (n, EPS, STEPS, a.seed))

    # --- 逐生成器（干净） ---
    p0 = run.predict()
    per_gen = {}
    print('\n%-12s %8s %10s' % ('生成器', 'n', '准确率%'))
    for g in ['ADM', 'BigGAN', 'GLIDE', 'Midjourney', 'SD15', 'VQDM', 'Wukong', 'Real']:
        m = (run.gen[:n] == g)
        if m.sum() == 0:
            continue
        acc = float((p0[m] == run.y[:n][m]).mean() * 100)
        per_gen[g] = {'n': int(m.sum()), 'acc': acc, 'hist': HIST.get(g)}
        print('%-12s %8d %10.2f   (历史 %.2f)' % (g, m.sum(), acc, HIST.get(g, 0)))
    clean_bare = float((p0 == run.y[:n]).mean() * 100)
    print('%-12s %8d %10.2f   (历史 90.09)' % ('整体', n, clean_bare))

    # --- 无防御攻击基线 ---
    a_bare = run.pgd()
    bare_attacked = run.acc_of(a_bare)
    print('\n无防御 PGD eps=%.2f -> %.2f%%' % (EPS, bare_attacked))

    # --- 主扫描：白盒自适应，每个点立刻落盘 ---
    rows = []
    csv = os.path.join(a.outdir, 'results_lowpass.csv')
    print('\n%-7s %-8s %-11s %-13s' % ('keep', '实留%', '干净滤波%', '白盒自适应%'))
    for r in RATIOS:
        frac = retained_fraction(r)
        cf = float((run.predict(r) == run.y[:n]).mean() * 100)
        a_wb = run.pgd(r=r)
        wb = run.acc_of(a_wb, r)
        rows.append({'keep_ratio': r, 'retained_frac': frac, 'clean_bare': clean_bare,
                     'clean_filtered': cf, 'whitebox_eps0.05': wb,
                     'bare_attacked_eps0.05': bare_attacked})
        pd.DataFrame(rows).to_csv(csv, index=False)      # 增量落盘，崩了不丢
        print('%-7.2f %-8.2f %-11.2f %-13.2f  (%.0fs)' % (r, frac * 100, cf, wb, time.time() - t0),
              flush=True)

    summary = {
        'ckpt': a.ckpt, 'n_eval': n, 'eps': EPS, 'steps': STEPS, 'seed': a.seed,
        'clean_bare': clean_bare, 'clean_filtered': None,
        'bare_attacked_eps0.05': bare_attacked,
        'per_generator': per_gen,
        'whitebox': {('%.2f' % d['keep_ratio']): d['whitebox_eps0.05'] for d in rows},
        'clean_filtered_by_ratio': {('%.2f' % d['keep_ratio']): d['clean_filtered'] for d in rows},
        'elapsed_sec': round(time.time() - t0, 1),
    }
    with open(os.path.join(a.outdir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print('\n总耗时 %.0fs' % (time.time() - t0))
    print('已保存: %s / summary.json' % csv)
    print('DONE')


if __name__ == '__main__':
    main()
