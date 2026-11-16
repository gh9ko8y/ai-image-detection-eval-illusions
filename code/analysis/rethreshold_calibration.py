#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TIFS 审稿 S16（可选但强烈建议）：校准漂移的直接证据。
对 full-training 检测器与 7 个 LOO 检测器（run1），计算：
  - 每类 ECE（10 bin）与真图上的 FPR；
  - "只用真图重定阈值"：把每个 LOO 检测器的阈值设到使真图 FPR 等于 full 检测器真图 FPR 的位置，
    重算逐类准确率 → 看非对角 Δacc 是否回到 ≈0（校准漂移可修复），对角格是否仍崩（真不可分）。
只需真图做重定阈值，不需要未见生成器的标注，部署上可行。
用法：python rethreshold_calibration.py --prep <DATA_ROOT>/data_npy \
        --full <DATA_ROOT>/runs/round_01/detector.pth \
        --loo-dir <DATA_ROOT>/runs/loo_01 --run 1 --out runs/rethreshold/result.json
GPU：每个模型推理 7000 张约 20 s，共 8 个模型 ≈ 3–5 分钟。"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_detector as td

GENS = ['ADM', 'BigGAN', 'GLIDE', 'Midjourney', 'SD15', 'VQDM', 'Wukong']


def scores(ckpt, X, dev):
    m = td.build_model(2)
    m.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=False)
    m.to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 200):
            out.append(F.softmax(m(td.to_tensor(np.asarray(X[i:i + 200])).to(dev)), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def ece(p, y, bins=10):
    e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prep', required=True)
    ap.add_argument('--full', required=True)
    ap.add_argument('--loo-dir', required=True)
    ap.add_argument('--run', type=int, default=1)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = np.load(os.path.join(a.prep, 'X_validation.npy'), mmap_mode='r')
    y = np.load(os.path.join(a.prep, 'y_validation.npy')).reshape(-1)
    gen = pd.read_csv(os.path.join(a.prep, 'validation_meta_named.csv'))['gen_name'].values
    real = gen == 'Real'
    fake_label = int(pd.Series(y[~real]).mode().iloc[0])
    yb = (y == fake_label).astype(float)

    def summarize(s, thr):
        d = {'thr': float(thr), 'real_fpr': float((s[real] >= thr).mean() * 100)}
        for g in GENS:
            d[g] = float((s[gen == g] >= thr).mean() * 100)
        d['ece_all'] = ece(s, yb)
        return d

    s_full = scores(a.full, X, dev)
    full = summarize(s_full, 0.5)
    target_fpr = full['real_fpr'] / 100
    res = {'full': full, 'loo': {}}
    for h in GENS:
        ck = os.path.join(a.loo_dir, 'detector_%s_r%d.pth' % (h, a.run))
        s = scores(ck, X, dev)
        thr_re = float(np.quantile(s[real], 1 - target_fpr))
        res['loo'][h] = {'fixed_0.5': summarize(s, 0.5), 'real_only_rethreshold': summarize(s, thr_re)}
        print(h, 'Δacc (fixed→rethr) per class:',
              {g: (round(res['loo'][h]['fixed_0.5'][g] - full[g], 1),
                   round(res['loo'][h]['real_only_rethreshold'][g] - full[g], 1)) for g in GENS}, flush=True)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)
    print('RETHRESHOLD_DONE')


if __name__ == '__main__':
    main()
