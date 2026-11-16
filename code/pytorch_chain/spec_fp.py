#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
频谱指纹直接测量（实验 14 候选）
================================
目的：检验"生成器家族连带"（实验 13）是否有频谱物理基础。
做法：7 个生成器 + Real 各取全部验证图（每生成器 500 张，Real 3500 张），
      灰度化 → 去均值 → FFT → 幅度谱(log1p) → 按生成器平均；
      另算径向功率谱（方位平均的一维曲线）。
输出：
  mean_spectra.npy    (9, 224, 224) 各类平均 log 幅度谱（含全图均值基准行）
  radial_profiles.csv (9, 113)      径向功率谱（低维、可解释）
  corr_matrix.csv     (9, 9)        平均 log 谱的 Pearson 相关矩阵
  spec_summary.json                  元信息与指纹
判读：若 corr 聚类呈 {SD15, Wukong, Midjourney, VQDM} vs {ADM, BigGAN} 两簇，
      与 LOO 家族连带结构一致 → "检测器学的是生成器频谱指纹"获得物理证据。
用法：
  python spec_fp.py --prep <DATA_ROOT>/data_npy \
      --out <DATA_ROOT>/runs/spec_01
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

GENS = ['ADM', 'BigGAN', 'GLIDE', 'Midjourney', 'SD15', 'VQDM', 'Wukong', 'Real']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prep', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--per-gen-limit', type=int, default=500,
                    help='每生成器最多用多少张（Real 默认也截到同量级，避免不均衡）')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = np.load(os.path.join(args.prep, 'X_validation.npy'), mmap_mode='r')
    meta = pd.read_csv(os.path.join(args.prep, 'validation_meta_named.csv'))
    names = meta['gen_name'].values

    acc = {g: None for g in GENS}          # log 谱累加
    cnt = {g: 0 for g in GENS}
    rng = np.random.default_rng(42)
    idx_by_gen = {}
    for g in GENS:
        idx = np.where(names == g)[0]
        if len(idx) > args.per_gen_limit:   # Real 3500 -> 均匀抽 500，与别家等量
            idx = rng.choice(idx, args.per_gen_limit, replace=False)
        idx_by_gen[g] = idx

    bs = 100
    for g in GENS:
        idx = idx_by_gen[g]
        run = None
        for i in range(0, len(idx), bs):
            xb = np.asarray(X[idx[i:i + bs]]).astype('float32')
            gray = xb.mean(axis=3)                    # (B,224,224) 灰度化
            gray = gray - gray.mean(axis=(1, 2), keepdims=True)   # 去均值压 DC
            t = torch.from_numpy(gray).to(dev)
            F = torch.fft.fftshift(torch.fft.fft2(t), dim=(-2, -1))
            ls = torch.log1p(F.abs())
            run = ls.sum(dim=0) if run is None else run + ls.sum(dim=0)
            cnt[g] += len(idx[i:i + bs])
        acc[g] = (run / max(cnt[g], 1)).cpu().numpy()
        print('%-11s n=%d 均值谱完成' % (g, cnt[g]), flush=True)

    spectra = np.stack([acc[g] for g in GENS])        # (8,224,224)
    np.save(os.path.join(args.out, 'mean_spectra.npy'), spectra)

    # 径向功率谱：到中心距离相同的系数取平均 -> 一维曲线（低维可解释）
    H = W = spectra.shape[-1]
    cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    rmax = min(cy, cx)
    prof = np.zeros((len(GENS), rmax + 1))
    for k, g in enumerate(GENS):
        s = spectra[k]
        for rr in range(rmax + 1):
            m = r == rr
            if m.sum():
                prof[k, rr] = spectra[k][m].mean()
    pd.DataFrame(prof, index=GENS,
                 columns=['r%d' % i for i in range(rmax + 1)]).to_csv(
        os.path.join(args.out, 'radial_profiles.csv'))

    # 平均 log 谱的 Pearson 相关矩阵（压掉 DC 区块，半径 <4 的中心不算）
    mask = r >= 4
    flat = spectra.reshape(len(GENS), -1)[:, mask.ravel()]
    corr = np.corrcoef(flat)
    pd.DataFrame(corr, index=GENS, columns=GENS).to_csv(
        os.path.join(args.out, 'corr_matrix.csv'))
    print('\ncorr matrix:')
    print(pd.DataFrame(corr, index=GENS, columns=GENS).round(3).to_string())

    json.dump({'per_gen_limit': args.per_gen_limit, 'counts': cnt,
               'preprocess': 'gray=channel-mean; per-image mean-subtracted; log1p|FFT2|fftshift',
               'note': 'Real 抽样到与其他生成器等量，避免不均衡'},
              open(os.path.join(args.out, 'spec_summary.json'), 'w'), indent=1)
    print('SPEC_DONE')


if __name__ == '__main__':
    main()
