#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""残差谱 + 零模型 + 置换/Holm —— TIFS 审稿 M5/M6 修正版（替代缺失的 spec_resid.py 与旧 spec_perm_holm.py）
=====================================================================================
修正点（相对 spec_perm_holm.py / spec_resid.py）：
  [M5] radial_profile 之前 np.fft.fftshift（旧版未 shift，DC 落在最后一个 bin，
       丢掉的 [1:] 第 0 格其实是 Nyquist 角，1/f 去趋势的频率轴是反的）；
       并且逐图减均值（与 spec_fp.py 一致），丢弃真正的 DC bin。
  [M6] 参考分布与观测值样本量对齐：类内 split-half 用 150 vs 150，
       跨类观测同时报告 300 vs 300（旧口径）与 150 vs 150（匹配口径）。
  [M6] 置换检验同时给出两个方向的单侧 p 和双侧 p（旧版只检验"比随机混组更相似"），
       并统计"低于全部置换值"的对数。
自检：脚本启动时先用合成图验证 DC 落在第 0 bin、k=3 余弦落在第 3 bin，不通过即退出。
输出（OUTDIR）：resid_corr.csv, resid_corr_matched150.csv, null_stats.csv,
               perm_holm_result.json, perm_holm_table.csv, radial_profiles.csv
用法：python spec_resid_fixed.py [--n-perm 5000] [--outdir ...]
CPU 即可（无 GPU）；2400 张 224×224 FFT + 5000 次置换，约 5–15 分钟。
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

BASE = os.environ.get('DATA_ROOT', '<DATA_ROOT>')
GENS7 = ['ADM', 'BigGAN', 'GLIDE', 'Midjourney', 'SD15', 'VQDM', 'Wukong']
ALL8 = GENS7 + ['Real']
SEED = 42
N_PER_CLASS = 300
N_SPLIT_HALF = 200          # 每类 200 次对半 -> 8 类共 1600（与旧版同规模）


def radial_profile(img_gray):
    """逐图减均值 -> fft2 -> fftshift -> 以频谱中心为原点的径向平均幅度。返回去掉 DC bin 后的剖面。"""
    x = img_gray - img_gray.mean()
    f = np.fft.fftshift(np.fft.fft2(x))
    p = np.abs(f) ** 2
    H, W = p.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    tbin = np.bincount(r.ravel(), p.ravel())
    nr = np.bincount(r.ravel())
    prof = np.sqrt(tbin / np.maximum(nr, 1))
    return prof[1:min(H, W) // 2]          # 去 DC；只保留完整圆环（到 Nyquist）


def _selftest():
    N = 224
    yy, xx = np.mgrid[0:N, 0:N]
    for k in (3, 20, 60):
        img = np.cos(2 * np.pi * xx * k / N) + 5.0
        pr = radial_profile(img)
        assert pr.argmax() + 1 == k, ('radial self-test failed', k, pr.argmax() + 1)
    print('radial_profile self-test OK (DC removed, bin index == spatial frequency)')


def detrend(spec_mean):
    n = len(spec_mean)
    design = np.vstack([np.log10(np.arange(1, n + 1)), np.ones(n)]).T
    coef, *_ = np.linalg.lstsq(design, spec_mean, rcond=None)
    return spec_mean - design @ coef


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def holm(p):
    order = np.argsort(p)
    m = len(p)
    out = np.empty(m)
    running = 0.0
    for rank, k in enumerate(order):
        running = max(running, (m - rank) * p[k])
        out[k] = min(1.0, running)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prep', default=BASE + '/data_npy')
    ap.add_argument('--outdir', default=BASE + '/runs/spec_resid_fixed')
    ap.add_argument('--n-perm', type=int, default=5000)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    _selftest()

    X = np.load(os.path.join(a.prep, 'X_validation.npy'), mmap_mode='r')
    g = pd.read_csv(os.path.join(a.prep, 'validation_meta_named.csv'))['gen_name'].values
    rng = np.random.RandomState(SEED)

    specs = {}
    for gen in ALL8:
        idx = rng.choice(np.where(g == gen)[0], N_PER_CLASS, replace=False)
        sp = [radial_profile(np.asarray(X[i], dtype='float64').mean(axis=2)) for i in idx]
        specs[gen] = np.log10(np.array(sp) + 1e-12)
        print('spec', gen, specs[gen].shape, flush=True)
    pd.DataFrame({gen: specs[gen].mean(0) for gen in ALL8}).to_csv(
        os.path.join(a.outdir, 'radial_profiles.csv'), index_label='freq_bin_minus1')

    # 原始（未去趋势）类均值谱相关 —— 对应论文 "raw spectra all pairwise 0.974–0.999"
    raw = {gen: specs[gen].mean(0) for gen in ALL8}
    Rraw = pd.DataFrame([[corr(raw[x], raw[y]) for y in ALL8] for x in ALL8], index=ALL8, columns=ALL8)
    Rraw.to_csv(os.path.join(a.outdir, 'raw_corr.csv'))

    # 观测：300 vs 300（旧口径）
    resid = {gen: detrend(specs[gen].mean(0)) for gen in ALL8}
    R = pd.DataFrame([[corr(resid[x], resid[y]) for y in ALL8] for x in ALL8], index=ALL8, columns=ALL8)
    R.to_csv(os.path.join(a.outdir, 'resid_corr.csv'))
    # 观测：150 vs 150（与 split-half 匹配；每类取前 150 张——各类独立随机样本）
    resid150 = {gen: detrend(specs[gen][:150].mean(0)) for gen in ALL8}
    R150 = pd.DataFrame([[corr(resid150[x], resid150[y]) for y in ALL8] for x in ALL8], index=ALL8, columns=ALL8)
    R150.to_csv(os.path.join(a.outdir, 'resid_corr_matched150.csv'))

    # 类内 split-half 参考（150 vs 150）
    rows = []
    for gen in ALL8:
        for _ in range(N_SPLIT_HALF):
            pm = rng.permutation(N_PER_CLASS)
            h1 = detrend(specs[gen][pm[:150]].mean(0))
            h2 = detrend(specs[gen][pm[150:]].mean(0))
            rows.append((gen, corr(h1, h2)))
    null = pd.DataFrame(rows, columns=['gen', 'split_half_corr'])
    null.to_csv(os.path.join(a.outdir, 'null_stats.csv'), index=False)

    # 跨类置换（7 生成器 2100 张打乱成 7×300），双向 p
    pair_idx = [(i, j) for i in range(7) for j in range(i + 1, 7)]
    names = ['%s-%s' % (GENS7[i], GENS7[j]) for i, j in pair_idx]
    obs = np.array([R.loc[GENS7[i], GENS7[j]] for i, j in pair_idx])
    pool = np.concatenate([specs[gen] for gen in GENS7], axis=0)
    L = pool.shape[1]
    perm = np.empty((a.n_perm, len(pair_idx)))
    for t in range(a.n_perm):
        grp = pool[rng.permutation(len(pool))].reshape(7, N_PER_CLASS, L).mean(axis=1)
        rr = [detrend(grp[k]) for k in range(7)]
        perm[t] = [corr(rr[i], rr[j]) for i, j in pair_idx]
        if (t + 1) % 1000 == 0:
            print('perm %d / %d' % (t + 1, a.n_perm), flush=True)
    m = a.n_perm
    p_greater = ((perm >= obs).sum(0) + 1.0) / (m + 1.0)   # 比随机混组更相似
    p_less = ((perm <= obs).sum(0) + 1.0) / (m + 1.0)      # 比随机混组更不相似（=存在类结构）
    p_two = np.minimum(1.0, 2 * np.minimum(p_greater, p_less))
    res = {
        'n_perm': m, 'seed': SEED, 'fftshift_fixed': True, 'pairs': names,
        'observed_corr': obs.tolist(),
        'p_greater': p_greater.tolist(), 'p_greater_holm': holm(p_greater).tolist(),
        'p_less': p_less.tolist(), 'p_less_holm': holm(p_less).tolist(),
        'p_two_sided': p_two.tolist(), 'p_two_sided_holm': holm(p_two).tolist(),
        'n_below_all_perm': int((obs < perm.min(0)).sum()),
        'null_mean': perm.mean(0).tolist(), 'null_q05': np.quantile(perm, 0.05, 0).tolist(),
        'null_q95': np.quantile(perm, 0.95, 0).tolist(),
        'within_class_q05_pooled': float(null.split_half_corr.quantile(0.05)),
        'within_class_sd_pooled': float(null.split_half_corr.std()),
        'within_class_q05_by_class': null.groupby('gen').split_half_corr.quantile(0.05).to_dict(),
    }
    json.dump(res, open(os.path.join(a.outdir, 'perm_holm_result.json'), 'w'), indent=1)
    pd.DataFrame({'pair': names, 'observed': obs, 'p_greater': p_greater,
                  'p_greater_holm': holm(p_greater), 'p_less': p_less,
                  'p_less_holm': holm(p_less), 'p_two_sided_holm': holm(p_two)}).to_csv(
        os.path.join(a.outdir, 'perm_holm_table.csv'), index=False)
    print(R.round(5).to_string())
    print('within-class q05 (pooled) = %.6f' % res['within_class_q05_pooled'])
    print('pairs below all permutations:', res['n_below_all_perm'])
    print('SPEC_RESID_FIXED_DONE')


if __name__ == '__main__':
    main()
