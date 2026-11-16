#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D批：跨类置换零分布 + Holm 校正（补频谱分析的统计收尾）。
==============================================================
目的：把论文中"21 对跨生成器残差相关"的显著性放到置换零分布下检验。
做法：
  1) 完整复刻 spec_resid.py 的观测值（同 seed 42、同 300 张/类、同径向
     剖面、同去趋势），并对照 resid_corr.csv 验证复现（报告最大偏差）；
  2) 把 7 个生成器共 2100 张的单图谱打乱，随机重分 7 组 x 300，
     重算 21 对残差相关 -> 单次置换的零值；重复 N_PERM 次；
  3) 每对 p 值按 Phipson & Smyth (2010) 取 (b+1)/(m+1)；
  4) Holm (1979) 步降校正 21 个 p；
  5) 输出 runs/spec_perm/perm_holm_result.json。
判读：预期 SD15-Wukong 校正后不显著（观测仅超类内参考约 1 sigma），
      与论文"证据只支撑到校准层耦合"的声明一致。
用法：
  python spec_perm_holm.py
"""
import json
import os

import numpy as np
import pandas as pd

BASE = os.environ.get('DATA_ROOT', '<DATA_ROOT>')
OUTDIR = os.path.join(BASE, 'runs', 'spec_perm')
os.makedirs(OUTDIR, exist_ok=True)
GENS7 = ['ADM', 'BigGAN', 'GLIDE', 'Midjourney', 'SD15', 'VQDM', 'Wukong']
ALL8 = GENS7 + ['Real']
N_PERM = 5000
SEED = 42


def radial_profile(img_f):
    p = np.abs(img_f) ** 2
    h = img_f.shape[0] // 2
    yy, xx = np.mgrid[0:img_f.shape[0], 0:img_f.shape[1]]
    r = np.sqrt((yy - h) ** 2 + (xx - h) ** 2).astype(int)
    tbin = np.bincount(r.ravel(), p.ravel())
    nr = np.bincount(r.ravel())
    return np.sqrt(tbin / np.maximum(nr, 1))


def detrend(spec_mean):
    n = len(spec_mean)
    design = np.vstack([np.log10(np.arange(1, n + 1)), np.ones(n)]).T
    coef, *_ = np.linalg.lstsq(design, spec_mean, rcond=None)
    return spec_mean - design @ coef


def main():
    X = np.load(os.path.join(BASE, 'data_npy', 'X_validation.npy'), mmap_mode='r')
    meta = pd.read_csv(os.path.join(BASE, 'data_npy', 'validation_meta_named.csv'))
    g = meta['gen_name'].values
    rng = np.random.RandomState(SEED)          # 与 spec_resid.py 完全同种子

    # ---- 1) 复刻观测值（与 spec_resid.py 逐行同逻辑）----
    specs = {}
    for gen in ALL8:
        idx = rng.choice(np.where(g == gen)[0], 300, replace=False)
        sp = []
        for i in idx:
            f = np.fft.fft2(np.asarray(X[i], dtype='float32').mean(axis=2))
            sp.append(radial_profile(f)[1:])
        specs[gen] = np.log10(np.array(sp) + 1e-12)
        print('spec', gen, specs[gen].shape, flush=True)

    resid = {gen: detrend(specs[gen].mean(0)) for gen in ALL8}
    R = np.array([[np.corrcoef(resid[a], resid[b])[0, 1] for b in ALL8]
                  for a in ALL8])
    old = pd.read_csv(os.path.join(BASE, 'runs', 'spec_resid', 'resid_corr.csv'),
                      index_col=0)
    max_diff = float(np.abs(R - old.values).max())
    print('reproduction check vs resid_corr.csv: max abs diff = %.2e' % max_diff,
          flush=True)

    # 7 生成器观测 21 对
    pair_idx = [(i, j) for i in range(7) for j in range(i + 1, 7)]
    pair_names = ['%s-%s' % (GENS7[i], GENS7[j]) for i, j in pair_idx]
    obs = np.array([np.corrcoef(resid[GENS7[i]], resid[GENS7[j]])[0, 1]
                    for i, j in pair_idx])

    # ---- 2) 跨类置换零分布（2100 张打乱重分 7 组）----
    pool = np.concatenate([specs[gen] for gen in GENS7], axis=0)  # (2100, L)
    n_img, L = pool.shape
    per_grp = 300
    null = np.empty((N_PERM, len(pair_idx)))
    for t in range(N_PERM):
        pm = rng.permutation(n_img)
        groups = pool[pm].reshape(7, per_grp, L).mean(axis=1)     # 7 组均值谱
        rr = np.array([detrend(groups[k]) for k in range(7)])
        null[t] = [np.corrcoef(rr[i], rr[j])[0, 1] for i, j in pair_idx]
        if (t + 1) % 1000 == 0:
            print('perm %d / %d' % (t + 1, N_PERM), flush=True)

    # ---- 3) Phipson & Smyth p 值 ----
    b = (null >= obs[None, :]).sum(axis=0)
    p_raw = (b + 1.0) / (N_PERM + 1.0)

    # ---- 4) Holm 步降 ----
    order = np.argsort(p_raw)
    m = len(p_raw)
    p_holm = np.empty(m)
    running = 0.0
    for rank, k in enumerate(order):
        adj = (m - rank) * p_raw[k]
        running = max(running, adj)
        p_holm[k] = min(1.0, running)

    result = {
        'n_perm': N_PERM,
        'seed': SEED,
        'reproduction_max_abs_diff_vs_resid_corr': max_diff,
        'n_pairs': m,
        'pairs': pair_names,
        'observed_corr': obs.tolist(),
        'p_raw': p_raw.tolist(),
        'p_holm': p_holm.tolist(),
        'null_mean': null.mean(axis=0).tolist(),
        'null_q95': np.quantile(null, 0.95, axis=0).tolist(),
        'null_max': null.max(axis=0).tolist(),
        'n_sig_raw_005': int((p_raw < 0.05).sum()),
        'n_sig_holm_005': int((p_holm < 0.05).sum()),
        'holm_min_p': float(p_holm.min()),
    }
    with open(os.path.join(OUTDIR, 'perm_holm_result.json'), 'w') as f:
        json.dump(result, f, indent=1)
    df = pd.DataFrame({'pair': pair_names, 'observed': obs,
                       'p_raw': p_raw, 'p_holm': p_holm,
                       'null_q95': np.quantile(null, 0.95, axis=0)})
    df.to_csv(os.path.join(OUTDIR, 'perm_holm_table.csv'), index=False)
    print(df.sort_values('p_holm').head(5).to_string(index=False))
    print('n significant (raw p < 0.05):', result['n_sig_raw_005'],
          '| after Holm:', result['n_sig_holm_005'])
    print('SPEC_PERM_DONE')


if __name__ == '__main__':
    main()
