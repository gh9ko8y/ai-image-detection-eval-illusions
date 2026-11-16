# -*- coding: utf-8 -*-
"""图 2（论文 Fig. 4）v2：TIFS 审稿 M5/M15 修正版。
- 读 spec_resid_fixed.py 的新输出（fftshift 修正后）；
- 不再硬编码 "only SD15–Wukong reaches it"：自动框出所有超过类内 5% 分位的生成器对，标题按结果生成；
- 色标下限按数据自动取。
用法：python redraw_fig2_v2.py <spec_outdir> <fig_outdir>"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else '.'
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(SRC, 'figs')
os.makedirs(OUT, exist_ok=True)
R = pd.read_csv(os.path.join(SRC, 'resid_corr.csv'), index_col=0)
null = pd.read_csv(os.path.join(SRC, 'null_stats.csv'))['split_half_corr']
thr = np.percentile(null, 5)
names = list(R.index)
C = R.values
off = C[~np.eye(len(names), dtype=bool)]
vmin = np.floor(off.min() * 100) / 100

fig, ax = plt.subplots(figsize=(4.8, 4.2))
im = ax.imshow(C, cmap='viridis', vmin=vmin, vmax=1.0)
ax.set_xticks(range(len(names)), names, rotation=35, ha='right')
ax.set_yticks(range(len(names)), names)
hits = []
for i in range(len(names)):
    for j in range(len(names)):
        if i == j:
            continue
        v = C[i, j]
        hot = v >= thr and 'Real' not in (names[i], names[j])
        if hot and i < j:
            hits.append((i, j))
        ax.text(j, i, '%.4f' % v, ha='center', va='center', fontsize=4.6,
                color='white' if v < (vmin + 1) / 2 else 'black',
                fontweight='bold' if hot else 'normal')
for i, j in hits:
    for a, b in [(i, j), (j, i)]:
        ax.add_patch(plt.Rectangle((b - 0.5, a - 0.5), 1, 1, fill=False, edgecolor='#e41a1c', lw=1.6))
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
cb.set_label('Residual-spectrum correlation (1/f detrended)')
desc = ('pairs at/above it (boxed): ' + ', '.join('%s–%s' % (names[i], names[j]) for i, j in hits)) if hits else 'no generator pair reaches it'
ax.set_title('Within-class split-half 5th pct = %.4f\n%s' % (thr, desc), fontsize=7.0)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig2_spectral_corr.png'), bbox_inches='tight', dpi=300)
print('FIG2_V2 thr=%.5f hits=%s' % (thr, hits))
