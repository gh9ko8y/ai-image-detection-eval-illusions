# -*- coding: utf-8 -*-
"""图 2 重画：1/f 去趋势后的残差谱相关 + 类内噪声底阈值（外审第7条修正版）。
数据：resid_corr.csv（服务器实测）+ null_stats.csv（类内对半，1600 次重采样）。"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
R = pd.read_csv(os.path.join(BASE, 'resid_corr.csv'), index_col=0)
null = pd.read_csv(os.path.join(BASE, 'null_stats.csv'))['split_half_corr']
thr = np.percentile(null, 5)  # 类内噪声底：同类对半相关的 5% 分位

names = list(R.index)
C = R.values

fig, ax = plt.subplots(figsize=(4.8, 4.2))
im = ax.imshow(C, cmap='viridis', vmin=0.96, vmax=1.0)
ax.set_xticks(range(8), names, rotation=35, ha='right')
ax.set_yticks(range(8), names)
for i in range(8):
    for j in range(8):
        if i != j:
            v = C[i, j]
            # 触及噪声底的格子：白字加粗
            hot = v >= thr and i < j
            ax.text(j, i, '%.4f' % v if v > 0.999 else '%.3f' % v,
                    ha='center', va='center', fontsize=5.0,
                    color='white' if hot or v < 0.975 else 'black',
                    fontweight='bold' if hot else 'normal')
# SD15-Wukong 高亮框
for (a, b) in [(names.index('SD15'), names.index('Wukong'))]:
    ax.add_patch(plt.Rectangle((a - 0.5, b - 0.5), 1, 1, fill=False,
                               edgecolor='#e41a1c', lw=1.8))
    ax.add_patch(plt.Rectangle((b - 0.5, a - 0.5), 1, 1, fill=False,
                               edgecolor='#e41a1c', lw=1.8))
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
cb.set_label('Residual-spectrum correlation (1/f detrended)')
ax.set_title('Same-source threshold (within-class 5%% pct) = %.4f\n'
             'only SD15–Wukong (boxed) reaches it' % thr,
             fontsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(BASE, 'figs', 'fig2_spectral_corr.png'),
            bbox_inches='tight', dpi=300)
print('FIG2_REDRAWN thr=%.5f' % thr)
