# -*- coding: utf-8 -*-
"""论文初稿配图：LOO 热图 / 频谱相关热图 / 口径对比柱图。
数据来源：第三遍复现实测（LOO 三轮均值矩阵 2026-10-18；频谱 corr_matrix 第二遍缓存，
确定性计算三遍逐位一致；低通铁三角 51.39/48.77/0.27，2026-10-17/18）。"""
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')
os.makedirs(OUT, exist_ok=True)

GENS = ['ADM', 'BigGAN', 'GLIDE', 'Midjourney', 'SD15', 'VQDM', 'Wukong']
COLS = GENS + ['Real']

# 第三遍 LOO 矩阵（rows = held-out generator, cols = evaluated class, 三轮均值）
LOO = np.array([
    [40.5, 98.7, 98.0, 78.1, 93.6, 87.9, 91.6, 85.1],
    [98.6, 62.6, 97.6, 80.9, 94.1, 86.0, 91.9, 87.2],
    [99.0, 97.1, 89.1, 69.0, 87.9, 83.3, 87.8, 92.2],
    [98.1, 97.1, 93.8, 14.9, 79.7, 81.1, 78.3, 96.6],
    [98.9, 96.0, 98.1, 67.6, 73.7, 77.7, 79.5, 95.6],
    [99.4, 99.5, 98.2, 74.3, 90.4, 27.7, 89.5, 93.1],
    [98.6, 98.9, 97.6, 64.8, 73.3, 76.8, 61.8, 96.6],
])

plt.rcParams.update({'font.size': 8, 'figure.dpi': 300})

# ---- Figure 1: LOO heatmap ----
fig, ax = plt.subplots(figsize=(5.6, 4.4))
im = ax.imshow(LOO, cmap='RdBu_r', vmin=10, vmax=100, aspect='auto')
ax.set_xticks(range(8), COLS, rotation=35, ha='right')
ax.set_yticks(range(7), GENS)
ax.set_xlabel('Evaluated class (validation set)')
ax.set_ylabel('Generator held out of training')
for i in range(7):
    for j in range(8):
        v = LOO[i, j]
        w = 'bold' if i == j else 'normal'
        c = 'white' if v < 35 or v > 92 else 'black'
        ax.text(j, i, '%.1f' % v, ha='center', va='center', fontsize=6.5,
                color=c, fontweight=w)
# 对角线框
for k in range(7):
    ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False,
                                edgecolor='black', lw=1.4))
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.set_label('Accuracy (%)')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig1_loo_heatmap.png'), bbox_inches='tight')
plt.close(fig)

# ---- Figure 2: spectral correlation heatmap ----
CSVP = os.path.join(os.environ.get('DATA_ROOT', '<DATA_ROOT>'), 'runs', 'spec_01', 'corr_matrix.csv')
names, rows = [], []
with open(CSVP, newline='') as f:
    rd = csv.reader(f)
    header = next(rd)
    for line in rd:
        names.append(line[0])
        rows.append([float(x) for x in line[1:]])
C = np.array(rows)
fig, ax = plt.subplots(figsize=(4.6, 4.0))
im = ax.imshow(C, cmap='viridis', vmin=0.970, vmax=1.0)
ax.set_xticks(range(8), names, rotation=35, ha='right')
ax.set_yticks(range(8), names)
for i in range(8):
    for j in range(8):
        if i != j:
            ax.text(j, i, '%.3f' % C[i, j], ha='center', va='center',
                    fontsize=5.2, color='white' if C[i, j] < 0.985 else 'black')
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.set_label('Pearson correlation of mean spectra')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig2_spectral_corr.png'), bbox_inches='tight')
plt.close(fig)

# ---- Figure 3: threat-model framing (low-pass defense) ----
labels = ['No attack\n(filtered)', 'Transfer attack\n(attacker blind to filter)',
          'Adaptive attack\n(attacker sees filter)']
vals = [51.39, 48.77, 0.27]
fig, ax = plt.subplots(figsize=(4.2, 3.0))
bars = ax.bar(labels, vals, width=0.55,
              color=['#8da0cb', '#fc8d62', '#d73027'])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.2, '%.2f%%' % v,
            ha='center', fontsize=9, fontweight='bold')
ax.set_ylabel('Detector accuracy under low-pass defense (%)')
ax.set_ylim(0, 62)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig3_threat_model.png'), bbox_inches='tight')
plt.close(fig)

print('FIGS_DONE', OUT)
for f in sorted(os.listdir(OUT)):
    print(' ', f)
