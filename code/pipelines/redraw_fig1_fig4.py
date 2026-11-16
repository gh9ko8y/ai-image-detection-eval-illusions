### FIGS_V2 ### strip marker lines
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = os.environ.get('DATA_ROOT', '<DATA_ROOT>')
OUT = BASE + '/runs/figs'
os.makedirs(OUT, exist_ok=True)
GENS = ['ADM', 'BigGAN', 'GLIDE', 'Midjourney', 'SD15', 'VQDM', 'Wukong']
COLS = GENS + ['Real']

ACC = np.array([
    [40.5, 98.7, 98.0, 78.1, 93.6, 87.9, 91.6, 85.1],
    [98.6, 62.6, 97.6, 80.9, 94.1, 86.0, 91.9, 87.2],
    [99.0, 97.1, 89.1, 69.0, 87.9, 83.3, 87.8, 92.2],
    [98.1, 97.1, 93.8, 14.9, 79.7, 81.1, 78.3, 96.6],
    [98.9, 96.0, 98.1, 67.6, 73.7, 77.7, 79.5, 95.6],
    [99.4, 99.5, 98.2, 74.3, 90.4, 27.7, 89.5, 93.1],
    [98.6, 98.9, 97.6, 64.8, 73.3, 76.8, 61.8, 96.6]])
FULL_BASE = [98.8, 99.4, 98.6, 80.2, 93.0, 88.8, 91.4, 90.14]
plt.rcParams.update({'font.size': 8, 'figure.dpi': 300})

fig, ax = plt.subplots(figsize=(6.0, 4.6))
im = ax.imshow(ACC, cmap='RdBu_r', vmin=10, vmax=100, aspect='auto')
ax.set_xticks(range(8), COLS, rotation=35, ha='right')
ax.set_yticks(range(7), GENS)
ax.set_xlabel('Evaluated class (validation set)')
ax.set_ylabel('Generator held out of training')
for i in range(7):
    for j in range(8):
        v = ACC[i, j]
        w = 'bold' if i == j else 'normal'
        c = 'white' if v < 35 or v > 92 else 'black'
        ax.text(j, i, '%.1f' % v, ha='center', va='center', fontsize=6.2,
                color=c, fontweight=w)
for k in range(7):
    ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False,
                               edgecolor='black', lw=1.4))
ax.axhline(6.6, color='#222222', lw=0.8, ls='--')
ax.text(-0.45, 6.75, 'full-training baseline $\\rightarrow$',
        fontsize=5.6, color='#222222', va='bottom')
for j in range(8):
    ax.text(j, 6.88, '%.1f' % FULL_BASE[j], ha='center', va='center',
            fontsize=5.6, color='#222222', style='italic')
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.set_label('Accuracy (%)')
fig.tight_layout()
fig.savefig(OUT + '/loo_heatmap.png', bbox_inches='tight')
plt.close(fig)
print('FIG1_REDRAWN', flush=True)

sdxl = json.load(open(BASE + '/runs/sdxl_zeroshot/result.json'))
archs = ['resnet', 'effnet', 'clip']
labels = ['ResNet-18', 'EffNet-B0', 'CLIP B/16']
tf_vals = [51.39, 47.4, 0.27]
tf_err = [0, 2.7, 0]
means, errs = [], []
for a in archs:
    v = [sdxl[a][k]['acc'] for k in ['r1', 'r2', 'r3']]
    means.append(sum(v) / 3)
    errs.append((max(v) - min(v)) / 2)
fig, ax = plt.subplots(figsize=(4.6, 3.1))
x = np.arange(3)
w = 0.28
b1 = ax.bar(x - w, tf_vals, w, yerr=tf_err, capsize=3,
            color=['#8da0cb', '#fc8d62', '#d73027'],
            label='low-pass defense (transfer / adaptive)')
b2 = ax.bar(x + w, means, w, yerr=errs, capsize=3, color='#66c2a5',
            label='SDXL (2023) zero-shot')
for xi, m, e in zip(x + w, means, errs):
    ax.text(xi + w, m + e + 1.2, '%.1f' % m, ha='center', fontsize=7.5,
            fontweight='bold')
ax.set_xticks(x, labels)
ax.set_ylabel('Fake-accuracy (%)')
ax.set_ylim(0, 108)
ax.legend(fontsize=6.5, loc='upper left')
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(OUT + '/threat_model.png', bbox_inches='tight')
plt.close(fig)
print('FIG4_REDRAWN', flush=True)
