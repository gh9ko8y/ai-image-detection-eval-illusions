# -*- coding: utf-8 -*-
"""图 4 v2（审稿 X1 修正）：拆两个子图。
(a) 低通防御三口径：x = 不攻击/迁移/自适应（迁移带三次测量误差棒+噪声带）
(b) SDXL 零样本：y = 假图准确率（与表 6 一致），三轮极差误差棒"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'figs')
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 8, 'figure.dpi': 300})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.0))

# ---- (a) 低通三口径 ----
labels = ['No attack', 'Transfer', 'Adaptive']
vals = [51.39, 47.4, 0.27]
errs = [0, 1.35, 0]  # 迁移 = 三次极差/2
colors = ['#8da0cb', '#fc8d62', '#d73027']
bars = ax1.bar(labels, vals, width=0.55, yerr=errs, capsize=4,
               color=colors)
ax1.axhspan(46.05, 48.75, color='#fc8d62', alpha=0.15)
ax1.text(2.35, 47.4, 'transfer\nnoise band\n(3 runs)', fontsize=5.6,
         va='center', color='#a0522d')
for b, v in zip(bars, vals):
    ax1.text(b.get_x() + b.get_width() / 2, v + 1.5, '%.2f' % v,
             ha='center', fontsize=8, fontweight='bold')
ax1.set_ylabel('Low-pass defense: fake-accuracy (%)')
ax1.set_ylim(0, 62)
ax1.set_title('(a) Threat-model framing (Table 2)', fontsize=8)
ax1.spines[['top', 'right']].set_visible(False)

# ---- (b) SDXL 假图准确率 ----
sdxl = json.load(open(os.path.join(BASE, 'sdxl_result.json')))
labels2 = ['ResNet-18', 'EffNet-B0', 'CLIP B/16']
means, errs2 = [], []
for a in ['resnet', 'effnet', 'clip']:
    v = [sdxl[a][k]['acc_fake'] for k in ['r1', 'r2', 'r3']]
    means.append(sum(v) / 3)
    errs2.append((max(v) - min(v)) / 2)
bars2 = ax2.bar(labels2, means, width=0.55, yerr=errs2, capsize=4,
                color='#66c2a5')
for b, m, e in zip(bars2, means, errs2):
    ax2.text(b.get_x() + b.get_width() / 2, m + e + 1.5, '%.1f' % m,
             ha='center', fontsize=8, fontweight='bold')
ax2.set_ylabel('SDXL (2023) zero-shot: fake-accuracy (%)')
ax2.set_ylim(0, 108)
ax2.set_title('(b) Out-of-sample probe (Table 5)', fontsize=8)
ax2.spines[['top', 'right']].set_visible(False)

fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig3_threat_model.png'), bbox_inches='tight')
print('FIG4_V2_DONE', means, errs2)
