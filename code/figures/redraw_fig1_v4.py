# -*- coding: utf-8 -*-
"""图 1 v4（R3 重跑：(a) 同检测器同预算 transfer/adaptive，三轮均值+极差/2 误差棒）
原 v3：图 4 v3（TIFS 审稿 M15 修正：(a) 纵轴=整体平衡准确率而非 fake-accuracy；面板标题表号改为 IEEE 罗马数字 Table II / Table VI；SDXL 路径可配置）
原 v2 说明：图 4 v2（审稿 X1 修正）：拆两个子图。
(a) 低通防御三口径：x = 不攻击/迁移/自适应（迁移带三次测量误差棒+噪声带）
(b) SDXL 零样本：y = 假图准确率（与表 6 一致），三轮极差误差棒"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import sys
BASE = os.path.dirname(os.path.abspath(__file__))
SDXL_JSON = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, 'sdxl_result.json')
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BASE, 'figs')
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 8, 'figure.dpi': 300})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.0))

# ---- (a) 低通：同一检测器、同一 eps，三轮 PyTorch 检测器（R3 重跑，lowpass_round_v2.py）----
# 数值取自 rerun_20261116/R3/r{1,2,3}/lowpass_v2.json（三轮 round_01..03）
R3 = {'clean_filtered': [51.385714, 50.942857, 51.0],
      0.02: {'transfer': [49.7, 49.642857, 49.442857], 'adaptive': [10.914286, 12.414286, 13.1]},
      0.05: {'transfer': [48.257143, 48.442857, 47.914286], 'adaptive': [0.271429, 0.442857, 0.557143]}}
mean = lambda v: sum(v) / len(v)
half = lambda v: (max(v) - min(v)) / 2
cf = R3['clean_filtered']
ax1.bar([0], [mean(cf)], width=0.55, yerr=[half(cf)], capsize=3, color='#8da0cb')
ax1.text(0, mean(cf) + 1.8, '%.1f' % mean(cf), ha='center', fontsize=7, fontweight='bold')
xs, w = [1.0, 2.2], 0.36
for i, e in enumerate([0.02, 0.05]):
    for j, (k, c) in enumerate([('transfer', '#fc8d62'), ('adaptive', '#d73027')]):
        v = R3[e][k]; x = xs[i] + (j - 0.5) * w * 1.05
        ax1.bar([x], [mean(v)], width=w, yerr=[half(v)], capsize=3, color=c,
                label=(k.capitalize() if i == 0 else None))
        ax1.text(x, mean(v) + 1.8, ('%.1f' if mean(v) >= 1 else '%.2f') % mean(v),
                 ha='center', fontsize=6.5, fontweight='bold')
ax1.set_xticks([0] + xs)
ax1.set_xticklabels(['No attack', r'$\varepsilon$=0.02', r'$\varepsilon$=0.05'])
ax1.set_ylabel('Low-pass defense: accuracy (%)')
ax1.axhline(50, color='#999999', lw=0.6, ls=':')
ax1.text(-0.42, 51.0, 'chance', fontsize=5.6, color='#777777', va='bottom')
ax1.set_ylim(0, 62)
ax1.legend(fontsize=6, frameon=False, loc='center', bbox_to_anchor=(0.62, 0.62))
ax1.set_title('(a) Same detector, same budget (3 rounds)', fontsize=8)
ax1.spines[['top', 'right']].set_visible(False)

# ---- (b) SDXL 假图准确率 ----
sdxl = json.load(open(SDXL_JSON))
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
ax2.set_ylabel('SDXL zero-shot: fake recall (%)')
ax2.set_ylim(0, 108)
ax2.set_title('(b) Out-of-sample probe (Table VI)', fontsize=8)
ax2.spines[['top', 'right']].set_visible(False)

fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig3_threat_model.png'), bbox_inches='tight')
print('FIG4_V2_DONE', means, errs2)
