# -*- coding: utf-8 -*-
"""图 5 v2（TIFS 审稿 M15/S16 修正：Stage 3 输出改为"有界的描述性相似度"、Stage 4 改为重复运行+种子、底部去掉 certifiable）
原说明：图 4：四阶段评测管线示意图（论文 fig4_pipeline.png）。
按 CCF-B 论文作图规范：无装饰、单色调、明确输入输出与判定规则。"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import sys
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')
os.makedirs(OUT, exist_ok=True)

fig, ax = plt.subplots(figsize=(7.2, 3.1))
ax.set_xlim(0, 100)
ax.set_ylim(0, 42)
ax.axis('off')

stages = [
    (2, 'Stage 1\nThreat-model audit',
     'transfer / naive WB / adaptive\n+ EOT, BPDA',
     'masking flag;\nadaptive column'),
    (27, 'Stage 2\nLOO matrix audit',
     'retrain per excluded\ngenerator (>=2 runs)',
     r'$\Delta_g$ vs. full-training' + '\nbaseline per class'),
    (52, 'Stage 3\nSpectral attribution',
     'class-mean spectra,\n1/f detrend + null',
     'descriptive similarity only\nunless corrected null clears'),
    (77, 'Stage 4\nReplication audit',
     '>=3 repeated runs\n(distinct seeds)',
     'mean + range;\nfloor vs. transition cells'),
]
for x, title, inp, outp in stages:
    box = FancyBboxPatch((x, 8), 21, 24, boxstyle='round,pad=0.6',
                         fc='#eef2f7', ec='#2b5d8a', lw=1.2)
    ax.add_patch(box)
    ax.text(x + 10.5, 27.5, title, ha='center', va='center', fontsize=8.2,
            fontweight='bold', color='#1a3a5c')
    ax.text(x + 10.5, 19.5, inp, ha='center', va='center', fontsize=6.2,
            color='#333333')
    ax.text(x + 10.5, 12.2, outp, ha='center', va='center', fontsize=6.2,
            color='#2b5d8a', style='italic')

for x in [23.6, 48.6, 73.6]:
    ax.add_patch(FancyArrowPatch((x, 20), (x + 3.4, 20),
                                 arrowstyle='-|>', mutation_scale=14,
                                 color='#2b5d8a', lw=1.4))

ax.text(50, 38.5, 'Four-stage AIGC-detector evaluation pipeline '
        '(single consumer GPU; every stage altered a conclusion on our substrate)',
        ha='center', fontsize=7.6, color='#555555')
ax.text(12.5, 3.2, 'input: detector + defense', ha='center', fontsize=6,
        color='#888888')
ax.text(87.5, 3.2, 'output: protocol-annotated report', ha='center', fontsize=6,
        color='#888888')

fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig4_pipeline.png'), dpi=300,
            bbox_inches='tight')
print('PIPELINE_FIG_DONE')
