# -*- coding: utf-8 -*-
"""图 5 v3：修复边框文字压线 + 消除 Stage 4 与主表同种子 42 的表述矛盾。"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = r'updates_20261127/e3_e4_e5_protocol\figs_e345'
os.makedirs(OUT, exist_ok=True)

fig, ax = plt.subplots(figsize=(7.6, 3.2))
ax.set_xlim(0, 104)
ax.set_ylim(0, 42)
ax.axis('off')

stages = [
    (1.5, 'Stage 1\nThreat-model audit',
     'transfer / naive WB /\nadaptive (EOT, BPDA)',
     'masking flag; clean\n+ adaptive columns'),
    (27.5, 'Stage 2\nLOO matrix audit',
     'retrain per excluded\ngenerator (>=2 runs)',
     r'$\Delta_g$ & $\Delta$AUROC vs.' + '\nfull baseline + recal.'),
    (53.5, 'Stage 3\nSpectral attribution',
     'class-mean spectra,\n1/f detrend + null',
     'descriptive similarity only\nunless corrected null clears'),
    (79.5, 'Stage 4\nReplication audit',
     '>=3 repeated runs\n(report seed scope;\ndistinct seeds preferred)',
     'mean + range;\nfloor vs. transition cells'),
]
for x, title, inp, outp in stages:
    box = FancyBboxPatch((x, 7.5), 22.0, 25.0, boxstyle='round,pad=0.6',
                         fc='#eef2f7', ec='#2b5d8a', lw=1.2)
    ax.add_patch(box)
    ax.text(x + 11.0, 28.2, title, ha='center', va='center', fontsize=7.8,
            fontweight='bold', color='#1a3a5c')
    ax.text(x + 11.0, 19.6, inp, ha='center', va='center', fontsize=6.0,
            color='#333333', linespacing=1.25)
    ax.text(x + 11.0, 11.8, outp, ha='center', va='center', fontsize=5.9,
            color='#2b5d8a', style='italic', linespacing=1.2)

for x in [24.2, 50.2, 76.2]:
    ax.add_patch(FancyArrowPatch((x, 20), (x + 2.8, 20),
                                 arrowstyle='-|>', mutation_scale=13,
                                 color='#2b5d8a', lw=1.4))

ax.text(52, 38.5, 'Four-stage AIGC-detector evaluation pipeline '
        '(single consumer GPU; every stage altered a conclusion on our substrate)',
        ha='center', fontsize=7.5, color='#555555')
ax.text(12.5, 3.0, 'input: detector + defense', ha='center', fontsize=6.0,
        color='#888888')
ax.text(90.5, 3.0, 'output: protocol-annotated report', ha='center', fontsize=6.0,
        color='#888888')

fig.tight_layout()
p_out = os.path.join(OUT, 'fig4_pipeline.png')
fig.savefig(p_out, dpi=300, bbox_inches='tight')
print('SAVED FIG5:', p_out)
