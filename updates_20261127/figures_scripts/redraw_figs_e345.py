# -*- coding: utf-8 -*-
"""重画图 1 (fig3_threat_model.png) 与图 3 (fig1_loo_heatmap.png) —— E3/E4/E5 三次同协议执行合并版 + FLUX v1 探针。
直接从 paper_e345_master_stats.json 读取数值，零手抄。"""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

P0 = r'updates_20261127/e3_e4_e5_protocol'
STATS = os.path.join(P0, 'E5', 'paper_e345_master_stats.json')
OUT = os.path.join(P0, 'E5', 'figs_e345')
os.makedirs(OUT, exist_ok=True)

d = json.load(open(STATS, encoding='utf-8'))
plt.rcParams.update({'font.size': 8, 'figure.dpi': 300})

# =====================================================================
# 图 1：两个协议探针 -> fig3_threat_model.png
# (a) 低通防御三口径 (E3/E4/E5 共 9 轮均值 + min-max 误差棒)
# (b) 样本外零样本探针：SDXL (2023) vs FLUX.1-schnell (v1, same prompts) (各 9 组均值 + min-max)
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.15))

lp = d['r3_lowpass']
cf = lp['clean_filtered_overall']
ax1.bar([0], [cf['mean']], width=0.55,
        yerr=[[cf['mean'] - cf['min9']], [cf['max9'] - cf['mean']]],
        capsize=3, color='#8da0cb')
ax1.text(0, cf['max9'] + 1.4, '%.1f' % cf['mean'], ha='center', fontsize=7, fontweight='bold')

xs, w = [1.0, 2.2], 0.36
for i, ep in enumerate(['eps002', 'eps005']):
    for j, (k, c) in enumerate([('transfer', '#fc8d62'), ('adaptive', '#d73027')]):
        blk = lp[f'{ep}_{k}']
        m, lo, hi = blk['mean'], blk['min9'], blk['max9']
        x = xs[i] + (j - 0.5) * w * 1.05
        ax1.bar([x], [m], width=w, yerr=[[m - lo], [hi - m]], capsize=3, color=c,
                label=(k.capitalize() if i == 0 else None))
        ax1.text(x, hi + 1.4, ('%.1f' if m >= 1 else '%.2f') % m,
                 ha='center', fontsize=6.5, fontweight='bold')

ax1.set_xticks([0] + xs)
ax1.set_xticklabels(['No attack', r'$\varepsilon$=0.02', r'$\varepsilon$=0.05'])
ax1.set_ylabel('Low-pass defense: accuracy (%)')
ax1.axhline(50, color='#999999', lw=0.6, ls=':')
ax1.text(2.58, 50.8, 'chance', fontsize=5.8, color='#666666', va='bottom', ha='right')
ax1.set_ylim(0, 62)
ax1.legend(fontsize=6.5, frameon=False, loc='center', bbox_to_anchor=(0.62, 0.64))
ax1.set_title('(a) Same detector, same budget (9 runs: E3–E5)', fontsize=8)
ax1.spines[['top', 'right']].set_visible(False)

# (b) SDXL vs FLUX.1-schnell (v1, same prompts)
labels2 = ['ResNet-18', 'EffNet-B0', 'CLIP B/16']
archs = ['resnet', 'effnet', 'clip']
x2 = np.arange(3)
w2 = 0.36

for idx, (tag, key, color) in enumerate([
    ('SDXL (2023)', 'sdxl', '#66c2a5'),
    ('FLUX.1-schnell (v1)', 'flux_v1', '#4c78a8'),
]):
    means, elo, ehi = [], [], []
    for a in archs:
        blk = d[key][a]['acc_fake']
        means.append(blk['mean'])
        elo.append(blk['mean'] - blk['min9'])
        ehi.append(blk['max9'] - blk['mean'])
    pos = x2 + (idx - 0.5) * w2 * 1.06
    ax2.bar(pos, means, width=w2, yerr=[elo, ehi], capsize=3, color=color, label=tag)
    for px, m, h in zip(pos, means, ehi):
        ax2.text(px, m + h + 1.5, '%.1f' % m, ha='center', fontsize=6.6, fontweight='bold')

ax2.set_xticks(x2)
ax2.set_xticklabels(labels2)
ax2.set_ylabel('Zero-shot fake recall (%)')
ax2.set_ylim(0, 122)
ax2.set_yticks([0, 20, 40, 60, 80, 100])
ax2.legend(fontsize=6.6, frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.01))
ax2.set_title('(b) Out-of-sample probes (Table VII, 9 runs)', fontsize=8)
ax2.spines[['top', 'right']].set_visible(False)

fig.tight_layout()
p_fig1 = os.path.join(OUT, 'fig3_threat_model.png')
fig.savefig(p_fig1, bbox_inches='tight')
plt.close(fig)
print('SAVED FIG1:', p_fig1)

# =====================================================================
# 图 3：LOO 留一矩阵热力图 -> fig1_loo_heatmap.png (E3/E4/E5 三次合并均值)
# =====================================================================
GENS = ['ADM', 'BigGAN', 'GLIDE', 'Midjourney', 'SD15', 'VQDM', 'Wukong']
COLS = GENS + ['Real']
ACC = np.array([[d['loo_matrix']['mean'][g][c] for c in COLS] for g in GENS])
FULL_BASE = [d['r3_baseline']['by_cls'][c]['mean'] for c in COLS]

fig, ax = plt.subplots(figsize=(6.0, 4.75))
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
        ax.text(j, i, '%.1f' % v, ha='center', va='center', fontsize=6.4,
                color=c, fontweight=w)

for k in range(7):
    ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False,
                               edgecolor='black', lw=1.4))

# 底部留白放全量训练基线行，不压虚线
ax.set_ylim(7.28, -0.5)
ax.axhline(6.62, color='#222222', lw=0.8, ls='--')
ax.text(-0.45, 6.84, 'full-training baseline (E3–E5)',
        fontsize=5.5, color='#222222', va='center')
for j in range(8):
    ax.text(j, 7.08, '%.1f' % FULL_BASE[j], ha='center', va='center',
            fontsize=5.8, color='#222222', style='italic')

cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.set_label('Accuracy (%)')
fig.tight_layout()
p_fig3 = os.path.join(OUT, 'fig1_loo_heatmap.png')
fig.savefig(p_fig3, bbox_inches='tight')
plt.close(fig)
print('SAVED FIG3:', p_fig3)
