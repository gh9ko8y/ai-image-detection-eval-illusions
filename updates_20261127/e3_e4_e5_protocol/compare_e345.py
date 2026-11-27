# -*- coding: utf-8 -*-
"""E3/E4/E5 同协议三次执行逐格对照（只读本地 JSON/CSV，不改任何原始文件）。"""
import csv
import json
import os
import statistics as st
import sys

B = r'.'
P0 = os.path.join(B, 'P0_审计_20261118')
SRC = {
    'E3': {
        'eval': os.path.join(P0, 'provenance_small', 'rerun_20261116'),
        'sdxl': os.path.join(P0, 'E4', 'sdxl_result_E3.json'),
        'auroc': os.path.join(B, 'github_repo_ai-image-detection-eval-illusions', 'data', 'results_json_csv', 'auroc_result.csv'),
    },
    'E4': {
        'eval': os.path.join(P0, 'E4', 'E4_eval'),
        'sdxl': os.path.join(P0, 'E4', 'E4_eval', 'sdxl', 'result_v2.json'),
        'auroc': os.path.join(P0, 'E4', 'E4_auroc_audit', 'result.csv'),
    },
    'E5': {
        'eval': os.path.join(P0, 'E5', 'E5_eval'),
        'sdxl': os.path.join(P0, 'E5', 'E5_eval', 'sdxl', 'result_v2.json'),
        'auroc': os.path.join(P0, 'E5', 'E5_auroc_audit', 'result_159.csv'),
    },
}
RUNS = ['E3', 'E4', 'E5']
GENS = ['ADM', 'BigGAN', 'GLIDE', 'Midjourney', 'SD15', 'VQDM', 'Wukong']
out = []


def J(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def row(label, vals, flag=5.0):
    v = [x for x in vals if x is not None]
    mean = st.mean(v) if v else None
    rng = (max(v) - min(v)) if len(v) > 1 else None
    cells = ['%.2f' % x if x is not None else '—' for x in vals]
    mark = ' ⚠' if (rng is not None and rng >= flag) else ''
    out.append('| %s | %s | %s | %s%s |' % (
        label, ' | '.join(cells),
        '%.2f' % mean if mean is not None else '—',
        '%.2f' % rng if rng is not None else '—', mark))


def head(title, note=None):
    out.append('')
    out.append('### ' + title)
    if note:
        out.append('')
        out.append(note)
    out.append('')
    out.append('| 格子 | E3 | E4 | E5 | 三次均值 | 极差 |')
    out.append('|---|---|---|---|---|---|')


# ---------- R3 干净基线 + 低通 ----------
r3 = {r: {k: J(os.path.join(SRC[r]['eval'], 'R3', 'r%d' % k, 'lowpass_v2.json')) for k in (1, 2, 3)} for r in RUNS}
head('1. 干净基线（R3 clean_bare，ResNet-18 全量三轮）')
for k in (1, 2, 3):
    for fld, nm in (('overall', '总体'), ('Real', '真实类'), ('fake_all', '假图')):
        row('r%d %s' % (k, nm), [r3[r][k]['clean_bare'][fld] for r in RUNS])
head('2. 低通防御（R3 overall，三口径）')
for eps in ('eps=0.020', 'eps=0.050', 'eps=0.100'):
    for k in (1, 2, 3):
        for mode, nm in (('bare_attacked', '白盒裸'), ('transfer', '迁移'), ('adaptive', '自适应')):
            vals = []
            for r in RUNS:
                blk = r3[r][k].get(eps, {})
                vals.append(blk.get(mode, {}).get('overall'))
            row('%s r%d %s' % (eps, k, nm), vals, flag=5.0)

# ---------- R7 ----------
r7 = {r: {n: J(os.path.join(SRC[r]['eval'], 'R7', 'result_run%d.json' % n)) for n in (1, 2)} for r in RUNS}
head('3. R7 样本内对照（固定 0.5 阈值）', '留出行 = 被留出生成器自身的检出率（held-out TPR）。')
for n in (1, 2):
    row('run%d full 真图 FPR' % n, [r7[r][n]['full']['real_fpr'] for r in RUNS])
    for g in GENS:
        row('run%d 留出 %s 检出率' % (n, g), [r7[r][n]['loo'].get(g, {}).get('fixed_0.5', {}).get(g) for r in RUNS], flag=10.0)
    for g in GENS:
        row('run%d 留出 %s 真图 FPR' % (n, g), [r7[r][n]['loo'].get(g, {}).get('fixed_0.5', {}).get('real_fpr') for r in RUNS])

# ---------- R2 UnivFD ----------
head('4. UnivFD（R2）')
for fn in ('eval_v2_eps005_r3_real700', 'eval_v2_eps005_r1_real100', 'eval_v2_eps002_r1_real700'):
    d = {r: J(os.path.join(SRC[r]['eval'], 'R2', fn + '.json')) for r in RUNS}
    for blk, fld in (('clean', 'fake_acc'), ('clean', 'real_acc'), ('whitebox_pgd', 'overall_acc'),
                     ('lowpass_transfer', 'overall_acc'), ('lowpass_adaptive', 'overall_acc')):
        row('%s %s.%s' % (fn.replace('eval_v2_', ''), blk, fld), [d[r].get(blk, {}).get(fld) for r in RUNS])

# ---------- R4 表 II ----------
r4 = {r: J(os.path.join(SRC[r]['eval'], 'R4', 'defenses_torch_port.json')) for r in RUNS}
head('5. 表 II 防御（R4，全量 7000）')
row('clean_bare', [r4[r].get('clean_bare') for r in RUNS])
for dfn in ('randomization', 'bit_reduce_0.2', 'jpeg_q50', 'lowpass_0.15'):
    row('%s 无攻击' % dfn, [r4[r][dfn]['all']['no_attack'] for r in RUNS])
    for eps in ('eps=0.02', 'eps=0.05', 'eps=0.10'):
        for mode in ('transfer', 'adaptive'):
            row('%s %s %s' % (dfn, eps, mode), [r4[r][dfn]['all'][eps][mode] for r in RUNS])

# ---------- SDXL ----------
sx = {r: J(SRC[r]['sdxl']) for r in RUNS}
head('6. SDXL 零样本（500 假 + 3500 真）')
for m in ('resnet', 'effnet', 'clip'):
    for k in ('r1', 'r2', 'r3'):
        for fld, nm in (('acc_fake', '假图'), ('acc_real', '真图'), ('auroc', 'AUROC')):
            row('%s %s %s' % (m, k, nm), [sx[r][m][k][fld] for r in RUNS], flag=10.0)

# ---------- R6 GenImage ----------
r6 = {r: J(os.path.join(SRC[r]['eval'], 'R6', 'result_v4.json')) for r in RUNS}
head('7. GenImage 零样本（R6，七生成器简单平均）')
for m in ('resnet', 'effnet', 'clip', 'univfd'):
    for fld, nm in (('acc_fake', '假图'), ('acc_balanced', '平衡准确率'), ('auroc', 'AUROC')):
        vals = []
        for r in RUNS:
            gs = [v for k, v in r6[r][m].items() if isinstance(v, dict) and fld in v]
            vals.append(st.mean(float(v[fld]) for v in gs) if gs else None)
        row('%s %s 七类均值' % (m, nm), vals)

# ---------- AUROC 审计 ----------
head('8. AUROC 审计（族内均值 / 最小值，×100）')
au = {}
for r in RUNS:
    with open(SRC[r]['auroc'], encoding='utf-8') as f:
        au[r] = list(csv.DictReader(f))
for fam in ('resnet', 'effnet', 'clip', 'loo'):
    for agg, fn in (('均值', st.mean), ('最小值', min)):
        vals = []
        for r in RUNS:
            a = [float(x['auroc']) * 100 for x in au[r] if x['model'].split('_')[0] == fam]
            vals.append(fn(a) if a else None)
        row('%s %s' % (fam, agg), vals)
out.append('')
out.append('行数：' + ' / '.join('%s=%d' % (r, len(au[r])) for r in RUNS))

text = '\n'.join(out)
dst = sys.argv[1] if len(sys.argv) > 1 else None
if dst:
    with open(dst, 'w', encoding='utf-8') as f:
        f.write(text)
print(text)
