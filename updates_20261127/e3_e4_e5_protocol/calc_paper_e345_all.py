# -*- coding: utf-8 -*-
"""全量论文数据复算脚本：E3 / E4 / E5 三次同协议执行均值与范围 + FLUX v1/v2。
只读本地原始 JSON/CSV/LOG，不修改任何原始文件。"""
import csv
import glob
import json
import os
import re
import statistics as st
from math import isnan

B = r'.'
P0 = os.path.join(B, 'P0_审计_20261118')
GENS = ['ADM', 'BigGAN', 'GLIDE', 'Midjourney', 'SD15', 'VQDM', 'Wukong']
COLS = GENS + ['Real']
GENS_T34 = ['VQDM', 'ADM', 'Midjourney', 'BigGAN', 'Wukong', 'SD15', 'GLIDE']
GENS_T9 = ['GLIDE', 'SD15', 'Wukong', 'BigGAN', 'ADM', 'VQDM', 'Midjourney']
RUNS = ['E3', 'E4', 'E5']

EVAL_DIR = {
    'E3': os.path.join(P0, 'provenance_small', 'rerun_20261116'),
    'E4': os.path.join(P0, 'E4', 'E4_eval'),
    'E5': os.path.join(P0, 'E5', 'E5_eval'),
}
SDXL_PATH = {
    'E3': os.path.join(P0, 'E4', 'sdxl_result_E3.json'),
    'E4': os.path.join(P0, 'E4', 'E4_eval', 'sdxl', 'result_v2.json'),
    'E5': os.path.join(P0, 'E5', 'E5_eval', 'sdxl', 'result_v2.json'),
}
AUROC_PATH = {
    'E3': os.path.join(B, 'github_repo_ai-image-detection-eval-illusions', 'data', 'results_json_csv', 'auroc_result.csv'),
    'E4': os.path.join(P0, 'E4', 'E4_auroc_audit', 'result.csv'),
    'E5': os.path.join(P0, 'E5', 'E5_auroc_audit', 'result_159.csv'),
}


def J(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def mean(xs):
    return sum(xs) / len(xs)


def rng(xs):
    return max(xs) - min(xs)


def fmt_mr(xs, dec=1):
    f = f'%.{dec}f'
    return f'{f % mean(xs)} ({f % min(xs)}–{f % max(xs)})'


def find_nested(root, name):
    hits = glob.glob(os.path.join(root, '**', name), recursive=True)
    assert hits, f'missing {name} in {root}'
    return hits[0]


res = {}

# =====================================================================
# 1. R3: ResNet-18 全量基线 + 低通三口径（E3/E4/E5 各 3 轮，共 9 轮）
# =====================================================================
r3 = {e: [J(os.path.join(EVAL_DIR[e], 'R3', f'r{k}', 'lowpass_v2.json')) for k in (1, 2, 3)] for e in RUNS}

rn_full_overall_by_e = {e: mean([x['clean_bare']['overall'] for x in r3[e]]) for e in RUNS}
rn_full_all9_overall = [x['clean_bare']['overall'] for e in RUNS for x in r3[e]]
rn_full_by_cls = {}
for c in COLS:
    by_e = [mean([x['clean_bare'][c] for x in r3[e]]) for e in RUNS]
    all9 = [x['clean_bare'][c] for e in RUNS for x in r3[e]]
    rn_full_by_cls[c] = {'by_e': by_e, 'mean': mean(by_e), 'all9_min': min(all9), 'all9_max': max(all9), 'e_rng': rng(by_e)}

lp = {}
for key, getter in [
    ('clean_filtered_overall', lambda x: x['clean_filtered']['overall']),
    ('clean_filtered_fake', lambda x: x['clean_filtered']['fake_all']),
    ('clean_filtered_real', lambda x: x['clean_filtered']['Real']),
    ('eps002_bare', lambda x: x['eps=0.020']['bare_attacked']['overall']),
    ('eps002_transfer', lambda x: x['eps=0.020']['transfer']['overall']),
    ('eps002_adaptive', lambda x: x['eps=0.020']['adaptive']['overall']),
    ('eps005_bare', lambda x: x['eps=0.050']['bare_attacked']['overall']),
    ('eps005_transfer', lambda x: x['eps=0.050']['transfer']['overall']),
    ('eps005_adaptive', lambda x: x['eps=0.050']['adaptive']['overall']),
    ('eps010_bare', lambda x: x['eps=0.100']['bare_attacked']['overall']),
    ('eps010_transfer', lambda x: x['eps=0.100']['transfer']['overall']),
    ('eps010_adaptive', lambda x: x['eps=0.100']['adaptive']['overall']),
]:
    by_e = [mean([getter(x) for x in r3[e]]) for e in RUNS]
    all9 = [getter(x) for e in RUNS for x in r3[e]]
    lp[key] = {'by_e': by_e, 'mean': mean(all9), 'min9': min(all9), 'max9': max(all9), 'rng9': rng(all9), 'e_min': min(by_e), 'e_max': max(by_e)}

res['r3_baseline'] = {'overall_by_e': rn_full_overall_by_e, 'overall_mean': mean(rn_full_all9_overall),
                      'overall_min9': min(rn_full_all9_overall), 'overall_max9': max(rn_full_all9_overall),
                      'by_cls': rn_full_by_cls}
res['r3_lowpass'] = lp

# =====================================================================
# 2. R4: 表 II 防御（移植 Paddle 检测器，E3/E4/E5 各 1 次，共 3 次）
# =====================================================================
r4 = {e: J(os.path.join(EVAL_DIR[e], 'R4', 'defenses_torch_port.json')) for e in RUNS}
t2 = {'clean_bare': [r4[e]['clean_bare'] for e in RUNS]}
for dfn in ('randomization', 'bit_reduce_0.2', 'jpeg_q50', 'lowpass_0.15'):
    t2[dfn] = {'no_attack': [r4[e][dfn]['all']['no_attack'] for e in RUNS]}
    for eps in ('eps=0.02', 'eps=0.05', 'eps=0.10'):
        t2[dfn][eps] = {m: [r4[e][dfn]['all'][eps][m] for e in RUNS] for m in ('transfer', 'naive_whitebox', 'adaptive')}
t2['jpeg_first2000'] = {'no_attack': [r4[e]['jpeg_q50']['first2000']['no_attack'] for e in RUNS]}
for eps in ('eps=0.02', 'eps=0.05', 'eps=0.10'):
    t2['jpeg_first2000'][eps] = {m: [r4[e]['jpeg_q50']['first2000'][eps][m] for e in RUNS] for m in ('transfer', 'naive_whitebox', 'adaptive')}
res['r4_table2'] = t2

# =====================================================================
# 3. LOO 矩阵（ResNet-18，图 3 + 表 III + 表 IV + 表 V）
#    E3 来自 make_paper_figs.py / 论文原矩阵（ADM/BigGAN 2轮均值，其余 3 轮均值）
#    E4/E5 来自本地 19 个 eval_*.json
# =====================================================================
E3_LOO = {
    'ADM':        dict(zip(COLS, [40.5, 98.7, 98.0, 78.1, 93.6, 87.9, 91.6, 85.1])),
    'BigGAN':     dict(zip(COLS, [98.6, 62.6, 97.6, 80.9, 94.1, 86.0, 91.9, 87.2])),
    'GLIDE':      dict(zip(COLS, [99.0, 97.1, 89.1, 69.0, 87.9, 83.3, 87.8, 92.2])),
    'Midjourney': dict(zip(COLS, [98.1, 97.1, 93.8, 14.9, 79.7, 81.1, 78.3, 96.6])),
    'SD15':       dict(zip(COLS, [98.9, 96.0, 98.1, 67.6, 73.7, 77.7, 79.5, 95.6])),
    'VQDM':       dict(zip(COLS, [99.4, 99.5, 98.2, 74.3, 90.4, 27.7, 89.5, 93.1])),
    'Wukong':     dict(zip(COLS, [98.6, 98.9, 97.6, 64.8, 73.3, 76.8, 61.8, 96.6])),
}

def load_loo_exec(exec_name):
    if exec_name == 'E3':
        return E3_LOO
    root = os.path.join(P0, f'{exec_name}_loo_01')
    mat = {}
    for g in GENS:
        ks = (1, 2) if g in ('ADM', 'BigGAN') else (1, 2, 3)
        items = [J(find_nested(root, f'eval_{g}_r{k}.json')) for k in ks]
        mat[g] = {c: mean([it['real_acc'] if c == 'Real' else it[c] for it in items]) for c in COLS}
    return mat

loo_by_e = {e: load_loo_exec(e) for e in RUNS}
loo_mean = {g: {c: mean([loo_by_e[e][g][c] for e in RUNS]) for c in COLS} for g in GENS}
loo_rng = {g: {c: rng([loo_by_e[e][g][c] for e in RUNS]) for c in COLS} for g in GENS}
loo_min = {g: {c: min([loo_by_e[e][g][c] for e in RUNS]) for c in COLS} for g in GENS}
loo_max = {g: {c: max([loo_by_e[e][g][c] for e in RUNS]) for c in COLS} for g in GENS}

# 行级加权总体准确率 (3500 Real + 7*500 Fake = 7000)
row_overall = {}
for g in GENS:
    row_overall[g] = (sum(loo_mean[g][c] * 500 for c in GENS) + loo_mean[g]['Real'] * 3500) / 7000.0
diag_vals = [loo_mean[g][g] for g in GENS]
diag_by_e = [mean([loo_by_e[e][g][g] for g in GENS]) for e in RUNS]
res['loo_matrix'] = {
    'by_e': loo_by_e, 'mean': loo_mean, 'min': loo_min, 'max': loo_max, 'rng': loo_rng,
    'row_overall': row_overall, 'diag_mean': mean(diag_vals), 'diag_by_e': diag_by_e,
}

# =====================================================================
# 4. AUROC 审计（159 行，E3/E4/E5 三次合并）
# =====================================================================
au = {}
for e in RUNS:
    with open(AUROC_PATH[e], encoding='utf-8') as f:
        au[e] = list(csv.DictReader(f))

def au_lookup(rows, model_prefix, gen):
    vals = [float(r['auroc']) for r in rows if r['model'].startswith(model_prefix) and r['generator'] == gen]
    tprs = [float(r['tpr_at_fpr1']) * 100.0 for r in rows if r['model'].startswith(model_prefix) and r['generator'] == gen]
    return vals, tprs

au_table4 = {}
for g in GENS:
    loo_by_e_au = []
    full_by_e_au = []
    loo_all_au = []
    for e in RUNS:
        lv, _ = au_lookup(au[e], f'loo_{g}_r', g)
        fv, _ = au_lookup(au[e], 'resnet_full_r', g)
        loo_by_e_au.append(mean(lv))
        full_by_e_au.append(mean(fv))
        loo_all_au.extend(lv)
    delta_acc = loo_mean[g][g] - rn_full_by_cls[g]['mean']
    au_table4[g] = {
        'acc': loo_mean[g][g],
        'acc_min': loo_min[g][g],
        'acc_max': loo_max[g][g],
        'full_acc': rn_full_by_cls[g]['mean'],
        'delta_acc': delta_acc,
        'loo_auroc': mean(loo_by_e_au),
        'loo_auroc_by_e': loo_by_e_au,
        'full_auroc': mean(full_by_e_au),
        'delta_auroc': mean(loo_by_e_au) - mean(full_by_e_au),
        'auroc_rng_all': rng(loo_all_au),
        'auroc_rng_e': rng(loo_by_e_au),
    }

# 全量检测器平均 TPR@1%FPR（三骨干 × 三次执行）
tpr1_full = {}
for fam, pref in [('resnet', 'resnet_full_r'), ('effnet', 'effnet_full_r'), ('clip', 'clip_full_r')]:
    by_e = []
    for e in RUNS:
        vals = [float(r['tpr_at_fpr1']) * 100.0 for r in au[e] if r['model'].startswith(pref)]
        by_e.append(mean(vals))
    tpr1_full[fam] = {'by_e': by_e, 'mean': mean(by_e)}

# 非对角线 Δacc, ΔAUROC (r1 across E3/E4/E5), ΔTPR@1% (r1 across E3/E4/E5)
offdiag = []
for g_out in GENS:
    for g_ev in GENS:
        if g_out == g_ev:
            continue
        d_acc = loo_mean[g_out][g_ev] - rn_full_by_cls[g_ev]['mean']
        d_au_e, d_tpr_e = [], []
        for e in RUNS:
            la, lt = au_lookup(au[e], f'loo_{g_out}_r1', g_ev)
            fa, ft = au_lookup(au[e], 'resnet_full_r1', g_ev)
            d_au_e.append(la[0] - fa[0])
            d_tpr_e.append(lt[0] - ft[0])
        offdiag.append({
            'pair': f'{g_out} → {g_ev}', 'd_acc': d_acc,
            'd_auroc': mean(d_au_e), 'd_tpr1': mean(d_tpr_e)
        })
offdiag.sort(key=lambda x: x['d_acc'])
res['auroc_t4'] = au_table4
res['tpr1_full'] = tpr1_full
res['offdiag_top10'] = offdiag[:10]
res['offdiag_auroc_range'] = (min(x['d_auroc'] for x in offdiag), max(x['d_auroc'] for x in offdiag))

# GLIDE TPR@1%FPR full vs LOO (ResNet)
glide_full_tpr = [mean(au_lookup(au[e], 'resnet_full_r', 'GLIDE')[1]) for e in RUNS]
glide_loo_tpr = [mean(au_lookup(au[e], 'loo_GLIDE_r', 'GLIDE')[1]) for e in RUNS]
res['glide_tpr1'] = {'full': mean(glide_full_tpr), 'loo': mean(glide_loo_tpr), 'delta': mean(glide_loo_tpr) - mean(glide_full_tpr)}

# CLIP MJ 5轮 AUROC across E3/E4/E5 (15 runs)
clip_mj_au_all = [float(r['auroc']) for e in RUNS for r in au[e] if r['model'].startswith('clip_holdout_MJ_r') and r['generator'] == 'Midjourney']
clip_full_mj_au = [float(r['auroc']) for e in RUNS for r in au[e] if r['model'].startswith('clip_full_r') and r['generator'] == 'Midjourney']
res['clip_mj_auroc'] = {'min': min(clip_mj_au_all), 'max': max(clip_mj_au_all), 'rng': rng(clip_mj_au_all), 'mean': mean(clip_mj_au_all), 'full_mean': mean(clip_full_mj_au)}

# Spearman rho (表 IV)
try:
    from scipy.stats import spearmanr
    dg = [au_table4[g]['delta_acc'] for g in GENS]
    la = [au_table4[g]['loo_auroc'] for g in GENS]
    ac = [au_table4[g]['acc'] for g in GENS]
    da = [au_table4[g]['delta_auroc'] for g in GENS]
    res['spearman'] = {
        'dg_vs_looauroc': tuple(float(x) for x in spearmanr(dg, la)),
        'acc_vs_looauroc': tuple(float(x) for x in spearmanr(ac, la)),
        'dg_vs_dauroc': tuple(float(x) for x in spearmanr(dg, da)),
    }
except Exception as ex:
    res['spearman'] = {'err': str(ex)}

# =====================================================================
# 5. R7: 仅真图阈值重校准（表 VI，E3/E4/E5 × run1/run2）
# =====================================================================
r7 = {e: {n: J(os.path.join(EVAL_DIR[e], 'R7', f'result_run{n}.json')) for n in (1, 2)} for e in RUNS}

def r7_summary(d):
    full = d['full']
    out = {'full_fpr': full['real_fpr'], 'full_ece': full['ece_all']}
    for mode in ('fixed_0.5', 'real_only_rethreshold'):
        off = []
        diag = []
        per_g = {}
        for g_out in GENS:
            blk = d['loo'][g_out][mode]
            for g_ev in GENS:
                diff = blk[g_ev] - full[g_ev]
                if g_ev == g_out:
                    diag.append(diff)
                    per_g[g_out] = diff
                else:
                    off.append(diff)
        out[mode] = {
            'off_abs_mean': mean([abs(x) for x in off]),
            'off_gt5': sum(1 for x in off if abs(x) > 5.0),
            'off_worst': min(off),
            'diag_mean': mean(diag),
            'per_g': per_g,
        }
    return out

r7_s = {e: {n: r7_summary(r7[e][n]) for n in (1, 2)} for e in RUNS}
res['r7_table6'] = r7_s

# =====================================================================
# 6. SDXL & FLUX v1 / v2（表 VII）
# =====================================================================
sdxl = {e: J(SDXL_PATH[e]) for e in RUNS}
probe_dir = os.path.join(P0, 'probe_non_sd')
flux_v1 = {
    'E3': J(os.path.join(probe_dir, 'flux_schnell_v1_eval.json')),
    'E4': J(os.path.join(probe_dir, 'flux_schnell_v1_eval_e4.json')),
    'E5': J(os.path.join(probe_dir, 'flux_schnell_v1_eval_e5.json')),
}
flux_v1_uni = J(os.path.join(probe_dir, 'flux_schnell_v1_eval_univfd.json'))
flux_v2 = {
    'E3': J(os.path.join(probe_dir, 'flux_schnell_eval.json')),
    'E4': J(os.path.join(probe_dir, 'flux_schnell_eval_e4.json')),
    'E5': J(os.path.join(probe_dir, 'flux_schnell_eval_e5.json')),
}
flux_v2_uni = J(os.path.join(probe_dir, 'flux_schnell_eval_univfd.json'))

def agg_probe(ds, bal_key='acc_balanced'):
    out = {}
    for m in ('resnet', 'effnet', 'clip'):
        out[m] = {}
        for fld in ('acc_fake', 'acc_real', bal_key, 'auroc', 'tpr1'):
            all9 = [ds[e][m][f'r{k}'][fld] for e in RUNS for k in (1, 2, 3)]
            by_e = [mean([ds[e][m][f'r{k}'][fld] for k in (1, 2, 3)]) for e in RUNS]
            out[m][fld] = {'mean': mean(all9), 'min9': min(all9), 'max9': max(all9), 'rng9': rng(all9),
                           'by_e': by_e, 'e_min': min(by_e), 'e_max': max(by_e)}
    return out

res['sdxl'] = agg_probe(sdxl, bal_key='acc')
res['flux_v1'] = agg_probe(flux_v1, bal_key='acc_balanced')
res['flux_v1']['univfd'] = flux_v1_uni
res['flux_v2'] = agg_probe(flux_v2, bal_key='acc_balanced')
res['flux_v2']['univfd'] = flux_v2_uni

# =====================================================================
# 7. 表 IX: EffNet-B0 与 CLIP ViT-B/16（E3/E4/E5 三次执行合并）
# =====================================================================
E3_EFF_FULL = [94.8, 95.3, 95.1]  # 论文锁定值，均值 95.07
E3_EFF_DIAG = {'GLIDE': 99.0, 'SD15': 87.1, 'Wukong': 82.1, 'BigGAN': 81.2, 'ADM': 37.7, 'VQDM': 22.9, 'Midjourney': 37.4}
E3_CLIP_FULL = [97.89, 97.73, 97.93]  # 均值 97.85
E3_CLIP_DIAG = {'GLIDE': 96.6, 'SD15': 96.2, 'Wukong': 80.2, 'BigGAN': 87.0, 'ADM': 44.8, 'VQDM': 42.2, 'Midjourney': 36.2}
E3_CLIP_MJ5 = [13.2, 63.2, 47.0, 40.0, 17.6]

eff_full_by_e = {'E3': E3_EFF_FULL}
eff_diag_by_e = {'E3': E3_EFF_DIAG}
eff_sd_wk = {}  # check SD15-Wukong coupling in E4/E5
for e in ('E4', 'E5'):
    root = os.path.join(P0, f'{e}_bb01')
    eff_full_by_e[e] = [J(find_nested(root, f'eval_efficientnet_b0_full_r{k}.json'))['overall'] for k in (1, 2, 3)]
    full_sd = mean([J(find_nested(root, f'eval_efficientnet_b0_full_r{k}.json'))['SD15'] for k in (1, 2, 3)])
    full_wk = mean([J(find_nested(root, f'eval_efficientnet_b0_full_r{k}.json'))['Wukong'] for k in (1, 2, 3)])
    eff_diag_by_e[e] = {}
    for g in GENS:
        vals = [J(find_nested(root, f'eval_efficientnet_b0_holdout_{g}_r{k}.json'))[g] for k in (1, 2, 3)]
        eff_diag_by_e[e][g] = mean(vals)
    wk_on_sd = mean([J(find_nested(root, f'eval_efficientnet_b0_holdout_Wukong_r{k}.json'))['SD15'] for k in (1, 2, 3)]) - full_sd
    sd_on_wk = mean([J(find_nested(root, f'eval_efficientnet_b0_holdout_SD15_r{k}.json'))['Wukong'] for k in (1, 2, 3)]) - full_wk
    eff_sd_wk[e] = (wk_on_sd, sd_on_wk)

clip_full_by_e = {'E3': E3_CLIP_FULL}
clip_diag_by_e = {'E3': E3_CLIP_DIAG}
clip_mj_all15 = list(E3_CLIP_MJ5)
for e in ('E4', 'E5'):
    root = os.path.join(P0, f'{e}_bb02')
    clip_full_by_e[e] = [J(find_nested(root, f'eval_clip_vitb16_full_r{k}.json'))['overall'] for k in (1, 2, 3)]
    clip_diag_by_e[e] = {}
    for g in GENS:
        if g == 'Midjourney':
            mj5 = [J(find_nested(root, f'eval_clip_vitb16_holdout_Midjourney_r{k}.json'))['Midjourney'] for k in (1, 2, 3, 4, 5)]
            clip_diag_by_e[e][g] = mean(mj5)
            clip_mj_all15.extend(mj5)
        else:
            clip_diag_by_e[e][g] = J(find_nested(root, f'eval_clip_vitb16_holdout_{g}_r1.json'))[g]

res['table9'] = {
    'eff_full_mean': mean([mean(eff_full_by_e[e]) for e in RUNS]),
    'eff_full_by_e': {e: mean(eff_full_by_e[e]) for e in RUNS},
    'eff_diag': {g: {'by_e': [eff_diag_by_e[e][g] for e in RUNS], 'mean': mean([eff_diag_by_e[e][g] for e in RUNS])} for g in GENS},
    'eff_sd_wk_e45': eff_sd_wk,
    'clip_full_mean': mean([mean(clip_full_by_e[e]) for e in RUNS]),
    'clip_full_by_e': {e: mean(clip_full_by_e[e]) for e in RUNS},
    'clip_diag': {g: {'by_e': [clip_diag_by_e[e][g] for e in RUNS], 'mean': mean([clip_diag_by_e[e][g] for e in RUNS])} for g in GENS},
    'clip_mj_all15': clip_mj_all15,
}

# =====================================================================
# 8. 表 X: GenImage 零样本（R6，E3/E4/E5 三次合并）
# =====================================================================
r6 = {e: J(os.path.join(EVAL_DIR[e], 'R6', 'result_v4.json')) for e in RUNS}
R6_GENS = ['ADM', 'BigGAN', 'GLIDE', 'Midjourney', 'SD1.4', 'VQDM', 'Wukong']
t10 = {}
for m in ('resnet', 'effnet', 'clip', 'univfd'):
    t10[m] = {'bal': {}, 'fake': {}, 'auroc': {}, 'real_by_e': []}
    for e in RUNS:
        t10[m]['real_by_e'].append(r6[e][m]['ADM']['acc_real'])
    t10[m]['real_mean'] = mean(t10[m]['real_by_e'])
    for g in R6_GENS:
        for fld, dst in [('acc_balanced', 'bal'), ('acc_fake', 'fake'), ('auroc', 'auroc')]:
            vals = [r6[e][m][g][fld] for e in RUNS]
            t10[m][dst][g] = {'by_e': vals, 'mean': mean(vals), 'min': min(vals), 'max': max(vals), 'rng': rng(vals)}
res['r6_table10'] = t10

# =====================================================================
# 9. 第 XII 节 历史 PGD-AT（E3 log_at_*.txt + E4/E5 eval_at_*.json）
# =====================================================================
def parse_e3_at(fn):
    txt = open(os.path.join(P0, 'provenance_small', 'runs', 'at01', fn), encoding='utf-8').read()
    blocks = txt.split('整体:')
    last = blocks[-1]
    ov = float(re.search(r'\s*([0-9.]+)%', last).group(1))
    d = {'overall': ov}
    for c in COLS:
        m = re.search(rf'{c}\s+\d+\s+([0-9.]+)', last)
        if m:
            d[c] = float(m.group(1))
    return d

at_data = {'E3': {}, 'E4': {}, 'E5': {}}
at_data['E3']['full'] = [parse_e3_at(f'log_at_resnet18_full_r{k}.txt')['overall'] for k in (1, 2, 3)]
for g in ('Midjourney', 'ADM', 'VQDM'):
    at_data['E3'][g] = parse_e3_at(f'log_at_resnet18_holdout_{g}_r1.txt')[g]
for e in ('E4', 'E5'):
    root = os.path.join(P0, f'{e}_at01')
    at_data[e]['full'] = [J(find_nested(root, f'eval_at_resnet18_r{k}.json'))['overall'] for k in (1, 2, 3)]
    for g in ('Midjourney', 'ADM', 'VQDM'):
        at_data[e][g] = J(find_nested(root, f'eval_at_resnet18_holdout_{g}_r1.json'))[g]
res['hist_at'] = at_data

# =====================================================================
# 保存完整 JSON 并打印人眼可读汇总
# =====================================================================
out_json = os.path.join(P0, 'E5', 'paper_e345_master_stats.json')
with open(out_json, 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=2)

print('=== 1. ResNet-18 干净基线 (E3/E4/E5 共9轮) ===')
print('  overall by E:', {k: round(v, 2) for k, v in rn_full_overall_by_e.items()},
      f'mean={mean(rn_full_all9_overall):.2f} range=[{min(rn_full_all9_overall):.2f}, {max(rn_full_all9_overall):.2f}]')
for c in COLS:
    x = rn_full_by_cls[c]
    print(f'  {c:10s}: mean={x["mean"]:.2f}  E3/4/5={[round(v,2) for v in x["by_e"]]}')

print('\n=== 2. 低通防御 (R3, 9轮) ===')
for k in ['clean_filtered_overall','clean_filtered_fake','clean_filtered_real',
          'eps002_bare','eps002_transfer','eps002_adaptive',
          'eps005_bare','eps005_transfer','eps005_adaptive',
          'eps010_bare','eps010_transfer','eps010_adaptive']:
    x = lp[k]
    print(f'  {k:24s}: mean={x["mean"]:6.2f}  [9轮 {x["min9"]:.2f}–{x["max9"]:.2f}]  E={[round(v,2) for v in x["by_e"]]}')

print('\n=== 3. 表 II (R4, 3次) ===')
print(f'  clean_bare: {fmt_mr(t2["clean_bare"], 2)}')
for dfn in ('randomization', 'bit_reduce_0.2', 'jpeg_q50', 'jpeg_first2000', 'lowpass_0.15'):
    na = t2[dfn]['no_attack']
    tr02, tr05 = t2[dfn]['eps=0.02']['transfer'], t2[dfn]['eps=0.05']['transfer']
    ad02, ad05 = t2[dfn]['eps=0.02']['adaptive'], t2[dfn]['eps=0.05']['adaptive']
    print(f'  {dfn:16s} | no_atk={mean(na):.2f} ({min(na):.2f}–{max(na):.2f}) | tr={mean(tr02):.2f}/{mean(tr05):.2f} | ad={mean(ad02):.2f}/{mean(ad05):.2f}')

print('\n=== 4. LOO 矩阵 (图3 / 表III / 表IV) ===')
print(f'  diag_mean={mean(diag_vals):.2f} (by E={[round(v,2) for v in diag_by_e]})')
print('  row_overall:', {k: round(v, 2) for k, v in row_overall.items()},
      f'range=[{min(row_overall.values()):.1f}–{max(row_overall.values()):.1f}]')
print('  表 III & 表 IV (顺序: VQDM/ADM/MJ/BigGAN/Wukong/SD15/GLIDE):')
for g in GENS_T34:
    t = au_table4[g]
    print(f'    {g:10s}: acc={t["acc"]:5.1f} ({t["acc_min"]:.1f}–{t["acc_max"]:.1f})  fake->real={100-t["acc"]:5.1f}  real={loo_mean[g]["Real"]:5.1f}  dAcc={t["delta_acc"]:+5.1f}  looAU={t["loo_auroc"]:.3f}  fullAU={t["full_auroc"]:.3f}  dAU={t["delta_auroc"]:+.3f}  auRngAll={t["auroc_rng_all"]:.3f}')
print('  Spearman:', res['spearman'])
print('  全量平均 TPR@1%:', {k: round(v['mean'], 1) for k, v in tpr1_full.items()})
print('  GLIDE TPR@1% (ResNet):', {k: round(v, 1) for k, v in res['glide_tpr1'].items()})
print('  非对角线 Top 10 Δacc:')
for x in offdiag[:10]:
    print(f'    {x["pair"]:24s} dAcc={x["d_acc"]:+6.1f}  dAU={x["d_auroc"]:+.3f}  dTPR1={x["d_tpr1"]:+5.1f}')
print('  全部42非对角 ΔAUROC 范围:', tuple(round(v, 3) for v in res['offdiag_auroc_range']))

print('\n=== 5. 表 VI (R7, 3次执行 × run1/run2) ===')
for n in (1, 2):
    fprs = [r7_s[e][n]['full_fpr'] for e in RUNS]
    print(f'  run{n} full_fpr={fmt_mr(fprs, 2)}')
    for mode in ('fixed_0.5', 'real_only_rethreshold'):
        am = [r7_s[e][n][mode]['off_abs_mean'] for e in RUNS]
        gt = [r7_s[e][n][mode]['off_gt5'] for e in RUNS]
        wr = [r7_s[e][n][mode]['off_worst'] for e in RUNS]
        dm = [r7_s[e][n][mode]['diag_mean'] for e in RUNS]
        print(f'    {mode:22s}: |Δ|={fmt_mr(am,2)}  >5={gt} (mean {mean(gt):.1f})  worst={fmt_mr(wr,1)}  diag={fmt_mr(dm,1)}')
        for g in ('SD15', 'Wukong', 'Midjourney', 'VQDM', 'ADM'):
            gv = [r7_s[e][n][mode]['per_g'][g] for e in RUNS]
            print(f'      {g:10s}: {fmt_mr(gv, 1)}')

print('\n=== 6. 表 VII (SDXL + FLUX v1 + FLUX v2, 各9组) ===')
for name, ds, bk in [('SDXL', res['sdxl'], 'acc'), ('FLUX-v1', res['flux_v1'], 'acc_balanced'), ('FLUX-v2', res['flux_v2'], 'acc_balanced')]:
    print(f'  [{name}]')
    for m in ('resnet', 'effnet', 'clip'):
        af = ds[m]['acc_fake']; ar = ds[m]['acc_real']; ba = ds[m][bk]; au_ = ds[m]['auroc']; tp = ds[m]['tpr1']
        print(f'    {m:7s}: fake={af["mean"]:.1f} ({af["min9"]:.1f}–{af["max9"]:.1f}, rng {af["rng9"]:.1f}) | real={ar["mean"]:.1f} | bal={ba["mean"]:.1f} | AUROC={au_["mean"]:.1f} (rng {au_["rng9"]:.1f}) | TPR1={tp["mean"]:.1f}')
    if 'univfd' in ds:
        u = ds['univfd']
        print(f'    univfd : fake={u["acc_fake"]:.1f} | real={u["acc_real"]:.1f} | bal={u["acc_balanced"]:.1f} | AUROC={u["auroc"]:.1f} | TPR1={u["tpr1"]:.1f}')

print('\n=== 7. 表 IX (三骨干 LOO 对角线, E3/E4/E5) ===')
print(f'  Full baselines: RN={mean(rn_full_all9_overall):.2f}  Eff={res["table9"]["eff_full_mean"]:.2f}  CLIP={res["table9"]["clip_full_mean"]:.2f}')
rn_d_list, ef_d_list, cl_d_list = [], [], []
for g in GENS_T9:
    rn_v = [loo_by_e[e][g][g] for e in RUNS]
    ef_v = res['table9']['eff_diag'][g]['by_e']
    cl_v = res['table9']['clip_diag'][g]['by_e']
    rn_d_list.append(mean(rn_v)); ef_d_list.append(mean(ef_v)); cl_d_list.append(mean(cl_v))
    print(f'  {g:10s} | RN={fmt_mr(rn_v,1):18s} | Eff={fmt_mr(ef_v,1):18s} | CLIP={fmt_mr(cl_v,1):18s}')
print(f'  对角均值   | RN={mean(rn_d_list):.1f} | Eff={mean(ef_d_list):.1f} | CLIP={mean(cl_d_list):.1f}')
print(f'  CLIP MJ 15轮: mean={mean(clip_mj_all15):.1f}, range=[{min(clip_mj_all15):.1f}, {max(clip_mj_all15):.1f}], rng={rng(clip_mj_all15):.1f}')
print(f'  CLIP MJ AUROC 15轮: {res["clip_mj_auroc"]}')
print(f'  EffNet SD15-Wukong coupling (E4/E5): {eff_sd_wk}')

print('\n=== 8. 表 X (GenImage, E3/E4/E5) ===')
for m in ('resnet', 'effnet', 'clip', 'univfd'):
    bals = [t10[m]['bal'][g]['mean'] for g in R6_GENS]
    faks = [t10[m]['fake'][g]['mean'] for g in R6_GENS]
    aus = [t10[m]['auroc'][g]['mean'] for g in R6_GENS]
    print(f'  {m:7s} real={t10[m]["real_mean"]:.1f} | bal={[round(x,1) for x in bals]} (range {min(bals):.1f}–{max(bals):.1f})')
    print(f'          fake={[round(x,1) for x in faks]} (range {min(faks):.1f}–{max(faks):.1f}) | AUROC range {min(aus):.1f}–{max(aus):.1f}')

print('\n=== 9. 第 XII 节 历史 PGD-AT (E3/E4/E5) ===')
at_full_all9 = [v for e in RUNS for v in at_data[e]['full']]
print(f'  full (9轮): mean={mean(at_full_all9):.2f}  range=[{min(at_full_all9):.2f}, {max(at_full_all9):.2f}]  byE={[round(mean(at_data[e]["full"]),2) for e in RUNS]}')
for g in ('Midjourney', 'ADM', 'VQDM'):
    vs = [at_data[e][g] for e in RUNS]
    print(f'  holdout {g:10s} (3次): {fmt_mr(vs, 1)}  raw={vs}  vs clean_LOO={loo_mean[g][g]:.1f} (diff {mean(vs)-loo_mean[g][g]:+.1f})')
print('\nSAVED:', out_json)
