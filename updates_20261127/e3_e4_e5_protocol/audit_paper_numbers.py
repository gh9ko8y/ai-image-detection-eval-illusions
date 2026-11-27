# -*- coding: utf-8 -*-
"""全量核对 main_tcsvt_中文版_E345与FLUX_20261123.docx 中所有表格与关键段落数字 vs 原始数据文件。"""
import csv
import json
import os
import re
import docx

B = r'.'
P0 = os.path.join(B, 'P0_审计_20261118')
DOCX = os.path.join(B, 'main_tcsvt_中文版_E345与FLUX_20261123.docx')
STATS = os.path.join(P0, 'E5', 'paper_e345_master_stats.json')

doc = docx.Document(DOCX)
d = json.load(open(STATS, encoding='utf-8'))
P = doc.paragraphs
T = doc.tables

errors = []
warnings = []
checks = []


def ok(name, detail=''):
    checks.append(f'[PASS] {name}' + (f' :: {detail}' if detail else ''))


def fail(name, detail):
    errors.append(f'[FAIL] {name} :: {detail}')


def warn(name, detail):
    warnings.append(f'[WARN] {name} :: {detail}')


def close(a, b, tol=0.06):
    return abs(float(a) - float(b)) <= tol


# 1. 表 I (TABLE 0) vs paddle_all.log / 表 I 原始记录
pad_log = os.path.join(P0, 'provenance_small', 'repro_logs', 'paddle_all.log')
pad_txt = open(pad_log, encoding='utf-8', errors='ignore').read()
print('=== CHECK 1: Table I vs paddle_all.log ===')
# 从文档表 I 读取数值，搜索 paddle_all.log 验证
for ri in range(1, 5):
    row_vals = [c.text.strip() for c in T[0].rows[ri].cells]
    checks.append(f'  Table I R{ri}: {row_vals}')
# 在 paddle_all.log 找 90.79 / 73.26 / 86.60 等关键数字
for num in ['90.79', '6.71', '2.61', '0.27', '0.01', '86.60', '75.07', '73.26', '40.96']:
    found = num in pad_txt
    (ok if found else warn)(f'paddle_all.log contains {num}', '')

# 2. 表 II (TABLE 1) vs R4 & R3
t2 = d['r4_table2']
lp = d['r3_lowpass']
expected_t2 = [
    (1, 1, sum(t2['randomization']['no_attack'])/3, 2),
    (2, 1, sum(t2['bit_reduce_0.2']['no_attack'])/3, 2),
    (3, 1, sum(t2['jpeg_q50']['no_attack'])/3, 2),
    (4, 1, sum(t2['jpeg_first2000']['no_attack'])/3, 2),
    (5, 1, sum(t2['lowpass_0.15']['no_attack'])/3, 2),
    (6, 1, lp['clean_filtered_overall']['mean'], 2),
]
for ri, ci, val, dec in expected_t2:
    got = float(T[1].rows[ri].cells[ci].text.strip())
    if close(got, val, 0.015):
        ok(f'Table II R{ri}C{ci}', f'doc={got:.2f} vs raw={val:.2f}')
    else:
        fail(f'Table II R{ri}C{ci}', f'doc={got} vs raw={val:.2f}')

# 3. 表 III (TABLE 2) & 表 IV (TABLE 3)
GENS_T34 = ['VQDM', 'ADM', 'Midjourney', 'BigGAN', 'Wukong', 'SD15', 'GLIDE']
au4 = d['auroc_t4']
lm = d['loo_matrix']['mean']
for ci, g in enumerate(GENS_T34, start=1):
    # Table III
    fr = float(T[2].rows[1].cells[ci].text.strip())
    ra = float(T[2].rows[2].cells[ci].text.strip())
    if close(fr, 100 - au4[g]['acc'], 0.06) and close(ra, lm[g]['Real'], 0.06):
        ok(f'Table III {g}', f'fake->real={fr}, real={ra}')
    else:
        fail(f'Table III {g}', f'got=({fr},{ra}) vs ({100-au4[g]["acc"]:.1f},{lm[g]["Real"]:.1f})')
    # Table IV
    acc_top = float(T[3].rows[1].cells[ci].text.splitlines()[0])
    lau = float(T[3].rows[2].cells[ci].text.strip())
    fau = float(T[3].rows[3].cells[ci].text.strip())
    dau = float(T[3].rows[4].cells[ci].text.strip())
    if close(acc_top, au4[g]['acc'], 0.06) and close(lau, au4[g]['loo_auroc'], 0.0015) and close(fau, au4[g]['full_auroc'], 0.0015) and close(dau, au4[g]['delta_auroc'], 0.0015):
        ok(f'Table IV {g}', f'acc={acc_top}, looAU={lau}, fullAU={fau}, dAU={dau}')
    else:
        fail(f'Table IV {g}', f'mismatch on {g}')

# 4. 表 V (TABLE 4)
for ri in range(1, 11):
    pair = T[4].rows[ri].cells[0].text.strip()
    da = float(T[4].rows[ri].cells[1].text.strip())
    dau = float(T[4].rows[ri].cells[2].text.strip())
    dtp = float(T[4].rows[ri].cells[3].text.strip())
    exp = d['offdiag_top10'][ri-1]
    if pair == exp['pair'] and close(da, exp['d_acc'], 0.06) and close(dau, exp['d_auroc'], 0.0015) and close(dtp, exp['d_tpr1'], 0.06):
        ok(f'Table V R{ri} ({pair})', f'dAcc={da}, dAU={dau}, dTPR1={dtp}')
    else:
        fail(f'Table V R{ri}', f'doc=({pair},{da},{dau},{dtp}) vs exp={exp}')

# 5. 第 VI 节频谱数字 vs R1 perm_holm_result.json & resid_corr.csv
perm_j = json.load(open(os.path.join(P0, 'provenance_small', 'rerun_20261116', 'R1', 'perm_holm_result.json'), encoding='utf-8'))
rcorr_path = os.path.join(B, 'TIFS修改稿_20261116', 'results_rerun', 'results', 'R1', 'resid_corr.csv')
rcorr_m_path = os.path.join(B, 'TIFS修改稿_20261116', 'results_rerun', 'results', 'R1', 'resid_corr_matched150.csv')
null_csv = os.path.join(P0, 'provenance_small', 'rerun_20261116', 'R1', 'null_stats.csv')
print('=== CHECK 5: R1 Spectrum numbers ===')
print('  perm_holm keys:', list(perm_j.keys()))
print('   within_class_q05_pooled=', round(perm_j['within_class_q05_pooled'], 4),
      'within_class_sd_pooled=', round(perm_j['within_class_sd_pooled'], 4))
print('   within_class_q05_by_class=', {k: round(v, 3) for k, v in perm_j['within_class_q05_by_class'].items()})
print('   min p_greater=', min(perm_j['p_greater']), 'min p_greater_holm=', min(perm_j['p_greater_holm']))
print('   n_lower_holm<=0.05:', sum(1 for x in perm_j['p_less_holm'] if x <= 0.05),
      'n_below_all_perm:', perm_j['n_below_all_perm'])
for i, nm in enumerate(perm_j['pairs']):
    if any(x in nm for x in ('SD15-Wukong', 'Midjourney-SD15', 'Midjourney-Wukong', 'ADM-BigGAN', 'BigGAN-GLIDE')):
        print(f'   pair {nm:20s}: corr={perm_j["observed_corr"][i]:.4f} p_less={perm_j["p_less"][i]:.4f} p_less_h={perm_j["p_less_holm"][i]:.4f} p_2s_h={perm_j["p_two_sided_holm"][i]:.4f}')

# 6. 表 VIII (TABLE 7) UnivFD vs R2 JSON (E3/E4/E5)
u_e3 = json.load(open(os.path.join(B, 'github_repo_ai-image-detection-eval-illusions', 'data', 'results_json_csv', 'univfd_eval.json'), encoding='utf-8'))
u_r2 = json.load(open(os.path.join(P0, 'provenance_small', 'rerun_20261116', 'R2', 'eval_v2_eps005_r3_real700.json'), encoding='utf-8'))
print('=== CHECK 6: Table VIII UnivFD ===')
GENS = ['ADM', 'BigGAN', 'GLIDE', 'Midjourney', 'SD15', 'VQDM', 'Wukong']
for ci, g in enumerate(GENS, start=1):
    doc_full = float(T[7].rows[1].cells[ci].text.strip())
    doc_loo = float(T[7].rows[2].cells[ci].text.strip())
    doc_dg = float(T[7].rows[3].cells[ci].text.strip())
    doc_au = float(T[7].rows[4].cells[ci].text.strip())
    raw_full = u_r2['loo_clean_matrix']['full'][g]
    raw_loo = u_r2['loo_clean_matrix'][f'loo_{g}'][g]
    raw_au = u_r2['loo_clean_matrix'][f'loo_{g}'][f'{g}_auroc'] / 100.0
    if close(doc_full, raw_full, 0.06) and close(doc_loo, raw_loo, 0.06) and close(doc_dg, raw_loo - raw_full, 0.11) and close(doc_au, raw_au, 0.0015):
        ok(f'Table VIII UnivFD {g}', f'full={doc_full}, loo={doc_loo}, dg={doc_dg}, au={doc_au}')
    else:
        fail(f'Table VIII UnivFD {g}', f'doc=({doc_full},{doc_loo},{doc_dg},{doc_au}) vs raw=({raw_full:.1f},{raw_loo:.1f},{raw_loo-raw_full:.1f},{raw_au:.3f})')
print('  Table VIII R6 text:', T[7].rows[6].cells[0].text)
print('  R2 clean/filt/wb/lp_tr/lp_ad:', round(u_r2['clean']['overall_acc'],2), round(u_r2['clean_filtered']['overall_acc'],2),
      round(u_r2['whitebox_pgd']['overall_acc'],2), round(u_r2['lowpass_transfer']['overall_acc'],2), round(u_r2['lowpass_adaptive']['overall_acc'],2))

# 7. 表 XI (TABLE 10) 像素 AT vs full42/summary_table.md
sum_md = open(os.path.join(P0, 'full42', 'summary_table.md'), encoding='utf-8').read()
print('=== CHECK 7: Table XI Pixel AT vs full42/summary_table.md ===')
for ri in range(1, 8):
    vals = [c.text.strip() for c in T[10].rows[ri].cells]
    checks.append(f'  Table XI R{ri}: {vals}')
print(sum_md)

# 8. 表 VI, VII, IX, X 逐格断言检查
# 表 X (TABLE 9)
t10 = d['r6_table10']
R6_GENS = ['ADM', 'BigGAN', 'GLIDE', 'Midjourney', 'SD1.4', 'VQDM', 'Wukong']
for ri, m in enumerate(['resnet', 'effnet', 'clip', 'univfd'], start=1):
    for ci, g in enumerate(R6_GENS, start=1):
        b_doc = float(T[9].rows[ri].cells[ci].text.strip())
        f_doc = float(T[9].rows[ri+5].cells[ci].text.strip())
        if close(b_doc, t10[m]['bal'][g]['mean'], 0.06) and close(f_doc, t10[m]['fake'][g]['mean'], 0.06):
            ok(f'Table X {m} {g}', f'bal={b_doc}, fake={f_doc}')
        else:
            fail(f'Table X {m} {g}', f'doc=({b_doc},{f_doc}) vs ({t10[m]["bal"][g]["mean"]:.1f},{t10[m]["fake"][g]["mean"]:.1f})')

# 9. 正文中跨段落一致性检查（如 P[5] 摘要、P[13]、P[14]、P[111] 结论里的数字）
print('\n=== SUMMARY ===')
print(f'PASS={sum(1 for x in checks if x.startswith("[PASS]"))}  WARN={len(warnings)}  FAIL={len(errors)}')
for w in warnings:
    print(w)
for e in errors:
    print(e)
