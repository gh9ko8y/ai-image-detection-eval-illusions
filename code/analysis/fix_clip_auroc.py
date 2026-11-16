#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AUROC 审计修复版：模型构建直接 import /root 的真实模块（杜绝结构漂移），
仅重跑 CLIP 系模型并替换 result.csv 中对应行。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from bb_models import build_resnet18, build_efficientnet_b0, RN18_MEAN, RN18_STD
from bb_clip_models import build_clip, CLIP_MEAN, CLIP_STD

BASE = os.environ.get('DATA_ROOT', '<DATA_ROOT>')
DEV = 'cuda'
GENS = ['ADM', 'BigGAN', 'GLIDE', 'Midjourney', 'SD15', 'VQDM', 'Wukong']
CSVP = os.path.join(BASE, 'runs/auroc_audit/result.csv')


def to_t(x_u8, mean, std):
    x = x_u8.astype('float32') / 255.0
    x = x.transpose(0, 3, 1, 2)
    return torch.from_numpy(np.ascontiguousarray((x - mean) / std))


def main():
    X = np.load(os.path.join(BASE, 'data_npy/X_validation.npy'), mmap_mode='r')
    meta = pd.read_csv(os.path.join(BASE, 'data_npy/validation_meta_named.csv'))
    g = meta['gen_name'].values
    jobs = []
    for r in [1, 2, 3]:
        jobs.append(('clip_full_r%d' % r, 'runs/bb02/det_clip_vitb16_full_r%d.pth' % r))
    for r in [1, 2, 3, 4, 5]:
        jobs.append(('clip_holdout_MJ_r%d' % r,
                     'runs/bb02/det_clip_vitb16_holdout_Midjourney_r%d.pth' % r))
    rows = []
    for tag, rel in jobs:
        model = build_clip()
        sd = torch.load(os.path.join(BASE, rel), map_location=DEV,
                        weights_only=True)
        model.load_state_dict(sd)  # strict=True：结构必须完全匹配
        model.to(DEV).eval()
        scores, gs = [], []
        with torch.no_grad():
            for i in range(0, len(X), 200):
                xb = to_t(np.asarray(X[i:i + 200]), CLIP_MEAN, CLIP_STD).to(DEV)
                scores.append(F.softmax(model(xb), 1)[:, 1].cpu().numpy())
                gs.append(g[i:i + 200])
        s = np.concatenate(scores)
        gg = np.concatenate(gs)
        neg = s[gg == 'Real']
        for gen in GENS:
            pos = s[gg == gen]
            thr = np.quantile(neg, 0.99)
            rows.append((tag, gen, roc_auc_score(
                np.r_[np.ones_like(pos), np.zeros_like(neg)],
                np.r_[pos, neg]), float((pos >= thr).mean())))
        print('DONE', tag, flush=True)
    df_new = pd.DataFrame(rows, columns=['model', 'generator', 'auroc', 'tpr_at_fpr1'])
    df_old = pd.read_csv(CSVP)
    df = pd.concat([df_old[~df_old.model.isin(df_new.model)], df_new])
    df = df.sort_values(['model', 'generator']).reset_index(drop=True)
    df.to_csv(CSVP, index=False)
    print('AUROC_CLIP_FIX_DONE', len(df))


if __name__ == '__main__':
    main()
