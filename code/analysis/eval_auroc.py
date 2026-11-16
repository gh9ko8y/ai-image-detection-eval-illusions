#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B批：阈值无关复算——逐生成器 AUROC + TPR@FPR=1%（外审 P0 项）。
模型清单：round×3(ResNet full) + loo 7列r1 + bb01 full×3(EffNet) + bb02 full×3 与 MJ×5(CLIP)。
输出：runs/auroc_audit/result.csv（model, generator, auroc, tpr@fpr1）"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

BASE = os.environ.get('DATA_ROOT', '<DATA_ROOT>')
DEV = 'cuda'
GENS = ['ADM', 'BigGAN', 'GLIDE', 'Midjourney', 'SD15', 'VQDM', 'Wukong']
RN_MEAN = np.array([0.485, 0.456, 0.406], dtype='float32').reshape(1, 3, 1, 1)
RN_STD = np.array([0.229, 0.224, 0.225], dtype='float32').reshape(1, 3, 1, 1)
CL_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype='float32').reshape(1, 3, 1, 1)
CL_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype='float32').reshape(1, 3, 1, 1)
OUT = os.path.join(BASE, 'runs/auroc_audit')
os.makedirs(OUT, exist_ok=True)


def to_t(x_u8, mean, std):
    x = x_u8.astype('float32') / 255.0
    x = x.transpose(0, 3, 1, 2)
    return torch.from_numpy(np.ascontiguousarray((x - mean) / std))


def build_resnet():
    import torchvision
    m = torchvision.models.resnet18(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, 2)
    return m


def build_effnet():
    import torchvision
    m = torchvision.models.efficientnet_b0(weights=None)
    m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, 2)
    return m


def build_clip():
    import open_clip
    m, _, _ = open_clip.create_model_and_transforms(
        'ViT-B-16', pretrained='openai', force_quick_gelu=True)

    class Wrap(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.b = backbone
            self.head = torch.nn.Linear(512, 2)

        def forward(self, x):
            return self.head(self.b.encode_image(x))
    return Wrap(m)


def auroc(scores_pos, scores_neg):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(np.r_[np.ones_like(scores_pos), np.zeros_like(scores_neg)],
                         np.r_[scores_pos, scores_neg])


def tpr_at_fpr(scores_pos, scores_neg, fpr_target=0.01):
    thr = np.quantile(scores_neg, 1 - fpr_target)
    return float((scores_pos >= thr).mean())


def evaluate(tag, ckpt, arch, mean, std, X, g, rows):
    model = {'resnet': build_resnet, 'effnet': build_effnet, 'clip': build_clip}[arch]()
    sd = torch.load(ckpt, map_location=DEV, weights_only=True)
    try:
        model.load_state_dict(sd)
    except RuntimeError:
        model.load_state_dict(sd, strict=False)
    model.to(DEV).eval()
    scores_all, g_all = [], []
    with torch.no_grad():
        for i in range(0, len(X), 200):
            xb = to_t(np.asarray(X[i:i + 200]), mean, std).to(DEV)
            p = F.softmax(model(xb), 1)[:, 1]
            scores_all.append(p.cpu().numpy())
            g_all.append(g[i:i + 200])
    s = np.concatenate(scores_all)
    g = np.concatenate(g_all)
    neg = s[g == 'Real']
    for gen in GENS:
        pos = s[g == gen]
        rows.append((tag, gen, auroc(pos, neg), tpr_at_fpr(pos, neg)))


def main():
    X = np.load(os.path.join(BASE, 'data_npy/X_validation.npy'), mmap_mode='r')
    meta = pd.read_csv(os.path.join(BASE, 'data_npy/validation_meta_named.csv'))
    g = meta['gen_name'].values
    rows = []
    jobs = []
    for r in [1, 2, 3]:
        jobs.append(('resnet_full_r%d' % r, 'runs/round_%02d/detector.pth' % r, 'resnet', RN_MEAN, RN_STD))
    for gen in GENS:
        jobs.append(('loo_%s_r1' % gen, 'runs/loo_01/detector_%s_r1.pth' % gen, 'resnet', RN_MEAN, RN_STD))
    for r in [1, 2, 3]:
        jobs.append(('effnet_full_r%d' % r, 'runs/bb01/det_efficientnet_b0_full_r%d.pth' % r, 'effnet', RN_MEAN, RN_STD))
    for r in [1, 2, 3]:
        jobs.append(('clip_full_r%d' % r, 'runs/bb02/det_clip_vitb16_full_r%d.pth' % r, 'clip', CL_MEAN, CL_STD))
    for r in [1, 2, 3, 4, 5]:
        jobs.append(('clip_holdout_MJ_r%d' % r, 'runs/bb02/det_clip_vitb16_holdout_Midjourney_r%d.pth' % r, 'clip', CL_MEAN, CL_STD))
    for tag, rel, arch, mean, std in jobs:
        ck = os.path.join(BASE, rel)
        if not os.path.exists(ck):
            print('MISS', tag)
            continue
        evaluate(tag, ck, arch, mean, std, X, g, rows)
        print('DONE', tag, flush=True)
    df = pd.DataFrame(rows, columns=['model', 'generator', 'auroc', 'tpr_at_fpr1'])
    df.to_csv(os.path.join(OUT, 'result.csv'), index=False)
    print('AUROC_AUDIT_DONE', len(df))


if __name__ == '__main__':
    main()
