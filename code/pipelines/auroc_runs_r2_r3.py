### AUROC_R23_v1 ### strip marker lines
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import bb_models

BASE = os.environ.get('DATA_ROOT', '<DATA_ROOT>')
DEV = 'cuda'
GENS = ['ADM', 'BigGAN', 'GLIDE', 'Midjourney', 'SD15', 'VQDM', 'Wukong']
CSVP = BASE + '/runs/auroc_audit/result.csv'

X = np.load(BASE + '/data_npy/X_validation.npy', mmap_mode='r')
meta = pd.read_csv(BASE + '/data_npy/validation_meta_named.csv')
g = meta['gen_name'].values


def to_norm(x_u8, mean, std):
    x = x_u8.astype('float32') / 255.0
    x = x.transpose(0, 3, 1, 2)
    return torch.from_numpy(np.ascontiguousarray((x - mean) / std))


model = bb_models.build_resnet18().to(DEV).eval()
mean, std = bb_models.RN18_MEAN, bb_models.RN18_STD

jobs = []
for r in [2, 3]:
    full_ck = BASE + '/runs/round_%02d/detector.pth' % r
    if os.path.exists(full_ck):
        jobs.append(('resnet_full_r%d' % r,
                     'runs/round_%02d/detector.pth' % r, None))
    for gen in GENS:
        ck = BASE + '/runs/loo_01/detector_%s_r%d.pth' % (gen, r)
        if os.path.exists(ck):
            jobs.append(('loo_%s_r%d' % (gen, r),
                         'runs/loo_01/detector_%s_r%d.pth' % (gen, r), gen))
print('jobs:', len(jobs), flush=True)

rows = []
for tag, rel, held in jobs:
    sd = torch.load(os.path.join(BASE, rel), map_location=DEV,
                    weights_only=True)
    model.load_state_dict(sd)
    scores, gs = [], []
    with torch.no_grad():
        for i in range(0, len(X), 200):
            xb = to_norm(np.asarray(X[i:i + 200]), mean, std).to(DEV)
            scores.append(F.softmax(model(xb), 1)[:, 1].cpu().numpy())
            gs.append(g[i:i + 200])
    s = np.concatenate(scores)
    gg = np.concatenate(gs)
    neg = s[gg == 'Real']
    targets = GENS if held is None else [held]
    for gen in targets:
        pos = s[gg == gen]
        thr = np.quantile(neg, 0.99)
        rows.append((tag, gen, roc_auc_score(
            np.r_[np.ones(len(pos)), np.zeros(len(neg))],
            np.r_[pos, neg]), float((pos >= thr).mean())))
    print('DONE', tag, flush=True)

df_new = pd.DataFrame(rows, columns=['model', 'generator', 'auroc',
                                     'tpr_at_fpr1'])
df_old = pd.read_csv(CSVP)
df = pd.concat([df_old[~df_old.model.isin(df_new.model)], df_new])
df = df.sort_values(['model', 'generator']).reset_index(drop=True)
df.to_csv(CSVP, index=False)
print('AUROC_R23_DONE', len(df), flush=True)
