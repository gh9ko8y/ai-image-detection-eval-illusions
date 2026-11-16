### EVAL_v1 ### strip marker lines
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from PIL import Image

import bb_models
import bb_clip_models

BASE = os.environ.get('DATA_ROOT', '<DATA_ROOT>')
GENDIR = os.path.join(os.environ.get('DATA_ROOT', '<DATA_ROOT>'), 'runs/sdxl_zeroshot/gen')
OUTJSON = os.path.join(os.environ.get('DATA_ROOT', '<DATA_ROOT>'), 'runs/sdxl_zeroshot/result.json')
DEV = 'cuda'

X_real_all = np.load(os.path.join(BASE, 'data_npy/X_validation.npy'),
                     mmap_mode='r')
meta = pd.read_csv(os.path.join(BASE, 'data_npy',
                                'validation_meta_named.csv'))
real_idx = np.where(meta['gen_name'].values == 'Real')[0][:3500]
X_real = np.asarray(X_real_all[real_idx])
fakes = sorted([f for f in os.listdir(GENDIR) if f.endswith('.png')])
print('fakes:', len(fakes), 'reals:', len(X_real), flush=True)
X_fake = np.stack([
    np.asarray(Image.open(os.path.join(GENDIR, f)).convert('RGB')
               .resize((224, 224), Image.BICUBIC), dtype=np.uint8)
    for f in fakes])


def to_norm(x_u8, mean, std):
    x = x_u8.astype('float32') / 255.0
    x = x.transpose(0, 3, 1, 2)
    return torch.from_numpy(np.ascontiguousarray((x - mean) / std))


result = {}
for arch in ['resnet', 'effnet', 'clip']:
    if arch == 'resnet':
        model = bb_models.build_resnet18()
        mean, std = bb_models.RN18_MEAN, bb_models.RN18_STD
        ckpts = ['runs/round_01/detector.pth',
                 'runs/round_02/detector.pth',
                 'runs/round_03/detector.pth']
    elif arch == 'effnet':
        model = bb_models.build_efficientnet_b0()
        mean, std = bb_models.RN18_MEAN, bb_models.RN18_STD
        ckpts = ['runs/bb01/det_efficientnet_b0_full_r1.pth',
                 'runs/bb01/det_efficientnet_b0_full_r2.pth',
                 'runs/bb01/det_efficientnet_b0_full_r3.pth']
    else:
        model = bb_clip_models.build_clip()
        mean, std = bb_clip_models.CLIP_MEAN, bb_clip_models.CLIP_STD
        ckpts = ['runs/bb02/det_clip_vitb16_full_r1.pth',
                 'runs/bb02/det_clip_vitb16_full_r2.pth',
                 'runs/bb02/det_clip_vitb16_full_r3.pth']
    model = model.to(DEV).eval()
    out = {}
    for ri, rel in enumerate(ckpts, 1):
        sd = torch.load(os.path.join(BASE, rel), map_location=DEV,
                        weights_only=True)
        model.load_state_dict(sd)
        s_fake = []
        s_real = []
        with torch.no_grad():
            for i in range(0, len(X_fake), 200):
                xb = to_norm(X_fake[i:i + 200], mean, std).to(DEV)
                s_fake.append(F.softmax(model(xb), 1)[:, 1].cpu().numpy())
            for i in range(0, len(X_real), 200):
                xb = to_norm(X_real[i:i + 200], mean, std).to(DEV)
                s_real.append(F.softmax(model(xb), 1)[:, 1].cpu().numpy())
        s_fake = np.concatenate(s_fake)
        s_real = np.concatenate(s_real)
        acc = float(((s_fake >= 0.5).mean() * 100
                     + (s_real < 0.5).mean() * 100) / 2)
        auroc = float(roc_auc_score(
            np.r_[np.ones(len(s_fake)), np.zeros(len(s_real))],
            np.r_[s_fake, s_real]) * 100)
        thr = np.quantile(s_real, 0.99)
        out['r%d' % ri] = {
            'acc_fake': float((s_fake >= 0.5).mean() * 100),
            'acc_real': float((s_real < 0.5).mean() * 100),
            'acc': acc,
            'auroc': auroc,
            'tpr1': float((s_fake >= thr).mean() * 100)}
        print(arch, 'r%d' % ri, out['r%d' % ri], flush=True)
    result[arch] = out
with open(OUTJSON, 'w') as fh:
    json.dump(result, fh, indent=1)
print('SDXL_EVAL_DONE', flush=True)
