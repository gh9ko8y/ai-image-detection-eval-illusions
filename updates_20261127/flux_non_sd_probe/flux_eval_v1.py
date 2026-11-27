import json, os, pickle, sys

sys.path.insert(0, '$DATA_ROOT/e4_code')
os.environ['HF_HOME'] = '/root/.cache/huggingface'
os.environ['HF_HUB_OFFLINE'] = '1'
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score

from bb_models import build_resnet18, build_efficientnet_b0, \
    RN18_MEAN, RN18_STD
from bb_clip_models import build_clip, CLIP_MEAN, CLIP_STD

BASE = '$DATA_ROOT'
GEN_DIR = BASE + '/runs/probe_non_sd/flux_schnell_500_v1'
DEV = 'cuda'
OUT_BASE = BASE + '/runs/probe_non_sd'

X_val = np.load(BASE + '/data_npy/X_validation.npy', mmap_mode='r')
meta = pd.read_csv(BASE + '/data_npy/validation_meta_named.csv')
g = meta['gen_name'].values
real_idx = np.where(g == 'Real')[0]

fake_imgs = np.zeros((500, 224, 224, 3), dtype=np.uint8)
for i in range(500):
    im = Image.open(GEN_DIR + ('/img_%05d.png' % i)).convert('RGB')
    fake_imgs[i] = np.asarray(im.resize((224, 224), Image.BICUBIC))
print('fake loaded', flush=True)


def to_t(x_u8, mean, std):
    x = torch.from_numpy(np.ascontiguousarray(x_u8)).float().div_(255.)
    x = x.permute(0, 3, 1, 2)
    return (x - mean) / std


def eval_model(tag, build, mean, std, ckpt):
    model = build()
    sd = torch.load(ckpt, map_location=DEV, weights_only=True)
    model.load_state_dict(sd)
    model = model.to(DEV).eval()
    sf, sr = [], []
    with torch.no_grad():
        for i in range(0, 500, 250):
            xb = to_t(fake_imgs[i:i + 250], mean, std).to(DEV)
            sf.append(F.softmax(model(xb), 1)[:, 1].cpu().numpy())
        real_np = np.asarray(X_val[real_idx])
        for i in range(0, len(real_idx), 250):
            xb = to_t(real_np[i:i + 250], mean, std).to(DEV)
            sr.append(F.softmax(model(xb), 1)[:, 1].cpu().numpy())
    s_f = np.concatenate(sf)
    s_r = np.concatenate(sr)
    acc_fake = float((s_f >= 0.5).mean() * 100)
    acc_real = float((s_r < 0.5).mean() * 100)
    auroc = roc_auc_score(
        np.r_[np.ones_like(s_f), np.zeros_like(s_r)],
        np.r_[s_f, s_r]) * 100
    thr = np.quantile(s_r, 0.99)
    tpr1 = float((s_f >= thr).mean() * 100)
    del model
    torch.cuda.empty_cache()
    print('%s: f=%.2f r=%.2f bal=%.2f auc=%.2f tpr=%.2f' % (
        tag, acc_fake, acc_real, (acc_fake + acc_real) / 2, auroc, tpr1),
        flush=True)
    return {'acc_fake': round(acc_fake, 2), 'acc_real': round(acc_real, 2),
            'acc_balanced': round((acc_fake + acc_real) / 2, 2),
            'auroc': round(auroc, 2), 'tpr1': round(tpr1, 2)}


# E3
R = {'resnet': {}, 'effnet': {}, 'clip': {}}
for r in (1, 2, 3):
    R['resnet']['r%d' % r] = eval_model('r3', build_resnet18, RN18_MEAN,
        RN18_STD, BASE + '/runs/round_0%d/detector.pth' % r)
for r in (1, 2, 3):
    R['effnet']['r%d' % r] = eval_model('e3', build_efficientnet_b0,
        RN18_MEAN, RN18_STD,
        BASE + '/runs/bb01/det_efficientnet_b0_full_r%d.pth' % r)
for r in (1, 2, 3):
    R['clip']['r%d' % r] = eval_model('c3', build_clip, CLIP_MEAN,
        CLIP_STD, BASE + '/runs/bb02/det_clip_vitb16_full_r%d.pth' % r)
json.dump(R, open(OUT_BASE + '/flux_schnell_v1_eval.json', 'w'), indent=1)
print('E3_DONE', flush=True)

# E4
R4 = {'resnet': {}, 'effnet': {}, 'clip': {}}
for r in (1, 2, 3):
    R4['resnet']['r%d' % r] = eval_model('r4', build_resnet18, RN18_MEAN,
        RN18_STD, BASE + '/runs/E4_round_0%d/detector.pth' % r)
for r in (1, 2, 3):
    R4['effnet']['r%d' % r] = eval_model('e4', build_efficientnet_b0,
        RN18_MEAN, RN18_STD,
        BASE + '/runs/E4_bb01/det_efficientnet_b0_full_r%d.pth' % r)
for r in (1, 2, 3):
    R4['clip']['r%d' % r] = eval_model('c4', build_clip, CLIP_MEAN,
        CLIP_STD, BASE + '/runs/E4_bb02/det_clip_vitb16_full_r%d.pth' % r)
json.dump(R4, open(OUT_BASE + '/flux_schnell_v1_eval_e4.json', 'w'), indent=1)
print('E4_DONE', flush=True)

# E5
R5 = {'resnet': {}, 'effnet': {}, 'clip': {}}
for r in (1, 2, 3):
    R5['resnet']['r%d' % r] = eval_model('r5', build_resnet18, RN18_MEAN,
        RN18_STD, BASE + '/runs/E5_round_0%d/detector.pth' % r)
for r in (1, 2, 3):
    R5['effnet']['r%d' % r] = eval_model('e5', build_efficientnet_b0,
        RN18_MEAN, RN18_STD,
        BASE + '/runs/E5_bb01/det_efficientnet_b0_full_r%d.pth' % r)
for r in (1, 2, 3):
    R5['clip']['r%d' % r] = eval_model('c5', build_clip, CLIP_MEAN,
        CLIP_STD, BASE + '/runs/E5_bb02/det_clip_vitb16_full_r%d.pth' % r)
json.dump(R5, open(OUT_BASE + '/flux_schnell_v1_eval_e5.json', 'w'), indent=1)
print('E5_DONE', flush=True)

# UnivFD
import open_clip
m, _, _ = open_clip.create_model_and_transforms(
    'ViT-L-14', pretrained='openai', force_quick_gelu=True)
visual = m.visual.to(DEV).eval()
with open(BASE + '/runs/univfd/heads/full.pkl', 'rb') as fh:
    clf = pickle.load(fh)
w = torch.tensor(clf.coef_, dtype=torch.float32)
b = torch.tensor(clf.intercept_, dtype=torch.float32)
W2 = torch.cat([-w, w], dim=0).to(DEV)
b2 = torch.cat([-b, b], dim=0).to(DEV)
CM = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(DEV)
CS = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(DEV)

def uni_score(x_u8):
    x = torch.from_numpy(np.ascontiguousarray(x_u8)).float().div_(255.).to(DEV)
    x = x.permute(0, 3, 1, 2)
    x = (x - CM) / CS
    feats = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        for i in range(0, len(x_u8), 125):
            f = visual(x[i:i + 125]).float()
            feats.append((f @ W2.T + b2).cpu())
    logits = torch.cat(feats).numpy()
    return torch.softmax(torch.from_numpy(logits), 1)[:, 1].numpy()

s_f = uni_score(fake_imgs)
real_np = np.asarray(X_val[real_idx])
s_r = uni_score(real_np)
acc_fake = float((s_f >= 0.5).mean() * 100)
acc_real = float((s_r < 0.5).mean() * 100)
auroc = roc_auc_score(
    np.r_[np.ones_like(s_f), np.zeros_like(s_r)], np.r_[s_f, s_r]) * 100
thr = np.quantile(s_r, 0.99)
tpr1 = float((s_f >= thr).mean() * 100)
UNI = {'acc_fake': round(acc_fake, 2), 'acc_real': round(acc_real, 2),
       'acc_balanced': round((acc_fake + acc_real) / 2, 2),
       'auroc': round(auroc, 2), 'tpr1': round(tpr1, 2)}
print('UNIVFD_V1:', UNI, flush=True)
json.dump(UNI, open(OUT_BASE + '/flux_schnell_v1_eval_univfd.json', 'w'),
          indent=1)
print('ALL_V1_EVAL_DONE', flush=True)
