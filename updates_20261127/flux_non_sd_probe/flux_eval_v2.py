import json, os, sys, time

sys.path.insert(0, '$DATA_ROOT/e5_code')
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score

from bb_models import build_resnet18, build_efficientnet_b0, \
    RN18_MEAN, RN18_STD
from bb_clip_models import build_clip, CLIP_MEAN, CLIP_STD

BASE = '$DATA_ROOT'
GEN_DIR = BASE + '/runs/probe_non_sd/flux_schnell_500'
DEV = 'cuda'

X_val = np.load(BASE + '/data_npy/X_validation.npy', mmap_mode='r')
meta = pd = __import__('pandas').read_csv(
    BASE + '/data_npy/validation_meta_named.csv')
g = meta['gen_name'].values
real_idx = np.where(g == 'Real')[0]
# 均匀抽 3500 真图(sdxl_eval 同口径: 全部 3500)
# 加载假图
fake_imgs = np.zeros((500, 224, 224, 3), dtype=np.uint8)
for i in range(500):
    im = Image.open(GEN_DIR + ('/img_%05d.png' % i)).convert('RGB')
    fake_imgs[i] = np.asarray(im.resize((224, 224), Image.BICUBIC))
print('fake loaded', fake_imgs.shape, flush=True)


def to_t(x_u8, mean, std):
    x = x_u8.astype('float32') / 255.0
    x = x.transpose(0, 3, 1, 2)
    return torch.from_numpy(np.ascontiguousarray((x - mean) / std))


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
        # 真图: 3500 全量按批
        real_np = np.asarray(X_val[real_idx])
        for i in range(0, len(real_idx), 250):
            xb = to_t(real_np[i:i + 250], mean, std).to(DEV)
            sr.append(F.softmax(model(xb), 1)[:, 1].cpu().numpy())
    s_f = np.concatenate(sf)
    s_r = np.concatenate(sr)
    pred_f = (s_f >= 0.5).astype(int)
    pred_r = (s_r >= 0.5).astype(int)
    acc_fake = pred_f.mean() * 100
    acc_real = (1 - pred_r).mean() * 100
    bal = (acc_fake + acc_real) / 2
    auroc = roc_auc_score(
        np.r_[np.ones_like(s_f), np.zeros_like(s_r)],
        np.r_[s_f, s_r]) * 100
    thr = np.quantile(s_r, 0.99)
    tpr1 = (s_f >= thr).mean() * 100
    del model
    torch.cuda.empty_cache()
    print('%s: fake=%.2f real=%.2f bal=%.2f auroc=%.2f tpr1=%.2f' % (
        tag, acc_fake, acc_real, bal, auroc, tpr1), flush=True)
    return {'acc_fake': round(acc_fake, 2), 'acc_real': round(acc_real, 2),
            'acc_balanced': round(bal, 2), 'auroc': round(auroc, 2),
            'tpr1': round(tpr1, 2)}


R = {'resnet': {}, 'effnet': {}, 'clip': {}}
for r in (1, 2, 3):
    R['resnet']['r%d' % r] = eval_model(
        'resnet_r%d' % r, build_resnet18, RN18_MEAN, RN18_STD,
        BASE + '/runs/round_0%d/detector.pth' % r)
for r in (1, 2, 3):
    R['effnet']['r%d' % r] = eval_model(
        'effnet_r%d' % r, build_efficientnet_b0, RN18_MEAN, RN18_STD,
        BASE + '/runs/bb01/det_efficientnet_b0_full_r%d.pth' % r)
for r in (1, 2, 3):
    R['clip']['r%d' % r] = eval_model(
        'clip_r%d' % r, build_clip, CLIP_MEAN, CLIP_STD,
        BASE + '/runs/bb02/det_clip_vitb16_full_r%d.pth' % r)

out_path = BASE + '/runs/probe_non_sd/flux_schnell_eval.json'
with open(out_path, 'w') as f:
    json.dump(R, f, indent=1)
print('EVAL_DONE', out_path, flush=True)

# E4 / E5 可选
for tag, prefix in (('e4', 'E4'), ('e5', 'E5')):
    R2 = {'resnet': {}, 'effnet': {}, 'clip': {}}
    for r in (1, 2, 3):
        R2['resnet']['r%d' % r] = eval_model(
            'res_r%d' % r, build_resnet18, RN18_MEAN, RN18_STD,
            BASE + '/runs/%s_round_0%d/detector.pth' % (prefix, r))
    for r in (1, 2, 3):
        R2['effnet']['r%d' % r] = eval_model(
            'eff_r%d' % r, build_efficientnet_b0, RN18_MEAN, RN18_STD,
            BASE + '/runs/%s_bb01/det_efficientnet_b0_full_r%d.pth' % (prefix, r))
    for r in (1, 2, 3):
        R2['clip']['r%d' % r] = eval_model(
            'clip_r%d' % r, build_clip, CLIP_MEAN, CLIP_STD,
            BASE + '/runs/%s_bb02/det_clip_vitb16_full_r%d.pth' % (prefix, r))
    with open(BASE + '/runs/probe_non_sd/flux_schnell_eval_%s.json' % tag,
              'w') as f:
        json.dump(R2, f, indent=1)
    print('EVAL_%s_DONE' % tag.upper(), flush=True)

print('ALL_EVAL_COMPLETE', flush=True)
