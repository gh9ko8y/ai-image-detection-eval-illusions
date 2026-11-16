### FIX_HEAD_V1 ### strip marker lines
"""Stage3 修复版：UnivFD 二分类头维度修正。
sklearn 二分类 coef_=[1,768]（单 logit）；P(f=1)=sigmoid(w·f+b)。
等价双输出头：W=[[-w],[+w]], b=[[-b],[+b]]，logits=[logit0, logit1]，
softmax 后 P(类1) 与单 logit 版严格一致。"""
import json
import os
import pickle
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

BASE = os.environ.get('DATA_ROOT', '<DATA_ROOT>')
FEATDIR = BASE + '/runs/univfd'
DEV = 'cuda'
GENS = ['ADM', 'BigGAN', 'GLIDE', 'Midjourney', 'SD15', 'VQDM', 'Wukong']
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073],
                     dtype='float32').reshape(1, 3, 1, 1)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711],
                    dtype='float32').reshape(1, 3, 1, 1)


def to_norm(x_u8):
    x = x_u8.astype('float32') / 255.0
    x = x.transpose(0, 3, 1, 2)
    return torch.from_numpy(np.ascontiguousarray((x - CLIP_MEAN) / CLIP_STD))


def load_visual():
    import open_clip
    m, _, _ = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='openai', force_quick_gelu=True)
    return m.visual.to(DEV).eval()


class UnivFDPixel(torch.nn.Module):
    def __init__(self, visual, clf):
        super().__init__()
        self.visual = visual
        w = torch.tensor(clf.coef_, dtype=torch.float32)  # [1,768]
        b = torch.tensor(clf.intercept_, dtype=torch.float32)  # [1]
        W2 = torch.cat([-w, w], dim=0)  # [2,768]
        b2 = torch.cat([-b, b], dim=0)  # [2]
        self.head_w = torch.nn.Parameter(W2)
        self.head_b = torch.nn.Parameter(b2)

    def forward(self, x):
        with torch.autocast('cuda', dtype=torch.float16):
            f = self.visual(x).float()
        return f @ self.head_w.T + self.head_b


def lowpass_corner(x_t, r=0.15):
    B, C, H, W = x_t.shape
    f = torch.fft.fft2(x_t, dim=(-2, -1))
    h = max(2, int(H * r))
    w = max(2, int(W * r))
    m = torch.zeros((H, W), dtype=f.dtype, device=x_t.device)
    m[:h, :w] = 1
    m[:h, W - w:] = 1
    m[H - h:, :w] = 1
    m[H - h:, W - w:] = 1
    return torch.fft.ifft2(f * m.view(1, 1, H, W), dim=(-2, -1)).real


def pgd(model, xb, yb, eps, alpha, steps):
    adv = xb.clone().detach()
    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(adv), yb)
        gr = torch.autograd.grad(loss, adv)[0]
        with torch.no_grad():
            adv = torch.clamp(adv + alpha * gr.sign(), xb - eps, xb + eps)
    return adv.detach()


def main():
    X = np.load(BASE + '/data_npy/X_validation.npy', mmap_mode='r')
    meta_v = pd.read_csv(BASE + '/data_npy/validation_meta_named.csv')
    gva = meta_v['gen_name'].values
    rng = np.random.RandomState(42)
    idx = np.concatenate(
        [rng.choice(np.where(gva == g)[0], 100, replace=False)
         for g in GENS + ['Real']])
    gsub = gva[idx]
    Xsub = np.asarray(X[idx])

    with open(FEATDIR + '/heads/full.pkl', 'rb') as fh:
        clf = pickle.load(fh)
    visual = load_visual()
    model = UnivFDPixel(visual, clf).to(DEV).eval()
    # 识别性自检：一个 batch 前向形状
    with torch.no_grad():
        probe = model(to_norm(Xsub[:4]).to(DEV))
    print('logit shape', tuple(probe.shape), flush=True)
    assert probe.shape == (4, 2), 'head shape wrong'

    def batch_scores(x_pixels, apply_lp=False):
        ss = []
        with torch.no_grad():
            for i in range(0, len(x_pixels), 50):
                xb = to_norm(x_pixels[i:i + 50]).to(DEV)
                if apply_lp:
                    xb = lowpass_corner(xb)
                p = F.softmax(model(xb), 1)[:, 1]
                ss.append(p.float().cpu().numpy())
        return np.concatenate(ss)

    s_clean_f, s_clean_r = [], []
    s_wb_f, s_wb_r = [], []
    s_lpt_f, s_lpt_r = [], []
    s_lpa_f, s_lpa_r = [], []
    B = 50
    for i in range(0, len(Xsub), B):
        xb = to_norm(Xsub[i:i + B]).to(DEV)
        yb = torch.tensor(
            [0 if gsub[j] == 'Real' else 1 for j in range(i, i + len(xb))],
            device=DEV, dtype=torch.long)
        adv = pgd(model, xb, yb, 0.05, 0.0125, 10)
        with torch.no_grad():
            pc = F.softmax(model(xb), 1)[:, 1].float().cpu().numpy()
            pw = F.softmax(model(adv), 1)[:, 1].float().cpu().numpy()
            plt_ = F.softmax(model(lowpass_corner(adv.detach())), 1)[:, 1].float().cpu().numpy()
            pla = F.softmax(model(lowpass_corner(adv)), 1)[:, 1].float().cpu().numpy()
        cm = (yb == 0).cpu().numpy()
        s_clean_f.append(pc[~cm]); s_clean_r.append(pc[cm])
        s_wb_f.append(pw[~cm]); s_wb_r.append(pw[cm])
        s_lpt_f.append(plt_[~cm]); s_lpt_r.append(plt_[cm])
        s_lpa_f.append(pla[~cm]); s_lpa_r.append(pla[cm])
        print('batch', i, flush=True)

    def agg(fp, rp):
        return np.concatenate(fp), np.concatenate(rp)

    out = {'n_eval': int(len(Xsub))}
    for name, fp, rp in [('clean', s_clean_f, s_clean_r),
                         ('whitebox_pgd', s_wb_f, s_wb_r),
                         ('lowpass_transfer', s_lpt_f, s_lpt_r),
                         ('lowpass_adaptive', s_lpa_f, s_lpa_r)]:
        sf, sr = agg(fp, rp)
        y = np.r_[np.ones(len(sf)), np.zeros(len(sr))]
        sc = np.r_[sf, sr]
        out[name] = {
            'fake_acc': float((sf >= 0.5).mean() * 100),
            'real_acc': float((sr < 0.5).mean() * 100),
            'overall_acc': float(((sf >= 0.5).mean() * 100
                                  + (sr < 0.5).mean() * 100) / 2),
            'auroc': float(roc_auc_score(y, sc) * 100),
            'tpr1': float((sf >= np.quantile(sr, 0.99)).mean() * 100)}
        print(name, out[name], flush=True)

    # LOO 干净矩阵（特征级，快速）
    Fva = np.load(FEATDIR + '/feat_validation.npy')
    loo = {}
    for tag in ['full'] + ['loo_' + g for g in GENS]:
        with open(FEATDIR + '/heads/%s.pkl' % tag, 'rb') as fh:
            c2 = pickle.load(fh)
        s = c2.predict_proba(Fva)[:, 1]
        s_real = s[gva == 'Real']
        thr = np.quantile(s_real, 0.99)
        e = {'overall_acc': float(((s[gva != 'Real'] >= 0.5).mean() * 100
                                   + (s_real < 0.5).mean() * 100) / 2),
             'real_acc': float((s_real < 0.5).mean() * 100)}
        for gen in GENS:
            pos = s[gva == gen]
            e[gen] = float((pos >= 0.5).mean() * 100)
            e[gen + '_auroc'] = float(roc_auc_score(
                np.r_[np.ones(len(pos)), np.zeros(len(s_real))],
                np.r_[pos, s_real]) * 100)
            e[gen + '_tpr1'] = float((pos >= thr).mean()) * 100
        loo[tag] = e
        print(tag, 'loo-done', flush=True)
    out['loo_clean_matrix'] = loo
    with open(FEATDIR + '/eval.json', 'w') as fh:
        json.dump(out, fh, indent=1)
    print('UNIVFD_EVAL_DONE', flush=True)


main()
