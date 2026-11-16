### UNIVFD_V2 ### strip marker lines
"""UnivFD (Ojha, CVPR 2023) external-detector replication on our substrate.
Stage 1: pre-extract frozen OpenCLIP ViT-L/14 features (fp16 autocast).
Stage 2: logistic-regression heads, full x1 (official protocol) -- LOO x7
         trained the same way, on pre-extracted features (fast).
Stage 3: three-framing pixel-level evaluation on the full pipeline
         (clean / white-box PGD / low-pass transfer / low-pass adaptive).
         UnivFD has no non-differentiable block, so adaptive == naive WB
         by construction -- itself a finding.
Output: runs/univfd/{feat_*.npy, heads/*.pkl, result.json, eval.json}"""
import json
import os
import pickle
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

BASE = os.environ.get('DATA_ROOT', '<DATA_ROOT>')
FEATDIR = BASE + '/runs/univfd'
os.makedirs(FEATDIR, exist_ok=True)
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


@torch.no_grad()
def extract(split, visual):
    outp = FEATDIR + '/feat_%s.npy' % split
    if os.path.exists(outp):
        print('skip', split, flush=True)
        return np.load(outp)
    X = np.load(BASE + '/data_npy/X_%s.npy' % split, mmap_mode='r')
    feats = []
    for i in range(0, len(X), 100):
        xb = to_norm(np.asarray(X[i:i + 100])).to(DEV)
        with torch.autocast('cuda', dtype=torch.float16):
            f = visual(xb).float()
        feats.append(f.cpu().numpy())
        if (i // 100) % 20 == 0:
            print(split, i, '/', len(X), flush=True)
    arr = np.concatenate(feats)
    np.save(outp, arr)
    print('extracted', split, arr.shape, flush=True)
    return arr


def stage1():
    visual = load_visual()
    extract('validation', visual)
    extract('train', visual)
    del visual
    torch.cuda.empty_cache()


def stage2():
    Ftr = np.load(FEATDIR + '/feat_train.npy')
    meta = pd.read_csv(BASE + '/data_npy/train_meta_named.csv')
    ytr = (meta['gen_name'].values != 'Real').astype('int64')
    head_dir = FEATDIR + '/heads'
    os.makedirs(head_dir, exist_ok=True)
    jobs = [('full', None)] + [('loo_' + g, g) for g in GENS]
    for tag, held in jobs:
        outp = head_dir + '/%s.pkl' % tag
        if os.path.exists(outp):
            print('skip', tag, flush=True)
            continue
        idx = np.arange(len(ytr))
        if held is not None:
            idx = idx[meta['gen_name'].values[idx] != held]
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
        clf.fit(Ftr[idx], ytr[idx])
        with open(outp, 'wb') as fh:
            pickle.dump(clf, fh)
        print('head', tag, 'n=%d' % len(idx), flush=True)
    print('HEADS_DONE', flush=True)


class UnivFDPixel(torch.nn.Module):
    """CLIP ViT-L visual + logistic head, fully differentiable pipeline."""

    def __init__(self, visual, clf):
        super().__init__()
        self.visual = visual
        w = torch.tensor(clf.coef_, dtype=torch.float32)
        b = torch.tensor(clf.intercept_, dtype=torch.float32)
        self.head_w = torch.nn.Parameter(w)
        self.head_b = torch.nn.Parameter(b)

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


def stage3():
    X = np.load(BASE + '/data_npy/X_validation.npy', mmap_mode='r')
    meta_v = pd.read_csv(BASE + '/data_npy/validation_meta_named.csv')
    gva = meta_v['gen_name'].values
    # 评估子集：每生成器 100 + Real 700（控制 PGD 时长），共 1400
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

    res = {'n_eval': int(len(Xsub))}
    s_clean_f, s_clean_r = [], []
    s_wb_f, s_wb_r = [], []
    s_lpt_f, s_lpt_r = [], []
    s_lpa_f, s_lpa_r = [], []
    B = 50
    for i in range(0, len(Xsub), B):
        xb = to_norm(Xsub[i:i + B]).to(DEV)
        yb = torch.tensor(
            [0 if gsub[j] == 'Real' else 1 for j in range(i, i + len(xb))],
            device=DEV)
        adv = pgd(model, xb, yb, 0.05, 0.0125, 10)
        with torch.no_grad():
            s_clean_f.append(F.softmax(model(xb), 1)[:, 1].float().cpu().numpy())
            s_wb_f.append(F.softmax(model(adv), 1)[:, 1].float().cpu().numpy())
            s_lpt_f.append(F.softmax(
                model(lowpass_corner(adv.detach())), 1)[:, 1].float().cpu().numpy())
            s_lpa_f.append(F.softmax(
                model(lowpass_corner(adv)), 1)[:, 1].float().cpu().numpy())
        real_mask = (yb == 0).cpu().numpy()
        clean_b = xb.detach().cpu().numpy()
        adv_b = adv.cpu().numpy()
        with torch.no_grad():
            pass
        # 按 real/fake 拆分累积
        def split_by(arr_list, mask):
            f = [a[mask] for a in arr_list]
            return f
        cm = real_mask
        s_clean_r.append(s_clean_f[-1][cm])
        s_clean_f[-1] = s_clean_f[-1][~cm]
        s_wb_r.append(s_wb_f[-1][cm])
        s_wb_f[-1] = s_wb_f[-1][~cm]
        s_lpt_r.append(s_lpt_f[-1][cm])
        s_lpt_f[-1] = s_lpt_f[-1][~cm]
        s_lpa_r.append(s_lpa_f[-1][cm])
        s_lpa_f[-1] = s_lpa_f[-1][~cm]
        del clean_b, adv_b
        print('batch', i, flush=True)

    def agg(fparts, rparts):
        return np.concatenate(fparts), np.concatenate(rparts)

    out = {}
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
    with open(FEATDIR + '/eval.json', 'w') as fh:
        json.dump(out, fh, indent=1)
    print('UNIVFD_EVAL_DONE', flush=True)


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if stage in ('all', '1'):
        stage1()
    if stage in ('all', '2'):
        stage2()
    if stage in ('all', '3'):
        stage3()
