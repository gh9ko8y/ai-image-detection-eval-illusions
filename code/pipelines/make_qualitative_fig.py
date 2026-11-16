### QUAL_V2 ### strip marker lines
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import bb_models
import bb_train_core
from bb_train_core import to_tensor

BASE = os.environ.get('DATA_ROOT', '<DATA_ROOT>')
OUTDIR = BASE + '/runs/figs'
os.makedirs(OUTDIR, exist_ok=True)
DEV = 'cuda'


def lowpass_corner(x_t, r=0.15):
    """主实验同款四角低频掩码低通（与 lowpass_round.py 一致，可微）。"""
    B, C, H, W = x_t.shape
    f = torch.fft.fft2(x_t, dim=(-2, -1))
    h = max(2, int(H * r))
    w = max(2, int(W * r))
    m = torch.zeros((H, W), dtype=f.dtype, device=x_t.device)
    m[:h, :w] = 1
    m[:h, W - w:] = 1
    m[H - h:, :w] = 1
    m[H - h:, W - w:] = 1
    return torch.fft.ifft2(f * m.view(1, 1, H, W),
                           dim=(-2, -1)).real


def main():
    import pandas as pd
    X = np.load(BASE + '/data_npy/X_validation.npy', mmap_mode='r')
    meta = pd.read_csv(BASE + '/data_npy/validation_meta_named.csv')
    idx_fake = int(np.where(meta['gen_name'].values == 'Midjourney')[0][7])

    model = bb_models.build_resnet18().to(DEV).eval()
    model.load_state_dict(torch.load(
        BASE + '/runs/round_01/detector.pth', map_location=DEV,
        weights_only=True))

    x_u8 = np.asarray(X[idx_fake])
    xb = to_tensor(x_u8[None], bb_models.RN18_MEAN,
                   bb_models.RN18_STD).to(DEV)
    yb = torch.tensor([1], device=DEV)

    eps, alpha, adv = 0.05, 0.0125, xb.clone().detach()
    for _ in range(10):
        adv = adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(adv), yb)
        gr = torch.autograd.grad(loss, adv)[0]
        with torch.no_grad():
            adv = torch.clamp(adv + alpha * gr.sign(), xb - eps, xb + eps)

    with torch.no_grad():
        p_clean = F.softmax(model(xb), 1)[0, 1].item()
        p_adv = F.softmax(model(adv), 1)[0, 1].item()
        p_lp_adv = F.softmax(model(lowpass_corner(adv)), 1)[0, 1].item()
        p_lp_clean = F.softmax(model(lowpass_corner(xb)), 1)[0, 1].item()
    print('P(fake): clean %.3f adv %.3f lp(adv) %.3f lp(clean) %.3f'
          % (p_clean, p_adv, p_lp_adv, p_lp_clean), flush=True)

    def denorm(t):
        x = t[0].cpu().numpy().transpose(1, 2, 0)
        return np.clip((x * [0.229, 0.224, 0.225]
                        + [0.485, 0.456, 0.406]), 0, 1)

    diff = np.abs(denorm(adv) - denorm(xb))
    diff = np.clip(diff * 15, 0, 1)

    titles = [
        'Original fake (Midjourney)\nP(fake)=%.2f' % p_clean,
        'PGD adversarial ($\\epsilon$=0.05)\nP(fake)=%.2f' % p_adv,
        'Perturbation ($\\times$15)',
        'Low-pass filtered (clean)\nP(fake)=%.2f' % p_lp_clean,
        'Low-pass filtered (adversarial)\nP(fake)=%.2f' % p_lp_adv,
    ]
    imgs = [denorm(xb), denorm(adv), diff,
            denorm(lowpass_corner(xb)), denorm(lowpass_corner(adv))]
    fig, axes = plt.subplots(1, 5, figsize=(13, 2.9))
    for ax, im, t in zip(axes, imgs, titles):
        ax.imshow(im)
        ax.set_title(t, fontsize=7.5)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(OUTDIR + '/qualitative.png', dpi=300, bbox_inches='tight')
    print('QUAL_V2_DONE', flush=True)


main()
