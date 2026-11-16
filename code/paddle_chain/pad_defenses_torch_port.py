#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R4（TIFS 审稿 M7/M18/S2）：Paddle 四种防御实验的 PyTorch 移植版。
为什么移植：当前 AutoDL 实例是 RTX PRO 6000 Blackwell（sm_120），Paddle 2.6.2 不支持该算力
（实测 GPU 前向输出全错、准确率 0%），因此把 Paddle exp2 检测器权重用 code/paddle_chain/convert_ckpt.py 的
同一映射转成 torchvision resnet18，在 PyTorch 下逐行复刻 pad_common.py 的预处理、PGD（α=2.5ε/10、无随机起点）、
EOT（σ=0.02, k=10）、BPDA（恒等直通）、bit-reduce(0.2，归一化空间)、JPEG q50（PIL）、四角低通 0.15。
校验：先报告干净准确率，应与 Paddle 原始 90.79% 一致（权重移植正确的判据）。
四列：no_attack | transfer（攻击裸模型→过防御）| naive white-box（梯度过防御、不修不可微）| adaptive（EOT/BPDA；低通=可微即 naive）
用法：venv_paddle/bin/python pad_defenses_torch_port.py convert   # 先转换权重
      venv/bin/python pad_defenses_torch_port.py --out R4/defenses_torch_port.json [--eps 0.02 0.05 0.10] [--n 0]
"""
import os

import argparse
import io
import json
import sys

import numpy as np
import pandas as pd
from PIL import Image
if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'convert':
    pass
else:
    import torch
    import torch.nn.functional as F
    import torchvision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PREP = os.path.join(os.environ.get('DATA_ROOT', '<DATA_ROOT>'), 'data_npy')
CKPT = os.path.join(os.environ.get('DATA_ROOT', '<DATA_ROOT>'), 'runs/paddle_01/exp2_detector.pdparams')
DEV = 'cuda'
MEAN = np.array([0.485, 0.456, 0.406], dtype='float32').reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype='float32').reshape(1, 3, 1, 1)


NPZ = os.path.join(os.environ.get('DATA_ROOT', '<DATA_ROOT>'), 'rerun_20261116/R4/exp2_detector_converted.npz')


def convert_with_paddle():
    """在 venv_paddle 中运行：paddle.load(CPU) -> convert_ckpt 映射 -> npz。原 pdparams 只读不改。"""
    import paddle
    paddle.set_device('cpu')
    from convert_ckpt import paddle_to_torch_keys
    t = paddle_to_torch_keys(paddle.load(CKPT))
    np.savez(NPZ, **t)
    print('converted', len(t), 'tensors ->', NPZ)


def load_model():
    t = np.load(NPZ)
    m = torchvision.models.resnet18(weights=None, num_classes=2)
    tsd = {}
    for k in t.files:
        v = torch.from_numpy(np.ascontiguousarray(t[k]))
        if k.endswith('running_mean') or k.endswith('running_var'):
            v = v.reshape(-1)
        tsd[k] = v
    miss, unexp = m.load_state_dict(tsd, strict=False)
    miss = [k for k in miss if not k.endswith('num_batches_tracked')]
    assert not miss and not unexp, (miss, unexp)
    return m.to(DEV).eval()


def to_tensor(x_u8):
    x = (x_u8.astype('float32') / 255.0).transpose(0, 3, 1, 2)
    return torch.from_numpy(np.ascontiguousarray((x - MEAN) / STD)).to(DEV)


def bit_reduce_t(x, step=0.2):
    return (x / step).round() * step


def jpeg_np(x_np, quality=50):
    outs = []
    for xi in x_np:
        arr = xi.transpose(1, 2, 0) * STD[0].reshape(1, 1, 3) + MEAN[0].reshape(1, 1, 3)
        arr = np.clip(arr * 255.0, 0, 255).astype('uint8')
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, 'JPEG', quality=quality)
        a2 = np.array(Image.open(buf).convert('RGB')).astype('float32') / 255.0
        outs.append(((a2 - MEAN[0].reshape(1, 1, 3)) / STD[0].reshape(1, 1, 3)).transpose(2, 0, 1))
    return np.stack(outs).astype('float32')


def jpeg_t(x):
    return torch.from_numpy(jpeg_np(x.detach().cpu().numpy())).to(DEV)


def lowpass(x, r=0.15):
    B, C, H, W = x.shape
    f = torch.fft.fft2(x)
    h = max(2, int(H * r)); w = max(2, int(W * r))
    m = torch.zeros((H, W), device=x.device)
    m[:h, :w] = 1; m[:h, W - w:] = 1; m[H - h:, :w] = 1; m[H - h:, W - w:] = 1
    return torch.fft.ifft2(f * m.view(1, 1, H, W)).real


def pgd(model, xb, yb, eps, steps=10, defense=None, eot_k=1, differentiable=True):
    """复刻 pad_common.pgd：α=2.5ε/steps、无随机起点；不可微防御的梯度为 None→补零（=naive 白盒盲打）。"""
    alpha = eps / steps * 2.5
    xa = xb.clone()
    for _ in range(steps):
        xa = xa.detach().requires_grad_(True)
        if eot_k > 1:
            g = torch.zeros_like(xa)
            for _k in range(eot_k):
                xn = xa + torch.randn_like(xa) * 0.02
                loss = F.cross_entropy(model(xn), yb)
                g += torch.autograd.grad(loss, xa)[0]
            g /= eot_k
        elif defense is not None and not differentiable:
            g = torch.zeros_like(xa)          # 防御在 numpy/round 中切断梯度（与 Paddle allow_unused 补零一致）
        else:
            inp = defense(xa) if defense is not None else xa
            g = torch.autograd.grad(F.cross_entropy(model(inp), yb), xa)[0]
        xa = xa.detach() + alpha * g.sign()
        xa = torch.max(torch.min(xa, xb + eps), xb - eps)
    return xa.detach()


def bpda(model, defense, xb, yb, eps, steps=10):
    alpha = eps / steps * 2.5
    xa = xb.clone()
    for _ in range(steps):
        xg = defense(xa).detach().requires_grad_(True)
        g = torch.autograd.grad(F.cross_entropy(model(xg), yb), xg)[0]
        xa = xa.detach() + alpha * g.sign()
        xa = torch.max(torch.min(xa, xb + eps), xb - eps)
    return xa.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eps', type=float, nargs='+', default=[0.02, 0.05, 0.10])
    ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.manual_seed(42); np.random.seed(42)
    X = np.load(PREP + '/X_validation.npy', mmap_mode='r')
    y = np.load(PREP + '/y_validation.npy').reshape(-1).astype('int64')
    gen = pd.read_csv(PREP + '/validation_meta_named.csv')['gen_name'].values
    model = load_model()
    N = a.n or len(y)
    idx_all = np.arange(N)

    @torch.no_grad()
    def rand_pred(x):
        return torch.stack([F.softmax(model(x + torch.randn_like(x) * 0.02), 1) for _ in range(10)]).mean(0).argmax(1)

    @torch.no_grad()
    def pred(x):
        return model(x).argmax(1)

    D = {
        'randomization': dict(fwd=rand_pred, prep=None, diff=True),
        'bit_reduce_0.2': dict(fwd=lambda x: pred(bit_reduce_t(x)), prep=bit_reduce_t, diff=False),
        'jpeg_q50': dict(fwd=lambda x: pred(jpeg_t(x)), prep=jpeg_t, diff=False),
        'lowpass_0.15': dict(fwd=lambda x: pred(lowpass(x)), prep=lowpass, diff=True),
    }
    res = {'note': 'Paddle exp2 detector weights ported to torchvision ResNet-18 (convert_ckpt mapping); '
                   'eps in ImageNet-normalized space; PGD-10 alpha=2.5eps/10, no random start; EOT sigma=0.02 k=10',
           'n_eval': int(N)}
    # 干净准确率（移植校验）
    c = 0
    for i in range(0, N, 200):
        c += int((pred(to_tensor(np.asarray(X[i:i + 200]))).cpu().numpy() == y[i:i + 200]).sum())
    res['clean_bare'] = 100.0 * c / N
    print('clean bare (port check, Paddle original 90.79):', res['clean_bare'], flush=True)
    BS = 100
    for name, d in D.items():
        r = {}
        for subset_name, idx in [('all', idx_all)] + ([('first2000', idx_all[:2000])] if name == 'jpeg_q50' and N > 2000 else []):
            rr = {'n': int(len(idx))}
            c = 0
            for i in range(0, len(idx), BS):
                b = idx[i:i + BS]
                c += int((d['fwd'](to_tensor(np.asarray(X[b]))).cpu().numpy() == y[b]).sum())
            rr['no_attack'] = 100.0 * c / len(idx)
            pcls = {}
            for eps in a.eps:
                ct = cw = ca = 0
                for i in range(0, len(idx), BS):
                    b = idx[i:i + BS]
                    xb = to_tensor(np.asarray(X[b])); yb = torch.from_numpy(y[b]).to(DEV); yn = y[b]
                    x_tr = pgd(model, xb, yb, eps)
                    ct += int((d['fwd'](x_tr).cpu().numpy() == yn).sum())
                    if name == 'randomization':
                        x_wb = x_tr
                        x_ad = pgd(model, xb, yb, eps, eot_k=10)
                    elif not d['diff']:
                        x_wb = pgd(model, xb, yb, eps, defense=d['prep'], differentiable=False)
                        x_ad = bpda(model, d['prep'], xb, yb, eps)
                    else:
                        x_wb = pgd(model, xb, yb, eps, defense=d['prep'])
                        x_ad = x_wb
                    cw += int((d['fwd'](x_wb).cpu().numpy() == yn).sum())
                    ca += int((d['fwd'](x_ad).cpu().numpy() == yn).sum())
                rr['eps=%.2f' % eps] = {'transfer': 100.0 * ct / len(idx), 'naive_whitebox': 100.0 * cw / len(idx),
                                        'adaptive': 100.0 * ca / len(idx)}
                print(name, subset_name, 'eps=%.2f' % eps, rr['eps=%.2f' % eps], flush=True)
            r[subset_name] = rr
            print(name, subset_name, 'no_attack', rr['no_attack'], flush=True)
        res[name] = r
        json.dump(res, open(a.out, 'w'), indent=1)
    print('R4_TORCH_PORT_DONE')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'convert':
        convert_with_paddle()
    else:
        main()
