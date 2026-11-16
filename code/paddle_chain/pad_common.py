#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Paddle 复现链公共库：数据加载 + 攻击器（按手册实验 1~10 的原始实现）。

坑位（手册记载，全部遵守）：
- clear_grad 在 step 之后（与 PyTorch 相反）
- clip 用 paddle.maximum/minimum（paddle.clip 不吃张量）
- 含梯度函数内不可用 paddle.no_grad
- paddle.grad(..., allow_unused=True) + None 补零（JPEG/位缩减白盒会拿到 None）
- round 梯度=0 → BPDA 反向直通
- ifft2 自带 1/N，不手动除
"""
import os

import io

import numpy as np
import paddle
import paddle.nn.functional as F

MEAN = np.array([0.485, 0.456, 0.406], dtype='float32').reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype='float32').reshape(1, 3, 1, 1)
PREP = os.path.join(os.environ.get('DATA_ROOT', '<DATA_ROOT>'), 'data_npy')


def to_tensor(x_uint8_nhwc):
    x = x_uint8_nhwc.astype('float32') / 255.0
    x = x.transpose(0, 3, 1, 2)
    x = (x - MEAN) / STD
    return paddle.to_tensor(np.ascontiguousarray(x))


def normed_tensor(a_np):
    """已归一化 NCHW float 数组 -> paddle 张量（不再反归一化/重归一化）。"""
    return paddle.to_tensor(np.ascontiguousarray(a_np.astype('float32')))


def load_val():
    X = np.load(PREP + '/X_validation.npy', mmap_mode='r')
    y = np.load(PREP + '/y_validation.npy').reshape(-1).astype('int64')
    import pandas as pd
    meta = pd.read_csv(PREP + '/validation_meta_named.csv')
    return X, y, meta


def load_train():
    X = np.load(PREP + '/X_train.npy', mmap_mode='r')
    y = np.load(PREP + '/y_train.npy').reshape(-1).astype('int64')
    import pandas as pd
    meta = pd.read_csv(PREP + '/train_meta_named.csv')
    return X, y, meta


def batches(idx, bs):
    for i in range(0, len(idx), bs):
        yield idx[i:i + bs]


@paddle.no_grad()
def eval_acc(model, X, y, idx, bs=200):
    correct = 0
    for b in batches(idx, bs):
        xb = to_tensor(np.asarray(X[b]))
        out = model(xb)
        correct += int((out.argmax(1).numpy() == y[b]).sum())
    return 100.0 * correct / len(idx)


def fgsm(model, xb, yb, eps):
    xb = xb.detach()
    xb.stop_gradient = False
    loss = F.cross_entropy(model(xb), yb)
    g = paddle.grad([loss], [xb])[0]
    return (xb + eps * g.sign()).detach()


def pgd(model, xb, yb, eps, steps, alpha=None, defense=None, eot_k=1):
    """标准 PGD。defense: 前向防御可微时可传入；eot_k>1 为 EOT（随机化防御用）。"""
    if alpha is None:
        alpha = eps / steps * 2.5
    xa = xb.clone().detach()
    for _ in range(steps):
        xa = xa.detach()
        xa.stop_gradient = False
        if eot_k > 1:
            gs = []
            for _k in range(eot_k):
                xn = xa + paddle.normal(0.0, 0.02, xa.shape)
                inp = defense(xn) if defense is not None else xn
                loss = F.cross_entropy(model(inp), yb)
                gs.append(paddle.grad([loss], [xa], allow_unused=True)[0])
            g = paddle.stack([paddle.zeros_like(xa) if gi is None else gi for gi in gs]).mean(0)
        else:
            inp = defense(xa) if defense is not None else xa
            loss = F.cross_entropy(model(inp), yb)
            g = paddle.grad([loss], [xa], allow_unused=True)[0]
            if g is None:
                g = paddle.zeros_like(xa)
        xa = xa.detach() + alpha * g.sign()
        xa = paddle.maximum(paddle.minimum(xa, xb + eps), xb - eps)
    return xa.detach()


def bpda(model, defense_np, xb, yb, eps, steps):
    """BPDA：前向走真实防御（numpy 世界），反向在 g(x_adv) 处取恒等梯度。"""
    alpha = eps / steps * 2.5
    xa = xb.clone().detach()
    for _ in range(steps):
        xa_np = xa.numpy()
        xg = normed_tensor(defense_np(xa_np))    # g(x_adv)：防御后的近似点（已归一化 NCHW）
        xg = xg.detach()
        xg.stop_gradient = False
        loss = F.cross_entropy(model(xg), yb)      # 梯度在 g(x) 处求（直通近似）
        g = paddle.grad([loss], [xg])[0]
        xa = xa.detach() + alpha * g.sign()        # 更新施加在 x_adv 上
        xa = paddle.maximum(paddle.minimum(xa, xb + eps), xb - eps)
    return xa.detach()


# ---- 防御（手册原始实现） ----

def bit_reduce(x_np, step=0.2):
    return (x_np / step).round() * step


def jpeg_np(x_np, quality=50):
    from PIL import Image
    outs = []
    for xi in x_np:
        # 反归一化到 uint8
        arr = xi.transpose(1, 2, 0)
        arr = arr * STD[0].reshape(1, 1, 3) + MEAN[0].reshape(1, 1, 3)
        arr = np.clip(arr * 255.0, 0, 255).astype('uint8')
        pil = Image.fromarray(arr)
        buf = io.BytesIO()
        pil.save(buf, 'JPEG', quality=quality)
        arr2 = np.array(Image.open(buf).convert('RGB')).astype('float32') / 255.0
        arr2 = (arr2 - MEAN[0].reshape(1, 1, 3)) / STD[0].reshape(1, 1, 3)
        outs.append(arr2.transpose(2, 0, 1))
    return np.stack(outs).astype('float32')


def lowpass(x, keep_ratio=0.15):
    B, C, H, W = x.shape
    f = paddle.fft.fft2(x)
    h = max(2, int(H * keep_ratio))
    w = max(2, int(W * keep_ratio))
    m = paddle.zeros((H, W))
    m[:h, :w] = 1.0
    m[:h, W - w:] = 1.0
    m[H - h:, :w] = 1.0
    m[H - h:, W - w:] = 1.0
    f = f * m.reshape([1, 1, H, W])
    return paddle.real(paddle.fft.ifft2(f))       # ifft 自带 1/N
