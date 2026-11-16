#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实验15 骨干扩展 · 模块B：数据加载 + 训练/评估循环。

路径策略：全部路径由常量 PREP_DIR 与 RUNS_ROOT 拼接常量子目录/文件名，
不接受任何 CLI 路径参数，不存在可控路径穿越面。
"""
import os
import time

import numpy as np
import pandas as pd
import torch

PREP_DIR = os.path.join(os.environ.get('DATA_ROOT', '<DATA_ROOT>'), 'data_npy')
RUNS_ROOT = os.path.join(os.environ.get('DATA_ROOT', '<DATA_ROOT>'), 'runs')
EXPERIMENT = 'bb01'

from bb_models import GENS  # noqa: E402


def to_tensor(x_uint8_nhwc, mean, std):
    x = x_uint8_nhwc.astype('float32') / 255.0
    x = x.transpose(0, 3, 1, 2)
    x = (x - mean) / std
    return torch.from_numpy(np.ascontiguousarray(x))


def load_val():
    X_val = np.load(os.path.join(PREP_DIR, 'X_validation.npy'), mmap_mode='r')
    y_val = np.load(os.path.join(PREP_DIR, 'y_validation.npy')).reshape(-1)
    meta_val = pd.read_csv(os.path.join(PREP_DIR, 'validation_meta_named.csv'))
    return X_val, y_val, meta_val


def load_train():
    X_train = np.load(os.path.join(PREP_DIR, 'X_train.npy'), mmap_mode='r')
    y_train = np.load(os.path.join(PREP_DIR, 'y_train.npy')).reshape(-1)
    meta_train = pd.read_csv(os.path.join(PREP_DIR, 'train_meta_named.csv'))
    return X_train, y_train, meta_train


@torch.no_grad()
def evaluate(model, X, y, meta, dev, mean, std, bs=200, tag=''):
    model.eval()
    preds = []
    for i in range(0, len(X), bs):
        xb = to_tensor(np.asarray(X[i:i + bs]), mean, std).to(dev)
        preds.append(model(xb).argmax(1).cpu().numpy())
    pred = np.concatenate(preds)
    overall = (pred == y).mean() * 100
    print('\n--- 评估 %s ---' % tag)
    print('整体: %.2f%% (n=%d)' % (overall, len(y)))
    per = {}
    for g in GENS:
        m = (meta['gen_name'] == g).values
        per[g] = float((pred[m] == y[m]).mean() * 100)
        print('%-11s %8d %8.2f' % (g, int(m.sum()), per[g]))
    return overall, per


def train_loop(model, X_train, y_train, train_idx, dev, mean, std,
               epochs=3, bs=64, lr=1e-4, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    steps = int(np.ceil(len(train_idx) / bs))
    print('训练: %d 张, %d 步/epoch, %d epoch, bs=%d' % (
        len(train_idx), steps, epochs, bs), flush=True)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(train_idx)
        tot, corr, run = 0, 0, 0.0
        for s in range(steps):
            bidx = perm[s * bs:(s + 1) * bs]
            if len(bidx) == 0:
                continue
            xb = to_tensor(np.asarray(X_train[bidx]), mean, std).to(dev)
            yb = torch.from_numpy(y_train[bidx].astype('int64')).to(dev)
            out = model(xb)
            loss = lossf(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += float(loss.detach())
            corr += int((out.argmax(1) == yb).sum())
            tot += len(bidx)
            if (s + 1) % 100 == 0:
                print('  ep%d %4d/%d loss=%.4f acc=%.2f%% %.0fs' % (
                    ep + 1, s + 1, steps, run / (s + 1), 100.0 * corr / tot,
                    time.time() - t0), flush=True)
        print('== epoch %d loss=%.4f acc=%.2f%% %.0fs' % (
            ep + 1, run / max(1, steps), 100.0 * corr / tot, time.time() - t0), flush=True)
