#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Paddle 实验 2+9 复现：检测器训练（含逐生成器评估 = 实验 9）。
口径：ResNet-18(ImageNet 预训练, num_classes=2) | Adam 1e-4 | bs 64 | 3 epoch。"""
import os
import time

import numpy as np
import paddle
import paddle.nn.functional as F

paddle.seed(42)
np.random.seed(42)

OUT = os.path.join(os.environ.get('DATA_ROOT', '<DATA_ROOT>'), 'runs/paddle_01/exp2_detector.pdparams')


def build():
    return paddle.vision.models.resnet18(pretrained=True, num_classes=2)


def main():
    from pad_common import batches, eval_acc, load_train, load_val, to_tensor
    X_val, y_val, meta_val = load_val()
    model = build()
    model.set_state_dict(paddle.load(OUT))
    model.eval()
    acc = eval_acc(model, X_val, y_val, np.arange(len(y_val)))
    print('整体准确率: %.2f%%' % acc)
    for g in ['ADM', 'BigGAN', 'GLIDE', 'Midjourney', 'SD15', 'VQDM', 'Wukong', 'Real']:
        idx = np.where(meta_val['gen_name'].values == g)[0]
        print('%-11s n=%d acc=%.2f%%' % (g, len(idx), eval_acc(model, X_val, y_val, idx)))
    print('PAD_EXP2_EXP9_DONE')


def train():
    from pad_common import batches, load_train, to_tensor
    X_train, y_train, meta = load_train()
    model = build()
    opt = paddle.optimizer.Adam(learning_rate=1e-4, parameters=model.parameters())
    idx_all = np.arange(len(y_train))
    t0 = time.time()
    for ep in range(3):
        model.train()
        perm = np.random.permutation(idx_all)
        c = n = 0
        run = 0.0
        nb = 0
        for b in batches(perm, 64):
            xb = to_tensor(np.asarray(X_train[b]))
            yb = paddle.to_tensor(y_train[b])
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()
            opt.clear_grad()
            run += float(loss)
            nb += 1
            c += int((out.argmax(1).numpy() == y_train[b]).sum())
            n += len(b)
        print('epoch %d loss=%.4f train_acc=%.2f%% %.0fs' % (
            ep + 1, run / nb, 100 * c / n, time.time() - t0), flush=True)
    paddle.save(model.state_dict(), OUT)
    print('saved', OUT)


if __name__ == '__main__':
    os.makedirs(os.path.join(os.environ.get('DATA_ROOT', '<DATA_ROOT>'), 'runs/paddle_01'), exist_ok=True)
    if not os.path.exists(OUT):
        train()
    main()
