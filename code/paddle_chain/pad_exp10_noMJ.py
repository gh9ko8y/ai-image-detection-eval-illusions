#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Paddle 实验 10 复现：留出 Midjourney 重训（pretrained=True）+ 全量逐生成器评估。"""
import os
import time

import numpy as np
import paddle
import paddle.nn.functional as F

paddle.seed(42)
np.random.seed(42)
OUT = os.path.join(os.environ.get('DATA_ROOT', '<DATA_ROOT>'), 'runs/paddle_01/exp10_noMJ.pdparams')


def main():
    from pad_common import batches, eval_acc, load_val, to_tensor
    X_val, y_val, meta = load_val()
    model = paddle.vision.models.resnet18(pretrained=False, num_classes=2)
    model.set_state_dict(paddle.load(OUT))
    model.eval()
    print('整体: %.2f%%' % eval_acc(model, X_val, y_val, np.arange(len(y_val))))
    for g in ['ADM', 'BigGAN', 'GLIDE', 'Midjourney', 'SD15', 'VQDM', 'Wukong', 'Real']:
        idx = np.where(meta['gen_name'].values == g)[0]
        print('%-11s acc=%.2f%%' % (g, eval_acc(model, X_val, y_val, idx)))
    print('PAD_EXP10_DONE')


def train():
    from pad_common import batches, load_train, to_tensor
    X_train, y_train, meta = load_train()
    keep = (meta['gen_name'] != 'Midjourney').values
    idx_all = np.where(keep)[0]
    print('留出 Midjourney: %d -> %d' % (len(y_train), len(idx_all)))
    model = paddle.vision.models.resnet18(pretrained=True, num_classes=2)
    opt = paddle.optimizer.Adam(learning_rate=1e-4, parameters=model.parameters())
    t0 = time.time()
    for ep in range(3):
        model.train()
        perm = np.random.permutation(idx_all)
        c = n = 0
        nb = 0
        run = 0.0
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
        print('epoch %d loss=%.4f acc=%.2f%% %.0fs' % (
            ep + 1, run / nb, 100 * c / n, time.time() - t0), flush=True)
    paddle.save(model.state_dict(), OUT)
    print('saved', OUT)


if __name__ == '__main__':
    os.makedirs(os.path.join(os.environ.get('DATA_ROOT', '<DATA_ROOT>'), 'runs/paddle_01'), exist_ok=True)
    if not os.path.exists(OUT):
        train()
    main()
