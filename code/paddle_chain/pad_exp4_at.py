#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Paddle 实验 4 复现：PGD-AT（eps=0.02、5 步，从实验 2 检测器微调 2+2 epoch）+ 攻击评估。"""
import os
import time

import numpy as np
import paddle
import paddle.nn.functional as F

paddle.seed(42)
np.random.seed(42)
CKPT = os.path.join(os.environ.get('DATA_ROOT', '<DATA_ROOT>'), 'runs/paddle_01/exp2_detector.pdparams')
OUT = os.path.join(os.environ.get('DATA_ROOT', '<DATA_ROOT>'), 'runs/paddle_01/exp4_advtrained.pdparams')


def main():
    from pad_common import batches, eval_acc, fgsm, load_val, pgd, to_tensor
    import pad_common as pc
    X_val, y_val, _ = load_val()
    model = paddle.vision.models.resnet18(pretrained=False, num_classes=2)
    model.set_state_dict(paddle.load(OUT))
    model.eval()
    idx = np.arange(len(y_val))
    print('AT 后干净: %.2f%%' % eval_acc(model, X_val, y_val, idx))
    for eps in [0.02, 0.05, 0.10, 0.15, 0.20]:
        cf = cp = 0
        for b in pc.batches(idx, 100):
            xb = to_tensor(np.asarray(X_val[b]))
            yb = paddle.to_tensor(y_val[b])
            xf = pc.fgsm(model, xb, yb, eps)
            xp = pc.pgd(model, xb, yb, eps, steps=10)
            with paddle.no_grad():
                cf += int((model(xf).argmax(1).numpy() == y_val[b]).sum())
                cp += int((model(xp).argmax(1).numpy() == y_val[b]).sum())
        print('eps=%.2f  FGSM %.2f%%  PGD %.2f%%' % (eps, 100 * cf / len(idx), 100 * cp / len(idx)), flush=True)
    print('PAD_EXP4_DONE')


def train():
    from pad_common import batches, load_train, pgd, to_tensor
    X_train, y_train, _ = load_train()
    model = paddle.vision.models.resnet18(pretrained=False, num_classes=2)
    model.set_state_dict(paddle.load(CKPT))
    opt = paddle.optimizer.Adam(learning_rate=1e-4, parameters=model.parameters())
    idx_all = np.arange(len(y_train))
    t0 = time.time()
    for ep in range(4):  # 原口径：两遍、每遍 2 epoch = 4 epoch 连续
        model.train()
        perm = np.random.permutation(idx_all)
        c = n = 0
        nb = 0
        run = 0.0
        for b in batches(perm, 64):
            xb = to_tensor(np.asarray(X_train[b]))
            yb = paddle.to_tensor(y_train[b])
            x_adv = pgd(model, xb, yb, eps=0.02, steps=5)
            out = model(paddle.concat([xb, x_adv]))
            loss = F.cross_entropy(out, paddle.concat([yb, yb]))
            loss.backward()
            opt.step()
            opt.clear_grad()
            run += float(loss)
            nb += 1
            c += int((out.argmax(1).numpy() == np.concatenate([y_train[b], y_train[b]])).sum())
            n += 2 * len(b)
        print('AT epoch %d loss=%.4f acc=%.2f%% %.0fs' % (
            ep + 1, run / nb, 100 * c / n, time.time() - t0), flush=True)
    paddle.save(model.state_dict(), OUT)
    print('saved', OUT)


if __name__ == '__main__':
    if not os.path.exists(OUT):
        train()
    main()
