#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Paddle 实验 3 复现：裸检测器 FGSM/PGD 基线攻击（eps 扫描 0.02~0.20，全集 7000）。"""
import os

import numpy as np
import paddle

paddle.seed(42)
np.random.seed(42)
CKPT = os.path.join(os.environ.get('DATA_ROOT', '<DATA_ROOT>'), 'runs/paddle_01/exp2_detector.pdparams')


def main():
    from pad_common import eval_acc, fgsm, load_val, pgd, to_tensor
    import pad_common as pc
    X_val, y_val, _ = load_val()
    model = paddle.vision.models.resnet18(pretrained=False, num_classes=2)
    model.set_state_dict(paddle.load(CKPT))
    model.eval()
    idx = np.arange(len(y_val))
    print('干净: %.2f%%' % eval_acc(model, X_val, y_val, idx))

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
    print('PAD_EXP3_DONE')


if __name__ == '__main__':
    main()
