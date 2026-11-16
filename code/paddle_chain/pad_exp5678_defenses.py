#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Paddle 实验 5/6/7/8 复现：随机化、位缩减+BPDA、JPEG+BPDA、低通（历史口径）。
依赖 exp2 检测器。逐项打印关键数字。"""
import os

import numpy as np
import paddle

paddle.seed(42)
np.random.seed(42)
CKPT = os.path.join(os.environ.get('DATA_ROOT', '<DATA_ROOT>'), 'runs/paddle_01/exp2_detector.pdparams')


def get_model():
    m = paddle.vision.models.resnet18(pretrained=False, num_classes=2)
    m.set_state_dict(paddle.load(CKPT))
    m.eval()
    return m


def rand_defense(model, xb, yb, sigma=0.02, n_samples=10):
    """随机化防御：10 次加噪平均预测。"""
    probs = []
    with paddle.no_grad():
        for _ in range(n_samples):
            xn = xb + paddle.normal(0.0, sigma, xb.shape)
            probs.append(paddle.nn.functional.softmax(model(xn), axis=1))
    return paddle.stack(probs).mean(0).argmax(1)


def main():
    from pad_common import batches, bit_reduce, eval_acc, jpeg_np, load_val, lowpass, pgd, bpda, to_tensor, normed_tensor
    import pad_common as pc
    X_val, y_val, _ = load_val()
    model = get_model()
    idx = np.arange(len(y_val))

    print('===== 实验 5：随机化防御（σ=0.02，10 采样）=====')
    c = 0
    for b in batches(idx, 100):
        xb = to_tensor(np.asarray(X_val[b]))
        yb = paddle.to_tensor(y_val[b])
        pred = rand_defense(model, xb, yb)
        c += int((pred.numpy() == y_val[b]).sum())
    print('防御下不攻击: %.2f%%' % (100 * c / len(idx)))
    for eps in [0.02, 0.05]:
        cw = ce = 0
        for b in batches(idx, 100):
            xb = to_tensor(np.asarray(X_val[b]))
            yb = paddle.to_tensor(y_val[b])
            xw = pc.pgd(model, xb, yb, eps, steps=10)
            xe = pc.pgd(model, xb, yb, eps, steps=10, eot_k=10)
            pw = rand_defense(model, xw, yb)
            pe = rand_defense(model, xe, yb)
            cw += int((pw.numpy() == y_val[b]).sum())
            ce += int((pe.numpy() == y_val[b]).sum())
        print('eps=%.2f  白盒PGD后 %.2f%%  EOT后 %.2f%%' % (eps, 100 * cw / len(idx), 100 * ce / len(idx)), flush=True)
    print('PAD_EXP5_DONE')

    print('===== 实验 6：位缩减 step=0.2 + BPDA =====')
    c = 0
    for b in batches(idx, 100):
        xb = to_tensor(np.asarray(X_val[b]))
        xr = normed_tensor(bit_reduce(xb.numpy()))
        with paddle.no_grad():
            c += int((model(xr).argmax(1).numpy() == y_val[b]).sum())
    print('防御下不攻击: %.2f%%' % (100 * c / len(idx)))
    for eps in [0.02, 0.05, 0.10]:
        cw = cb = 0
        for b in batches(idx, 100):
            xb = to_tensor(np.asarray(X_val[b]))
            yb = paddle.to_tensor(y_val[b])
            xw = pc.pgd(model, xb, yb, eps, steps=10,
                        defense=lambda t: normed_tensor(bit_reduce(t.numpy())))
            xb2 = pc.bpda(model, bit_reduce, xb, yb, eps, steps=10)
            with paddle.no_grad():
                xr = normed_tensor(bit_reduce(xw.numpy()))
                cw += int((model(xr).argmax(1).numpy() == y_val[b]).sum())
                xr2 = normed_tensor(bit_reduce(xb2.numpy()))
                cb += int((model(xr2).argmax(1).numpy() == y_val[b]).sum())
        print('eps=%.2f  白盒后 %.2f%%  BPDA后 %.2f%%' % (eps, 100 * cw / len(idx), 100 * cb / len(idx)), flush=True)
    print('PAD_EXP6_DONE')

    print('===== 实验 7：JPEG q=50 + BPDA（前 2000 子集）=====')
    idx2 = idx[:2000]
    c = 0
    for b in batches(idx2, 100):
        xb = to_tensor(np.asarray(X_val[b]))
        xj = normed_tensor(jpeg_np(xb.numpy()))
        with paddle.no_grad():
            c += int((model(xj).argmax(1).numpy() == y_val[b]).sum())
    print('防御下不攻击: %.2f%%' % (100 * c / len(idx2)))
    for eps in [0.02, 0.05, 0.10]:
        cw = cb = 0
        for b in batches(idx2, 50):
            xb = to_tensor(np.asarray(X_val[b]))
            yb = paddle.to_tensor(y_val[b])
            xb2 = pc.bpda(model, jpeg_np, xb, yb, eps, steps=10)
            # 白盒：对"防御后模型"做 PGD——JPEG 切断梯度，用 allow_unused+补零（=白盒盲打，历史现象）
            xw = pc.pgd(model, xb, yb, eps, steps=10,
                        defense=lambda t: normed_tensor(jpeg_np(t.numpy())))
            with paddle.no_grad():
                xj = normed_tensor(jpeg_np(xw.numpy()))
                cw += int((model(xj).argmax(1).numpy() == y_val[b]).sum())
                xj2 = normed_tensor(jpeg_np(xb2.numpy()))
                cb += int((model(xj2).argmax(1).numpy() == y_val[b]).sum())
        print('eps=%.2f  白盒后 %.2f%%  BPDA后 %.2f%%' % (eps, 100 * cw / len(idx2), 100 * cb / len(idx2)), flush=True)
    print('PAD_EXP7_DONE')

    print('===== 实验 8：低通滤波 keep=0.15（历史口径复现——已知该口径为迁移，结论已被实验11推翻）=====')
    c = 0
    for b in batches(idx, 100):
        xb = to_tensor(np.asarray(X_val[b]))
        with paddle.no_grad():
            xl = lowpass(xb, 0.15)
            c += int((model(xl).argmax(1).numpy() == y_val[b]).sum())
    print('防御下不攻击: %.2f%%' % (100 * c / len(idx)))
    for eps in [0.02, 0.05, 0.10]:
        ct = 0
        for b in batches(idx, 100):
            xb = to_tensor(np.asarray(X_val[b]))
            yb = paddle.to_tensor(y_val[b])
            xadv = pc.pgd(model, xb, yb, eps, steps=10)      # 攻击裸模型
            with paddle.no_grad():
                xl = lowpass(xadv, 0.15)                     # 再套滤波（迁移口径）
                ct += int((model(xl).argmax(1).numpy() == y_val[b]).sum())
        print('eps=%.2f  迁移口径攻击后 %.2f%%' % (eps, 100 * ct / len(idx)), flush=True)
    print('PAD_EXP8_DONE')


if __name__ == '__main__':
    main()
