#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Paddle 实验 5/6/7/8 v2 —— TIFS 审稿 M7/M18/S2 修正版（原 pad_exp5678_defenses.py 保持不变）。
修正：
  [M18] JPEG 默认在全部 7000 张验证图上评估（旧版 idx[:2000]，类别构成未知）；
        --jpeg-subset stratified 可改为分层抽样 2000 张（每类按比例），用于省时。
  [S2]  四种防御统一输出四列：No attack | Transfer（攻击裸模型→再过防御）| Naive white-box | Adaptive
        （随机化=EOT k=10；位缩减/JPEG=BPDA；低通=滤波在回路内的可微 PGD）。
  [M7]  每个 ε 单独成行并写入 JSON，论文 Table II 可以注明 ε。
依赖 exp2 检测器（runs/paddle_01/exp2_detector.pdparams）与 pad_common.py。
输出：runs/paddle_01/exp5678_v2.json
用法：python pad_exp5678_defenses_v2.py [--eps 0.02 0.05] [--jpeg-subset full|stratified]
"""
import os

import argparse
import json

import numpy as np
import paddle

paddle.seed(42)
np.random.seed(42)
CKPT = os.path.join(os.environ.get('DATA_ROOT', '<DATA_ROOT>'), 'runs/paddle_01/exp2_detector.pdparams')
OUT = os.path.join(os.environ.get('DATA_ROOT', '<DATA_ROOT>'), 'rerun_20261116/R4/exp5678_v2.json')
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def get_model():
    m = paddle.vision.models.resnet18(pretrained=False, num_classes=2)
    m.set_state_dict(paddle.load(CKPT))
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eps', type=float, nargs='+', default=[0.02, 0.05])
    ap.add_argument('--jpeg-subset', choices=['full', 'stratified'], default='full')
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--n', type=int, default=0, help='smoke test: use first n images only')
    a = ap.parse_args()
    import pad_common as pc
    from pad_common import batches, bit_reduce, jpeg_np, load_val, lowpass, normed_tensor, to_tensor
    X_val, y_val, meta = load_val()
    gen = meta['gen_name'].values
    model = get_model()
    idx_all = np.arange(a.n if a.n else len(y_val))
    if a.jpeg_subset == 'full':
        idx_jpeg = idx_all
    else:
        rng = np.random.RandomState(42)
        idx_jpeg = np.sort(np.concatenate([
            rng.choice(np.where(gen == g)[0], int(round(2000 * (gen == g).mean())), replace=False)  # noqa
            for g in np.unique(gen)]))
    print('val class counts (all):', dict(zip(*np.unique(gen, return_counts=True))))
    print('val class counts (first 2000, old JPEG subset):', dict(zip(*np.unique(gen[:2000], return_counts=True))))

    def rand_pred(xb, sigma=0.02, k=10):
        with paddle.no_grad():
            ps = [paddle.nn.functional.softmax(model(xb + paddle.normal(0.0, sigma, xb.shape)), axis=1)
                  for _ in range(k)]
        return paddle.stack(ps).mean(0).argmax(1).numpy()

    def plain_pred(xb):
        with paddle.no_grad():
            return model(xb).argmax(1).numpy()

    D = {
        'randomization': dict(idx=idx_all, fwd=lambda xb: rand_pred(xb),
                              diff=None, np_def=None, eot=10),
        'bit_reduce_0.2': dict(idx=idx_all, fwd=lambda xb: plain_pred(normed_tensor(bit_reduce(xb.numpy()))),
                               diff=lambda t: normed_tensor(bit_reduce(t.numpy())), np_def=bit_reduce, eot=1),
        'jpeg_q50': dict(idx=idx_jpeg, fwd=lambda xb: plain_pred(normed_tensor(jpeg_np(xb.numpy()))),
                         diff=lambda t: normed_tensor(jpeg_np(t.numpy())), np_def=jpeg_np, eot=1),
        'lowpass_0.15': dict(idx=idx_all, fwd=lambda xb: plain_pred(lowpass(xb, 0.15)),
                             diff=lambda t: lowpass(t, 0.15), np_def=None, eot=1),
    }
    res = {'note': 'eps in ImageNet-normalized space; PGD-10, alpha=2.5*eps/10, no random start'}
    for name, d in D.items():
        idx = d['idx']
        r = {'n_eval': int(len(idx)),
             'class_counts': {k: int(v) for k, v in zip(*np.unique(gen[idx], return_counts=True))}}
        c = sum(int((d['fwd'](to_tensor(np.asarray(X_val[b]))) == y_val[b]).sum()) for b in batches(idx, 100))
        r['no_attack'] = 100.0 * c / len(idx)
        for eps in a.eps:
            ct = cw = ca = 0
            for b in batches(idx, 50):
                xb = to_tensor(np.asarray(X_val[b]))
                yb = paddle.to_tensor(y_val[b])
                x_tr = pc.pgd(model, xb, yb, eps, steps=10)                       # transfer：攻击裸模型
                ct += int((d['fwd'](x_tr) == y_val[b]).sum())
                if name == 'randomization':
                    x_wb = pc.pgd(model, xb, yb, eps, steps=10)                   # naive：不建模随机性
                    x_ad = pc.pgd(model, xb, yb, eps, steps=10, eot_k=d['eot'])   # EOT
                elif d['np_def'] is not None:
                    x_wb = pc.pgd(model, xb, yb, eps, steps=10, defense=d['diff'])  # 梯度被切断 → 掩蔽
                    x_ad = pc.bpda(model, d['np_def'], xb, yb, eps, steps=10)
                else:  # 低通可微：naive WB == adaptive
                    x_wb = pc.pgd(model, xb, yb, eps, steps=10, defense=d['diff'])
                    x_ad = x_wb
                cw += int((d['fwd'](x_wb) == y_val[b]).sum())
                ca += int((d['fwd'](x_ad) == y_val[b]).sum())
            r['eps=%.2f' % eps] = {'transfer': 100.0 * ct / len(idx), 'naive_whitebox': 100.0 * cw / len(idx),
                                   'adaptive': 100.0 * ca / len(idx)}
            print(name, 'eps=%.2f' % eps, r['eps=%.2f' % eps], flush=True)
        res[name] = r
        json.dump(res, open(a.out, 'w'), indent=1)
    print('PAD_EXP5678_V2_DONE')


if __name__ == '__main__':
    main()
