#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TIFS 审稿 S4（可选）：AutoAttack（standard, L_inf）评测 ResNet-18 全量检测器，替代论文中"AutoAttack untested"的断言。
pip install git+https://github.com/fra31/auto-attack
AutoAttack 期望输入在 [0,1] 像素域，因此这里把归一化放进模型里，ε 按像素给出（默认 2.9/255 ≈ 归一化空间 0.05）。
用法：python autoattack_resnet.py --ckpt runs/round_01/detector.pth --n 1000 --eps255 2.9
GPU：1000 张 standard 版约 20–40 分钟（RTX 4080S）。"""
import os

import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_detector as td


class Wrapped(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.register_buffer('mean', torch.tensor(td.MEAN))
        self.register_buffer('std', torch.tensor(td.STD))

    def forward(self, x):
        return self.m((x - self.mean) / self.std)


def main():
    from autoattack import AutoAttack
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--prep', default=os.path.join(os.environ.get('DATA_ROOT', '<DATA_ROOT>'), 'data_npy'))
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--eps255', type=float, default=2.9)
    ap.add_argument('--out', default='autoattack_result.json')
    a = ap.parse_args()
    X = np.load(a.prep + '/X_validation.npy', mmap_mode='r')
    y = np.load(a.prep + '/y_validation.npy').reshape(-1)
    rng = np.random.RandomState(42)
    idx = np.sort(rng.choice(len(y), a.n, replace=False))       # 随机子集（类别按总体比例）
    x = torch.tensor(np.asarray(X[idx]).astype('float32').transpose(0, 3, 1, 2) / 255.0)
    yt = torch.tensor(y[idx]).long()
    m = td.build_model(2)
    m.load_state_dict(torch.load(a.ckpt, map_location='cpu'), strict=False)
    model = Wrapped(m).cuda().eval()
    adv = AutoAttack(model, norm='Linf', eps=a.eps255 / 255, version='standard')
    xa = adv.run_standard_evaluation(x.cuda(), yt.cuda(), bs=100)
    with torch.no_grad():
        acc = (model(xa).argmax(1).cpu() == yt).float().mean().item() * 100
    json.dump({'n': a.n, 'eps255': a.eps255, 'robust_acc': acc}, open(a.out, 'w'), indent=1)
    print('AUTOATTACK robust acc = %.2f%%' % acc)


if __name__ == '__main__':
    main()
