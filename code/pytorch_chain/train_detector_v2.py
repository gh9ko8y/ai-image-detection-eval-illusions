#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
[v2 · TIFS 审稿 M13/S18（可选）：新增 --deterministic，用于把"同 seed 下 GPU 非确定性"与"seed 方差"分开；原 train_detector.py 保持不变]
  例：python train_detector_v2.py --seed 0 --out det_seed0            # 换 seed
      python train_detector_v2.py --seed 42 --deterministic --out det_det42   # 同 seed 确定性，重复两次应逐位一致
检测器训练（PyTorch，AutoDL RTX 4080 SUPER）
============================================
复现历史配置（手册实验 2 / 实验 B），并统一预处理口径：
  ResNet-18(ImageNet 预训练) + fc->2 | Adam lr=1e-4 | batch 64 | 3 epoch
  预处理：uint8 NHWC -> /255 -> NCHW -> ImageNet 归一化

用法：
  # 全量训练（复现 detector_resnet18）
  python train_detector.py --prep <DATA_ROOT>/prep_XXXX --out detector_resnet18_torch

  # 留出某个生成器重训（复现 detector_noMJ，为 LOO 矩阵铺路）
  python train_detector.py --prep <DATA_ROOT>/prep_XXXX --exclude Midjourney --out detector_noMJ_torch

  # 只做评估，不训练
  python train_detector.py --prep <DATA_ROOT>/prep_XXXX --eval-only --ckpt detector_resnet18_torch.pth
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision

MEAN = np.array([0.485, 0.456, 0.406], dtype='float32').reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype='float32').reshape(1, 3, 1, 1)

HIST = {
    'ADM': 96.60, 'BigGAN': 98.80, 'GLIDE': 98.60, 'Midjourney': 64.80,
    'SD15': 83.20, 'VQDM': 78.20, 'Wukong': 78.00, 'Real': 94.71,
}


def to_tensor(x_uint8_nhwc):
    """uint8 NHWC -> torch float32 NCHW 已归一化"""
    x = x_uint8_nhwc.astype('float32') / 255.0
    x = x.transpose(0, 3, 1, 2)
    x = (x - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x))


def find_prep(explicit=None):
    """定位预处理数据目录。

    固定路径优先（fetch3.py 的布局），找不到再回退到时间戳目录。
    """
    if explicit:
        return explicit
    fixed = os.path.join(os.environ.get('DATA_ROOT', '<DATA_ROOT>'), 'data_npy')
    if os.path.exists(os.path.join(fixed, 'X_validation.npy')):
        return fixed
    cands = sorted(glob.glob(os.path.join(os.environ.get('DATA_ROOT', '<DATA_ROOT>'), 'prep_*')))
    assert cands, '找不到数据目录（既无 data_npy 也无 prep_*）'
    return cands[-1]


@torch.no_grad()
def evaluate(model, X, y, meta, dev, bs=200, tag=''):
    model.eval()
    preds = []
    for i in range(0, len(X), bs):
        xb = to_tensor(X[i:i + bs]).to(dev)
        preds.append(model(xb).argmax(1).cpu().numpy())
    pred = np.concatenate(preds)
    overall = (pred == y).mean() * 100
    print('\n--- 评估 %s ---' % tag)
    print('整体准确率: %.2f%%  (n=%d)' % (overall, len(y)))
    if meta is not None:
        print('%-12s %8s %10s %10s' % ('生成器', 'n', '本次%', '历史%'))
        for g in ['ADM', 'BigGAN', 'GLIDE', 'Midjourney', 'SD15', 'VQDM', 'Wukong', 'Real']:
            m = (meta['gen_name'] == g).values
            if m.sum() == 0:
                print('%-12s %8d %10s %10s' % (g, 0, '-', '%.2f' % HIST.get(g, 0)))
                continue
            a = (pred[m] == y[m]).mean() * 100
            h = HIST.get(g)
            if h is None or m.sum() < 100:
                print('%-12s %8d %10.2f %10s' % (g, m.sum(), a, '-' if h is None else '%.2f' % h))
            else:
                print('%-12s %8d %10.2f %10.2f  Δ=%+.2f' % (g, m.sum(), a, h, a - h))
    return overall


def build_model(num_classes=2):
    """构建 ImageNet 预训练的 ResNet-18 并替换分类头。

    坑：新版 torchvision 不允许同时传 weights= 和 num_classes=（会抛
        "expected value 1000 but got 2"）。必须先按默认 1000 类构建，
        再手动替换 fc —— 这也是"迁移学习"的标准写法。
    """
    m = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    in_f = m.fc.in_features
    m.fc = nn.Linear(in_f, num_classes)   # 分类头重新初始化，其余层用预训练权重
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prep', default=None)
    ap.add_argument('--out', default='detector_resnet18_torch')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--exclude', default=None, help='留出的生成器名，如 Midjourney')
    ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--deterministic', action='store_true')
    args = ap.parse_args()
    if args.deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    D = find_prep(args.prep)
    print('device:', dev, '| prep:', D)

    # ---- 检查数据齐了没 ----
    need = ['X_train.npy', 'y_train.npy', 'train_meta_named.csv',
            'X_validation.npy', 'y_validation.npy', 'validation_meta_named.csv']
    missing = [f for f in need if not os.path.exists(os.path.join(D, f))]
    if missing:
        print('!! 数据还没准备好，缺:', missing)
        print('   当前目录内容:', sorted(os.listdir(D)))
        sys.exit(3)

    X_val = np.load(os.path.join(D, 'X_validation.npy'), mmap_mode='r')
    y_val = np.load(os.path.join(D, 'y_validation.npy'))
    meta_val = pd.read_csv(os.path.join(D, 'validation_meta_named.csv'))
    print('X_val', X_val.shape, '| X_train 文件 %.2f GB' % (
        os.path.getsize(os.path.join(D, 'X_train.npy')) / 1e9))

    model = build_model(2)

    if args.eval_only:
        ck = args.ckpt or (args.out + '.pth')
        sd = torch.load(ck, map_location='cpu')
        model.load_state_dict(sd, strict=False)
        model.to(dev)
        evaluate(model, X_val, y_val, meta_val, dev, tag=os.path.basename(ck))
        return

    X_train = np.load(os.path.join(D, 'X_train.npy'), mmap_mode='r')
    y_train = np.load(os.path.join(D, 'y_train.npy'))
    meta_train = pd.read_csv(os.path.join(D, 'train_meta_named.csv'))

    if args.exclude:
        keep = (meta_train['gen_name'] != args.exclude).values
        print('留出 %s: 训练集 %d -> %d' % (args.exclude, len(y_train), keep.sum()))
        train_idx = np.where(keep)[0]
    else:
        train_idx = np.arange(len(y_train))

    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()

    steps = int(np.ceil(len(train_idx) / args.bs))
    print('训练: %d 张, %d 步/epoch, %d epoch' % (len(train_idx), steps, args.epochs))
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(train_idx)
        tot, corr, run = 0, 0, 0.0
        for s in range(steps):
            bidx = perm[s * args.bs:(s + 1) * args.bs]
            if len(bidx) == 0:
                continue
            xb = to_tensor(np.asarray(X_train[bidx])).to(dev)
            yb = torch.from_numpy(y_train[bidx].astype('int64')).to(dev)
            out = model(xb)
            loss = lossf(out, yb)
            opt.zero_grad()          # PyTorch: zero_grad 在 step 之前（与 Paddle 相反）
            loss.backward()
            opt.step()
            run += float(loss)
            corr += int((out.argmax(1) == yb).sum())
            tot += len(bidx)
            if (s + 1) % 50 == 0:
                print('  ep%d %4d/%d  loss=%.4f  train_acc=%.2f%%  %.0fs' % (
                    ep + 1, s + 1, steps, run / (s + 1), 100.0 * corr / tot,
                    time.time() - t0), flush=True)
        print('== epoch %d 完成  loss=%.4f  train_acc=%.2f%%  用时 %.0fs' % (
            ep + 1, run / max(1, steps), 100.0 * corr / tot, time.time() - t0), flush=True)
        evaluate(model, X_val, y_val, meta_val, dev, tag='epoch %d 后' % (ep + 1))

    torch.save(model.state_dict(), args.out + '.pth')
    print('\n已保存:', args.out + '.pth')
    json.dump({'out': args.out, 'epochs': args.epochs, 'bs': args.bs, 'lr': args.lr,
               'exclude': args.exclude, 'seed': args.seed,
               'train_n': int(len(train_idx))},
              open(args.out + '_config.json', 'w'), indent=1)
    print('TRAIN_DONE')


if __name__ == '__main__':
    main()
