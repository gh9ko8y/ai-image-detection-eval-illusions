#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实验17 · 对抗训练 x 未见生成器泛化。

训练口径：PGD-AT（eps=0.02、5 步，复刻实验 4），ResNet-18，
其余超参与轮次一致。AT 训练后逐生成器评估。
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bb_models import GENS, RN18_MEAN, RN18_STD, build_resnet18
from bb_train_core import PREP_DIR, RUNS_ROOT, evaluate, load_train, load_val, to_tensor

EXPERIMENT = 'at01'
EPS = 0.02
AT_STEPS = 5
AT_ALPHA = EPS / AT_STEPS


def pgd_attack(model, xb, yb):
    xa = xb.clone().detach()
    for _ in range(AT_STEPS):
        xa.requires_grad_(True)
        loss = F.cross_entropy(model(xa), yb)
        g = torch.autograd.grad(loss, xa)[0]
        xa = xa.detach() + AT_ALPHA * g.sign()
        xa = torch.max(torch.min(xa, xb + EPS), xb - EPS)
    return xa.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exclude', default=None, choices=[None] + GENS)
    ap.add_argument('--run', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    holdout_part = ''
    if args.exclude is not None:
        holdout_part = '_holdout_' + args.exclude
    tag = 'at_resnet18' + holdout_part + '_r' + str(int(args.run))
    out_dir = RUNS_ROOT + '/' + EXPERIMENT
    model_path = out_dir + '/det_' + tag + '.pth'
    cfg_path = out_dir + '/cfg_' + tag + '.json'

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X_val, y_val, meta_val = load_val()
    X_train, y_train, meta_train = load_train()
    if args.exclude is not None:
        keep = (meta_train['gen_name'] != args.exclude).values
        train_idx = np.where(keep)[0]
        print('AT 留出 %s: 训练集 %d -> %d' % (args.exclude, len(y_train), len(train_idx)))
    else:
        train_idx = np.arange(len(y_train))

    model = build_resnet18().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()

    steps = int(np.ceil(len(train_idx) / args.bs))
    print('AT 训练: %d 张, %d 步/epoch, %d epoch, eps=%.3f, PGD %d 步' % (
        len(train_idx), steps, args.epochs, EPS, AT_STEPS), flush=True)
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(train_idx)
        tot, corr, run = 0, 0, 0.0
        for s in range(steps):
            bidx = perm[s * args.bs:(s + 1) * args.bs]
            if len(bidx) == 0:
                continue
            xb = to_tensor(np.asarray(X_train[bidx]), RN18_MEAN, RN18_STD).to(dev)
            yb = torch.from_numpy(y_train[bidx].astype('int64')).to(dev)
            xadv = pgd_attack(model, xb, yb)
            out = model(torch.cat([xb, xadv]))
            loss = lossf(out, torch.cat([yb, yb]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += float(loss.detach())
            corr += int((out.argmax(1) == torch.cat([yb, yb])).sum())
            tot += 2 * len(bidx)
            if (s + 1) % 50 == 0:
                print('  ep%d %4d/%d loss=%.4f acc=%.2f%% %.0fs' % (
                    ep + 1, s + 1, steps, run / (s + 1), 100.0 * corr / tot,
                    time.time() - t0), flush=True)
        print('== epoch %d loss=%.4f acc=%.2f%% %.0fs' % (
            ep + 1, run / max(1, steps), 100.0 * corr / tot, time.time() - t0), flush=True)
        evaluate(model, X_val, y_val, meta_val, dev, RN18_MEAN, RN18_STD,
                 tag='epoch %d' % (ep + 1))

    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    cfg_log = {'backbone': 'resnet18_at', 'exclude': args.exclude,
               'run': int(args.run), 'epochs': int(args.epochs),
               'bs': int(args.bs), 'lr': float(args.lr),
               'seed': int(args.seed), 'eps': EPS, 'at_steps': AT_STEPS}
    json.dump(cfg_log, open(cfg_path, 'w'), indent=1)
    print('MODEL_SAVED', model_path)
    print('TRAIN_DONE')


if __name__ == '__main__':
    main()
