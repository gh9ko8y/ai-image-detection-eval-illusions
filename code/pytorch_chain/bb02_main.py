#!/usr/bin/env python
import argparse
import json
import os

import numpy as np
import torch

from bb_models import GENS
from bb_clip_models import CLIP_MEAN, CLIP_STD, build_clip
from bb_train_core import evaluate, load_train, load_val, train_loop

RUNS_ROOT = os.path.join(os.environ.get('DATA_ROOT', '<DATA_ROOT>'), 'runs')
EXPERIMENT = 'bb02'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exclude', default=None, choices=[None] + GENS)
    ap.add_argument('--run', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    tag = 'clip_vitb16' + ('_holdout_' + str(args.exclude) if args.exclude else '_full')
    tag = tag + '_r' + str(int(args.run))
    out_dir = os.path.join(RUNS_ROOT, EXPERIMENT)
    model_path = os.path.join(out_dir, 'det_' + tag + '.pth')
    cfg_path = os.path.join(out_dir, 'cfg_' + tag + '.json')

    mean, std = CLIP_MEAN, CLIP_STD
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    X_val, y_val, meta_val = load_val()
    X_train, y_train, meta_train = load_train()
    if args.exclude:
        keep = (meta_train['gen_name'] != args.exclude).values
        train_idx = np.where(keep)[0]
        print('留出 %s: 训练集 %d -> %d' % (args.exclude, len(y_train), len(train_idx)))
    else:
        train_idx = np.arange(len(y_train))

    model = build_clip().to(dev)
    train_loop(model, X_train, y_train, train_idx, dev, mean, std,
               epochs=args.epochs, bs=args.bs, lr=args.lr, seed=args.seed)
    evaluate(model, X_val, y_val, meta_val, dev, mean, std, tag='final ' + tag)

    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    json.dump({'backbone': 'clip_vitb16', 'exclude': args.exclude,
               'run': args.run, 'epochs': args.epochs, 'bs': args.bs,
               'lr': args.lr, 'seed': args.seed},
              open(cfg_path, 'w'), indent=1)
    print('MODEL_SAVED', model_path)
    print('TRAIN_DONE')


if __name__ == '__main__':
    main()
