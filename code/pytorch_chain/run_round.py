#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验轮次运行器（AutoDL）
========================
设计原则：**每一轮全量重跑，不复用上一轮任何结果。**
  理由：整轮约 20 分钟（训练 6.5min + 低通 8.7min），
        用 20 分钟换一个完全独立、可对照的重复，远好过"省时间但说不清哪里变了"。

每轮产出目录结构：
  <DATA_ROOT>/runs/round_<NN>/
    ├── env.json              环境快照（GPU/驱动/torch 版本/数据校验和）
    ├── config.json           本轮全部超参
    ├── data_checksum.txt     数据集指纹（证明两轮用的是同一份数据）
    ├── detector.pth          本轮训练出的检测器
    ├── log_train.txt         训练日志
    ├── log_lowpass.txt       低通实验日志
    ├── results_lowpass.csv   主结果表
    └── summary.json          汇总（便于跨轮自动比对）

用法：
  python run_round.py 1
  python run_round.py 2 --epochs 3 --seed 42
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time

BASE = os.environ.get('DATA_ROOT', '<DATA_ROOT>')
DATA = os.path.join(BASE, 'data_npy')
RUNS = os.path.join(BASE, 'runs')
VPY = os.path.join(BASE, 'venv/bin/python')


def sh(cmd, log=None, timeout=None):
    print('$ ' + cmd, flush=True)
    t0 = time.time()
    if log:
        with open(log, 'w') as f:
            r = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
        with open(log, encoding='utf-8', errors='replace') as f:
            txt = f.read()
        print(txt[-3000:], flush=True)
    else:
        r = subprocess.run(cmd, shell=True, timeout=timeout)
    print('[rc=%d %.0fs]' % (r.returncode, time.time() - t0), flush=True)
    return r.returncode


def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def env_snapshot():
    def cap(c):
        return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
    vers = {}
    for m in ['torch', 'torchvision', 'numpy', 'pandas']:
        try:
            vers[m] = __import__(m).__version__
        except Exception:
            vers[m] = None
    return {
        'host': platform.node(),
        'python': sys.version.split()[0],
        'packages': vers,
        'gpu': cap("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
        'cuda': cap("nvcc --version 2>/dev/null | tail -2 | head -1"),
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
    }


def data_fingerprint():
    """数据集指纹：形状 + 校验和 + 分布。证明两轮用的是同一份数据。"""
    import numpy as np
    import pandas as pd
    out = {}
    for split in ['train', 'validation']:
        xp = os.path.join(DATA, 'X_%s.npy' % split)
        yp = os.path.join(DATA, 'y_%s.npy' % split)
        mp = os.path.join(DATA, '%s_meta_named.csv' % split)
        if not os.path.exists(xp):
            continue
        X = np.load(xp, mmap_mode='r')
        y = np.load(yp)
        meta = pd.read_csv(mp)
        out[split] = {
            'shape': list(X.shape),
            'dtype': str(X.dtype),
            'size_bytes': os.path.getsize(xp),
            'md5': md5(xp),
            'y_md5': md5(yp),
            'label_counts': {str(k): int(v) for k, v in pd.Series(y).value_counts().items()},
            'gen_counts': {str(k): int(v) for k, v in meta['gen_name'].value_counts().items()},
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('round', type=int, help='轮次编号，如 1')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--exclude', default=None, help='留出的生成器，如 Midjourney')
    ap.add_argument('--skip-train', action='store_true')
    ap.add_argument('--skip-lowpass', action='store_true')
    a = ap.parse_args()

    R = os.path.join(RUNS, 'round_%02d' % a.round)
    if os.path.exists(R):
        print('!! 轮次目录已存在: %s' % R)
        print('   多轮实验的原则是每轮独立，请换编号或先归档旧目录。')
        sys.exit(2)
    os.makedirs(R)
    print('=' * 66)
    print('实验轮次 %02d -> %s' % (a.round, R))
    print('=' * 66)

    cfg = {'round': a.round, 'epochs': a.epochs, 'bs': a.bs, 'lr': a.lr,
           'seed': a.seed, 'exclude': a.exclude,
           'data_dir': DATA, 'started': time.strftime('%Y-%m-%d %H:%M:%S')}
    with open(os.path.join(R, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=1, ensure_ascii=False)

    print('\n[1/4] 环境快照')
    env = env_snapshot()
    with open(os.path.join(R, 'env.json'), 'w') as f:
        json.dump(env, f, indent=1, ensure_ascii=False)
    print(json.dumps(env, indent=1, ensure_ascii=False))

    print('\n[2/4] 数据集指纹')
    fp = data_fingerprint()
    with open(os.path.join(R, 'data_checksum.txt'), 'w') as f:
        json.dump(fp, f, indent=1, ensure_ascii=False)
    for split, d in fp.items():
        print('  %-12s shape=%s md5=%s' % (split, d['shape'], d['md5'][:16]))

    ckpt = os.path.join(R, 'detector.pth')

    if not a.skip_train:
        print('\n[3/4] 训练检测器')
        cmd = ('%s -u code/pytorch_chain/train_detector.py --prep %s --out %s --epochs %d --bs %d '
               '--lr %g --seed %d' % (VPY, DATA, os.path.join(R, 'detector'),
                                      a.epochs, a.bs, a.lr, a.seed))
        if a.exclude:
            cmd += ' --exclude %s' % a.exclude
        rc = sh(cmd, log=os.path.join(R, 'log_train.txt'), timeout=10800)
        if rc != 0:
            print('!! 训练失败')
            sys.exit(3)
        if os.path.exists(os.path.join(R, 'detector.pth')):
            pass
        elif os.path.exists(os.path.join(R, 'detector.pth.pth')):
            shutil.move(os.path.join(R, 'detector.pth.pth'), ckpt)
    else:
        print('\n[3/4] 跳过训练')

    if not a.skip_lowpass:
        print('\n[4/4] 低通自适应实验')
        rc = sh('%s -u code/pytorch_chain/lowpass_round.py --ckpt %s --outdir %s' % (VPY, ckpt, R),
                log=os.path.join(R, 'log_lowpass.txt'), timeout=10800)
        if rc != 0:
            print('!! 低通实验失败')
            sys.exit(4)

    print('\n' + '=' * 66)
    print('ROUND_%02d_DONE %s' % (a.round, R))
    print('=' * 66)
    for f in sorted(os.listdir(R)):
        p = os.path.join(R, f)
        print('  %-28s %10.2f KB' % (f, os.path.getsize(p) / 1024))


if __name__ == '__main__':
    main()
