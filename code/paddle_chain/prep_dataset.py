#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tiny-GenImage -> npy 预处理（AutoDL 版，用 hf-mirror）
======================================================
目的：不依赖从 AI Studio 上传 5GB 数据，直接在 AutoDL 上重建与历史完全一致的
      X_val/y_val（NHWC uint8 未归一化，224x224）+ 元数据 CSV。

与 AI Studio 历史做法保持一致：
  - HF_ENDPOINT=https://hf-mirror.com（必须在 import datasets 之前设好）
  - repo: TheKernel01/Tiny-GenImage
  - resize 224x224 -> npy（NHWC、uint8、未归一化）
  - 另存 val_meta_named.csv（label, generator, gen_name），用于逐生成器评估

用法：
  HF_ENDPOINT=https://hf-mirror.com python prep_dataset.py            # 只做验证集
  HF_ENDPOINT=https://hf-mirror.com python prep_dataset.py --full     # 训练集+验证集
"""
import os
import sys
import io
import numpy as np
import pandas as pd
from PIL import Image

OUT_BASE = os.environ.get('DATA_ROOT', '<DATA_ROOT>')

GEN_NAMES = {0: 'Real', 1: 'ADM', 2: 'BigGAN', 3: 'GLIDE',
             4: 'Midjourney', 6: 'SD15', 7: 'VQDM', 8: 'Wukong'}


def build(split, outdir):
    from datasets import load_dataset
    print('=== 加载 %s split ===' % split, flush=True)
    ds = load_dataset('TheKernel01/Tiny-GenImage', split=split)
    print('样例数:', len(ds), '| 特征:', ds.features, flush=True)

    n = len(ds)
    # 先探测第一张图确定尺寸
    first = np.array(ds[0]['image'].convert('RGB').resize((224, 224), Image.BICUBIC))
    H, W, C = first.shape
    X = np.zeros((n, H, W, C), dtype=np.uint8)
    y = np.zeros((n,), dtype=np.int64)
    g = np.zeros((n,), dtype=np.int64)
    X[0] = first
    y[0] = int(ds[0]['label'])
    g[0] = int(ds[0]['generator'])

    for i in range(1, n):
        im = ds[i]['image'].convert('RGB').resize((224, 224), Image.BICUBIC)
        X[i] = np.array(im)
        y[i] = int(ds[i]['label'])
        g[i] = int(ds[i]['generator'])
        if (i + 1) % 1000 == 0:
            print('  已处理 %d/%d' % (i + 1, n), flush=True)

    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, 'X_%s.npy' % split), X)
    np.save(os.path.join(outdir, 'y_%s.npy' % split), y)
    meta = pd.DataFrame({
        'label': y,
        'generator': g,
        'gen_name': [GEN_NAMES.get(int(v), 'UNK_%d' % v) for v in g],
    })
    meta.to_csv(os.path.join(outdir, '%s_meta_named.csv' % split), index=False)

    print('--- %s 分布 ---' % split)
    print('shape', X.shape, X.dtype, '| label', dict(pd.Series(y).value_counts()))
    print(dict(meta['gen_name'].value_counts()))
    print('已保存到', outdir, flush=True)


if __name__ == '__main__':
    full = '--full' in sys.argv
    ts = pd.Timestamp.now().strftime('prep_%Y%m%d_%H%M')
    outdir = os.path.join(OUT_BASE, ts)
    print('输出目录:', outdir)
    print('HF_ENDPOINT =', os.environ.get('HF_ENDPOINT'))
    build('validation', outdir)
    if full:
        build('train', outdir)
    print('PREP_DONE', outdir)
