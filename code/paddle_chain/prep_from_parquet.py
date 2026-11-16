#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""复现专用：从本地 data_raw parquet 分片重导 npy（不走网络）。

与历史 prep_dataset.py 完全同构的解码链：
  image.bytes -> PIL open -> convert('RGB') -> resize(224, BICUBIC) -> uint8 NHWC
行序 = 分片序（与 datasets load_dataset 一致）。
输出与历史同名文件到指定目录，并打印 md5 供与指纹比对。

用法：python prep_from_parquet.py --out <DATA_ROOT>/data_npy
"""
import argparse
import glob
import hashlib
import io
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

RAW = os.path.join(os.environ.get('DATA_ROOT', '<DATA_ROOT>'), 'data_raw')
GEN_NAMES = {0: 'Real', 1: 'ADM', 2: 'BigGAN', 3: 'GLIDE',
             4: 'Midjourney', 6: 'SD15', 7: 'VQDM', 8: 'Wukong'}


def md5_of(path, chunk=1 << 22):
    h = hashlib.md5()
    h2 = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
            h2.update(b)
    return h.hexdigest(), h2.hexdigest()[:16]


def build(split, outdir):
    shards = sorted(glob.glob(os.path.join(RAW, split + '-*.parquet')))
    assert shards, 'no shards for ' + split
    rows_total = sum(pq.ParquetFile(s).metadata.num_rows for s in shards)
    print('=== %s: %d shards, %d rows ===' % (split, len(shards), rows_total), flush=True)

    X = np.zeros((rows_total, 224, 224, 3), dtype=np.uint8)
    y = np.zeros((rows_total,), dtype=np.int64)
    g = np.zeros((rows_total,), dtype=np.int64)
    i = 0
    for sp in shards:
        tbl = pq.read_table(sp)
        imgs = tbl.column('image').to_pylist()
        labels = tbl.column('label').to_pylist()
        gens = tbl.column('generator').to_pylist()
        for rec, lb, ge in zip(imgs, labels, gens):
            im = Image.open(io.BytesIO(rec['bytes'])).convert('RGB')
            im = im.resize((224, 224), Image.BICUBIC)
            X[i] = np.asarray(im, dtype=np.uint8)
            y[i] = int(lb)
            g[i] = int(ge)
            i += 1
            if i % 2000 == 0:
                print('  %d/%d' % (i, rows_total), flush=True)
    assert i == rows_total

    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, 'X_%s.npy' % split), X)
    np.save(os.path.join(outdir, 'y_%s.npy' % split), y)
    meta = pd.DataFrame({
        'label': y,
        'generator': g,
        'gen_name': [GEN_NAMES.get(int(v), 'UNK_%d' % v) for v in g],
    })
    meta.to_csv(os.path.join(outdir, '%s_meta_named.csv' % split), index=False)
    xp = os.path.join(outdir, 'X_%s.npy' % split)
    yp = os.path.join(outdir, 'y_%s.npy' % split)
    xm, xs = md5_of(xp)
    ym, _ = md5_of(yp)
    print('%s X shape %s | md5(X)=%s | sha256_16(X)=%s | md5(y)=%s' % (
        split, X.shape, xm, xs, ym), flush=True)
    print('%s 分布: %s' % (split, dict(meta['gen_name'].value_counts())), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    build('validation', args.out)
    build('train', args.out)
    print('PREP_FROM_PARQUET_DONE')


if __name__ == '__main__':
    main()
