#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LOO 结构化评估：加载留出重训的检测器，按生成器评估，输出机器可读 JSON + 增量 CSV。
与 train_detector.py 同目录运行（复用其预处理/建模函数，保证口径完全一致）。

用法：
  python loo_eval.py --prep <DATA_ROOT>/data_npy --ckpt detector_Midjourney_r1.pth \
      --held-out Midjourney --run 1 --outdir <DATA_ROOT>/runs/loo_01

输出：
  eval_<GEN>_r<N>.json   该配置的完整逐生成器结果
  loo_acc.csv            准确率矩阵（增量追加，每完成一个配置立刻落盘）
  loo_realrate.csv       "预测为真实"比例矩阵（验证崩塌是否=全猜真）
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

import train_detector as td

GENS = ['ADM', 'BigGAN', 'GLIDE', 'Midjourney', 'SD15', 'VQDM', 'Wukong', 'Real']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prep', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--held-out', dest='held_out', required=True)
    ap.add_argument('--run', type=int, default=1)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    X_val = np.load(os.path.join(args.prep, 'X_validation.npy'), mmap_mode='r')
    y_val = np.load(os.path.join(args.prep, 'y_validation.npy')).reshape(-1)
    meta = pd.read_csv(os.path.join(args.prep, 'validation_meta_named.csv'))

    model = td.build_model(2)
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu'), strict=False)
    model.to(dev).eval()

    preds = []
    with torch.no_grad():
        for i in range(0, len(X_val), 200):
            xb = td.to_tensor(np.asarray(X_val[i:i + 200])).to(dev)
            preds.append(model(xb).argmax(1).cpu().numpy())
    pred = np.concatenate(preds)

    # 真实图的标签值（0/1）从数据本身确定，不假设
    real_label = int(pd.Series(y_val[(meta['gen_name'] == 'Real').values]).mode().iloc[0])

    acc = {'overall': float((pred == y_val).mean() * 100)}
    realrate = {'overall': float((pred == real_label).mean() * 100)}
    for g in GENS:
        m = (meta['gen_name'] == g).values
        acc[g] = float((pred[m] == y_val[m]).mean() * 100)
        realrate[g] = float((pred[m] == real_label).mean() * 100)

    payload = {'held_out': args.held_out, 'run': args.run, 'ckpt': args.ckpt,
               'real_label': real_label, 'n_val': int(len(y_val)),
               'acc': acc, 'pred_real_rate': realrate}
    jp = os.path.join(args.outdir, 'eval_%s_r%d.json' % (args.held_out, args.run))
    json.dump(payload, open(jp, 'w'), indent=1)
    print('JSON ->', jp)
    print('held_out=%s run=%d real_label=%d overall=%.2f' % (
        args.held_out, args.run, real_label, acc['overall']))
    print('acc:', {g: round(acc[g], 2) for g in GENS})
    print('pred_real_rate:', {g: round(realrate[g], 2) for g in GENS})

    # 增量 CSV：崩了也只丢当前配置，已完成的都在
    for name, d in [('loo_acc.csv', acc), ('loo_realrate.csv', realrate)]:
        p = os.path.join(args.outdir, name)
        if not os.path.exists(p):
            with open(p, 'w') as f:
                f.write('held_out,run,overall,' + ','.join(GENS) + '\n')
        with open(p, 'a') as f:
            f.write('%s,%d,%.2f,' % (args.held_out, args.run, d['overall'])
                    + ','.join('%.2f' % d[g] for g in GENS) + '\n')
    print('EVAL_DONE')


if __name__ == '__main__':
    main()
