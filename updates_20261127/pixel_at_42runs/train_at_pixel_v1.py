#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_at_pixel_v1 —— AT_PIXEL_V1 主协议训练器（SPEC-V1.1, 2026-11-18）

契约: P0_审计_20261118/spec_train_at_pixel_v1.md。冻结后不得修改; 修改=V2+重跑。

与旧 at_main.py 的区别（执行书第 3 节全部落实）:
  像素空间 PGD(eps=8/255, alpha=2/255, 5步, 随机起点, 双投影) / 归一化在模型
  内部恰好一次 / 每批先 zero_grad 再攻击 / 内层 eval+参数冻结+仅输入梯度+BN快照 /
  外层 [x,x]/[x,x_adv] 128项 0.5CE+0.5CE 单次 BN 遍历 / 训练零 validation 接触 /
  显式 --prep 禁自动发现 / RNG 三流分离 / 逐 batch 样本 ID 哈希(sha256) /
  分数定义 s=sigmoid(l_fake-l_real) 由 eval_frozen_v1 使用。

用法:
  python train_at_pixel_v1.py --prep DIR --out-root DIR --exclude ADM \
      --seed 0 --method at [--smoke-batches 2] [--resume]
"""
import argparse
import csv
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

SPEC = 'SPEC-V1.1 2026-11-18'
GENS = ['ADM', 'BigGAN', 'GLIDE', 'Midjourney', 'SD15', 'VQDM', 'Wukong']
EPS = 8.0 / 255.0
ALPHA = 2.0 / 255.0
STEPS = 5

# 历史数据集指纹(与三轮执行 data_checksum 对齐的记录口径)
EXPECT_LEGACY_DIGEST = {
    'X_train.npy': '060071725e58b4b053c821f766064c25',
    'y_train.npy': 'c4301f1fa4784a258f472e214dbe3625',
}
EXPECT_N = 28000


def _digest(path, algo):
    h = hashlib.new(algo)
    fh = open(path, 'rb')
    while True:
        blk = fh.read(1 << 22)
        if not blk:
            break
        h.update(blk)
    fh.close()
    return h.hexdigest()


def _digest_int64(arr):
    h = hashlib.sha256()
    h.update(arr.astype(np.int64).tobytes())
    return h.hexdigest()[:16]


class Normalize(nn.Module):
    """ImageNet 归一化作为模型内部层(恰好一次)。buffer 进 state_dict。"""

    def __init__(self):
        super().__init__()
        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


def build_model():
    core = models.resnet18(weights=None)
    core.fc = nn.Linear(core.fc.in_features, 2)
    return nn.Sequential(Normalize(), core)


def load_data(prep, exclude):
    X = np.load(os.path.join(prep, 'X_train.npy'), mmap_mode='r')
    y = np.load(os.path.join(prep, 'y_train.npy'), mmap_mode='r')
    meta = pd.read_csv(os.path.join(prep, 'train_meta_named.csv'))
    assert len(X) == EXPECT_N and len(y) == EXPECT_N and len(meta) == EXPECT_N
    real_mask = (meta['gen_name'] == 'Real').values
    assert set(np.unique(y[real_mask])) == {0}, 'Real 标签非 0, 停'
    assert set(np.unique(y[~real_mask])) == {1}, 'Fake 标签非 1, 停'
    if exclude is not None:
        keep = (meta['gen_name'] != exclude).values
        n = int(keep.sum())
        assert n == 26000, 'LOO 后应为 26000, 实际 %d, 停' % n
        idx_all = np.where(keep)[0]
    else:
        n = EXPECT_N
        idx_all = np.arange(n)
    return X, y, idx_all, n


def bn_snapshot(model):
    out = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            out.append((m.running_mean.clone(), m.running_var.clone(),
                        m.num_batches_tracked.clone()))
    return out


def bn_equal(a, b):
    for (m1, v1, t1), (m2, v2, t2) in zip(a, b):
        if not (torch.equal(m1, m2) and torch.equal(v1, v2)
                and torch.equal(t1, t2)):
            return False
    return True


def pgd5(model, x, y, gen):
    for p in model.parameters():
        assert p.grad is None, '进入攻击前参数 .grad 非 None(时序错误)'
    was_training = model.training
    model.eval()
    saved = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in saved:
        p.requires_grad_(False)
    bn_before = bn_snapshot(model)
    delta = (torch.rand_like(x, generator=gen) * 2 - 1) * EPS
    x_adv = torch.clamp(x + delta, 0.0, 1.0).detach()
    for _ in range(STEPS):
        x_adv = x_adv.detach().requires_grad_(True)
        with torch.enable_grad():
            loss = F.cross_entropy(model(x_adv), y)
            g, = torch.autograd.grad(loss, x_adv)
        assert torch.isfinite(g).all(), '攻击梯度非有限值'
        with torch.no_grad():
            x_adv = torch.clamp(
                torch.min(torch.max(x_adv + ALPHA * g.sign(), x - EPS),
                          x + EPS), 0.0, 1.0)
        max_dev = (x_adv - x).abs().max().item()
        assert max_dev <= EPS + 1e-6, '超出 L_inf 球: %.3e' % max_dev
    for p, flag in saved:
        p.requires_grad_(flag)
    if was_training:
        model.train()
    assert bn_equal(bn_before, bn_snapshot(model)), '攻击阶段 BN buffer 变化'
    return x_adv.detach()


def out_path(out_dir, name):
    return out_dir + '/' + name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prep', required=True, help='显式数据目录(禁自动发现)')
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--exclude', default=None, choices=GENS)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--method', required=True, choices=['clean_dup', 'at'])
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--smoke-batches', type=int, default=0)
    ap.add_argument('--save-every', type=int, default=0,
                    help='>0 时每 N 批写 resume_state(批级断点恢复测试用)')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()

    excl = args.exclude or 'full'
    out = out_path(out_path(out_path(out_path(
        args.out_root, 'AT_PIXEL_V1'), excl), 'seed_%d' % args.seed),
        args.method)
    os.makedirs(out, exist_ok=True)
    smoke = args.smoke_batches > 0

    if not args.resume:
        cfg = {'spec': SPEC, 'code_sha256': _digest(__file__, 'sha256'),
               'prep': args.prep, 'exclude': args.exclude, 'seed': args.seed,
               'method': args.method, 'epochs': args.epochs, 'bs': args.bs,
               'lr': args.lr, 'eps': EPS, 'alpha': ALPHA, 'steps': STEPS,
               'smoke_batches': args.smoke_batches,
               'env': {'torch': torch.__version__,
                       'gpu': torch.cuda.get_device_name(0)
                       if torch.cuda.is_available() else 'CPU'}}
        with open(out_path(out, 'config.json'), 'w') as f:
            json.dump(cfg, f, indent=1)

    got_x = _digest(args.prep + '/X_train.npy', 'md5')
    got_y = _digest(args.prep + '/y_train.npy', 'md5')
    assert got_x == EXPECT_LEGACY_DIGEST['X_train.npy'], 'X_train 指纹不符, 停'
    assert got_y == EXPECT_LEGACY_DIGEST['y_train.npy'], 'y_train 指纹不符, 停'
    X, y, idx_all, n = load_data(args.prep, args.exclude)
    steps_per_epoch = (n + args.bs - 1) // args.bs
    if not smoke:
        assert steps_per_epoch == 407, 'steps/epoch 应 407, 实际 %d' % steps_per_epoch
    print('data ok: n=%d steps/epoch=%d' % (n, steps_per_epoch), flush=True)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    shuffle_rng = np.random.default_rng(args.seed)
    atk_gen = torch.Generator(device=dev).manual_seed(args.seed + 2000)
    perms = [shuffle_rng.permutation(n) for _ in range(args.epochs)]

    def batch_ids(ep):
        ids = []
        for b in range(steps_per_epoch):
            sel = idx_all[perms[ep][b * args.bs:(b + 1) * args.bs]]
            ids.append(_digest_int64(sel))
        return ids

    all_batch_ids = [batch_ids(e) for e in range(args.epochs)]

    init_path = out_path(out, 'init_state_dict.pth')
    if args.resume and os.path.exists(init_path):
        init_sd = torch.load(init_path, map_location='cpu', weights_only=True)
    else:
        gstate = torch.get_rng_state()
        torch.manual_seed(args.seed + 1000)
        m0 = build_model()
        init_sd = {k: v.clone() for k, v in m0.state_dict().items()}
        torch.set_rng_state(gstate)
        torch.save(init_sd, init_path)
    with open(out_path(out, 'init_sha256.txt'), 'w') as f:
        f.write(_digest(init_path, 'sha256'))

    model = build_model()
    res = model.load_state_dict(init_sd, strict=True)
    assert not res.missing_keys and not res.unexpected_keys
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                           eps=1e-8, weight_decay=0, amsgrad=False)

    start_ep, start_step = 0, 0
    log_rows = []
    log_path = out_path(out, 'train_log.csv')
    p_resume = out_path(out, 'resume_state.pt')
    if args.resume and os.path.exists(p_resume):
        st = torch.load(p_resume, map_location=dev, weights_only=True)
        model.load_state_dict(st['model'])
        opt.load_state_dict(st['opt'])
        start_ep, start_step = int(st['epoch']), int(st['step'])
        atk_gen.set_state(st['atk_rng'].cpu() if dev == 'cuda' else st['atk_rng'])
        with open(log_path) as f:
            log_rows = list(csv.DictReader(f))
        print('resumed at ep%d step%d' % (start_ep, start_step), flush=True)

    t0 = time.time()
    max_mem = 0
    n_updates = 0
    done = False
    for ep in range(start_ep, args.epochs):
        step0 = start_step if ep == start_ep else 0
        for b in range(step0, steps_per_epoch):
            sel = idx_all[perms[ep][b * args.bs:(b + 1) * args.bs]]
            xb = torch.from_numpy(
                np.ascontiguousarray(X[sel]).astype(np.float32) / 255.0
            ).permute(0, 3, 1, 2).contiguous().to(dev)
            yb = torch.from_numpy(
                np.ascontiguousarray(y[sel]).astype(np.int64)).to(dev)
            assert xb.min() >= -1e-6 and xb.max() <= 1 + 1e-6

            opt.zero_grad(set_to_none=True)
            if args.method == 'at':
                x_adv = pgd5(model, xb, yb, atk_gen)
                inp = torch.cat([xb, x_adv], 0)
            else:
                inp = torch.cat([xb, xb], 0)
            logits = model(inp)
            loss = 0.5 * F.cross_entropy(logits[:len(yb)], yb) \
                 + 0.5 * F.cross_entropy(logits[len(yb):], yb)
            assert torch.isfinite(loss), 'loss 非有限值'
            loss.backward()
            opt.step()
            n_updates += 1
            with torch.no_grad():
                acc = (logits.argmax(1)[:len(yb)] == yb).float().mean().item()
            log_rows.append({'epoch': ep + 1, 'step': b + 1,
                             'batch_id': all_batch_ids[ep][b],
                             'loss': float(loss.item()), 'train_acc': acc,
                             'sec': round(time.time() - t0, 1)})
            if b % 50 == 0:
                print('ep%d %d/%d loss=%.4f acc=%.3f' % (
                    ep + 1, b + 1, steps_per_epoch, loss.item(), acc),
                    flush=True)
            if dev == 'cuda':
                max_mem = max(max_mem, torch.cuda.max_memory_allocated())
            if args.save_every and (b + 1) % args.save_every == 0:
                torch.save({'model': model.state_dict(),
                            'opt': opt.state_dict(),
                            'epoch': ep, 'step': b + 1,
                            'atk_rng': atk_gen.get_state()}, p_resume)
                with open(log_path, 'w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=[
                        'epoch', 'step', 'batch_id', 'loss', 'train_acc', 'sec'])
                    w.writeheader()
                    w.writerows(log_rows)
            if smoke and b + 1 >= args.smoke_batches:
                done = True
                break
        if smoke and done:
            break
        torch.save(model.state_dict(), out_path(out, 'epoch%d.pth' % (ep + 1)))
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                    'epoch': ep + 1, 'step': 0,
                    'atk_rng': atk_gen.get_state()}, p_resume)
        with open(log_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=[
                'epoch', 'step', 'batch_id', 'loss', 'train_acc', 'sec'])
            w.writeheader()
            w.writerows(log_rows)
        print('== epoch %d done, ckpt saved' % (ep + 1), flush=True)

    if smoke:
        with open(log_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=[
                'epoch', 'step', 'batch_id', 'loss', 'train_acc', 'sec'])
            w.writeheader()
            w.writerows(log_rows)
    if not smoke:
        p_final = out_path(out, 'final_ckpt.pth')
        torch.save(model.state_dict(), p_final)
        with open(out_path(out, 'final_sha256.txt'), 'w') as f:
            f.write(_digest(p_final, 'sha256'))
    with open(out_path(out, 'resource.json'), 'w') as f:
        json.dump({'method': args.method, 'updates': n_updates,
                   'elapsed_sec': round(time.time() - t0, 1),
                   'max_mem_alloc_bytes': int(max_mem),
                   'smoke': smoke, 'deterministic_mode': True}, f, indent=1)
    print('SMOKE_DONE' if smoke else 'TRAIN_DONE',
          'updates=%d' % n_updates, flush=True)


if __name__ == '__main__':
    main()
