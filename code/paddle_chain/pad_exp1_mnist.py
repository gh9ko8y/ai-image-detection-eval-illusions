#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Paddle 实验 1 复现：MNIST 闭环 + FGSM/PGD/噪声（原始 week1 脚本 + 攻击节）。
输出：runs/paddle_01/exp1_mnist.log 关键数字。"""
import time

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.io import DataLoader
from paddle.vision.datasets import MNIST
from paddle.vision.transforms import Compose, ToTensor, Normalize

paddle.seed(42)
np.random.seed(42)

OUT = os.path.join(os.environ.get('DATA_ROOT', '<DATA_ROOT>'), 'runs/paddle_01/exp1_mnist.pdparams')


class SimpleCNN(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = paddle.nn.Conv2D(1, 32, 3, padding=1)
        self.conv2 = paddle.nn.Conv2D(32, 64, 3, padding=1)
        self.pool = paddle.nn.MaxPool2D(2)
        self.fc1 = paddle.nn.Linear(64 * 7 * 7, 128)
        self.fc2 = paddle.nn.Linear(128, 10)
        self.dropout = paddle.nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = paddle.flatten(x, start_axis=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def main():
    transform = Compose([ToTensor(), Normalize(mean=[0.1307], std=[0.3081], data_format='CHW')])
    test_set = MNIST(mode='test', transform=transform, download=True)
    model = SimpleCNN()
    model.set_state_dict(paddle.load(OUT))
    model.eval()
    loader = DataLoader(test_set, batch_size=512, shuffle=False)

    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    X = paddle.concat(xs)
    Y = paddle.concat(ys).astype('int64').reshape([-1])
    print('test set:', X.shape)

    with paddle.no_grad():
        acc0 = float((model(X).argmax(1).numpy() == Y.numpy()).mean())
    print('干净准确率: %.2f%%' % (acc0 * 100))

    eps = 0.3
    # 噪声对照
    with paddle.no_grad():
        xn = X + paddle.normal(0.0, eps, X.shape)
        accn = float((model(xn).argmax(1).numpy() == Y.numpy()).mean())
    print('随机噪声(σ=%.2f): %.2f%%' % (eps, accn * 100))

    # FGSM（标准化空间；MNIST 归一化后范围约 [-0.42, 2.82]，clip 到 [-3,3] 足够）
    Xa = X.detach()
    Xa.stop_gradient = False
    loss = F.cross_entropy(model(Xa), Y)
    g = paddle.grad([loss], [Xa])[0]
    xfg = paddle.clip(Xa + eps * g.sign(), -3.0, 3.0).detach()
    with paddle.no_grad():
        accf = float((model(xfg).argmax(1).numpy() == Y.numpy()).mean())
    print('FGSM(ε=%.2f): %.2f%%' % (eps, accf * 100))

    # PGD 10 步
    xa = X.detach()
    for _ in range(10):
        xa = xa.detach()
        xa.stop_gradient = False
        loss = F.cross_entropy(model(xa), Y)
        g = paddle.grad([loss], [xa])[0]
        xa = paddle.clip(xa.detach() + 0.075 * g.sign(), -3.0, 3.0)
        xa = paddle.maximum(paddle.minimum(xa, X + eps), X - eps)
    with paddle.no_grad():
        accp = float((model(xa).argmax(1).numpy() == Y.numpy()).mean())
    print('PGD-10(ε=%.2f): %.2f%%' % (eps, accp * 100))
    print('PAD_EXP1_DONE')


def train():
    transform = Compose([ToTensor(), Normalize(mean=[0.1307], std=[0.3081], data_format='CHW')])
    tr = MNIST(mode='train', transform=transform, download=True)
    te = MNIST(mode='test', transform=transform, download=True)
    tl = DataLoader(tr, batch_size=128, shuffle=True, num_workers=2)
    el = DataLoader(te, batch_size=512, shuffle=False, num_workers=2)
    model = SimpleCNN()
    opt = paddle.optimizer.Adam(learning_rate=1e-3, parameters=model.parameters())
    for ep in range(5):
        model.train()
        t0 = time.time()
        for x, y in tl:
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            opt.step()
            opt.clear_grad()
        model.eval()
        c = n = 0
        with paddle.no_grad():
            for x, y in el:
                c += int((model(x).argmax(1).numpy() == y.numpy().reshape([-1])).sum())
                n += len(y)
        print('epoch %d test acc %.2f%% (%.0fs)' % (ep + 1, 100 * c / n, time.time() - t0), flush=True)
    paddle.save(model.state_dict(), OUT)
    print('saved', OUT)


if __name__ == '__main__':
    import os
    os.makedirs(os.path.join(os.environ.get('DATA_ROOT', '<DATA_ROOT>'), 'runs/paddle_01'), exist_ok=True)
    if not os.path.exists(OUT):
        train()
    main()
