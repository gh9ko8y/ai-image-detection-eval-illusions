#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实验15 骨干扩展 · 模块A：模型构建与常数（无任何文件IO）。"""
import numpy as np
import torch.nn as nn
import torchvision

RN18_MEAN = np.array([0.485, 0.456, 0.406], dtype='float32').reshape(1, 3, 1, 1)
RN18_STD = np.array([0.229, 0.224, 0.225], dtype='float32').reshape(1, 3, 1, 1)


def build_resnet18():
    m = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m


def build_efficientnet_b0():
    m = torchvision.models.efficientnet_b0(
        weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    return m


def build_clip():
    raise SystemExit('CLIP 需先安装 open_clip_torch 后在此实现')


BACKBONES = {
    'resnet18': (build_resnet18, RN18_MEAN, RN18_STD),
    'efficientnet_b0': (build_efficientnet_b0, RN18_MEAN, RN18_STD),
    'clip': (build_clip, None, None),
}

GENS = ['ADM', 'BigGAN', 'GLIDE', 'Midjourney', 'SD15', 'VQDM', 'Wukong', 'Real']
