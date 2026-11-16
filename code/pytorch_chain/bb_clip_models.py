#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实验16 骨干扩展第二批 · CLIP ViT-B/16 模块A：建模与常数。

注意：
1. openai 权重必须 quick_gelu=True（否则激活函数不匹配，特征劣化）
2. CLIP 归一化常数与 ImageNet 不同（std 不是 0.225 系）
3. 全模型微调口径（与 ResNet/EfficientNet 一致），分类头接在 512 维图像特征上
"""
import numpy as np
import torch
import torch.nn as nn

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype='float32').reshape(1, 3, 1, 1)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype='float32').reshape(1, 3, 1, 1)


class CLIPDetector(nn.Module):
    """CLIP ViT-B/16 visual 编码器（全量微调）+ 线性分类头。"""

    def __init__(self):
        super().__init__()
        import open_clip
        m, _, _ = open_clip.create_model_and_transforms(
            'ViT-B-16', pretrained='openai', force_quick_gelu=True)
        self.visual = m.visual
        self.head = nn.Linear(512, 2)

    def forward(self, x):
        feat = self.visual(x).float()
        return self.head(feat)


def build_clip():
    return CLIPDetector()
