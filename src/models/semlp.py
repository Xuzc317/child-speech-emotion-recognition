"""SE-MLP 分类器

轻量级通道门控 MLP 分类头。在新框架中不是核心贡献，
只是经过 adapter + pooling 后的自然分类汇聚。

结构: 768 → 512 → SE-gating → 256 → 128 → 6
含 BatchNorm、Dropout、SE channel gating。
"""
import torch
import torch.nn as nn


class SEBlock1D(nn.Module):
    """Channel-wise gating on (B, C) feature vectors."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class SEMLP(nn.Module):
    """轻量 SE-MLP 分类器。

    输入 (B, 768) 特征向量 → 6 类情绪输出。
    可训练参数 ~590K。
    """

    def __init__(self, input_dim=768, num_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            SEBlock1D(512),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)
