"""DrseNet 分类器

在新框架中不是核心贡献，只作为"自然汇聚"的分类模块。
相比原版做了以下调整：
  - 输入维度改为 768（SSL 特征经 adapter + pooling 后）
  - 移除了 AdaptiveAvgPool1d（ONNX 不兼容）
  - 简化结构，减少参数量（分类器不需要太复杂）

注意：这个 DrseNet 的输入是 (B, 768) 经过 adapter + pooling 后的特征向量，
而非原始论文中的 (B, 1, 162)。它只做分类，不做特征提取。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock1D(nn.Module):
    """简化的 channel-wise gating 模块。

    注意：由于 DrseCNN 的输入是 (B, C) 特征向量而非 (B, C, T) 时序特征，
    不存在空间维度可 squeeze，因此此处省略了标准 SE 的 global pooling 步骤，
    等价于 sigmoid(Linear(ReLU(Linear(x)))) ⊙ x 的 learnable gating。
    在论文中不作为创新点，仅作为分类器的激活正则化手段。
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C) — 特征向量
        y = self.fc(x)
        return x * y


class DrseCNN(nn.Module):
    """轻量 DrseNet 分类器。

    相比原版：
      - 降级为纯分类器（特征提取由 SSL backbone + adapter + pooling 完成）
      - 输入维度 768 → 分类输出
      - 保留 SE 模块是因为其本身有消融价值
      - 参数量：~1M（原版 ~6.3M）

    注意：
      这个模型在论文中只需要 1-2 段描述，不需要作为创新点展开。
      消融实验里可以作为"基线分类器"出现。
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
        # x: (B, 768)
        return self.net(x)
