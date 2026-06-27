"""Module 1：声学校准适配器 (Acoustic Calibration Adapter)

在 frozen SSL backbone 之上加一个轻量校准层（scale + bias + gate），
补偿儿童语音在成人预训练模型特征空间中的分布偏移。

v6 结论：6:2:2 协议下 Adapter 贡献仅 ~0.5pp test，已从核心方法路径移除。
保留此模块用于消融讨论和未来的分布偏移分析。
"""

import torch
import torch.nn as nn


class AcousticCalibrationAdapter(nn.Module):
    """声学补偿型适配器（推荐 Phase 2 优先实现）。

    设计思路：
      - 儿童 F0 高（250-400 Hz vs 成人 80-200 Hz）
        → 高频通道的响应需要更强调
      - 儿童 formant 分散
        → 特定频段的 attention 需要调整
      - 不是简单地学习一个 affine 变换，而是用 gate 机制
        让每个时间步的特征根据其频谱内容做自适应校准

    输入/输出维度：(B, T, 768)

    结构：
      scale_init: 由儿童/成人统计差异估计的初始缩放向量
      bias_init:  由儿童/成人统计差异估计的初始偏置向量
      gate:       轻量 gate 网络，做非线性自适应调整

    为什么不是纯 affine：
      同一段儿童语音的不同帧，频谱内容不同，所需的校准也不同。
      gate 网络可以根据帧内容动态调整校准强度。
    """

    def __init__(self, dim=768, reduction=4, init_scale=None, init_bias=None):
        super().__init__()

        if init_scale is not None:
            self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        else:
            self.scale = nn.Parameter(torch.ones(dim))
        if init_bias is not None:
            self.bias = nn.Parameter(torch.tensor(init_bias, dtype=torch.float32))
        else:
            self.bias = nn.Parameter(torch.zeros(dim))

        # 轻量 gate: 根据帧内容做非线性调整
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(),
            nn.Linear(dim // reduction, dim),
            nn.Tanh(),  # [-1, 1] 范围，作为残差调节
        )

    def forward(self, x):
        """x: (B, T, 768)"""
        # 1. 基础校准（线性）
        out = x * self.scale.unsqueeze(0).unsqueeze(0) + \
              self.bias.unsqueeze(0).unsqueeze(0)

        # 2. 动态微调（非线性）
        gate_out = self.gate(x)  # (B, T, 768)
        out = out + 0.1 * gate_out * x  # 残差缩放，保持稳定性

        return out
