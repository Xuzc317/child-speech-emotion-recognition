"""Module 1：分布校准适配器 (Distribution Calibration Adapter)

核心思想：
  emotion2vec/wav2vec 在成人数据上预训练，儿童语音在其隐空间中
  存在系统性分布偏移。我们不 fine-tune 整个 backbone，而是在
  frozen SSL 之上加一个轻量校准层，显式补偿已知的声学差异。

与 LoRA 的区别：
  LoRA 是通用参数高效微调，在目标数据上直接拟合；
  我们的适配器以可解释的方式补偿已知的声学偏移（F0、formant），
  第一个 scale/bias 参数由儿童语音统计先验初始化，而非随机初始化。

整体校准策略——有两种设计方向，Phase 2 时需要实验对比：
  A. 声学补偿型 (AcousticCalibrationAdapter) —— 用统计先验初始化
  B. 分布对齐型 (DomainAdversarialAdapter) —— 对抗训练 + 域混淆

两个方向可以独立实现，也可以串联使用。
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


class DomainAdversarialAdapter(nn.Module):
    """分布对齐型适配器（备选，需要成人对比数据）。

    核心思想：
      用一个 domain classifier（成人 vs 儿童）来判断适配器的输出
      是否还保留着域信息。训练目标：
        - 适配器：最大化 domain classifier 的 loss（让分不出儿童/成人）
        - domain classifier：最小化分类 loss

      当 domain classifier 无法区分时，说明适配器的输出已经
      "域无关"，分布偏移被校准了。

    依赖条件：
      - 需要有成人语音数据（如 IEMOCAP 或 RAVDESS 的样本）
      - 训练时多一个对抗 loss，调参更复杂

    建议：Phase 2 内先实现 AcousticCalibrationAdapter，
    如果效果不够再用 DomainAdversarialAdapter。
    """

    def __init__(self, dim=768):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        # 残差连接
        self.residual = True

    def forward(self, x):
        out = self.adapter(x)
        if self.residual:
            out = x + 0.1 * out  # 残差 + 缩放
        return out
