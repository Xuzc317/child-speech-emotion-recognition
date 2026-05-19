"""Module 2：时序重要性池化 (Temporal Importance Pooling)

核心思想：
  儿童语音的韵律变异性比成人更大（F0 range 宽、语速波动大），
  不同帧的情感信息密度不均匀。用儿童韵律特征来引导哪些帧更重要，
  替代现有方法中的 mean pooling 或 self-attention pooling。

与标准 attention pooling 的区别：
  - 标准方法：attention weights 仅由特征本身决定（纯粹数据驱动）
  - 我们的方法：显式注入韵律先验（F0、energy），可解释性更强
  - 例：F0 突变的帧 = 可能包含情绪转折 → 权重升高
  - 例：F0 平坦的帧 = 可能为中性/过渡 → 权重降低

输入：
  - ssl_feats: (B, T, 768)    emotion2vec 帧级特征
  - f0:        (B, T, 1)      帧级 F0 曲线（通过 librosa 提取）
  - energy:    (B, T, 1)      帧级 RMS 能量/

输出：
  - pooled: (B, 768)          加权融合后的全局特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np


def extract_prosody(waveform, sr=16000, hop_length=320):
    """从波形提取帧级韵律特征。

    Args:
        waveform: (T_wav,) numpy array
        sr: 采样率
        hop_length: 帧移（对应 SSL 模型的帧移，emotion2vec 为 20ms = 320 samples @ 16kHz）
    Returns:
        f0: (T_frames,) F0 曲线（未 voiced 帧为 0）
        energy: (T_frames,) RMS 能量
    """
    # F0 — 使用 librosa.yin（比 pyin 快 10x，精度略低但足够）
    f0 = librosa.yin(
        waveform, sr=sr,
        fmin=librosa.note_to_hz('C2'),   # ~65 Hz
        fmax=librosa.note_to_hz('C7'),   # ~2093 Hz（覆盖儿童高频）
        hop_length=hop_length,
    )
    f0 = np.nan_to_num(f0, nan=0.0)

    # RMS energy
    energy = librosa.feature.rms(y=waveform, hop_length=hop_length)
    energy = energy.squeeze(0)  # (T_frames,)

    # 对齐帧数
    min_len = min(len(f0), len(energy))
    f0 = f0[:min_len]
    energy = energy[:min_len]

    return f0, energy


class TemporalImportancePooling(nn.Module):
    """时序重要性池化。

    Phase 2 分两步实现：
      Step 1: 只做韵律引导的 attention（主要创新点）
      Step 2: 可选加入帧级 confidence 校准

    结构：
      1. 韵律特征投影 (prosody_proj)
         - 输入：F0 + energy（每帧 2 维）
         - 输出：64 维韵律嵌入
      2. 融合 attention (attn)
         - 输入：SSL 特征 (768) + 韵律嵌入 (64) = 832 维拼接
         - 输出：每帧的 attention 权重 (1 维)
      3. 加权求和得到全局特征
    """

    def __init__(self, ssl_dim=768, prosody_dim=2, proj_dim=64, dropout: float = 0.0):
        super().__init__()

        self.prosody_proj = nn.Sequential(
            nn.Linear(prosody_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
        )

        # attention 分数
        self.attn = nn.Sequential(
            nn.Linear(ssl_dim + proj_dim, 128),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, ssl_feats, f0, energy, mask=None):
        """
        Args:
            ssl_feats: (B, T, 768)  SSL 帧级特征
            f0: (B, T, 1)  F0（归一化到 [0, 1]）
            energy: (B, T, 1)  能量（归一化到 [0, 1]）
            mask: (B, T) bool, True=valid frame, False=padding (optional)
        Returns:
            pooled: (B, 768)  加权融合特征
        """
        # 韵律特征投影（先归一化到 [0,1] 尺度）
        f0_norm = f0 / 2093.0  # C7 = ~2093 Hz, covers children's F0 range
        energy_norm = energy / (energy.max(dim=1, keepdim=True).values.clamp(min=1e-8))
        prosody = torch.cat([f0_norm, energy_norm], dim=-1)  # (B, T, 2)
        prosody_emb = self.prosody_proj(prosody)    # (B, T, 64)

        # 融合特征
        combined = torch.cat([ssl_feats, prosody_emb], dim=-1)  # (B, T, 832)

        # Attention 权重
        attn_weights = self.attn(combined)            # (B, T, 1)
        attn_weights = attn_weights.squeeze(-1)       # (B, T)

        # Mask padding frames: set attention score to -inf before softmax
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=1)  # (B, T)

        # 加权求和
        pooled = torch.bmm(attn_weights.unsqueeze(1), ssl_feats)  # (B, 1, 768)
        pooled = pooled.squeeze(1)  # (B, 768)

        return pooled


class SelfAttentionPooling(nn.Module):
    """Pure self-attention pooling — strict ablation baseline.

    Uses ONLY the SSL features (B, T, 768) to compute per-frame attention
    weights. No prosody features (F0, energy). Designed to have exactly
    the same number of trainable parameters as TemporalImportancePooling
    (111,105) for a fair comparison.

    Architecture: 4-layer MLP attention
      Linear(768, 116) → ReLU → Linear(116, 100) → ReLU
      → Linear(100, 100) → Tanh → Linear(100, 1)

    Parameter count proof:
      Linear(768, 116): 768*116 + 116 = 89,204
      Linear(116, 100): 116*100 + 100 = 11,700
      Linear(100, 100): 100*100 + 100 = 10,100
      Linear(100, 1):   100*1 + 1     =    101
      Total: 89,204 + 11,700 + 10,100 + 101 = 111,105
    """

    def __init__(self, ssl_dim: int = 768, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(ssl_dim, 116),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(116, 100),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(100, 1),
        )

    def forward(self, ssl_feats, mask=None):
        """
        Args:
            ssl_feats: (B, T, 768) SSL frame-level features
            mask: (B, T) bool, True=valid, False=padding (optional)
        Returns:
            pooled: (B, 768) weighted-sum pooled features
        """
        attn_weights = self.attn(ssl_feats)      # (B, T, 1)
        attn_weights = attn_weights.squeeze(-1)   # (B, T)

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=1)  # (B, T)

        pooled = torch.bmm(attn_weights.unsqueeze(1), ssl_feats)  # (B, 1, 768)
        pooled = pooled.squeeze(1)  # (B, 768)
        return pooled


def create_pooling(pooling_type: str = 'prosody_guided', ssl_dim: int = 768, dropout: float = 0.0):
    """Factory function with configurable toggle between pooling types.

    Args:
        pooling_type: 'prosody_guided' or 'self_attention'
        ssl_dim: SSL feature dimension (default 768)

    Returns:
        nn.Module — the pooling module

    Raises:
        AssertionError: if parameter counts don't match
        ValueError: if pooling_type is unknown
    """
    if pooling_type == 'prosody_guided':
        pooler = TemporalImportancePooling(ssl_dim=ssl_dim, dropout=dropout)
    elif pooling_type == 'self_attention':
        pooler = SelfAttentionPooling(ssl_dim=ssl_dim, dropout=dropout)
    else:
        raise ValueError(
            f"Unknown pooling_type '{pooling_type}'. "
            f"Must be 'prosody_guided' or 'self_attention'."
        )

    return pooler


def verify_parameter_parity():
    """Runtime assertion: both poolers must have identical parameter counts.

    Returns True if they match, raises AssertionError if not.
    """
    prosody_pooler = TemporalImportancePooling(ssl_dim=768, dropout=0.3)
    self_attn_pooler = SelfAttentionPooling(ssl_dim=768, dropout=0.3)

    n_prosody = sum(p.numel() for p in prosody_pooler.parameters())
    n_self = sum(p.numel() for p in self_attn_pooler.parameters())

    assert n_prosody == n_self, (
        f"PARAMETER COUNT MISMATCH: "
        f"TemporalImportancePooling={n_prosody}, "
        f"SelfAttentionPooling={n_self}"
    )
    return True
