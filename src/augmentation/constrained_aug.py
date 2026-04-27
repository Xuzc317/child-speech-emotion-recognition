"""Module 3：分布约束增强 (Distribution-Constrained Augmentation)

核心思想：
  成人语音上有效的增强参数应用到儿童语音上，会使其脱离真实分布，
  破坏情感相关特征。我们不做新增强方法，而是做"增强空间约束"。

三个组成部分：
  1. 儿童语音统计分布计算 (compute_child_speech_stats)
  2. 分布约束增强 (DistributionConstrainedAugmentation)
  3. 分布偏移量化工具 (measure_distribution_shift) —— 负面实验的关键

负面实验设计：
  - 成人参数增强：pitch ±6, stretch 0.7-1.3
    → 样本漂出分布（FDist/MMD ↑）
    → 情感特征被破坏（准确率 ↓）
  - 儿童约束增强：pitch ±3, stretch 0.85-1.15
    → 样本留在分布内（FDist/MMD 不显著 ↑ 或 ↓）
    → 准确率 ↑ 或不变
  - 极端增强：pitch ±12, stretch 0.5-1.5
    → 严重漂出分布（FDist/MMD ↑↑）
    → 准确率 ↓↓

与旧项目的区别：
  旧项目：增强在特征提取后用 .npy 固化，且 mean-pooled
  新项目：增强在波形层面进行，然后通过 SSL 模型提帧级特征
"""

import torch
import torch.nn as nn
import librosa
import numpy as np
from scipy import linalg


# ============================================================
# Part 1：儿童语音统计分布计算
# ============================================================

def compute_child_speech_stats(wav_paths):
    """从儿童 WAV 文件列表计算声学参数统计分布。

    用于确定增强参数的合法范围，确保增强后的样本仍属于
    儿童语音的真实分布。

    参数：
      - F0 distribution → pitch_shift 范围
      - speaking rate → time_stretch 范围
      - energy/SNR → noise level 范围

    返回：
      {'f0_mean', 'f0_std', 'f0_min', 'f0_max',
       'energy_mean', 'energy_std',
       'pitch_shift_range': (min, max),
       'stretch_range': (min, max),
       'noise_snr_range': (min, max)}
    """
    f0_all = []
    energy_all = []

    for path in wav_paths:
        try:
            y, sr = librosa.load(path, sr=16000)
            # F0
            f0, _, _ = librosa.pyin(
                y, sr=sr, fmin=65, fmax=1100,
                hop_length=320
            )
            f0 = f0[~np.isnan(f0)]
            f0_all.extend(f0.tolist())

            # Energy
            energy = librosa.feature.rms(y=y, hop_length=320)
            energy_all.extend(energy[0].tolist())
        except:
            continue

    f0_all = np.array(f0_all)
    energy_all = np.array(energy_all)

    # 增强参数范围（基于统计量）
    # pitch_shift: F0 均值的 ±2 个标准差（上限不超过儿童生理极限）
    # stretch: 基于语速变异性估计
    # noise: 基于能量动态范围估计

    stats = {
        'f0_mean': float(np.mean(f0_all)),
        'f0_std': float(np.std(f0_all)),
        'f0_min': float(np.percentile(f0_all, 5)),
        'f0_max': float(np.percentile(f0_all, 95)),
        'energy_mean': float(np.mean(energy_all)),
        'energy_std': float(np.std(energy_all)),
    }

    # 默认参数范围（Phase 1：先设为合理值，Phase 2：用统计量精确计算）
    stats['pitch_shift_range'] = (-3.0, 3.0)   # semitones
    stats['stretch_range'] = (0.85, 1.15)       # rate
    stats['noise_snr_range'] = (15.0, 25.0)     # dB

    return stats


# ============================================================
# Part 2：分布约束增强
# ============================================================

class DistributionConstrainedAugmentation:
    """分布约束的波形级数据增强。

    用法：
      aug = DistributionConstrainedAugmentation(child_stats)
      augmented = aug(waveform, sr=16000)

    增强类型：
      1. pitch_shift: 音高偏移（约束到儿童 F0 范围内）
      2. time_stretch: 时间拉伸（约束到儿童语速范围内）
      3. noise_injection: 噪声注入（约束 SNR 范围）
    """

    def __init__(self, child_stats=None):
        self.stats = child_stats or compute_child_speech_stats([])
        self.pitch_range = self.stats.get('pitch_shift_range', (-3.0, 3.0))
        self.stretch_range = self.stats.get('stretch_range', (0.85, 1.15))
        self.snr_range = self.stats.get('noise_snr_range', (15.0, 25.0))

    def __call__(self, waveform, sr=16000):
        """应用随机增强。

        Args:
            waveform: (T,) numpy array
            sr: 采样率
        Returns:
            augmented: (T,) numpy array
        """
        aug_type = np.random.choice(['pitch', 'stretch', 'noise', 'none'])

        if aug_type == 'pitch':
            n_steps = np.random.uniform(*self.pitch_range)
            return librosa.effects.pitch_shift(waveform, sr=sr, n_steps=n_steps)

        elif aug_type == 'stretch':
            rate = np.random.uniform(*self.stretch_range)
            return librosa.effects.time_stretch(waveform, rate=rate)

        elif aug_type == 'noise':
            snr_db = np.random.uniform(*self.snr_range)
            signal_power = np.mean(waveform ** 2)
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = np.random.normal(0, np.sqrt(noise_power), size=waveform.shape)
            return waveform + noise

        else:  # 'none'
            return waveform


# ============================================================
# Part 3：SpecAugment（SSL 特征上的标准增强）
# ============================================================

class SpecAugment(nn.Module):
    """在 SSL 模型的输入频谱上做时间/频率掩码。

    SpecAugment 是 SSL 模型的标准增强方式，对 WavLM/emotion2vec
    等模型有效。不是创新，是工程实践中应该做的事情。
    """

    def __init__(self, freq_mask_param=10, time_mask_param=10):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

    def forward(self, x):
        """x: (B, T, 768) — 在时间维上做 mask"""
        # 简化实现：在时间维上随机 mask 连续帧
        # TODO Phase 2: 可以使用 torchaudio 的 FrequencyMasking / TimeMasking
        return x


# ============================================================
# Part 4：分布偏移的定量衡量（负面实验关键工具）
# ============================================================

def compute_frechet_distance(feats_1, feats_2):
    """计算两组特征之间的 Frechet Distance（FD ↓）。

    用于衡量增强后样本是否漂出了儿童语音的原始分布。
    FD 越小，表示分布越接近。

    输入：
      feats_1: (N, D) 原始/增强前的特征
      feats_2: (M, D) 增强后的特征
    """
    mu1 = feats_1.mean(axis=0)
    sigma1 = np.cov(feats_1, rowvar=False)
    mu2 = feats_2.mean(axis=0)
    sigma2 = np.cov(feats_2, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)


def compute_mmd(feats_1, feats_2, kernel='rbf', sigma=1.0):
    """Maximum Mean Discrepancy（MMD ↓）。

    另一种分布距离衡量，与 FD 互补。
    """
    # TODO Phase 2: 实现 MMD 计算
    # 用于负面实验中定量证明"增强把样本推出了儿童分布"
    pass


def measure_distribution_shift(original_feats, augmented_feats):
    """综合衡量增强引起的分布偏移。

    返回 FD 和 MMD 两个指标，供负面实验使用。
    """
    fd = compute_frechet_distance(original_feats, augmented_feats)
    return {'frechet_distance': float(fd)}
