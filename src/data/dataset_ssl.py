"""基于 SSL 特征的数据集。

提供两种使用模式：
  模式 A（推荐 Phase 1）：预提取特征模式
    - 为所有 WAV 提前跑一次 emotion2vec，将 (T, 768) 帧级特征存为 .npy
    - 训练时直接从 .npy 加载，不需要实时跑 SSL backbone
    - 优点：省 GPU 显存、加速训练、方便反复实验

  模式 B（Phase 2+）：在线模式
    - DataLoader 中实时加载 WAV → emotion2vec
    - 优点：支持波形级增强（分布约束增强、SpecAugment）
    - 缺点：需要更多显存和计算

Phase 1 先用模式 A 快速验证 SSL 基线效果。
Phase 2 切到模式 B，支持波形级增强。
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa


class SSLFeatureDataset(Dataset):
    """模式 A：从预提取的 .npy 特征文件加载。

    Args:
        data_path: .npy 文件路径，形状 (N, T, 768) float32
        label_path: .npy 文件路径，形状 (N,) int64
        prosody_path: 可选 .npy，形状 (N, T, 2) float32 (F0 + energy)
    """

    def __init__(self, data_path, label_path, prosody_path=None):
        self.datas = np.load(data_path).astype(np.float32)
        self.labels = np.load(label_path)
        if prosody_path and os.path.exists(prosody_path):
            self.prosody = np.load(prosody_path).astype(np.float32)
        else:
            self.prosody = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.datas[idx]  # (T, 768)
        label = torch.tensor(self.labels[idx], dtype=torch.int64)
        if self.prosody is not None:
            p = self.prosody[idx]  # (T, 2)
            return torch.from_numpy(x), label, torch.from_numpy(p)
        return torch.from_numpy(x), label


class SSLOnlineDataset(Dataset):
    """模式 B：在线加载 WAV + SSL backbone。

    适合 Phase 2+ 使用，支持波形级增强。

    注意：
      每个 epoch 内每次 __getitem__ 调用都会跑一次 SSL 模型，
      速度较慢。建议 batch_size 不要太大，或者用多进程。
    """

    def __init__(self, wav_dir, entries, backbone, augmentation=None,
                 sr=16000, duration=4.0):
        """
        Args:
            wav_dir: 数据集根目录
            entries: [(filepath, label, speaker_id), ...]
            backbone: SSLBackbone 实例
            augmentation: DistributionConstrainedAugmentation 实例
        """
        self.wav_dir = wav_dir
        self.entries = entries
        self.backbone = backbone
        self.augmentation = augmentation
        self.sr = sr
        self.target_length = int(sr * duration)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        path, label, _ = self.entries[idx]
        waveform, _ = librosa.load(path, sr=self.sr)

        # 截断/填充到固定长度
        if len(waveform) > self.target_length:
            waveform = waveform[:self.target_length]
        else:
            waveform = np.pad(waveform, (0, self.target_length - len(waveform)))

        # 波形级增强
        if self.augmentation:
            waveform = self.augmentation(waveform, self.sr)

        # SSL 特征提取
        waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)
        with torch.no_grad():
            feats = self.backbone(waveform_tensor)  # (1, T, 768)
        feats = feats.squeeze(0)  # (T, 768)

        return feats, torch.tensor(label, dtype=torch.int64)


def collate_fn_ssl_features(batch):
    """SSL 特征序列的 collate 函数。

    支持可选的韵律特征（3 元组模式）。
    """
    has_prosody = len(batch[0]) == 3
    features = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    prosody = [item[2] for item in batch] if has_prosody else None

    max_t = max(f.shape[0] for f in features)
    feat_dim = features[0].shape[1]

    padded, masks = [], []
    for f in features:
        t = f.shape[0]
        if t < max_t:
            pad = torch.zeros(max_t - t, feat_dim)
            padded.append(torch.cat([f, pad], dim=0))
            mask = torch.cat([torch.ones(t), torch.zeros(max_t - t)])
        else:
            padded.append(f)
            mask = torch.ones(t)
        masks.append(mask)

    result = [torch.stack(padded), labels, torch.stack(masks)]
    if has_prosody:
        # Pad prosody to same max_t
        prosody_padded = []
        for p in prosody:
            if p.shape[0] < max_t:
                pad = torch.zeros(max_t - p.shape[0], p.shape[1])
                prosody_padded.append(torch.cat([p, pad], dim=0))
            else:
                prosody_padded.append(p)
        result.append(torch.stack(prosody_padded))  # (B, T, 2)
    return tuple(result)
