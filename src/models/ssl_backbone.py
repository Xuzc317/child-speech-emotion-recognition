"""emotion2vec / WavLM 封装

使用 HuggingFace transformers 或 fairseq 加载预训练 SSL 模型，
输出帧级特征供下游模块使用。

Phase 1 目标：先跑 linear probe 验证 SSL 基线效果。
"""

import torch
import torch.nn as nn
import librosa
import numpy as np


class SSLBackbone(nn.Module):
    """预训练 SSL 模型封装，支持 emotion2vec 和 WavLM。

    emotion2vec（推荐，ACL 2024）:
      - Base ~90M 参数，12 层 Transformer
      - 输出帧率 50Hz，帧间 20ms
      - 帧级特征维度 768
      - HuggingFace: `emotion2vec_base` / `emotion2vec_plus_base`

    WavLM Base（备选，Microsoft 2022）:
      - 94.6M 参数
      - 同样输出帧级 768-dim 特征
      - HuggingFace: `microsoft/wavlm-base`

    用法：
      model = SSLBackbone(model_name='emotion2vec', frozen=True)
      feats = model(waveforms)  # (B, T, 768)
    """

    def __init__(self, model_name='emotion2vec', frozen=True, device='cuda'):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.frozen = frozen
        self.backbone = self._load_model()
        if frozen:
            self._freeze()

    def _load_model(self):
        """加载预训练 SSL 模型。

        emotion2vec（推荐）: 通过 FunASR + ModelScope 加载 'iic/emotion2vec_base'
        WavLM（备选）: 'microsoft/wavlm-base' 或 'microsoft/wavlm-base-sv'
          - wavlm-base-sv 是 speaker verification 版本，encoder 与 base 相同
        """
        model_name = self.model_name  # 局部变量，避免 fallback 污染原始值

        if 'emotion2vec' in model_name:
            try:
                # 优先走 transformers（HF 网络可访问时）
                from transformers import Wav2Vec2Model
                model = Wav2Vec2Model.from_pretrained("emotion2vec/emotion2vec_base")
                print(f"[SSLBackbone] Loaded emotion2vec via HuggingFace")
            except Exception:
                # Fallback: 本地缓存的 FunASR emotion2vec
                try:
                    model = self._load_emotion2vec_funasr()
                except Exception as e:
                    print(f"[SSLBackbone] emotion2vec unavailable ({e}), falling back to WavLM")
                    return self._load_wavlm()
        elif 'wavlm' in model_name:
            model = self._load_wavlm()
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        return model.to(self.device)

    def _load_emotion2vec_funasr(self):
        """通过 FunASR 从 ModelScope 加载 emotion2vec（离线可用）。"""
        from funasr import AutoModel
        funasr_model = AutoModel(
            model='iic/emotion2vec_base',
            hub='ms',
            device=str(self.device),
            disable_update=True,
        )
        print(f"[SSLBackbone] Loaded emotion2vec via FunASR/ModelScope")
        return Emotion2vecFunASRWrapper(funasr_model)

    def _load_wavlm(self):
        """加载 WavLM base 模型。"""
        from transformers import WavLMModel
        for model_id in ['microsoft/wavlm-base-sv', 'microsoft/wavlm-base']:
            try:
                model = WavLMModel.from_pretrained(model_id)
                print(f"[SSLBackbone] Loaded {model_id}")
                return model
            except Exception:
                continue
        raise RuntimeError("Cannot load WavLM — check network or cache")

    def _freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f"[SSLBackbone] Frozen {self.model_name} — {sum(p.numel() for p in self.backbone.parameters()):,} params frozen")

    def forward(self, waveforms, sr=16000):
        """输入波形，输出帧级特征。

        Args:
            waveforms: (B, T_wav) 或 (B, 1, T_wav)，float32，已重采样到 16kHz
            sr: 采样率（必须匹配 SSL 模型要求）
        Returns:
            feats: (B, T_frames, 768) 帧级特征
        """
        if waveforms.dim() == 3:
            waveforms = waveforms.squeeze(1)

        if self.frozen:
            with torch.no_grad():
                outputs = self.backbone(waveforms, output_hidden_states=True)
        else:
            outputs = self.backbone(waveforms, output_hidden_states=True)

        # 取最后一层隐藏状态作为帧级特征
        # outputs.last_hidden_state: (B, T_frames, 768)
        return outputs.last_hidden_state


class Emotion2vecFunASRWrapper(nn.Module):
    """将 FunASR emotion2vec 封装为与 HuggingFace WavLMModel 一致的接口。

    暴露标准的 .forward(waveforms) → last_hidden_state 和 .parameters()，
    使 SSLBackbone 无需区分底层加载方式。
    """

    def __init__(self, funasr_model):
        super().__init__()
        self._funasr_model = funasr_model
        self.inner = funasr_model.model  # funasr.models.emotion2vec.model.Emotion2vec
        self.config = type('Config', (), {'hidden_size': 768, 'num_hidden_layers': 12})()

    def forward(self, waveforms, output_hidden_states=False):
        if waveforms.dim() == 3:
            waveforms = waveforms.squeeze(1)
        result = self.inner.extract_features(waveforms, mode='default')
        # result['x']: (B, T, 768) — 最后一层帧级特征
        # 模拟 HuggingFace BaseModelOutput
        from transformers.modeling_outputs import BaseModelOutput
        return BaseModelOutput(
            last_hidden_state=result['x'],
            hidden_states=result.get('layer_results') if output_hidden_states else None,
        )

    def parameters(self, recurse=True):
        return self.inner.parameters(recurse=recurse)

    def to(self, device):
        self.inner = self.inner.to(device)
        return self


def preprocess_wav(path, target_sr=16000, duration=4.0):
    """加载 WAV 并重采样到 16kHz，填充/截断到固定长度。

    Args:
        path: WAV 文件路径
        target_sr: 目标采样率（SSL 模型要求 16kHz）
        duration: 目标长度（秒），短则补零，长则截断
    Returns:
        waveform: (1, T_wav) float32 tensor
        sr: int
        padding_mask: (1, T_wav) bool tensor (True = padding 位置)
    """
    data, sr = librosa.load(path, sr=target_sr)
    target_length = int(target_sr * duration)

    if len(data) < target_length:
        pad_len = target_length - len(data)
        data = np.pad(data, (0, pad_len), 'constant')
    else:
        data = data[:target_length]

    return torch.from_numpy(data).float().unsqueeze(0), sr, None


def collate_fn_ssl(batch):
    """SSL 数据集的 collate 函数。

    batch 内各条音频在 collate 时 padding 到相同长度，
    同时生成 attention_mask。
    """
    waveforms = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])

    # 找到 batch 内最大长度，padding
    max_len = max(w.shape[1] for w in waveforms)
    padded = []
    attention_masks = []
    for w in waveforms:
        pad_len = max_len - w.shape[1]
        if pad_len > 0:
            padded_w = torch.nn.functional.pad(w, (0, pad_len))
            mask = torch.cat([torch.ones(w.shape[1]), torch.zeros(pad_len)])
        else:
            padded_w = w
            mask = torch.ones(w.shape[1])
        padded.append(padded_w)
        attention_masks.append(mask)

    return torch.stack(padded), labels, torch.stack(attention_masks).bool()
