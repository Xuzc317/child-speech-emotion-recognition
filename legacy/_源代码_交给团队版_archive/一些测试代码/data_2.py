import os
import torch
import torchaudio
from torch.utils.data import Dataset
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence

class EmotionDataset(Dataset):
    def __init__(self, wav_dir, label_file, max_length=10):
        """
        Args:
            wav_dir (str): 包含WAV文件的目录路径
            label_file (str): 标签文件路径
            max_length (int): MFCC特征的最大时间步长
        """
        self.wav_dir = wav_dir
        self.max_length = max_length
        self.labels = self._parse_labels(label_file)
        self.file_list = list(self.labels.keys())
        
        # 定义MFCC转换器
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=40,
            melkwargs={
                'n_fft': 512,
                'win_length': 400,
                'hop_length': 160,
                'n_mels': 40,
                'center': False
            }
        )
        
        # 定义重采样器（假设原始采样率可能不同）
        self.resample = torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000)

    def _parse_labels(self, label_file):
        labels = defaultdict(list)
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                labels[parts[1]].append(self._label_to_index(parts[0]))
        return labels

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_id = self.file_list[idx]
        wav_path = os.path.join(self.wav_dir, f"{file_id}.wav")
        
        # 加载音频
        waveform, sample_rate = torchaudio.load(wav_path)
        
        # 转换为单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 重采样到16000Hz
        if sample_rate != 16000:
            waveform = self.resample(waveform)
        
        # 计算MFCC特征
        mfcc = self.mfcc_transform(waveform)  # 输出形状: (1, 25, time)
        # print(waveform.shape)
        # 调整时间维度到固定长度
        if mfcc.shape[2] < self.max_length:
            pad = self.max_length - mfcc.shape[2]
            mfcc = torch.nn.functional.pad(mfcc, (0, pad))
        else:
            mfcc = mfcc[:, :, :self.max_length]
        
        # 调整形状为CNN输入格式 (通道数, 特征数, 时间步)
        mfcc = mfcc.squeeze(0)  # (25, max_length)
        mfcc = mfcc.unsqueeze(0)  # (1, 25, max_length)
        
        label = torch.tensor(self.labels[file_id][0])
        return mfcc, label

    def _label_to_index(self, label):
        label_mapping = {'N': 0, 'R': 1, 'E': 2, 'A': 3, 'P': 4}
        return label_mapping[label]

def collate_fn(batch):
    """处理固定长度的MFCC特征"""
    inputs = torch.stack([item[0] for item in batch])  # (batch_size, 1, 25, max_length)
    labels = torch.stack([item[1] for item in batch])
    return inputs, labels
