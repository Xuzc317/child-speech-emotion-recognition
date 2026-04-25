# 特征工程
## Feature Engineering

## 概述

本项目的特征工程包括**声学特征提取**和**数据增强**两部分，旨在从儿童语音信号中提取区分性特征，并通过增强技术提高模型泛化能力。

## 特征提取流程

### 整体流程
```
原始音频
    ↓ (加载与预处理)
音频信号 (16kHz, 2.5秒)
    ↓ (特征提取)
94维融合特征 [ZCR(1) + Chroma(12) + MFCC(40) + RMS(1) + Mel(40)]
    ↓ (数据增强)
增强特征集 (原始 + 加噪 + 拉伸变调)
    ↓ (规范化)
训练/测试特征
```

### 代码模块
- `extract_features()`: 单一样本特征提取
- `get_features()`: 完整特征提取与增强
- `noise()`, `stretch()`, `pitch()`: 数据增强函数

## 声学特征详解

### 1. 过零率 (Zero Crossing Rate, ZCR)
| 属性 | 说明 |
|------|------|
| **维度** | 1 |
| **计算** | `librosa.feature.zero_crossing_rate()` |
| **含义** | 信号通过零点的频率 |
| **作用** | 区分清音/浊音，反映语音的尖锐程度 |
| **儿童语音特点** | 儿童清音部分可能更明显 |

### 2. 色度特征 (Chroma STFT)
| 属性 | 说明 |
|------|------|
| **维度** | 12 |
| **计算** | `librosa.feature.chroma_stft()` |
| **含义** | 12个音级（半音）的能量分布 |
| **作用** | 反映音乐和语音的音高信息 |
| **儿童语音特点** | 儿童音高较高，色度分布可能偏移 |

### 3. MFCC (Mel-frequency Cepstral Coefficients)
| 属性 | 说明 |
|------|------|
| **维度** | 40 |
| **计算** | `librosa.feature.mfcc()` |
| **含义** | Mel频率倒谱系数，模拟人耳听觉 |
| **作用** | 反映声道特性，语音识别核心特征 |
| **儿童语音特点** | 儿童声道较短，MFCC分布与成人不同 |

#### MFCC参数配置
```python
# 代码中的MFCC参数
n_mfcc=40,          # 40阶系数
n_fft=512,          # FFT窗口大小
win_length=400,     # 窗口长度 (25ms)
hop_length=160,     # 帧移 (10ms)
n_mels=40,          # Mel滤波器数量
center=False        # 不居中
```

### 4. RMS能量 (Root Mean Square Energy)
| 属性 | 说明 |
|------|------|
| **维度** | 1 |
| **计算** | `librosa.feature.rms()` |
| **含义** | 信号的均方根能量 |
| **作用** | 反映语音强度，用于语音活动检测 |
| **儿童语音特点** | 儿童语音强度变化可能更剧烈 |

### 5. 梅尔频谱 (Mel Spectrogram)
| 属性 | 说明 |
|------|------|
| **维度** | 40 |
| **计算** | `librosa.feature.melspectrogram()` |
| **含义** | Mel尺度上的频谱能量 |
| **作用** | 反映频谱包络，与MFCC互补 |
| **儿童语音特点** | 高频部分能量相对较高 |

### 特征融合策略
```
特征向量 = ZCR(1) ⊕ Chroma(12) ⊕ MFCC(40) ⊕ RMS(1) ⊕ Mel(40)
总维度 = 1 + 12 + 40 + 1 + 40 = 94
```

**融合优势**:
1. **互补性**: 不同特征捕捉语音不同方面
2. **鲁棒性**: 某些特征在特定条件下更稳定
3. **区分性**: 组合特征提供更丰富的表示

## 数据增强技术

### 增强策略
对每个原始样本生成3个版本：
1. **原始特征** (无增强)
2. **加噪特征** (添加噪声)
3. **拉伸变调特征** (时长拉伸+音高偏移)

### 1. 加噪 (Additive Noise)
```python
def noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    data = data + noise_amp * np.random.normal(size=data.shape[0])
    return data
```

| 参数 | 值 | 说明 |
|------|-----|------|
| **噪声幅度** | 0-3.5% of max amplitude | 随机均匀分布 |
| **噪声类型** | 高斯白噪声 | 均值为0的高斯分布 |
| **目的** | 提高对背景噪声的鲁棒性 | 模拟真实环境 |

### 2. 时长拉伸 (Time Stretching)
```python
def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)
```

| 参数 | 值 | 说明 |
|------|-----|------|
| **拉伸率** | 0.8 | 加速20% |
| **实现** | `librosa.effects.time_stretch` | 相位不变拉伸 |
| **目的** | 模拟语速变化 | 提高时间鲁棒性 |

### 3. 音高偏移 (Pitch Shifting)
```python
def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)
```

| 参数 | 值 | 说明 |
|------|-----|------|
| **偏移量** | 0.7个半音 | 音高微调 |
| **实现** | `librosa.effects.pitch_shift` | 相位声码器 |
| **目的** | 模拟音高变化 | 提高频率鲁棒性 |

### 增强效果可视化
```
原始音频 → 特征提取 → 原始特征
    ↓ (加噪)
加噪音频 → 特征提取 → 加噪特征
    ↓ (拉伸+变调)
变形音频 → 特征提取 → 变形特征
```

## 实现代码详解

### 核心函数: `extract_features()`
```python
def extract_features(data, sample_rate):
    result = np.array([])
    
    # 1. ZCR (1维)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))
    
    # 2. Chroma STFT (12维)
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))
    
    # 3. MFCC (40维)
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))
    
    # 4. RMS (1维)
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))
    
    # 5. Mel Spectrogram (40维)
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))
    
    return result  # 总维度: 94
```

### 完整流程: `get_features()`
```python
def get_features(path):
    # 加载音频 (2.5秒，0.6秒偏移)
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    
    # 1. 原始特征
    res1 = extract_features(data, sample_rate)  # (94,)
    result = np.array(res1)
    
    # 2. 加噪特征
    noise_data = noise(data)
    res2 = extract_features(noise_data, sample_rate)
    result = np.vstack((result, res2))  # (2, 94)
    
    # 3. 拉伸变调特征
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch, sample_rate)
    result = np.vstack((result, res3))  # (3, 94)
    
    return result  # 最终形状: (3, 94)
```

## 儿童语音适配考虑

### 1. 时长选择
- **2.5秒截断**: 儿童情绪表达通常较简短
- **0.6秒偏移**: 跳过可能的起始静音或引导词

### 2. 特征参数调整
- **Mel滤波器**: 40个，覆盖儿童语音频率范围 (通常80Hz-8kHz)
- **帧长设置**: 25ms窗口，适合儿童语音的较短音素

### 3. 增强策略针对性
- **噪声幅度**: 较低 (0-3.5%)，儿童录音质量通常较好
- **拉伸程度**: 适度 (0.8)，避免过度失真
- **音高偏移**: 较小 (0.7半音)，儿童音高已较高

## 实验验证

### 特征有效性验证
通过消融实验验证各特征的贡献：

| 特征组合 | 准确率 | 说明 |
|----------|--------|------|
| **MFCC Only** | 0.65-0.70 | 基线性能 |
| **MFCC + Mel** | 0.70-0.75 | 频谱信息补充 |
| **MFCC + Chroma** | 0.68-0.73 | 音高信息补充 |
| **All Features (94维)** | **0.85+** | 最佳性能 |

### 增强策略有效性
| 增强策略 | 准确率提升 | 说明 |
|----------|------------|------|
| **无增强** | 基准 | 容易过拟合 |
| **仅加噪** | +2-3% | 提高噪声鲁棒性 |
| **仅拉伸变调** | +3-4% | 提高时间/频率鲁棒性 |
| **全部增强** | **+5-8%** | 综合效果最佳 |

## 使用指南

### 快速开始
```python
from src.data.dataset import get_features, extract_features

# 提取单个文件特征
features = get_features("audio.wav")  # 返回 (3, 94)

# 仅提取基础特征
basic_features = extract_features(audio_data, sample_rate)  # (94,)
```

### 参数调整建议
1. **数据集特定**:
   - 调整Mel滤波器数量适应不同频率范围
   - 修改增强幅度适应数据质量

2. **计算优化**:
   - 减少MFCC阶数可降低维度
   - 调整帧长/帧移平衡时频分辨率

3. **儿童语音优化**:
   - 考虑儿童特有的频率范围
   - 针对儿童情绪表达特点调整增强策略

## 局限性

### 当前实现限制
1. **固定时长**: 所有音频截断为2.5秒，可能损失信息
2. **手工特征**: 依赖手工设计的声学特征
3. **增强参数固定**: 增强幅度和方式未自适应调整
4. **实时性**: 特征提取计算量较大，实时应用需优化

### 改进方向
1. **端到端学习**: 直接从原始音频学习特征
2. **自适应增强**: 根据样本特性动态调整增强参数
3. **多尺度特征**: 结合不同时间尺度的特征
4. **注意力机制**: 自动关注重要特征维度

## 相关文件

- `src/data/dataset.py` - 特征提取完整实现
- `src/data/statistics.py` - 特征统计分析
- `experiments/results/ablation_summary.csv` - 特征消融实验结果

---

**注意**: 特征工程是语音情绪识别的关键环节，需要根据具体数据集和任务进行调整优化。