# 模型架构
## Model Architecture

## 概述

本项目实现了多种深度学习模型用于儿童语音情绪识别，包括**DrseCNN** (核心模型)、CNN、LSTM/BiLSTM、Transformer、SigWavNet等。模型设计考虑了儿童语音特点和数据集规模限制。

## 模型总览

| 模型 | 参数量 | 最佳准确率 | 特点 |
|------|--------|------------|------|
| **DrseCNN** | ~45.5M | 0.8599 | 残差SE注意力CNN，专门设计 |
| **CNN** | ~5.8M | 0.7214 | 标准卷积网络，基线模型 |
| **BiLSTM** | ~12.1M | 0.7226 | 序列建模，捕捉时序依赖 |
| **Transformer** | ~2.0M | 0.6948 | 自注意力机制，全局建模 |
| **SigWavNet** | ~16.9M | 0.7318 | 信号处理网络，原始音频输入 |

## 核心模型: DrseCNN

### 设计理念
DrseCNN (Deep Residual Squeeze-Excitation CNN) 专为儿童语音情绪识别设计，结合了：
1. **残差连接** (Residual Connection): 缓解梯度消失，促进深层网络训练
2. **SE注意力** (Squeeze-and-Excitation): 通道注意力机制，增强重要特征
3. **多阶段卷积**: 分层特征提取，从局部到全局

### 网络结构
```
输入: (B, 1, T, 94)  # Batch, Channel, Time, Features
    ↓
[Stage 1] 3×卷积层 + BN + ReLU + 最大池化
    ↓
[Stage 2] 3×卷积层 + BN + ReLU + 最大池化
    ↓
[Stage 3] 3×卷积层 + BN + ReLU + 最大池化
    ↓
[Stage 4] 3×卷积层 + BN + ReLU + 全局平均池化
    ↓
全连接层 (2048 → 1024 → 512 → 6)
    ↓
输出: (B, 6)  # 6类情绪概率
```

### 关键组件

#### 1. ResSE Block (残差SE模块)
```python
class ResSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # SE注意力
        self.se = SELayer(out_channels)
        
        # 残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)  # SE注意力
        out += self.shortcut(residual)  # 残差连接
        return F.relu(out)
```

#### 2. SE注意力层 (Squeeze-and-Excitation)
```python
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
```

### 参数配置
| 参数 | 值 | 说明 |
|------|-----|------|
| **输入维度** | (B, 1, T, 94) | Batch, Channel, Time, Features |
| **卷积核** | 3×3 | 标准卷积核大小 |
| **通道数** | [64, 128, 256, 512] | 逐阶段翻倍 |
| **池化** | 最大池化 2×2 | 下采样 |
| **SE缩减比** | 16 | 通道压缩比例 |
| **Dropout** | 0.3-0.5 | 防止过拟合 |
| **最终特征维度** | 512 | 全局平均池化后 |

## 基线模型

### 1. CNN模型 (CNNModel)
标准卷积神经网络，作为基础基线。

#### 结构特点
```
输入: (B, 1, 163)  # 1D卷积输入
    ↓
Conv1D(1→256, k=5) + BN + ReLU
    ↓
Conv1D(256→256, k=5) + BN + ReLU
    ↓
MaxPool1d(5, stride=2)
    ↓
Conv1D(256→128, k=3) + BN + ReLU
    ↓
Conv1D(128→128, k=3) + BN + ReLU
    ↓
MaxPool1d(5, stride=2) + Dropout(0.4)
    ↓
全连接层 (4608 → 128 → 64 → 6)
```

#### 参数量: ~5.8M

### 2. BiLSTM模型 (BiLSTMModel)
双向长短时记忆网络，擅长序列建模。

#### 结构特点
```
输入: (B, T, 94)  # 序列输入
    ↓
BiLSTM (94 → 128, 2层, dropout=0.3)
    ↓
取最后时刻隐藏状态 (256维)
    ↓
全连接层 (256 → 128 → 64 → 6)
```

#### 参数量: ~12.1M

### 3. Transformer模型
自注意力机制，全局序列建模。

#### 结构特点
```
输入: (B, T, 94)
    ↓
线性投影 (94 → 1024)
    ↓
位置编码
    ↓
Transformer编码器 (6层, 16头, hidden=1024)
    ↓
全局平均池化
    ↓
全连接层 (1024 → 512 → 6)
```

#### 参数量: ~2.0M (轻量版)

### 4. SigWavNet模型
直接处理原始音频信号的网络。

#### 结构特点
```
输入: (B, 1, 音频长度)
    ↓
多尺度卷积 (不同核大小)
    ↓
特征融合
    ↓
深度可分离卷积
    ↓
全局池化 + 全连接
```

#### 参数量: ~16.9M

## 模型比较分析

### 性能对比
| 模型 | 准确率 | 训练速度 | 内存占用 | 适用场景 |
|------|--------|----------|----------|----------|
| **DrseCNN** | **0.8599** | 中等 | 高 | 最佳性能需求 |
| **CNN** | 0.7214 | 快 | 低 | 快速基线 |
| **BiLSTM** | 0.7226 | 慢 | 中等 | 时序建模 |
| **Transformer** | 0.6948 | 慢 | 中等 | 全局依赖 |
| **SigWavNet** | 0.7318 | 中等 | 高 | 端到端学习 |

### 消融实验

#### DrseCNN组件消融
| 变体 | 准确率 | 说明 |
|------|--------|------|
| **DrseCNN-Full** | **0.8599** | 完整模型 |
| **No Residual** | 0.8321 | 移除残差连接 (-2.78%) |
| **No SE** | 0.8415 | 移除SE注意力 (-1.84%) |
| **Plain CNN** | 0.8210 | 简化卷积 (-3.89%) |

#### 输入特征消融
| 特征 | 准确率 | 说明 |
|------|--------|------|
| **All Features (94)** | **0.8599** | 全部特征 |
| **MFCC Only (40)** | 0.7214 | 仅MFCC (-13.85%) |
| **MFCC+Mel (80)** | 0.8012 | 无音高特征 (-5.87%) |

## 实现细节

### 1. 输入处理
```python
# 输入形状调整
# 原始特征: (B, 3, 94) → 重塑为 (B, 1, 3, 94)
x = x.view(x.size(0), 1, x.size(1), x.size(2))
```

### 2. 训练配置
| 参数 | DrseCNN | CNN | BiLSTM | Transformer |
|------|---------|-----|--------|-------------|
| **学习率** | 5e-4 | 5e-4 | 1e-3 | 5e-4 |
| **权重衰减** | 5e-5 | 5e-5 | 1e-4 | 5e-5 |
| **Batch Size** | 128 | 128 | 64 | 64 |
| **Epochs** | 128 | 128 | 100 | 100 |
| **优化器** | AdamW | AdamW | Adam | AdamW |

### 3. 正则化策略
- **Dropout**: 全连接层后 (0.3-0.5)
- **BatchNorm**: 卷积层后，加速收敛
- **权重衰减**: L2正则化，防止过拟合
- **早停法**: 验证损失不再下降时停止

## 儿童语音适配设计

### 1. 频谱特性考虑
- **高频敏感**: 儿童语音高频成分较多，网络需能捕捉高频特征
- **共振峰偏移**: 儿童声道较短，共振频率较高，需要适应的特征提取

### 2. 时序特性考虑
- **情绪表达节奏**: 儿童情绪表达可能更快速变化，需要较强的时间建模能力
- **非平稳性**: 儿童语音非平稳性更强，需要鲁棒的特征表示

### 3. 数据量适配
- **参数量控制**: 数据集有限，避免过参数化
- **正则化加强**: 使用多种正则化技术防止过拟合
- **数据增强**: 充分利用有限数据

## 使用指南

### 模型初始化
```python
from src.models.models import DrseCNN, CNNModel, BiLSTMModel, TransformerModel

# 初始化模型
model = DrseCNN(num_classes=6)
model = CNNModel(num_classes=6)
model = BiLSTMModel(num_classes=6)
model = TransformerModel(input_dim=94, num_classes=6)
```

### 训练示例
```python
import torch
from src.training.train import train, evaluate

# 训练配置
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-5)

# 训练循环
for epoch in range(epochs):
    train_loss, train_acc = train(model, criterion, optimizer, train_loader)
    val_loss, val_acc = evaluate(model, criterion, val_loader)
```

### 模型保存与加载
```python
# 保存
torch.save(model.state_dict(), "best_model.pth")

# 加载
model.load_state_dict(torch.load("best_model.pth"))
```

## 局限性

### 当前实现限制
1. **固定输入尺寸**: 要求固定的时间长度和特征维度
2. **计算复杂度**: DrseCNN参数量较大，需要较多GPU内存
3. **过拟合风险**: 数据集较小，复杂模型容易过拟合
4. **实时性**: 部分模型推理速度较慢

### 改进方向
1. **轻量化设计**: 模型压缩和剪枝
2. **动态输入**: 支持变长输入
3. **多模态融合**: 结合文本和视觉信息
4. **自监督预训练**: 利用无标签数据预训练

## 相关文件

- `src/models/models.py` - 全部模型实现
- `experiments/results/ablation_summary.csv` - 消融实验结果
- `docs/model-comparison.md` - 模型对比表格

---

**注意**: 模型选择需平衡性能、复杂度和计算资源，根据具体应用场景选择合适模型。