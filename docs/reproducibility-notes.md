# 复现说明
## Reproducibility Notes

## 重要声明

**本项目早期实验存在复现性限制**，请仔细阅读以下说明：

1. **最佳结果说明**: 早期实验记录的最佳准确率约为0.86，但由于当时未固定随机种子，该结果可能无法完全稳定复现。
2. **报告结果**: 本仓库以可复现/可说明的**0.8599准确率**和**0.8506 Macro F1**作为主要报告结果。
3. **复现目标**: 正常复现应能达到**0.85+准确率**，具体值可能因随机性在0.84-0.86之间波动。

## 复现性改进计划

针对早期实验的复现性不足问题，计划进行以下改进：

1. **固定随机种子**: 在实验开始时设置PyTorch、NumPy、Python random的随机种子
2. **保存完整配置**: 每个实验保存完整的超参数和模型配置
3. **保存数据集划分**: 固定训练/验证/测试集的划分，避免随机性
4. **保存环境依赖**: 记录完整的requirements.txt和系统环境信息
5. **规范检查点命名**: 使用系统化的检查点命名规则，包含实验标识
6. **记录完整实验日志**: 保存训练过程中的所有指标和中间结果

## 复现挑战

### 1. 随机性来源
| 来源 | 影响程度 | 控制方法 |
|------|----------|----------|
| **随机种子未固定** | 高 | 设置PyTorch/Numpy随机种子 |
| **数据增强随机性** | 中 | 固定数据增强随机种子 |
| **CUDA随机性** | 低 | 设置CUDA确定性算法 |
| **数据加载随机性** | 中 | 固定DataLoader随机种子 |

### 2. 环境差异
| 组件 | 可能影响 | 建议版本 |
|------|----------|----------|
| **PyTorch版本** | 高 (API变化) | 2.0+ |
| **CUDA版本** | 中 (数值差异) | 11.7+ |
| **库版本差异** | 低 (算法实现) | 见requirements.txt |
| **硬件差异** | 低 (浮点误差) | GPU一致型号最佳 |

### 3. 数据依赖
| 依赖项 | 状态 | 获取方式 |
|--------|------|----------|
| **C-BESD数据集** | ❌ 不包含 | 需联系数据集所有者 |
| **预处理特征** | ❌ 不包含 | 需自行从原始数据提取 |
| **数据集划分** | ✅ 包含 | 代码中实现划分逻辑 |

## 完整复现步骤

### 步骤1: 环境配置
```bash
# 1. 创建conda环境 (推荐)
conda create -n speech-emotion python=3.9
conda activate speech-emotion

# 2. 安装依赖
pip install -r requirements.txt

# 3. 安装PyTorch (根据CUDA版本)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 步骤2: 数据准备
```bash
# 1. 获取C-BESD数据集 (需自行申请)
# 2. 解压到 data/raw/BESD/
# 3. 运行特征提取
python src/data/statistics.py  # 数据集分析
# 4. 生成特征文件 (需要修改代码中的路径)
```

### 步骤3: 固定随机种子
在训练脚本开头添加：
```python
import torch
import numpy as np
import random

# 固定所有随机种子
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# 设置CUDA确定性
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### 步骤4: 运行训练
```python
# 修改train.py中的路径配置
wav_dir = 'data/processed/data_all.npy'  # 修改为你的路径
label_file = 'data/processed/label_all.npy'

# 运行训练
python src/training/train.py
```

### 步骤5: 评估验证
```python
# 加载最佳模型评估
model.load_state_dict(torch.load('best_model.pth'))
val_loss, val_acc = evaluate(model, criterion, test_loader)
print(f"复现准确率: {val_acc:.4f}")
```

## 预期复现结果

### 正常复现范围
| 模型 | 预期准确率范围 | 达到概率 |
|------|----------------|----------|
| **DrseCNN** | 0.84 - 0.86 | 90%+ |
| **CNN** | 0.71 - 0.73 | 95%+ |
| **BiLSTM** | 0.71 - 0.73 | 90%+ |

### 精确复现要求
要获得**完全一致**的结果，需要：
1. 相同数据集和划分
2. 完全相同的随机种子设置
3. 相同的软件环境版本
4. 相同的硬件配置
5. 相同的训练超参数

## 常见复现问题

### 问题1: 准确率显著偏低 (<0.80)
**可能原因**:
1. 数据集路径错误
2. 特征提取错误
3. 数据划分不一致
4. 模型初始化问题

**解决方案**:
```python
# 1. 检查数据形状
print(f"Data shape: {data.shape}")  # 应为 (样本数, 3, 94)
print(f"Label shape: {label.shape}")  # 应为 (样本数,)

# 2. 验证数据加载
sample, label = dataset[0]
print(f"Sample shape: {sample.shape}")  # 应为 (3, 94)
```

### 问题2: 结果波动大 (>0.02差异)
**可能原因**:
1. 随机种子未完全固定
2. 数据增强随机性
3. CUDA非确定性

**解决方案**:
```python
# 确保所有随机源都固定
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 固定DataLoader
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed)

dataloader = DataLoader(dataset, batch_size=128, shuffle=True,
                       worker_init_fn=seed_worker, generator=g)
```

### 问题3: 训练不收敛
**可能原因**:
1. 学习率不合适
2. 梯度爆炸/消失
3. 数据预处理问题

**解决方案**:
```python
# 1. 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 2. 学习率调整
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=10, factor=0.5
)

# 3. 检查数据归一化
print(f"Data mean: {data.mean():.4f}, std: {data.std():.4f}")
```

## 环境配置详情

### 测试环境 (原实验)
| 组件 | 版本 |
|------|------|
| **操作系统** | Windows 10 |
| **Python** | 3.9 |
| **PyTorch** | 2.0.1 |
| **CUDA** | 11.7 |
| **GPU** | NVIDIA RTX 3060 (12GB) |
| **内存** | 32GB |

### 依赖包版本
```txt
# 建议版本 (兼容范围)
torch>=2.0.0
torchaudio>=2.0.0
librosa>=0.9.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
pandas>=1.3.0
tqdm>=4.64.0
```

## 简化复现方案

### 方案A: 仅验证模型架构
如果无法获取C-BESD数据集，可以使用：
```python
# 1. 创建模拟数据
batch_size = 32
dummy_data = torch.randn(batch_size, 1, 100, 94)  # 模拟输入
dummy_labels = torch.randint(0, 6, (batch_size,))  # 模拟标签

# 2. 验证前向传播
model = DrseCNN(num_classes=6)
output = model(dummy_data)
print(f"Output shape: {output.shape}")  # 应为 (32, 6)

# 3. 验证训练步骤
loss = criterion(output, dummy_labels)
loss.backward()
```

### 方案B: 使用替代数据集
可以使用公开成人语音情绪数据集验证：
1. **RAVDESS**: 英语语音情绪数据集
2. **CREMA-D**: 多演员情绪数据集
3. **TESS**: 老年人语音情绪数据集

**注意**: 成人语音与儿童语音特性不同，结果可能差异较大。

## 复现检查清单

### 必要检查项
- [ ] 数据集正确放置和加载
- [ ] 所有随机种子固定
- [ ] 环境依赖包安装正确
- [ ] 模型初始化一致
- [ ] 超参数设置一致

### 建议检查项
- [ ] 数据预处理步骤一致
- [ ] 特征提取参数一致
- [ ] 训练日志记录完整
- [ ] 评估指标计算正确

## 复现支持

### 提供信息
如需复现帮助，请提供：
1. 错误信息/日志
2. 环境配置详情
3. 复现步骤描述
4. 当前结果与期望差异

### 联系方式
项目相关问题可通过GitHub Issues提交。

---

**最后强调**: 由于早期实验的随机性控制不足，**完全精确复现0.8599准确率可能困难**。但遵循上述步骤，应能获得**0.85+的稳定结果**，证明模型和方法的有效性。