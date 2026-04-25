# 测试代码
## Test Code

## 说明

测试代码模块包含项目的单元测试和功能测试。当前版本未包含完整的测试套件，但提供了基本的测试示例。

## 测试类型

### 1. 单元测试 (Unit Tests)
测试单个函数或类的正确性：
- 特征提取函数测试
- 数据加载类测试
- 模型前向传播测试

### 2. 集成测试 (Integration Tests)
测试模块间的协作：
- 完整训练流程测试
- 数据预处理管道测试
- 模型保存和加载测试

### 3. 功能测试 (Functional Tests)
测试特定功能需求：
- 模型性能基准测试
- 数据增强效果测试
- 内存和速度测试

## 测试示例

### 示例1: 模型前向传播测试
```python
# tests/test_model_forward.py
import torch
from src.models.models import DrseCNN, CNNModel

def test_model_forward():
    """测试模型前向传播"""
    # 测试输入
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, 100, 94)  # DrseCNN输入
    
    # 测试DrseCNN
    model = DrseCNN(num_classes=6)
    output = model(dummy_input)
    assert output.shape == (batch_size, 6), f"DrseCNN输出形状错误: {output.shape}"
    
    # 测试CNNModel (1D输入)
    dummy_input_1d = torch.randn(batch_size, 1, 163)
    model_cnn = CNNModel(num_classes=6)
    output_cnn = model_cnn(dummy_input_1d)
    assert output_cnn.shape == (batch_size, 6), f"CNN输出形状错误: {output_cnn.shape}"
    
    print("模型前向传播测试通过")
```

### 示例2: 特征提取测试
```python
# tests/test_feature_extraction.py
import numpy as np
import librosa
from src.data.dataset import extract_features

def test_feature_extraction():
    """测试特征提取函数"""
    # 创建模拟音频数据
    duration = 2.5
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * 440 * t)  # 440Hz正弦波
    
    # 提取特征
    features = extract_features(audio_data, sample_rate)
    
    # 验证特征维度
    expected_dim = 94  # ZCR1 + Chroma12 + MFCC40 + RMS1 + Mel40
    assert features.shape == (expected_dim,), f"特征维度错误: {features.shape}"
    
    # 验证特征值范围
    assert not np.any(np.isnan(features)), "特征包含NaN值"
    assert not np.any(np.isinf(features)), "特征包含Inf值"
    
    print(f"特征提取测试通过，特征维度: {features.shape}")
```

### 示例3: 数据加载测试
```python
# tests/test_dataloader.py
import numpy as np
import torch
from src.data.dataset import npyDataset
from torch.utils.data import DataLoader

def test_dataloader():
    """测试数据加载器"""
    # 创建模拟数据
    n_samples = 100
    data = np.random.randn(n_samples, 3, 94).astype(np.float32)
    labels = np.random.randint(0, 6, (n_samples,)).astype(np.int64)
    
    # 保存临时文件
    np.save('test_data.npy', data)
    np.save('test_labels.npy', labels)
    
    # 创建数据集
    dataset = npyDataset('test_data.npy', 'test_labels.npy')
    
    # 验证数据集大小
    assert len(dataset) == n_samples, f"数据集大小错误: {len(dataset)}"
    
    # 测试数据加载
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for batch_data, batch_labels in dataloader:
        assert batch_data.shape[0] == batch_labels.shape[0], "批次大小不匹配"
        assert batch_data.dtype == torch.float32, "数据类型错误"
        break
    
    # 清理
    import os
    os.remove('test_data.npy')
    os.remove('test_labels.npy')
    
    print("数据加载器测试通过")
```

## 测试运行

### 使用pytest运行测试
```bash
# 安装测试依赖
pip install pytest

# 运行所有测试
pytest tests/

# 运行特定测试文件
pytest tests/test_model_forward.py

# 显示详细输出
pytest -v tests/

# 生成覆盖率报告
pytest --cov=src tests/
```

### 手动运行测试
```bash
# 直接运行Python测试脚本
python tests/test_model_forward.py
python tests/test_feature_extraction.py
python tests/test_dataloader.py
```

## 测试覆盖率目标

### 当前状态
由于是展示版本，测试覆盖率有限。建议在实际项目中达到：

| 模块 | 建议覆盖率 | 当前状态 |
|------|-----------|----------|
| 模型定义 | 80%+ | 基本前向传播测试 |
| 数据加载 | 70%+ | 基本加载测试 |
| 特征提取 | 60%+ | 基本功能测试 |
| 训练流程 | 50%+ | 未系统测试 |

### 需要添加的测试
1. **模型测试**
   - 梯度反向传播测试
   - 参数初始化测试
   - 不同输入尺寸测试

2. **数据测试**
   - 数据增强正确性测试
   - 数据集划分测试
   - 特征归一化测试

3. **训练测试**
   - 优化器收敛测试
   - 学习率调度测试
   - 早停法测试

4. **评估测试**
   - 指标计算正确性测试
   - 混淆矩阵测试
   - 性能基准测试

## 测试最佳实践

### 1. 测试隔离
每个测试应独立，不依赖其他测试状态：
```python
# 好: 使用setup和teardown
class TestDataset:
    def setup_method(self):
        self.test_data = np.random.randn(10, 3, 94)
        self.test_labels = np.random.randint(0, 6, 10)
    
    def teardown_method(self):
        del self.test_data
        del self.test_labels
    
    def test_dataset_size(self):
        assert len(self.test_data) == 10
```

### 2. 模拟数据
使用模拟数据避免真实数据依赖：
```python
def create_mock_audio(duration=2.5, sample_rate=16000):
    """创建模拟音频数据用于测试"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    return np.sin(2 * np.pi * 440 * t)  # 简单正弦波
```

### 3. 参数化测试
测试不同参数组合：
```python
import pytest

@pytest.mark.parametrize("batch_size", [16, 32, 64, 128])
def test_batch_size(batch_size):
    """测试不同批次大小"""
    # 测试逻辑
    pass
```

## 相关文件

- `src/models/models.py` - 模型定义，需要测试
- `src/data/dataset.py` - 数据加载和特征提取，需要测试
- `src/training/train.py` - 训练流程，需要测试
- `docs/reproducibility-notes.md` - 测试对复现性的帮助

---

**注**: 当前版本主要展示项目核心功能，测试代码相对简单。在实际生产或研究项目中，建议建立完整的测试套件确保代码质量。