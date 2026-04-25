# 实验配置
## Experiment Configurations

## 说明

此目录用于存储实验配置文件。当前版本使用代码中的硬编码参数，未来可迁移到配置文件管理。

## 实验参数

### 当前硬编码参数
在 `src/training/train.py` 中定义的参数：

```python
# 训练参数
lr = 5e-4
weight_decay = 5e-5
epochs = 128
batch_size = 128

# 数据参数
wav_dir = 'data_all.npy'      # 特征数据路径
label_file = 'label_all.npy'  # 标签数据路径
scale = 0.7                    # 训练集比例

# 模型参数
num_classes = 6                # 情绪类别数
save_path = "best_model.pth"   # 模型保存路径
```

### 模型特定参数
在 `src/models/models.py` 中各模型的定义参数：

1. **DrseCNN**
   - 卷积层通道数: [64, 128, 256, 512]
   - SE缩减比例: 16
   - Dropout率: 0.3-0.5

2. **CNNModel**
   - 卷积核大小: [5, 5, 3, 3]
   - 通道数: [256, 256, 128, 128]
   - Dropout率: 0.4

3. **Transformer**
   - 隐藏维度: 1024
   - 注意力头数: 16
   - 编码器层数: 6
   - Dropout率: 0.3

## 配置文件格式建议

### YAML格式示例
```yaml
# configs/drsenet_baseline.yaml
experiment:
  name: "drseNet_baseline"
  description: "DrseCNN baseline experiment with default parameters"
  timestamp: "2026-04-17"

data:
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  batch_size: 128
  num_workers: 4
  feature_dim: 94
  num_classes: 6

model:
  type: "DrseCNN"
  params:
    num_classes: 6
    channels: [64, 128, 256, 512]
    se_reduction: 16
    dropout_rate: 0.3

training:
  optimizer: "AdamW"
  learning_rate: 0.0005
  weight_decay: 0.00005
  epochs: 128
  early_stopping_patience: 20
  save_best_only: true

evaluation:
  metrics: ["accuracy", "f1", "recall"]
  save_confusion_matrix: true
  save_training_curves: true
```

### JSON格式示例
```json
{
  "experiment": {
    "name": "cnn_baseline",
    "description": "CNN baseline model"
  },
  "data": {
    "batch_size": 128,
    "train_ratio": 0.7,
    "feature_dim": 94
  },
  "model": {
    "type": "CNNModel",
    "num_classes": 6
  },
  "training": {
    "learning_rate": 0.0005,
    "epochs": 100
  }
}
```

## 配置管理系统

### 简单配置加载器
```python
# src/utils/config_loader.py
import yaml
import json

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            config = json.load(f)
        else:
            raise ValueError("Unsupported config format")
    return config

def update_args_with_config(args, config):
    """用配置更新参数"""
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            args.__dict__[key] = value
    return args
```

### 命令行参数与配置结合
```python
import argparse
from src.utils.config_loader import load_config

def parse_args():
    parser = argparse.ArgumentParser()
    
    # 基础参数
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--model', type=str, default='DrseCNN', help='模型类型')
    parser.add_argument('--lr', type=float, default=5e-4, help='学习率')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 如果提供了配置文件，覆盖命令行参数
    if args.config:
        config = load_config(args.config)
        args = update_args_with_config(args, config)
    
    return args
```

## 实验参数记录

### 重要实验配置
| 实验名称 | 配置文件 | 关键参数 | 结果 |
|----------|----------|----------|------|
| DrseCNN_Full | (硬编码) | lr=5e-4, wd=5e-5, bs=128 | 0.8599 |
| DrseCNN_NoSE | (硬编码) | lr=5e-4, wd=5e-5, bs=128 | 0.8424 |
| CNN_Baseline | (硬编码) | lr=5e-4, wd=5e-5, bs=128 | 0.7214 |
| BiLSTM | (硬编码) | lr=1e-3, wd=1e-4, bs=64 | 0.7226 |

### 参数敏感性分析
基于实验记录的参数影响：

1. **学习率**
   - 最佳范围: 1e-4 到 1e-3
   - 太大: 训练不稳定
   - 太小: 收敛慢

2. **权重衰减**
   - 最佳范围: 5e-5 到 1e-4
   - 重要正则化手段

3. **批量大小**
   - 受GPU内存限制
   - 128在性能和内存间平衡

## 超参数搜索建议

### 网格搜索示例
```python
# 搜索空间定义
search_space = {
    'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3],
    'weight_decay': [1e-5, 5e-5, 1e-4, 5e-4],
    'dropout_rate': [0.2, 0.3, 0.4, 0.5],
    'batch_size': [64, 128, 256]
}

# 自动生成配置
def generate_configs(search_space):
    import itertools
    keys = search_space.keys()
    values = search_space.values()
    
    configs = []
    for combination in itertools.product(*values):
        config = dict(zip(keys, combination))
        configs.append(config)
    
    return configs
```

### 贝叶斯优化建议
对于更高效的搜索：
1. 使用Optuna或Hyperopt库
2. 定义目标函数（验证准确率）
3. 设置合理的搜索边界
4. 并行运行多个试验

## 配置版本控制

### 最佳实践
1. **配置文件与代码一起版本控制**
2. **每个实验独立的配置文件**
3. **配置文件包含实验完整描述**
4. **配置与结果对应记录**

### 命名规范
```
configs/
├── drsenet/
│   ├── baseline.yaml
│   ├── no_se_attention.yaml
│   └── no_residual.yaml
├── cnn/
│   ├── baseline.yaml
│   └── deep.yaml
└── experiments_log.csv     # 实验记录总表
```

## 相关文件

- `src/training/train.py` - 当前参数定义位置
- `experiments/results/` - 实验结果记录
- `docs/experiment-results.md` - 实验参数和结果分析

---

**注**: 当前版本使用硬编码参数简化实现，实际研究项目中建议使用配置文件管理系统，便于实验复现和参数管理。