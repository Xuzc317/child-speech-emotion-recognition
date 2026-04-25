# 工具函数
## Utility Functions

## 说明

工具函数模块包含项目中的通用辅助函数。当前版本中，主要工具函数已集成在各个模块中。

## 功能概述

### 原项目中的工具函数
原始项目 `code_屎山版/utils/` 目录包含多种工具函数，但由于代码结构整理，部分函数已集成到相应模块：

1. **数据工具**
   - 数据加载和预处理辅助函数
   - 特征提取工具函数
   - 数据增强实现

2. **模型工具**
   - 模型参数统计和可视化
   - 模型保存和加载工具
   - 训练过程监控

3. **实验工具**
   - 实验配置管理
   - 结果记录和可视化
   - 日志管理

### 当前集成情况
| 功能 | 当前位置 | 说明 |
|------|----------|------|
| 特征提取 | `src/data/dataset.py` | `extract_features()`, `get_features()` |
| 数据增强 | `src/data/dataset.py` | `noise()`, `stretch()`, `pitch()` |
| 数据加载 | `src/data/dataset.py` | `EnglishDataset`, `npyDataset` |
| 模型评估 | `src/training/train.py` | `evaluate()` 函数 |
| 训练循环 | `src/training/train.py` | `train()` 函数 |
| 可视化 | `src/training/train.py` | `plot_training_curves()` |

## 常用工具函数示例

### 1. 模型参数统计
```python
def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 使用示例
from src.models.models import DrseCNN
model = DrseCNN(num_classes=6)
print(f"模型参数量: {count_parameters(model):,}")
```

### 2. 训练设备设置
```python
def setup_device():
    """设置训练设备 (GPU/CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("使用CPU")
    return device
```

### 3. 随机种子设置
```python
def set_random_seed(seed=42):
    """设置所有随机种子"""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # CUDA确定性设置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### 4. 学习率调度
```python
def create_scheduler(optimizer, scheduler_type='plateau'):
    """创建学习率调度器"""
    if scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=10, factor=0.5, verbose=True
        )
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100
        )
    return scheduler
```

## 扩展建议

### 需要添加的工具函数
1. **数据可视化工具**
   - 特征分布可视化
   - 混淆矩阵绘制
   - 训练曲线美化

2. **实验管理工具**
   - 实验配置保存和加载
   - 实验结果自动记录
   - 实验复现工具

3. **模型分析工具**
   - 特征重要性分析
   - 模型解释性工具
   - 注意力可视化

### 模块化建议
建议将工具函数分类组织：
```
src/utils/
├── data_utils.py      # 数据相关工具
├── model_utils.py     # 模型相关工具  
├── train_utils.py     # 训练相关工具
├── eval_utils.py      # 评估相关工具
└── vis_utils.py       # 可视化工具
```

## 相关文件

- `src/data/dataset.py` - 数据相关工具函数
- `src/training/train.py` - 训练相关工具函数
- `docs/reproducibility-notes.md` - 随机种子设置说明
- `docs/known-issues.md` - 工具函数改进建议

---

**注**: 当前项目工具函数较为分散，建议在实际应用中根据需要进行模块化整理。