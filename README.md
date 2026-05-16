# 新方案-分布驱动儿童SER

儿童语音情绪识别 —— 从儿童语音的统计分布出发，重新约束 SER 工程流程。

## 项目结构

```
├── docs/                        # 文档
│   ├── module1_data_pipeline.md # Module 1: 数据管道标准化
│   └── ...
├── src/
│   ├── data/                    # 数据管道 (Module 1)
│   │   ├── audio_processor.py   # 标准化音频预处理
│   │   ├── label_mapper.py      # 统一4类标签映射
│   │   ├── speaker_splitter.py  # 说话人独立划分
│   │   ├── dataset.py           # 统一数据集类
│   │   └── data_loader.py       # DataLoader 构建器
│   ├── models/                  # 模型 (Module 2+)
│   ├── augmentation/            # 数据增强 (Module 3)
│   └── training/                # 训练脚本
├── experiments/logs/            # 训练日志
├── checkpoints/                 # 模型检查点
└── scripts/                     # 评估和工具脚本
```

## Module 文档

- [Module 1: Data Pipeline Standardization](docs/module1_data_pipeline.md) — 统一数据管道、跨语料库基础设施、4类标签映射

## 快速开始

```python
from src.data import get_dataloaders

# 在 C-BESD 上训练
dataloaders = get_dataloaders(['c-besd'], batch_size=32, seed=42)

for waveforms, labels, lengths, speaker_ids, dataset_names in dataloaders['train']:
    # waveforms: (B, T_max), labels: (B,) 0-3
    ...
```
