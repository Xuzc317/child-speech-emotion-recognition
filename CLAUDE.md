# 新方案-分布驱动儿童SER

## 项目定位

儿童语音情绪识别 —— 从儿童语音的统计分布出发，重新约束 SER 工程流程。

## 目录结构

```
新方案-分布驱动儿童SER/
├── docs/                        # 文档
│   ├── 方向对比与方案设计.md      # 三方对比 + 框架设计
│   └── 实施步骤指南.md           # 分阶段实施指导
├── src/
│   ├── data/
│   │   ├── preprocess.py        # speaker 划分逻辑（从旧项目复用）
│   │   ├── dataset_ssl.py       # SSL 特征提取 + 数据集类
│   │   └── statistics.py        # 儿童语音统计分析
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ssl_backbone.py      # emotion2vec/WavLM 封装
│   │   ├── adapter.py           # Module 1: 分布校准适配器
│   │   ├── pooling.py           # Module 2: 时序重要性池化
│   │   └── drse_cnn.py          # DrseNet 分类器
│   ├── augmentation/
│   │   ├── __init__.py
│   │   └── constrained_aug.py   # Module 3: 分布约束增强 + SpecAugment
│   ├── training/
│   │   ├── __init__.py
│   │   └── train_ssl.py         # 新的训练脚本
│   └── utils/
│       ├── __init__.py
│       └── tracker.py           # 实验追踪（从旧项目复用）
├── data/                        # 预处理数据
├── experiments/logs/            # 训练日志
├── checkpoints/                 # 模型检查点
├── scripts/                     # 评估和工具脚本
├── requirements.txt
├── CLAUDE.md
└── README.md
```

## 关键约束

- 所有 Conv1d 操作必须在**真实的帧级时间轴**上滑动，而非特征拼接维度
- ONNX 兼容性要求：不要使用 `nn.AdaptiveAvgPool1d`，改用 `MaxPool1d` 或固定 `AvgPool1d`
- 特征维度：emotion2vec 帧级 768-dim（不再是 162-dim mean-pooled）
