# 新方案-分布驱动儿童SER

## 项目定位

儿童语音情绪识别 —— 从儿童语音的统计分布出发，重新约束 SER 工程流程。

## 目录结构

```
新方案-分布驱动儿童SER/
├── docs/                        # 文档
│   ├── module1_data_pipeline.md # Module 1: 数据管道标准化文档
│   ├── 方向对比与方案设计.md      # 三方对比 + 框架设计
│   └── 实施步骤指南.md           # 分阶段实施指导
├── src/
│   ├── data/
│   │   ├── audio_processor.py   # 标准化音频预处理 (16kHz, mono, peak-norm)
│   │   ├── label_mapper.py      # 统一4类标签映射 (angry/happy/neutral/sad)
│   │   ├── speaker_splitter.py  # 确定性hash说话人独立划分 (70/15/15)
│   │   ├── dataset.py           # 统一跨语料库数据集类
│   │   ├── data_loader.py       # DataLoader 构建器 + 跨语料库支持
│   │   ├── preprocess.py        # [legacy] speaker 划分逻辑（BESD-only）
│   │   ├── dataset_ssl.py       # [legacy] SSL 特征提取 + 数据集类
│   │   └── statistics.py        # 儿童语音统计分析
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ssl_backbone.py      # emotion2vec/WavLM 封装
│   │   ├── adapter.py           # Module 1: 分布校准适配器
│   │   ├── pooling.py           # Module 2: 时序重要性池化
│   │   └── drse_cnn.py          # DrseNet 分类器
│   ├── augmentation/
│   │   ├── __init__.py
│   │   ├── safe_augmentation.py  # SafeAWGN: 加性高斯白噪声 (SNR 10-20dB)
│   │   └── constrained_aug.py   # [legacy] 分布约束增强 (已废弃)
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
