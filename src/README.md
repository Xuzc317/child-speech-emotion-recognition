# src/ — 源代码

AC 套件 (ac_suite_2026-06) 的核心训练和推理代码。

## models/ — 模型组件

| 文件 | 用途 |
|------|------|
| `ssl_backbone.py` | WavLM/emotion2vec 封装 |
| `layer_fusion.py` | 12 层可学习加权求和 |
| `pooling.py` | Mean / Self-Attention / Prosody Guided 池化 |
| `semlp.py` | SEMLP 分类器 (~593K params) |

## data/ — 数据加载

| 文件 | 用途 |
|------|------|
| `speaker_splitter.py` | MD5 hash 说话人独立划分 (70/15/15) |
| `dataset.py` | 统一跨语料数据集 |
| `data_loader.py` | Dataloader 构建器 |

## evaluation/ — 评估与诊断

| 文件 | 用途 |
|------|------|
| `distribution_metrics.py` | FD + SMMD 分布偏移测量 |
| `xai_visualizer.py` | APC 计算 + 注意力-韵律可视化 |

## augmentation/ — 数据增强

| 文件 | 用途 |
|------|------|
| `safe_augmentation.py` | 加性高斯白噪声增强 |
