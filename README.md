# 儿童语音情绪识别

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)

## 📋 项目概述

本项目聚焦于**儿童语音情绪识别**，使用C-BESD数据集。目标是通过声学特征分析和深度学习模型，将儿童语音信号分类为六种情绪类别（愤怒、厌恶、恐惧、快乐、中性、悲伤）。

### 主要特点
- **特征工程**: MFCC、Chroma STFT、ZCR、RMS、梅尔频谱等94维特征，含数据增强
- **模型架构**: DrseCNN（核心模型）、CNN、BiLSTM、Transformer、SigWavNet等变体
- **训练流程**: 可复现的训练和评估脚本
- **实验记录**: 完整的实验结果和可视化

## 📊 结果摘要

| 模型 | 最佳准确率 | Macro F1 | 参数量 |
|-------|---------------|----------|------------|
| DrseCNN (完整) | 0.8599 | 0.8506 | ~45.5M |
| CNN | 0.7214 | - | ~5.8M |
| SigWavNet | 0.7318 | - | ~16.9M |
| Transformer | 0.6948 | - | ~2.0M |

**注意**: 早期实验记录的最佳准确率约为0.86，但由于当时未固定随机种子，该结果可能无法完全稳定复现。本仓库以可复现/可说明的**0.8599准确率和0.8506 Macro F1**作为主要报告结果。

## 🏗️ 项目结构

```
children-speech-emotion-recognition/
├── src/                    # 源代码
│   ├── data/              # 数据集和特征提取
│   ├── models/            # 模型定义
│   ├── training/          # 训练脚本
│   └── utils/             # 工具函数
├── experiments/           # 实验记录
│   ├── configs/          # 配置文件
│   ├── results/          # 结果文件 (CSV, 日志)
│   └── visualizations/   # 训练曲线
├── docs/                 # 技术文档
├── assets/               # 图像和图表
├── checkpoints/          # 模型权重 (不包含)
├── data/                 # 数据集 (不包含)
└── legacy/               # 历史代码 (不维护)
```

## 💻 核心代码功能

| 文件路径 | 主要功能 | 说明 |
|----------|----------|------|
| `src/training/train.py` | 模型训练主脚本 | 包含训练循环、评估函数、命令行参数解析和模型保存功能 |
| `src/data/dataset.py` | 数据加载与特征提取 | 实现音频特征提取（MFCC、Chroma等）、数据增强（加噪、拉伸、变调）和数据加载器 |
| `src/models/models.py` | 模型定义 | 包含DrseCNN、CNN、BiLSTM、Transformer等多种模型架构的实现 |
| `src/data/statistics.py` | 数据集统计分析 | 提供数据集的基本统计信息分析和可视化功能 |
| `experiments/configs/README.md` | 实验配置说明 | 记录实验参数配置和超参数设置指南 |
| `docs/model-structure.md` | 模型架构文档 | 详细说明DrseCNN及其他模型的架构设计和实现细节 |

**主要代码文件关系**:
1. **数据流程**: `dataset.py` → 提取特征 → `train.py`加载训练
2. **模型流程**: `models.py`定义模型 → `train.py`训练评估
3. **实验流程**: 配置参数 → 训练模型 → 记录结果 → 分析可视化

## 🚀 快速开始

### 环境要求
```bash
pip install -r requirements.txt
```

### 基本使用
1. **准备数据**: 获取C-BESD数据集并放置在 `data/raw/` 目录 (不包含在本仓库)
2. **提取特征**: 运行 `python src/data/statistics.py` 进行数据集分析
3. **训练模型**: 运行 `python src/training/train.py` (需要数据文件)
4. **评估**: 查看 `experiments/results/` 中的评估指标

**重要提示**: 本仓库仅包含代码结构和文档。运行代码需要:
- C-BESD数据集 (需联系数据集所有者获取使用许可)
- 提取的特征文件 (.npy) 或原始音频文件
- 足够的GPU内存进行模型训练

## 📚 文档

- [项目总结](docs/project-summary.md) - 项目背景、目标和成果总结
- [数据集说明](docs/dataset.md) - C-BESD数据集详细信息
- [特征工程](docs/feature-engineering.md) - 声学特征和数据增强技术
- [模型架构](docs/model-structure.md) - DrseCNN及其他模型详细说明
- [实验结果](docs/experiment-results.md) - 详细实验记录和分析
- [复现说明](docs/reproducibility-notes.md) - 复现性挑战和改进计划
- [已知问题](docs/known-issues.md) - 实现限制和待改进之处
- [访谈记录](docs/interview-notes.md) - 项目讨论要点

## 🔧 技术细节

### 特征提取
从2.5秒16kHz音频中提取94维特征:
- ZCR (1维)
- Chroma STFT (12维)
- MFCC (40维)
- RMS (1维)
- 梅尔频谱 (40维)

### 数据增强
每个样本生成3个版本:
1. 原始特征
2. 加噪版本
3. 拉伸+变调版本

### 核心模型: DrseCNN
深度残差压缩-激励卷积网络，专为儿童语音设计:
- **残差连接**: 缓解梯度消失，支持深层网络
- **SE注意力**: 通道注意力机制，增强重要特征
- **多阶段卷积**: 分层特征提取

## ⚠️ 限制与注意事项

### 技术限制
- **小规模数据集**: C-BESD每类情绪样本有限
- **复现性挑战**: 早期实验未固定随机种子
- **计算需求**: DrseCNN需要较多GPU内存
- **路径依赖**: 代码需要用户自行设置数据集路径

### 使用限制
- **数据集许可**: C-BESD数据集有使用限制，需遵守相关协议
- **代码参考**: 部分实现参考了开源项目和团队协作成果
- **学术用途**: 仅供研究和学习使用，不用于临床诊断或商业部署

## 🤝 贡献与协作

本项目为本科毕业论文/科研实践项目。部分实现细节参考了开源代码并在团队协作中开发。

### 参与贡献
- 参与了儿童语音声学特性分析
- 组织了多特征融合方案
- 参与了DrseCNN模型结构迭代和理解
- 参与了模型训练、对比实验和消融分析
- 整理了实验结果、论文材料和项目文档
- 重新组织了项目仓库用于GitHub展示和技术交流

### 致谢
- C-BESD数据集提供者
- 开源语音情绪识别项目
- 研究小组成员的讨论和调试帮助

## 📄 使用许可与免责声明

### 许可说明
本仓库内容遵循 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) 许可:
- **署名**: 使用时请注明本项目
- **非商业性**: 禁止商业用途

### 重要免责声明
1. **数据集限制**: C-BESD数据集受原始许可约束，使用者需自行获取并遵守相关协议
2. **代码参考**: 部分代码参考了开源实现，请遵守原项目的许可条款
3. **学术诚信**: 使用本项目内容时请恰当引用，尊重知识产权
4. **使用风险**: 本代码为研究性质，不保证完全无错误，使用者自行承担风险
5. **非医疗用途**: 模型仅供学术研究，不用于临床诊断或医疗决策

### 引用建议
如需在学术工作中引用本项目，请参考:
```
[项目名称]. Children Speech Emotion Recognition - Lightweight Public Version. GitHub repository. 2026.
```

---
**最后说明**: 本项目展示了从实验探索到代码整理的全过程，重点在于研究方法和技术实现的交流，而非数据或权重的分发。