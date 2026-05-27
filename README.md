# 分布驱动儿童语音情绪识别

从儿童语音的统计分布出发，重新约束 SER 工程流程。AC 套件 (`ac_suite_2026-05`) 投稿状态。

## 项目结构

```
├── src/                         # 源代码（每个子目录有 README）
│   ├── data/                    # 数据管道：预处理、标签映射、说话人划分
│   ├── models/                  # 模型：WavLM骨干、层融合、池化、分类器
│   ├── evaluation/              # 评估：分布偏移测量(FD/SMMD)、XAI可视化
│   ├── training/                # 训练入口
│   ├── augmentation/            # 数据增强
│   └── utils/                   # 实验日志
├── scripts/                     # 脚本工具集（含 README）
├── results/                     # ★ 权威实验结果（投稿数据源）
│   └── logs/                    # 6 主实验 + 多种子 + APC
├── publication_package/         # 投稿资产包（与 results 同步）
├── paper_draft/                 # 论文 LaTeX 源文件
│   └── figures/                 # 配图（fig01-08, figA1-A4）
├── checkpoints/                 # 模型权重（autodl/ = AC套件, v5_622/ = 历史）
├── docs/                        # 项目文档（全中文命名）
├── experiments/                 # 历史实验数据（v5_622 等）
├── data/                        # 预处理特征（旧协议，AC 套件不再使用）
├── references/                  # 参考文献 PDF
├── results_remote/              # AutoDL 云端原始备份
├── submission_bundle/           # 投稿打包输出
└── last/                        # 外部工具归档
```

## 核心数据入口

| 需求 | 入口 |
|------|------|
| 查看所有实验数值 | `docs/权威数据手册.md` |
| 验收实验 JSON | `python scripts/verify_experiment_jsons.py` |
| 查看投稿前待办 | 桌面 `SER项目全量台账_数据路径与任务清单.md` §3 |
| 查看 AC 实验协议 | `docs/AC实验协议.md` |
| 查看项目路线图 | `docs/实施步骤指南.md` |

## 快速开始

```python
from src.data import get_dataloaders

# C-BESD 域内训练 + 评估
dataloaders = get_dataloaders(['c-besd'], batch_size=16, seed=42)

# 跨语料零样本（C-BESD → FAU Aibo）
from src.data import get_cross_corpus_dataloaders
dls = get_cross_corpus_dataloaders(['c-besd'], ['fau-aibo'], batch_size=16, seed=42, test_split='test')
```

## 环境变量

| 变量 | 数据集 |
|------|--------|
| `SER_C_BESD_PATH` | C-BESD 儿童语音 |
| `SER_IEMOCAP_PATH` | IEMOCAP 成人语音 |
| `SER_FAU_AIBO_PATH` | FAU Aibo 儿童自发性语音 |
| `SER_CREMA_D_PATH` | CREMA-D 成人语音 |

## Git

- **仓库**: https://github.com/Xuzc317/child-speech-emotion-recognition
- **当前分支**: `research/interpretability-fd`
