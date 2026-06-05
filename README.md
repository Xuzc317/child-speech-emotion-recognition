# 分布驱动儿童语音情绪识别

从儿童语音的统计分布出发，重新约束 SER 工程流程。AC 套件 (`ac_suite_2026-05`) 投稿状态。

## 核心实验结果（canonical，2026-05-26 冻结）

### C-BESD 儿童演绎式（3-seed）

| Pooling | Test WA | Δ |
|---------|---------|---|
| Self-Attention | **94.97% ± 1.66%** | — |
| Prosody Guided | **95.10% ± 2.78%** | −0.13pp (n.s.) |

### FAU Aibo 儿童自发性（3-seed）

| Pooling | Test WA | Test UAR |
|---------|---------|----------|
| Self-Attention | **66.46% ± 0.72%** | 55.39% ± 2.05% |
| Prosody Guided | **65.15% ± 1.12%** | 53.95% ± 2.22% |

### IEMOCAP 成人（2-seed，s456 val=0）

| Pooling | Test WA | Δ vs 儿童 |
|---------|---------|-----------|
| Self-Attention | **75.96% ± 7.94%** | — |
| Prosody Guided | **66.23% ± 7.56%** | 韵律有害（−2.12pp in v5） |

### 四条可复现规律

1. **FD↑ ⇒ WA↓ 严格单调** — 增强/年龄/语言三维度均成立
2. **WavLM 中层(L8)权值最高** — 熵≈2.48，分布近均匀
3. **韵律先验儿童特异性** — +2.24pp(儿童)→−2.12pp(成人)，方向反转
4. **成人增强经验不可迁移** — 成人参数损害儿童模型 ~28pp

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
│   └── logs/                    # 6 主实验 + 多种子 + APC + 零样本矩阵
├── publication_package/         # 投稿资产包（与 results 同步）
├── paper_draft/                 # 论文 LaTeX 源文件
│   └── figures/                 # 配图（fig01-08, figA1-A4）
├── checkpoints/                 # 模型权重（autodl/ = AC套件, v5_622/ = 历史）
├── docs/                        # 项目文档（全中文命名）
│   ├── 权威数据手册.md           # ★ 投稿唯一数据口径
│   ├── 全部实验结果汇总.md        # 全部实验历史记录
│   ├── AC实验协议.md             # AC Suite 实验协议
│   └── SER项目全量台账_数据路径与任务清单.md
├── experiments/                 # 历史实验数据（v5_622 等）
├── data/                        # 预处理特征（旧协议，AC 套件不再使用）
├── references/                  # 参考文献 PDF
├── results_remote/              # AutoDL 云端原始备份
├── submission_bundle/           # 投稿打包输出
└── last/                        # 外部工具归档 + 旧版代码脚本
```

## 核心数据入口

| 需求 | 入口 |
|------|------|
| 查看论文引用数值 | `docs/权威数据手册.md` ★ |
| 查看全部实验历史 | `docs/全部实验结果汇总.md` |
| 查看实验协议与验收 | `docs/AC实验协议.md` |
| 验收实验 JSON | `python scripts/verify_experiment_jsons.py` |
| 查看投稿前待办 | `docs/SER项目全量台账_数据路径与任务清单.md` |
| 零样本全矩阵 | `results/logs/exp_zeroshot_*.json`（6方向） |
| 全量 APC（540样本） | `results/logs/apc_full_test.json` |
| FD 矩阵（AutoDL GPU 重算） | `results/canonical_fd_pairs.json` |

## 快速开始

```python
from src.data import get_dataloaders

# C-BESD 域内训练 + 评估
dataloaders = get_dataloaders(['c-besd'], batch_size=16, seed=42)

# 跨语料零样本（C-BESD → FAU Aibo）
from src.data import get_cross_corpus_dataloaders
dls = get_cross_corpus_dataloaders(['c-besd'], ['fau-aibo'], batch_size=16, seed=42, test_split='test')
```

## 模型架构

| 组件 | 规格 | 参数量 |
|------|------|--------|
| SSL Backbone | WavLM Base (wavlm-base-sv), frozen | 94M（冻结） |
| Layer Fusion | WavLMLayerFusion（12 learnable weights） | 12 |
| Pooling | Self-Attn / Prosody Guided（二选一） | 111,105 |
| Classifier | SEMLP（SE + MLP） | ~593K |
| **总可训** | | **~704K** |

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
