# 分布驱动儿童语音情绪识别

从儿童语音的统计分布出发，构建分布偏移诊断框架 (FD-WA)，系统验证"分布偏移→性能下降"的因果关系。

> **协议**: `ac_suite_2026-06` | **状态**: 🎉 **192/192 实验全部完成**
> **分支**: `research/interpretability-fd`

## 三数据集

| 数据集 | 样本 | 类别 | 说话人 | 年龄 | 风格 |
|--------|------|------|--------|------|------|
| C-BESD (MY) | 4,179 | 6类 | 70 children | 6-12y | 演绎式 (EN+TE 双语) |
| FAU Aibo | 18,216 | 4类 | 51 children | 10-13y | 自然式儿童-机器人交互 |
| IEMOCAP | ~9,794 | 4类 | 10 adults | — | 演绎式 (成人对照) |

## 实验矩阵 (B1-B7)

| Phase | E系列 | 内容 | 实验数 | 状态 |
|-------|-------|------|--------|------|
| B1 | E1 | Pooling × Dataset 基线 (frozen) | 9×3=27 | ✅ |
| B2 | E3 | Zero-shot 跨语料迁移 | 18×1=18 | ✅ |
| B3 | E4 | 数据增强敏感性 (C1-C4) | 12×3=36 | ✅ |
| B4 | E5 | LayerFusion 消融 | 54 | ✅ |
| B5 | E2 | WavLM Unfreeze 对比 | 3×3=9 | ✅ |
| B6 | E6 | 模块消融 (Adapter/Pooling/Fusion) | 10×3=30 | ✅ |
| B7 | E7 | 模型迁移 Fine-tune | 6×3=18 | ✅ |
| **总计** | | | **192** | ✅ |

### 核心结果速览

| 数据集 | 最佳配置 | 最高 WA |
|--------|---------|---------|
| C-BESD | self_attn + unfreeze | **97.13%** |
| FAU Aibo | MeanPool + WeightedFusion | **92.24%** |
| IEMOCAP | prosody + unfreeze | **66.36%** |

## 项目结构

```
├── src/                    # 源代码
│   ├── models/             # WavLM骨干、层融合、池化、分类器
│   ├── data/               # 数据管道（预处理、说话人划分）
│   ├── evaluation/         # 分布偏移测量、XAI可视化
│   ├── augmentation/       # 数据增强
│   ├── training/           # 训练入口
│   └── utils/              # 实验日志
├── scripts/                # 脚本工具集
├── checkpoints/            # 模型权重
│   ├── autodl/b1~b7/       # AC套件 B1-B7 权重 (151个)
│   └── archive/            # 历史权重
├── results_remote/         # ★ 云端完整结果 (192 JSONs)
│   └── results/logs/       # E1~E7 全量实验 JSON
├── results/                # 本地分析结果 (FD/XAI/layer weights)
│   └── logs/               # 早期结果快照 + DATA_FREEZE
├── paper_draft/            # 论文 LaTeX + 配图 + PPT
│   ├── current/            # v9 版本 LaTeX
│   └── figures/            # 论文配图
├── docs/                   # 项目文档
│   ├── current/            # 当前有效文档
│   ├── discussion/         # 设计讨论
│   └── ops/                # 运维手册
├── references/             # 参考文献 PDF
├── experiments/            # 历史实验数据（旧协议）
└── _legacy/                # 已废弃的旧数据和工具
```

## 核心数据入口

| 需求 | 入口 |
|------|------|
| 查看论文引用数值 | `CLAUDE.md` |
| 查看全部实验协议 | `docs/current/权威数据手册.md` |
| 验收实验 JSON (192 runs) | `python scripts/verify_all_192.py` |
| 下载云端权重 | `python scripts/download_checkpoints.py` |
| FD 矩阵 | `results/canonical_fd_pairs.json` |

## 模型架构

| 组件 | 规格 | 参数量 |
|------|------|--------|
| SSL Backbone | WavLM Base (frozen/unfrozen) | 94M |
| Layer Fusion | 12 learnable weights | 12 |
| Pooling | Mean / Self-Attn / Prosody | ~111K |
| Classifier | SEMLP | ~593K |
| **总可训** | | **~704K** |

## 关键约束

- 数据划分：说话人独立 MD5 hash，70/15/15，`data_split_seed=42` 固定
- 训练：在线提取特征，batch_size=16, epochs=100, patience=15
- 所有 Conv1d 在真实帧级时间轴上滑动，非特征拼接维度
- 云端：AutoDL RTX 4090D 24GB（可关机，数据已同步）

## Git

- **仓库**: https://github.com/Xuzc317/child-speech-emotion-recognition
- **当前分支**: `research/interpretability-fd`
