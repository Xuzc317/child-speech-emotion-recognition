# 新方案-分布驱动儿童SER

## 项目定位

儿童语音情绪识别 —— 从儿童语音的统计分布出发，重新约束 SER 工程流程。

## 目录结构

```
新方案-分布驱动儿童SER/
├── docs/                        # 文档
│   ├── 方向对比与方案设计.md      # 设计方案
│   ├── 实施步骤指南.md           # 实施指南 (Phase 1-5)
│   ├── 修改方案_v6.md            # 最新修改方案
│   ├── 论文初稿_v6.md            # 论文 v6
│   └── 论文初稿_v6.docx          # 论文 v6 Word 版
├── src/
│   ├── data/
│   │   ├── preprocess.py        # speaker 划分 (split_speakers_7_3_with_inner_val)
│   │   ├── dataset_ssl.py       # SSL 特征数据集
│   │   └── statistics.py        # 儿童语音统计分析
│   ├── models/
│   │   ├── ssl_backbone.py      # WavLM/emotion2vec 封装
│   │   ├── adapter.py           # 声学校准适配器 (消融讨论)
│   │   ├── pooling.py           # 韵律引导时序重要性池化 (核心方法)
│   │   └── semlp.py          # SEMLP 分类器
│   ├── augmentation/
│   │   └── constrained_aug.py   # 分布约束增强 + FD 计算
│   ├── training/
│   │   └── train_ssl.py         # 训练脚本
│   └── utils/
│       └── experiment_logger.py # 实验日志系统
├── experiments/
│   ├── v5_622/                  # 6:2:2 协议完整实验数据
│   ├── phase3_ablation.json     # 历史 60/20/20 协议结果
│   └── cross_language_results.json
├── checkpoints/v5_622/          # 24 个模型权重 (.pth)
├── scripts/                     # 工具脚本
├── requirements.txt
├── AGENTS.md
└── README.md
```

## 关键约束

- 所有 Conv1d 操作必须在**真实的帧级时间轴**上滑动，而非特征拼接维度
- ONNX 兼容性要求：不要使用 `nn.AdaptiveAvgPool1d`，改用 `MaxPool1d` 或固定 `AvgPool1d`
- 特征维度：WavLM 帧级 768-dim
- 数据划分：外部 8:2 + 内部 val（`split_speakers_7_3_with_inner_val`），目标 6:2:2

## 当前状态 (2026-05-04)

- **Phase 6.1-6.5 全部完成**: 实验 + 审查 + 论文 v7 中英文
- **最终推荐模型**: A3 = WavLM frozen + Prosody Pooling + SEMLP (无 Adapter)
  - C-BESD 6:2:2 Test: 80.9% WA / 80.8% UAR
  - Prosody Pooling 是主驱动力 (+2.24 percentage points)
- **跨数据集三组对比**:
  - C-BESD (儿童, 指定表达): +2.24 percentage points
  - CREMA-D (成人, 指定表达): +0.75 percentage points
  - IEMOCAP (成人, 自然对话): −2.12 percentage points
- **增强实验**: 所有增强均有害, FD-Accuracy 严格单调
- **跨语言**: English↔Telugu 迁移几乎完全失效 (19-28%)
- **可解释性**: Attention Entropy-Accuracy 负相关
- **统一FD框架**: 年龄/增强/语言三维 FD-Accuracy 单调
- **论文 v7**: `docs/论文初稿_v7_cn.docx` (中文), `docs/论文初稿_v7.docx` (English)
- **参考文献**: 7 篇 PDF 下载到 `references/`

## 实验数据位置

- `experiments/v5_622/` — 完整实验数据 (3 数据集对比 + attention + FD)
- `checkpoints/v5_622/` — 24 个 .pth 模型文件
- `references/` — 7 篇参考文献 PDF
