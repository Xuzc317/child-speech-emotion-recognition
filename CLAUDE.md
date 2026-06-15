# 新方案-分布驱动儿童SER

> **协议**: `ac_suite_2026-06` | **最后更新**: 2026-06-16
> **权威设计文档**: `docs/实验设计方案_v3_含学习笔记.md`

## 项目定位

儿童语音情绪识别 —— 从儿童语音的统计分布出发，构建分布偏移诊断框架 (FD-WA)，系统验证"分布偏移→性能下降"的因果关系。

## 当前架构

```
WavLM Base (frozen/unfrozen) → 12层 LayerFusion → Pooling → SEMLP 分类器
                                  (learnable weights)   (mean/self_attn/prosody)  (~704K params)
```

- **主干**: `microsoft/wavlm-base-sv`, 768-dim 帧级特征 @ 50Hz
- **数据划分**: 说话人独立 MD5 hash, 70/15/15, `data_split_seed=42` 固定
- **训练**: 在线提取特征（非预存），batch_size=16, epochs=100, patience=15
- **云端**: AutoDL RTX 4090D 24GB, conda env `speech`
- **SSH**: `connect.cqa1.seetacloud.com:25808` (root/9HmcVfCXUFVD)
  - paramiko: `look_for_keys=False, allow_agent=False, disabled_algorithms={'pubkeys':['rsa-sha2-256','rsa-sha2-512']}`

## 三数据集

| 数据集 | 样本 | 类别 | 说话人 | 年龄 | 风格 |
|--------|------|------|--------|------|------|
| C-BESD (MY) | 4,179 | 6类 | 70 children | 6-12y | 演绎式 (EN+TE 双语) |
| FAU Aibo | 18,216 | 4类 (A+E→Angry, P→Happy, N, R→Sad) | 51 children | 10-13y | 自然式儿童-机器人交互 |
| IEMOCAP | ~9,794 | 4类 (angry/happy/neutral/sad) | 10 adults | — | 演绎式 (成人对照) |

## 实验矩阵 (B1-B7)

| Phase | E系列 | 内容 | 实验数 | 状态 |
|-------|-------|------|--------|------|
| B1 | E1 | Pooling × Dataset 基线 (frozen) | 9×3=27 | ✅ |
| B2 | E3 | Zero-shot 跨语料迁移 | 18×1=18 | ✅ |
| B3 | E4 | 数据增强敏感性 (C1-C4) | 12×3=36 | ✅ |
| B4 | E5 | LayerFusion 消融 (last/weighted/L1-L12) | 54 | ✅ |
| B5 | E2 | WavLM Unfreeze 对比 | 3×3=9 | ✅ |
| B6 | E6 | 模块消融 (Adapter/Pooling/Fusion) | 7×3=21 | ✅ |
| B7 | E7 | 模型迁移 Fine-tune | 6×3=18 | ⏳ 待启动 |

**总计**: 183 runs | 已完成: 165 | 待跑: 18 (B7)

## 关键数值 (最新 AC 套件)

| 实验 | 数据集 | 配置 | WA | 备注 |
|------|--------|------|-----|------|
| E2-01 | C-BESD | self_attn + **unfreeze** | **96.91%** (3-seed) | 🔥 最高分 |
| E1-02 | C-BESD | self_attn + frozen | 92.92% | 冻结天花板 |
| E2-03 | IEMOCAP | prosody + unfreeze | 66.36% (3-seed) | |
| E1-05 | FAU | self_attn + frozen | 67.81% | |
| E3 best | IEMOCAP→C-BESD | self_attn zero-shot | 34.68% | 分布偏移上限 |

**B4 结论**: last ≈ weighted ≈ 任何单层 L≥7，Fusion 策略不重要
**B3 结论**: C3 child aug 微弱正收益 (+0.25~0.74pp)，C2/C4 外域混合显著损害
**B6 结论**: Adapter 负面或中性，可移除

## 目录结构（当前有效部分）

```
├── docs/
│   └── 实验设计方案_v3_含学习笔记.md   # ⭐ 权威实验设计文档
├── src/
│   ├── train.py                    # 统一训练入口 (当前使用)
│   ├── models/ssl_backbone.py      # WavLM 封装 + LayerFusion
│   ├── models/pooling.py           # Mean/Self-Attn/Prosody Pooling
│   ├── models/drse_cnn.py          # SEMLP 分类器
│   └── augmentation/safe_augmentation.py
├── scripts/
│   ├── launch_b1.sh ~ launch_b7.sh # 云端批量启动脚本
│   ├── tmp_paramiko_autodl_runner.py # 云端同步工具
│   └── verify_experiments.py       # 实验完整性校验
├── results/logs/                   # 本地结果 (仅核心6实验)
├── results_remote/results/logs/    # ⭐ 云端完整结果 (163+ JSON)
├── results_remote/results/figures/ # 混淆矩阵图
├── checkpoints/                    # 本地权重 (README.md 有索引)
├── experiments/                    # 历史实验数据 (Phase 3-5, 旧协议)
└── CLAUDE.md
```

## 结果数据权威来源

- **完整结果**: `results_remote/results/logs/` (163+ JSON, 从云端同步)
- **核心快照**: `results/logs/DATA_FREEZE.json`
- **同步命令**: `python scripts/tmp_paramiko_autodl_runner.py --pull-all`
- **注意**: `results/logs/` 只有 6 个核心实验，不要从这里推断完整实验状态

## 关键约束

- 所有 Conv1d 在真实帧级时间轴上滑动，非特征拼接维度
- 不用 `nn.AdaptiveAvgPool1d`，用 `MaxPool1d` 或固定 `AvgPool1d`
- `data_split_seed=42` 固定，`--seed` 控制模型初始化
