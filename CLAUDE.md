# 新方案-分布驱动儿童SER

> **协议**: `ac_suite_2026-06` | **最后更新**: 2026-06-16
> **权威设计文档**: `docs/current/实验设计方案_v3_含学习笔记.md`
> **状态**: 🎉 **192/192 全部完成** (本地+云端双检通过)

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

## 实验矩阵 (B1-B7) — 全部完成

| Phase | E系列 | 内容 | 实验数 | 状态 |
|-------|-------|------|--------|------|
| B1 | E1 | Pooling × Dataset 基线 (frozen) | 9×3=27 | ✅ |
| B2 | E3 | Zero-shot 跨语料迁移 | 18×1=18 | ✅ |
| B3 | E4 | 数据增强敏感性 (C1-C4) | 12×3=36 | ✅ |
| B4 | E5 | LayerFusion 消融 (last/weighted/L1-L12) | 54 | ✅ |
| B5 | E2 | WavLM Unfreeze 对比 | 3×3=9 | ✅ |
| B6 | E6 | 模块消融 (Adapter/Pooling/Fusion) | 10×3=30 | ✅ |
| B7 | E7 | 模型迁移 Fine-tune | 6×3=18 | ✅ |

**总计**: 192/192 runs ✅

## 关键数值

### 各数据集天花板

| 实验 | 数据集 | 配置 | WA (3-seed) | 备注 |
|------|--------|------|-------------|------|
| E2-01 | C-BESD | self_attn + **unfreeze** | **96.91%** | 🔥 全局最高 |
| E1-02 | C-BESD | self_attn + frozen | 92.92% | 冻结天花板 |
| E1-05 | FAU | self_attn + frozen | 67.81% | FAU天花板 |
| E2-03 | IEMOCAP | prosody + unfreeze | 66.36% | IEMOCAP天花板 |
| E3 best | IEMOCAP→C-BESD | self_attn zero-shot | 34.68% | 分布偏移上限 |

### B6 模块消融 (E6, FAU Aibo, 3-seed)

| 实验 | 配置 | WA | 结论 |
|------|------|-----|------|
| E6-01 | Full (Adapter+WeightedFusion+SelfAttn) | 80.89% | 全量基线 |
| E6-02 | w/o Adapter | 81.17% | Adapter移除→+0.28pp |
| E6-03 | MeanPool+WeightedFusion | **92.24%** | 🔥 MeanPool最优 |
| E6-04 | MeanPool only (no Fusion) | 91.96% | 接近天花板 |
| E6-05 | SelfAttn only (no Fusion) | 91.12% | SelfAttn次优 |
| E6-06 | Prosody+WeightedFusion | 67.97% | Prosody显著差 |
| E6-07 | Prosody only (no Fusion) | 68.15% | Prosody独立 |
| E6-08 | Prosody+Mean联合 | 67.80% | 无效组合 |
| E6-09 | SelfAttn+Adapter (no Fusion) | 66.41% | Adapter损害 |
| E6-10 | SelfAttn+Adapter+WeightedFusion | 66.11% | 全Adapter+SelfAttn |

**B6 结论**: 
- MeanPooling 在 FAU 上最优 (91.96-92.24%)，远超 SelfAttn (91.12%)
- Adapter 在所有配置中均负面 (-0.28~-24.7pp)
- LayerFusion 贡献微弱 (+0.28pp for MeanPool)
- Prosody pooling 在 FAU 上严重不足 (~68%)

### B7 模型迁移 (E7, 3-seed)

| 实验 | 配置 | WA | 结论 |
|------|------|-----|------|
| E7-01 | IEMOCAP(Prosody)→C-BESD | 66.82% | 源域Prosody一般 |
| E7-02 | IEMOCAP(Prosody)→C-BESD(Prosody) | 63.25% | Prosody→Prosody更差 |
| E7-03 | C-BESD(SelfAttn)→FAU | **91.57%** | 🔥 跨语料最佳迁移 |
| E7-04 | C-BESD(MeanPool)→FAU(SelfAttn) | 62.81% | MeanPool源域差 |
| E7-05 | C-BESD(WeightedFusion)→FAU(SelfAttn) | **91.17%** | 接近E7-03 |
| E7-06 | C-BESD(DeepFusion L1-8)→FAU | 65.97% | DeepFusion源域差 |

**B7 结论**:
- C-BESD→FAU 迁移效果好 (91%+)，与FAU in-domain天花板有约24pp差距
- 源域 Pooling 策略至关重要：SelfAttn/WeightedFusion 好，Mean/Prosody 差
- IEMOCAP→C-BESD 迁移仅 63-67%，成人→儿童域偏移严重

### 各Phase关键结论汇总

| Phase | 核心结论 |
|-------|---------|
| B1 | SelfAttn > Mean >> Prosody; C-BESD >> FAU ≈ IEMOCAP |
| B2 | Zero-shot 跨语料 34-41%，分布偏移显著 |
| B3 | C3 child aug 微弱正收益 (+0.25~0.74pp)；C2/C4 外域混合显著损害 |
| B4 | last ≈ weighted ≈ 任何单层 L≥7，Fusion 策略不重要 |
| B5 | Unfreeze 在 C-BESD 贡献 +4pp，FAU/IEMOCAP 约 +8pp |
| B6 | MeanPool 最优；Adapter 全面负面；Fusion 微弱正面 |
| B7 | C-BESD→FAU 91%+；源域pooling选择关键；成人→儿童域偏移大 |

## 目录结构

```
├── docs/
│   ├── current/                       # ⭐ 当前使用的核心文档
│   │   ├── 实验设计方案_v3_含学习笔记.md
│   │   ├── AC实验协议.md
│   │   ├── 权威数据手册.md
│   │   ├── SER项目全量台账_数据路径与任务清单.md
│   │   ├── 全部实验结果汇总.md
│   │   └── 模块文档 (1-6)
│   ├── discussion/                    # 在讨论的设计方案
│   │   ├── 讨论纪要_解冻WavLM效果分析与FAU重标注方案.md
│   │   ├── 跨数据集儿童SER数据方案_v2.md
│   │   └── 方向对比与方案设计.md
│   ├── ops/                           # 运维操作手册
│   └── archive/                       # 已迭代的旧版文档
├── paper_draft/
│   ├── current/                       # ⭐ 当前v9版本 (LaTeX + Markdown)
│   ├── archive/                       # v8及更早版本
│   ├── presentations/                 # PPT汇报文件
│   └── figures/                       # 论文图表
├── src/
│   ├── train.py                       # 统一训练入口
│   ├── models/                        # ssl_backbone, pooling, semlp, layer_fusion, adapter
│   ├── augmentation/                  # safe_augmentation, constrained_aug
│   ├── data/                          # 数据加载
│   └── evaluation/                    # 评估工具
├── scripts/
│   ├── launch_b1.sh ~ launch_b7.sh    # 云端批量启动脚本
│   ├── launch_b6_fill.sh              # B6补完脚本
│   ├── tmp_paramiko_autodl_runner.py  # 云端同步工具
│   ├── verify_experiments.py          # 实验完整性校验 (B1/B3)
│   ├── verify_all_192.py              # 全量 192 实验校验 (B1-B7)
│   └── archive/                       # 已迭代脚本 (v2-v6等)
├── results/
│   └── logs/                          # 本地核心结果快照
├── results_remote/
│   └── results/logs/                  # ⭐ 云端完整结果 (192 runs)
├── checkpoints/                       # 本地权重
├── experiments/
│   └── archive/                       # 旧Phase实验数据 (Phase 1-5)
└── CLAUDE.md
```

## 实验完整性校验

- **校验脚本**: `python scripts/verify_all_192.py` — 覆盖 B1-B7 全部 192 runs
- **本地**: `results_remote/results/logs/` — **192/192 ✅** (2026-06-16 复验通过)
- **云端**: `/root/autodl-tmp/d-ser/results/logs/` — **192/192 ✅** (与本地一致)
- **192 + 14 extra JSONs** (exp1~5b 单文件副本, apc_metrics, fau_multiseed_summary 等)

## AutoDL 云端状态

- **当前状态**: 所有实验已完成，数据已全量同步到本地
- **可以关机**: ✅ — 无待跑实验，数据已安全存储在 `results_remote/` + GitHub
- **下次需要时再开机**: 需补充实验、重新训练、或提取 checkpoint 时
- **开机后同步命令**: `python scripts/tmp_paramiko_autodl_runner.py --pull-all`

## 结果数据权威来源

- **完整结果**: `results_remote/results/logs/` (192 runs, 从云端同步)
- **核心快照**: `results/logs/DATA_FREEZE.json`
- **同步命令**: `python scripts/tmp_paramiko_autodl_runner.py --pull-all`
- **完整校验**: `python scripts/verify_all_192.py`

## 关键约束

- 所有 Conv1d 在真实帧级时间轴上滑动，非特征拼接维度
- 不用 `nn.AdaptiveAvgPool1d`，用 `MaxPool1d` 或固定 `AvgPool1d`
- `data_split_seed=42` 固定，`--seed` 控制模型初始化
