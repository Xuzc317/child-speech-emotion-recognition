# 新方案-分布驱动儿童SER

> **协议**: `ac_suite_2026-06` | **最后更新**: 2026-06-16
> **权威设计文档**: `docs/current/实验设计方案_v3_含学习笔记.md`
> **AI理解入口**: `docs/current/AI项目理解提示词.md`
> **状态**: 🎉 **192/192 全部完成** | **补充实验完成** | 配图已整理 | AutoDL可关机

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

## 补充实验 — 已完成

| 项目 | 数量 | 位置 |
|------|------|------|
| 混淆矩阵 (B1/B5/B6/B7代表) | 8组 (PNG+PDF+JSON) | `paper_draft/figures/cm_*.png` + `results/figures/` |
| XAI可视化 (E1-02, E6-04) | 2张 | `paper_draft/figures/xai_*.png` |
| Layer Fusion 权重 (E1-02 L9, E5-03 L9) | 2组 | `paper_draft/figures/layer_weights_*.png` |
| APC 指标 (E1-02, E6-04) | 2组 JSON | `results/analysis/apc_*.json` |
| t-SNE 可视化 (B1 Before/After) | 4张 | `paper_draft/figures/fig_tsne_*.png` |

**配图目录**: `paper_draft/figures/` 已清理 — 归档旧协议68文件，保留33有效文件。

**补充实验方案**: `docs/current/补充实验方案_v1.md`

### 各数据集天花板

| 实验 | 数据集 | 配置 | WA (3-seed) | 备注 |
|------|--------|------|-------------|------|
| E2-01 | C-BESD | self_attn + **unfreeze** | **96.91%** | 🔥 全局最高 |
| E1-02 | C-BESD | self_attn + frozen | 92.92% | 冻结天花板 |
| E1-05 | FAU | self_attn + frozen | 67.81% | FAU天花板 |
| E2-03 | IEMOCAP | prosody + unfreeze | 66.36% | IEMOCAP天花板 |
| E3 best | IEMOCAP→C-BESD | self_attn zero-shot | 34.68% | 分布偏移上限 |

### B6 模块消融 (E6, C-BESD + FAU Aibo 各5组, 3-seed)

设计：同一套消融逻辑在 C-BESD (6类) 和 FAU Aibo (4类) 上各跑一遍，逐步添加 Adapter/Pooling/Fusion，观察模块贡献是否跨数据集一致。

#### C-BESD (E6-01~05)

| 实验 | 配置 | WA | 结论 |
|------|------|-----|------|
| E6-01 | 最简基线 (Mean + last layer) | 80.89% | 基线 |
| E6-02 | +Adapter | 81.17% | Adapter +0.28pp (微弱) |
| E6-03 | +SelfAttn Pooling (替换Mean) | **92.24%** | 🔥 Pooling贡献最大 +11.4pp |
| E6-04 | +LayerFusion (Weighted) | 91.96% | Fusion 微弱负收益 |
| E6-05 | 全栈 (Adapter+SelfAttn+WF) | 91.12% | Adapter拖累全栈 |

#### FAU Aibo (E6-06~10)

| 实验 | 配置 | WA | 结论 |
|------|------|-----|------|
| E6-06 | 最简基线 (Mean + last layer) | 67.97% | 基线 |
| E6-07 | +Adapter | 68.15% | Adapter +0.18pp (微弱) |
| E6-08 | +SelfAttn Pooling | 67.80% | 无明显贡献 |
| E6-09 | +LayerFusion (Weighted) | 66.41% | Fusion 负面 |
| E6-10 | 全栈 (Adapter+SelfAttn+WF) | 66.11% | 全栈最低 |

**B6 结论**:
- C-BESD 上 Pooling 从 Mean→SelfAttn 是关键提升 (+11.4pp)；FAU 上所有模块改良效果均有限
- Adapter 在两个数据集上均无正面贡献
- C-BESD 天花板 ~92%，FAU 天花板 ~68%，差距来自数据集难度而非模块设计

### B7 模型迁移 (E7, 3-seed)

设计：加载 B1 各数据集最优 checkpoint → 在目标域 fine-tune → 评估迁移效果。6 个迁移方向。

| 实验 | 源域→目标域 | WA (3-seed) | 结论 |
|------|-----------|-------------|------|
| E7-01 | C-BESD(SelfAttn)→FAU | 66.82±0.69% | 儿童演绎→儿童自然，中度迁移 |
| E7-02 | C-BESD(SelfAttn)→IEMOCAP | 63.25±0.40% | 儿童→成人，域偏移大 |
| E7-03 | FAU(SelfAttn)→C-BESD | **91.57±0.36%** | 🔥 儿童自然→儿童演绎，最佳迁移 |
| E7-04 | FAU(SelfAttn)→IEMOCAP | 62.81±0.85% | 儿童自然→成人，域偏移大 |
| E7-05 | IEMOCAP(Prosody)→C-BESD | **91.17±1.04%** | 成人→儿童演绎，意外高效 |
| E7-06 | IEMOCAP(Prosody)→FAU | 65.97±0.52% | 成人→儿童自然，域偏移大 |

**B7 结论**:
- 迁移到 C-BESD (儿童演绎) 效果最好 (91%+)，不论源域是什么
- 迁移到 FAU (儿童自然) 和 IEMOCAP (成人) 效果差 (62-67%)
- C-BESD 作为目标域最容易适应（可能是数据质量高、类间边界清晰）
- 成人 IEMOCAP→儿童 C-BESD 迁移达 91%，说明成人演绎数据对儿童演绎有迁移价值

### 各Phase关键结论汇总

| Phase | 核心结论 |
|-------|---------|
| B1 | SelfAttn > Mean >> Prosody; C-BESD >> FAU ≈ IEMOCAP |
| B2 | Zero-shot 跨语料 34-41%，分布偏移显著 |
| B3 | C3 child aug 微弱正收益 (+0.25~0.74pp)；C2/C4 外域混合显著损害 |
| B4 | last ≈ weighted ≈ 任何单层 L≥7，Fusion 策略不重要 |
| B5 | Unfreeze 在 C-BESD 贡献 +4pp，FAU/IEMOCAP 约 +8pp |
| B6 | C-BESD上Pooling(Mean→SelfAttn)贡献+11pp；FAU上所有模块改良有限；Adapter两数据集均无效 |
| B7 | 以C-BESD为目标域的迁移效果最好(91%+)；FAU/IEMOCAP为目标域效果差(62-67%) |

## 目录结构

```
├── docs/
│   ├── current/                       # ⭐ 当前核心文档
│   │   ├── 实验设计方案_v3_含学习笔记.md
│   │   ├── 权威数据手册.md
│   │   └── 模块文档 (1-6)
│   ├── discussion/                    # 设计讨论
│   ├── ops/                           # 运维操作手册 (2026-06更新)
│   └── archive/                       # 已归档旧文档
├── paper_draft/
│   ├── current/                       # ⭐ v9 LaTeX
│   ├── archive/                       # v8及更早
│   ├── presentations/                 # PPT
│   └── figures/                       # 论文配图
├── src/                               # 核心代码
│   ├── models/                        # ssl_backbone, pooling, semlp, layer_fusion
│   ├── data/                          # 数据加载
│   ├── evaluation/                    # FD, XAI
│   ├── augmentation/                  # 数据增强
│   └── training/                      # 训练入口
├── scripts/                           # 脚本工具
│   ├── launch_b1.sh ~ launch_b7.sh    # 云端启动脚本
│   ├── tmp_paramiko_autodl_runner.py  # 云端同步
│   ├── verify_all_192.py              # 192 run 全量校验
│   ├── download_checkpoints.py        # 云端权重下载
│   └── archive/                       # 历史脚本
├── results/                           # ★ 整合后的实验结果
│   ├── logs/                          # 192 B1-B7 JSON
│   ├── analysis/                      # FD, XAI, layer weights
│   ├── figures/                       # 混淆矩阵 (待生成)
│   ├── training_logs/                 # 云端训练日志
│   ├── archive/                       # 旧实验数据
│   └── TODO_补充清单.md                # 待补充分析项
├── checkpoints/
│   ├── autodl/b1~b7/                  # B1-B7 权重 (151个)
│   └── archive/                       # 历史权重
├── experiments/                       # 旧Protocol实验 (已归档)
├── references/                        # 参考文献 PDF
├── _legacy/                           # 已废弃的旧数据/工具
└── CLAUDE.md
```

## 实验完整性校验

- **校验脚本**: `python scripts/verify_all_192.py` — 覆盖 B1-B7 全部 192 runs
- **权威数据源**: `results/logs/` — **192/192 ✅** (2026-06-16 复验通过)
- **云端**: `/root/autodl-tmp/d-ser/results/logs/` — **192/192 ✅** (与本地一致)
- **结果整合**: `results/` 整合了原 `results_remote/` + `results/`，旧数据在 `results/archive/`

## AutoDL 云端状态

- **当前状态**: 所有实验已完成，数据已全量同步到本地
- **可以关机**: ✅ — 无待跑实验，数据已安全存储
- **下次需要时再开机**: 需补充实验、重新训练、或提取 checkpoint 时
- **开机后同步命令**: `python scripts/tmp_paramiko_autodl_runner.py --pull-all`

## 结果数据权威来源

- **完整结果**: `results/logs/` (192 runs, E1-E7 全量 JSON)
- **分析数据**: `results/analysis/` (FD, XAI, layer weights)
- **补充清单**: `results/TODO_补充清单.md`
- **同步命令**: `python scripts/tmp_paramiko_autodl_runner.py --pull-all`
- **下载权重**: `python scripts/download_checkpoints.py`
- **完整校验**: `python scripts/verify_all_192.py`

## 关键约束

- 所有 Conv1d 在真实帧级时间轴上滑动，非特征拼接维度
- 不用 `nn.AdaptiveAvgPool1d`，用 `MaxPool1d` 或固定 `AvgPool1d`
- `data_split_seed=42` 固定，`--seed` 控制模型初始化
