# 新方案-分布驱动儿童SER

> **协议**: `ac_suite_2026-06-validated` | **最后更新**: 2026-06-27 | **校验**: Phase 0-4 通过, 0 INVALID
> **权威设计文档**: `docs/current/实验设计方案_v3_含学习笔记.md`
> **AI理解入口**: `docs/current/AI项目理解提示词.md`
> **状态**: 🎉 **210/210 全部完成** | **补充实验完成** | 配图已整理 | AutoDL可关机

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
| B7 | E7 | 模型迁移 Fine-tune (frozen) | 6×3=18 | ✅ |
| B7-ext | E7-ext | 解冻迁移对比 (unfrozen) | 6×3=18 | ✅ |

**总计**: 210/210 runs ✅ (B1-B7 192 + B7-ext 18)

## 补充实验 — 已完成

| 项目 | 数量 | 位置 |
|------|------|------|
| 混淆矩阵 (B1/B5/B6/B7代表) | 8组 (PNG+PDF+JSON) | `paper_draft/figures/confusion_matrices/` + `results/figures/` |
| XAI可视化 (E1-02, E6-04) | 2张 | `paper_draft/figures/xai/` |
| Layer Fusion 权重 (E1-02 L9) | 1组 | `paper_draft/figures/layer_weights/` |
| APC 指标 (E1-02, E6-04) | 2组 JSON | `results/analysis/apc_*.json` |
| t-SNE 可视化 (B1 Before/After) | 4张 | `paper_draft/figures/tsne/` |

**配图目录**: `paper_draft/figures/` 已清理 — 归档旧协议68文件，保留33有效文件。

**补充实验方案**: `docs/current/补充实验方案_v1.md`

### 各数据集天花板

| 实验 | 数据集 | 配置 | WA (3-seed) | 备注 |
|------|--------|------|-------------|------|
| E2-01 | C-BESD | self_attn + **unfreeze** | **96.91%** | 🔥 全局最高 |
| E1-02 | C-BESD | self_attn + frozen | 91.87% | 冻结天花板 (3-seed样本mean, ddof=1) |
| E1-05 | FAU | self_attn + frozen | 67.05% | FAU天花板 (3-seed样本mean, ddof=1) |
| E2-03 | IEMOCAP | prosody + unfreeze | 66.36% | IEMOCAP天花板 |
| E3 best | IEMOCAP→C-BESD | self_attn zero-shot | 34.68% | 分布偏移上限 |

### B6 模块消融 (E6, C-BESD + FAU Aibo 各5组, 3-seed)

设计：cumulative build-up design（累加式），观察各模块负贡献。C-BESD (E6-01~05) + FAU (E6-06~10)。**权威配置以 `docs/current/权威数据手册.md` 为准。**

#### C-BESD (E6-01~05)

| 实验 | 配置（数据手册权威版） | WA | 结论 |
|------|------|-----|------|
| E6-01 |Mean+Last基线| 80.89% | 全栈基线 |
| E6-02 | +Adapter| 81.17% | 加Adapter +0.28pp |
| E6-03 | +SelfAttn,−Adapter†| **91.91%** | 🔥 换SelfAttn +11pp |
| E6-04 |+WeightedFusion | **91.96%** | 🔥 加WF微弱+0.05pp |
| E6-05 | 全栈| 91.12% | 加Adapter反而-0.84pp |

#### FAU Aibo (E6-06~10)

| 实验 | 配置（数据手册权威版） | WA | 结论 |
|------|------|-----|------|
| E6-06 | Mean+Last基线  | 67.97% | FAU基线 |
| E6-07 | +Adapter| 68.15% | 加Adapter +0.18pp |
| E6-08 | +SelfAttn,−Adapter† | 67.79% | 无明显贡献 |
| E6-09 | +WeightedFusion | 66.41% | 负面 |
| E6-10 | 全栈 | 66.11% | 全栈最低 |

**B6 结论**:
- C-BESD 上 Pooling 从 Mean→SelfAttn 是关键提升 (+11pp)；FAU 上所有模块改良效果有限
- Adapter 在两个数据集上均无正面贡献（可安全移除）
- LayerFusion 正收益 <1pp，可视为冗余
- C-BESD 天花板 ~91.9%，FAU ~67.1%，差距来自数据集难度而非模块设计

### B7 模型迁移 (E7, 3-seed)

设计：加载 B1 各数据集最优 checkpoint → 在目标域 fine-tune → 评估迁移效果。`launch_b7.sh` 对全部6组统一传入 `--pooling_type self_attention`，故微调阶段 Pooling 头均为 self_attention（IEMOCAP 源的 prosody_guided 头不参与迁移，仅 backbone+LayerFusion 权重迁移）。

以下为权威数据手册记录（`docs/current/权威数据手册.md` §B7），已与 `results/logs/E7-*.json` 的 `train_data`/`test_data` 字段核对一致：

| 实验 | 源域→目标域 | 源Pooling (E1预训练) | WA (3-seed) | 结论 |
|------|-----------|----------|-------------|------|
| E7-01 | C-BESD→FAU | self_attention | 66.82±0.84% | 儿童→儿童跨语料，中等 |
| E7-02 | C-BESD→IEMOCAP | self_attention | 63.25±0.49% | 儿童→成人，中等 |
| E7-03 | FAU→C-BESD | self_attention | **91.57±0.44%** | 🔥 最佳迁移 |
| E7-04 | FAU→IEMOCAP | self_attention | 62.81±0.85% | 目标域仍是IEMOCAP，效果一般 |
| E7-05 | IEMOCAP→C-BESD | prosody_guided | **91.17±1.28%** | 🔥 源Pooling不同但效果相近 |
| E7-06 | IEMOCAP→FAU | prosody_guided | 65.97±0.64% | 目标域仍是FAU，效果一般 |

**B7 结论**:
- **目标域自身天花板主导迁移结果，源域影响较小**：C-BESD 为目标 (E7-03/05) 收敛至91%+，接近 C-BESD 自身天花板 (91.87%)；FAU 为目标 (E7-01/06) 聚集在66-67%，接近 FAU 天花板 (67.05%)；IEMOCAP 为目标 (E7-02/04) 聚集在62-63%
- FAU→C-BESD 迁移效果最佳 (91.57%)，与 C-BESD 域内 (91.87%) 差距仅 0.30pp
- 源域 (FAU vs IEMOCAP) 对同一目标的影响有限（C-BESD 目标: 91.57% vs 91.17%，仅差0.4pp），fine-tune 后模型很大程度回归目标域固有难度

### B7-ext 解冻迁移对比 (E7-07~E7-12, 3-seed)

设计：B7 仅做 frozen backbone 微调迁移。B7-ext 以相同 source checkpoint + 相同 transfer direction，唯一变量 `--unfreeze_ssl` + 差分学习率 (backbone 1e-5, head 3e-4)，形成严格 apples-to-apples 对照。batch_size=8（与 B5 一致，解冻 ~95M 参数后 24GB 显存安全值）。

| 实验 (frozen) | 实验 (unfrozen) | 源域→目标域 | WA frozen | WA unfrozen | Δ | UAR unfrozen | 结论 |
|---------------|----------------|-----------|-----------|-------------|---|-------------|------|
| E7-01 | **E7-07** | C-BESD→FAU | 66.82±0.84% | 65.95±2.16% | −0.87 | 41.37±4.06% | FAU目标解冻无效 |
| E7-02 | **E7-08** | C-BESD→IEMOCAP | 63.25±0.49% | 66.85±0.58% | +3.60 | 61.35±0.67% | IEMOCAP目标解冻增益 |
| E7-03 | **E7-09** | FAU→C-BESD | 91.57±0.44% | **96.96±0.61%** | +5.39 | 96.93±0.61% | 🔥 解冻逼近C-BESD天花板 |
| E7-04 | **E7-10** | FAU→IEMOCAP | 62.81±0.85% | 67.19±0.25% | +4.38 | 62.10±0.99% | IEMOCAP目标解冻增益 |
| E7-05 | **E7-11** | IEMOCAP→C-BESD | 91.17±1.28% | **96.40±0.59%** | +5.23 | 96.38±0.60% | 🔥 解冻逼近C-BESD天花板 |
| E7-06 | **E7-12** | IEMOCAP→FAU | 65.97±0.64% | 65.30±1.09% | −0.67 | 42.31±1.93% | FAU目标解冻无效 |

**B7-ext 结论**:
- **解冻增益完全由目标域决定**：C-BESD 为目标 +5.2~5.4pp → 逼近 C-BESD from-scratch 天花板 (96.91%)；IEMOCAP 为目标 +3.6~4.4pp；FAU 为目标 −0.7~−0.9pp（解冻反而略差，batch_size=8 可能欠拟合或 FAU 数据噪声主导）
- E7-09 (FAU→C-BESD unfrozen) WA=96.96%，已超越 C-BESD from-scratch unfrozen (E2-01, 96.91%)，说明 FAU 预训练 + C-BESD fine-tune 的组合优于纯 C-BESD 训练
- **实用建议**：若目标域为高资源清晰数据集（如 C-BESD），解冻迁移值得做；若目标域为低天花板数据集（如 FAU），冻结迁移即可，解冻不带来收益

### 各Phase关键结论汇总

| Phase | 核心结论 |
|-------|---------|
| B1 | SelfAttn > Mean >> Prosody; C-BESD >> FAU ≈ IEMOCAP |
| B2 | Zero-shot 跨语料 19.17%-35.47%，分布偏移显著 |
| B3 | C3 child aug 微弱正收益 (+0.25~0.74pp)；C2/C4 外域混合显著损害 |
| B4 | last ≈ weighted ≈ 任何单层 L≥7，Fusion 策略不重要 |
| B5 | Unfreeze 在 C-BESD 贡献 +4pp，FAU/IEMOCAP 约 +8pp |
| B6 | C-BESD上Pooling(Mean→SelfAttn)贡献+11pp；FAU上所有模块改良有限；Adapter两数据集均无效 |
| B7 | 目标域自身天花板主导迁移结果：以C-BESD为目标(91%+) > 以FAU为目标(66-67%) > 以IEMOCAP为目标(62-63%)，源域影响较小 |
| B7-ext | 解冻增益完全由目标域决定：C-BESD目标+5.2~5.4pp逼近天花板，IEMOCAP目标+3.6~4.4pp，FAU目标解冻无效(−0.7~−0.9pp)；FAU预训练+C-BESD微调(96.96%)超越纯C-BESD训练(96.91%) |

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
│   ├── current/                       # ⭐ v10 LaTeX (v10_main.tex + v10_*.tex)
│   │   # ⚠️ v10_main.tex 的 \input 命令仍指向旧文件 (0_Abstract 等)
│   │   # 应改为 \input{v10_0_Abstract} 等才能使用 v10 版章节内容
│   ├── archive/                       # v9及更早
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
│   ├── logs/                          # 210 B1-B7+B7-ext JSON
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

- **校验脚本**: `python scripts/verify_all_192.py` — 覆盖 B1-B7 全部 192 runs（B7-ext E7-07~E7-12 含 18 runs 未纳入此脚本，需手动验证）
- **权威数据源**: `results/logs/` — **210/210 ✅** (2026-06-27 复验通过)
- **云端**: `/root/autodl-tmp/d-ser/results/logs/` — **210/210 ✅** (与本地一致)
- **结果整合**: `results/` 整合了原 `results_remote/` + `results/`，旧数据在 `results/archive/`

## AutoDL 云端状态

- **当前状态**: 所有实验已完成，数据已全量同步到本地
- **可以关机**: ✅ — 无待跑实验，数据已安全存储
- **下次需要时再开机**: 需补充实验、重新训练、或提取 checkpoint 时
- **开机后同步命令**: `python scripts/tmp_paramiko_autodl_runner.py --pull-all`

## 结果数据权威来源

- **完整结果**: `results/logs/` (210 runs, E1-E7-ext 全量 JSON)
- **分析数据**: `results/analysis/` (FD, XAI, layer weights)
- **补充清单**: `results/TODO_补充清单.md`
- **同步命令**: `python scripts/tmp_paramiko_autodl_runner.py --pull-all`
- **下载权重**: `python scripts/download_checkpoints.py`
- **完整校验**: `python scripts/verify_all_192.py`

## 关键约束

- 所有 Conv1d 在真实帧级时间轴上滑动，非特征拼接维度
- 不用 `nn.AdaptiveAvgPool1d`，用 `MaxPool1d` 或固定 `AvgPool1d`
- `data_split_seed=42` 固定，`--seed` 控制模型初始化

## 已知问题与数据不一致记录 (2026-06-17)

### ⚠️ v10_main.tex \input 路径错误
`paper_draft/current/v10_main.tex` 中 `\input{0_Abstract}` 等命令指向旧版文件，而 v10 章节内容在 `v10_0_Abstract.tex` 等文件中。编译时会加载旧内容。**修复方式**: 将所有 `\input{X}` 改为 `\input{v10_X}`。

### ✅ B6 配置方向已澄清 (2026-06-19, 以训练日志为准)
B6 实为**累加式 (build-up)** 消融：从极简基线逐步叠加模块。依据 `results/logs/E6-0X_s42.json` 的 `pooling_type`/`use_adapter`/`fusion_mode` 实测：E6-01 (mean/无Adapter/last, 80.89%) → E6-02 +Adapter (81.17%) → E6-03 换SelfAttn (91.91%) → E6-04 +WeightedFusion (91.96%) → E6-05 +Adapter全栈 (91.12%)。**E6-01 是极简基线，不是全栈基线**；关键提升来自换 Pooling (+11pp)。
注：2026-06-17 曾误将其"更正"为"从全栈逐步移除、E6-01 是全栈基线"，与训练日志不符，已废弃。`权威数据手册.md` §B6 核心结论 (约 line 158/199/208) 仍把 80.89% 误标为"全栈 Adapter+WF+SA"，待同步更正。

### ✅ B7 迁移方向标注分歧 (已于 2026-06-19 修复)
CLAUDE.md 与权威数据手册曾共享同一份错误的 B7 表格：source→target 方向标反，且每行虚构了不同的"源Pooling"(mean/weighted_fusion/deep_fusion)，但 `launch_b7.sh` 实际对全部6组统一传入 `--pooling_type self_attention`。已对照 `scripts/launch_b7.sh` 逐行逻辑 + `results/logs/E7-*.json` 的 `train_data`/`test_data`/`pooling_type` 字段 + `paper_draft/current/v10_4_Experiments_and_Results.tex`（三方互证一致）重写两份文档的 B7 表格与结论。新结论：目标域自身天花板主导迁移结果，源域影响较小。

### ✅ 2026-06-22 校验修正 (Phase 0-4) — ALL CLEAN (2026-06-23 更新)
- **天花板数字修正**: E1-02 91.87%, E1-05 67.05%（3-seed 样本 mean, ddof=1）
- **标准差口径**: 统一 ddof=1 (样本标准差)
- **5 个实验重跑完成**: E1-08_s42, E4-04_s42, E4-10_s42, E4-10_s123, E4-12_s123 已修复，210/210 全量有效
  - E1-08: mean=64.05±0.22% (3-seed) ✅
  - E4-10: mean=65.44±0.26% (3-seed) ✅
  - E4-12: mean=61.16±0.67% (3-seed) ✅
- **0 INVALID experiments**
- **6 个配置字段不可信**: augment_condition/fusion_mode/use_adapter/unfreeze_ssl/reg_profile/fusion_best_layer 为代码默认值（旧协议文件）
- **B7 源域无法独立确认**: 日志未记录源 checkpoint，依赖 launch_b7.sh 正确执行
- 详见 `validation/reproducibility_report.md`

### ℹ️ 部分 JSON 协议字段为旧版
`results/logs/` 下部分 JSON 文件含 `"protocol": "ac_suite_2026-05"`，这是协议迭代前生成的。数值本身有效，协议字段忽略即可。
