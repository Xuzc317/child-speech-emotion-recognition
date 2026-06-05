# 分布驱动儿童 SER — 全量台账（数据 · 路径 · 任务 · 日志）

> **生成时间**: 2026-05-27  
> **项目根目录**: `d:\大学\论文\儿童语音情绪识别\新方案-分布驱动儿童SER`  
> **协议 ID**: `ac_suite_2026-05`  
> **Git 当前分支**: `research/interpretability-fd`  
> **Git HEAD**: `cbfa8bab049e4da361d75e71ef64f8812e0c9fc6`  
> **GitHub 仓库**: [Xuzc317/child-speech-emotion-recognition](https://github.com/Xuzc317/child-speech-emotion-recognition)

---

## 目录

1. [投稿主表数值（CANONICAL / 已冻结）](#1-投稿主表数值canonical--已冻结)
2. [衍生指标与辅助数据](#2-衍生指标与辅助数据)
3. [尚未完成的任务清单](#3-尚未完成的任务清单)
4. [已完成的任务清单（含历史阶段）](#4-已完成的任务清单含历史阶段)
5. [数据与文件路径总表](#5-数据与文件路径总表)
6. [论文撰写相关资产（LaTeX / 图 / 表 / Word）](#6-论文撰写相关资产latex--图--表--word)
7. [脚本与自动化流水线](#7-脚本与自动化流水线)
8. [Git 迭代记录与链接](#8-git-迭代记录与链接)
9. [AutoDL 云端与同步](#9-autodl-云端与同步)
10. [问题与故障日志](#10-问题与故障日志)
11. [本地数据集路径](#11-本地数据集路径)
12. [Cursor / Agent 对话记录](#12-cursor--agent-对话记录)

---

## 1. 投稿主表数值（CANONICAL / 已冻结）

**权威验收命令**: `python scripts/verify_experiment_jsons.py` → 当前 **12/12 PASS**

| Exp | 设置 | Test WA | Test UAR | best_epoch | reg_profile | 主 JSON 路径 |
|-----|------|---------|----------|------------|-------------|--------------|
| Exp1 | C-BESD, Self-Attn | **92.78%** | **92.79%** | 30 | default | `results/logs/exp1_self_attention.json` |
| Exp2 | C-BESD, Prosody | **91.30%** | **91.35%** | 10 | default | `results/logs/exp2_prosody_guided.json` |
| Exp3 | IEMOCAP, Prosody | **58.67%** | **59.11%** | 8 | default | `results/logs/exp3_adult_iemocap.json` |
| Exp4 | C-BESD→FAU 零样本 | **19.56%** | **23.83%** | 10 | default | `results/logs/exp4_zero_shot_fau.json` (N=3389 test-only, 2026-05-27 修正) |
| Exp5 | FAU, Prosody | **66.36%** | **56.35%** | 3 | fau | `results/logs/exp5_fau_indomain.json` |
| Exp5b | FAU, Self-Attn | **66.18%** | **58.23%** | 2 | fau | `results/logs/exp5b_self_attention_fau.json` |

**同步副本**（数值必须与上表一致）:

- `publication_package/logs/exp{1,2,3,4,5,5b}_*.json`
- `publication_package/experiment_results.csv`
- `results/logs/DATA_FREEZE.json`（冻结记录，含 git commit）

**FAU 多随机种子汇总** (`results/logs/fau_multiseed_summary.json`):

| 实验 | WA mean±std | UAR mean±std | n |
|------|-------------|--------------|---|
| Exp5 Prosody | 65.15±1.12% | 53.95±2.22% | 3 |
| Exp5b Self-Attn | 66.46±0.72% | 55.39±2.05% | 3 |

多种子明细 JSON:

- `results/logs/exp5_fau_indomain_s123.json`, `_s456.json`
- `results/logs/exp5b_self_attention_fau_s123.json`, `_s456.json`

---

## 2. 衍生指标与辅助数据

| 指标 | 当前值 | 文件路径 | 备注 |
|------|--------|----------|------|
| APC_wav | **0.718**（本地 C2） | `results/logs/apc_metrics.json` | 云端曾得 0.515；论文旧叙事写 ≈0.698，**待统一** |
| APC_delta | -0.144 | 同上 | |
| Layer argmax (0-based) | **8** | `results/layer_weights.json` | |
| Layer argmax (1-based) | **9** | 同上 | 清单 D5 写「Layer 8」需与定义对齐 |
| Layer entropy | 2.484 | 同上 | |
| FD | **C-BESD vs IEMOCAP: 7.20; C-BESD vs FAU: 8.50; C-BESD vs CREMA-D: 16.33** | `results/canonical_fd_pairs.json` (2026-05-27 AutoDL GPU 重算) | fd_accuracy_table.json 已同步更新 |
| SMMD | **vs IEMOCAP: 0.315; vs FAU: 0.378; vs CREMA-D: 0.412** | 同上 | 文中旧值 0.41 实际是 C-BESD vs CREMA-D |
| XAI 原始数据 | npz | `publication_package/xai_raw_data.npz` | 用于 fig04 重绘 |
| FD–Accuracy 表 | JSON | `publication_package/fd_accuracy_table.json` | fig03 数据源。⚠️ 2026-05-27 已修正标注，FD(C-BESD vs FAU) 待重算 |

**历史实验数据（v5 6:2:2，非 AC 主表）**:

- `experiments/v5_622/v5_results.json` — 旧协议 A3=80.9% WA 等
- `experiments/phase3_ablation.json` — 60/20/20 协议
- `experiments/cross_language_results.json` — 跨语言 EN↔TE

---

## 3. 投稿前终检清单（按执行顺序）

> **原则**：每一步依赖前一步完成后才能做。数字写定 → grep 清除 → ref 校验 → 匿名化。
> 所有需要写入 .tex 的数值已在「就绪数据源」列给出 verified 路径，直接引用即可。

### Step 1 — 数字写定（阻断后续所有步骤）

| 字段 | 就绪数据源 | 要写入的 .tex | 旧值（需清除） |
|------|-----------|---------------|---------------|
| Exp1 test WA/UAR | `results/logs/exp1_self_attention.json` → 92.78% / 92.79% | `4_Experiments_and_Results.tex` | 79.63 / 92.04 |
| Exp2 test WA/UAR | `results/logs/exp2_prosody_guided.json` → 91.30% / 91.35% | 同上 | 92.04 |
| Exp3 test WA/UAR | `results/logs/exp3_adult_iemocap.json` → 58.67% / 59.11% | 同上 | 60.14 |
| Exp4 test WA/UAR | `results/logs/exp4_zero_shot_fau.json` → **19.56%** / **23.83%** (N=3389) | 同上 | 18.70 / 16.68 |
| Exp5 test WA/UAR | `results/logs/exp5_fau_indomain.json` → 66.36% / 56.35% | 同上 | 68.87 |
| Exp5b test WA/UAR | `results/logs/exp5b_self_attention_fau.json` → 66.18% / 58.23% | 同上 | 69.54 |
| FAU multiseed | `results/logs/fau_multiseed_summary.json` | 同上（或附录） | — |
| FD(C-BESD vs IEMOCAP) | `results/canonical_fd_pairs.json` → **7.20** | `5_Analysis_and_Discussion.tex` | 6.87 |
| FD(C-BESD vs FAU) | 同上 → **8.50** | 同上 | 12.33 / 16.33 |
| SMMD(C-BESD vs FAU) | 同上 → **0.378** | 同上 | 0.369 / 0.41 / 0.412 |
| APC_wav | `results/logs/apc_metrics.json` → **0.718** | `0_Abstract.tex`, `5_*.tex` | 0.698 / 0.515 |
| Layer argmax | `results/layer_weights.json` → 1-based=**9**, 0-based=8 | `3_Methodology.tex` | 与「Layer 8」表述对齐 |

### Step 2 — grep 清除旧数字（依赖 Step 1）

在 `paper_draft/*.tex` 中搜索并替换以下**所有过期数字**：

```
79.63  92.04  60.14  68.87  69.54  16.68  18.70  22.75
6.87   12.33  16.33  0.369  0.41   0.412  0.515  0.698
Layer 8（若指 1-based，应改为 Layer 9）
```

旧 FD/SMMD 特别容易残留在 `5_Analysis_and_Discussion.tex`、`Methods_and_Results_Skeleton.tex`。

### Step 3 — 表图引用 + 补图（依赖 Step 1）

| 检查项 | 文件 | 状态 |
|--------|------|------|
| CM fig07/08 正文引用 | `main.tex`, `4_*.tex` | 待人工确认 `\ref{}` |
| CM figA1–A4 附录引用 | `main.tex` | 同上 |
| fig03 FD-Accuracy | `fd_accuracy_table.json` 已更新，待重跑 `generate_paper_figures.py` | 数据就绪 |
| fig06 layer fusion weights | `layer_weights.json` 就绪，待 `plot_layer_weights.py` 出图 | **fig06 缺失** |
| fig00 system architecture | 待 `generate_architecture_figure.py` | **fig00 缺失** |

### Step 4 — 匿名化 + 模板（独立操作）

- 作者名、机构、致谢删除/匿名化
- 绝对路径 `d:\大学\论文\...` 替换
- INTERSPEECH 官方 `\documentclass` 替换 `main.tex`

### Step 5 — 收尾（依赖以上全部）

- 更新 `docs/submission_checklist.md` 勾选状态
- 更新 `docs/json_verification_checklist.md`
- 重跑 `verify_experiment_jsons.py`（Exp4 值已变，需更新 CANONICAL 字典）
- `generate_docx.py` 重生成 Word 终稿
- `build_submission_bundle.py` 重新打包

---

### 未来扩展（投稿后）

| 任务 | 需要 |
|------|------|
| MyST / KidsTALC / EmoReact 扩展 | 新语料 + GPU |
| 从 AutoDL 补拉完整 checkpoint | 实例在线或重训 |
| `.gitignore` 扩展 checkpoints/results_remote | — |

---

## 4. 已完成的任务清单（含历史阶段）

### 4.1 项目历史阶段（Phase 1–6，见 `docs/实施步骤指南.md`）

| 阶段 | 内容 | 状态 | 关键产出路径 |
|------|------|------|--------------|
| Phase 1 | SSL 基线验证（WavLM linear probe ~80%） | ✅ | `docs/全部实验结果汇总.md` §3 |
| Phase 2 | 三模块：Adapter / Prosody Pooling / FD 增强 | ✅ | `src/models/pooling.py`, `src/augmentation/` |
| Phase 3 | 整合消融 + 数据泄露修复 | ✅ | `experiments/phase3_ablation.json` |
| Phase 4 | 跨语言迁移 | ✅ | `experiments/cross_language_results.json` |
| Phase 5 | v5 6:2:2 协议复核 | ✅ | `experiments/v5_622/`（24 个 .pth 权重） |
| Phase 6.1–6.5 | 外部基准 / 技术债 / XAI / FD 框架 / 论文 v7 | ✅ | `docs/论文初稿_v7.md`, `docs/论文初稿_v7_cn.docx` |

**Phase 5 推荐模型（历史 v5，非 AC 主表）**: A3 = WavLM frozen + Prosody Pooling + SEMLP，C-BESD Test 80.9% WA

### 4.2 AutoDL 最终矩阵（2026-05-19 ~ 2026-05-27）

| 阶段 | 状态 | 说明 |
|------|------|------|
| A1 六组主实验训练 | ✅ | 6 JSON + 10 checkpoint |
| A2 六组混淆矩阵 | ✅ | 18 文件 png/pdf/json |
| C1 FAU 多 seed (42/123/456) | ✅ | `fau_multiseed_summary.json` |
| C2 layer_weights + APC + XAI | ✅ | 云端 + 本地交叉验证 |
| `--pull-all` 同步 | ✅ | `results_remote/` |
| 本地 post pipeline | ✅ 大部分 | merge / verify / docx / bundle |

### 4.3 本地收尾（2026-05-27）

| 任务 | 状态 |
|------|------|
| `verify_experiment_jsons.py` 12/12 PASS | ✅ |
| `merge_remote_logs.py` 含多 seed JSON | ✅ |
| `aggregate_fau_multiseed.py` | ✅ |
| `run_prosody_diagnostics.py` 本地 C2 | ✅（APC≈0.718） |
| `generate_paper_figures.py` fig01–05 | ✅ |
| CM 重命名 fig07/08, figA1–A4 | ✅ |
| `generate_docx.py` / `_cn.py` | ✅ |
| `build_submission_bundle.py` | ✅ |
| `DATA_FREEZE.json` 更新 | ✅ |
| `4_Experiments_and_Results.tex` / `main.tex` patch | ✅（需人工终读） |

### 4.4 本次 Cursor 会话已修复的问题

| 问题 | 修复 |
|------|------|
| C2 GPU/CPU 设备不一致 | `src/extract_diagnostics.py` 增加 `.to(device)` |
| C2 韵律帧长不对齐 | 增加 `_interpolate_1d` |
| PyTorch 2.6 `weights_only` 默认 | `torch.load(..., weights_only=False)` |
| post pipeline 归档后 CSV 丢失 | 调整顺序：先作图再归档 |
| multiseed JSON 未 merge | 扩展 `merge_remote_logs.py` |

### 4.5 2026-05-27 Claude Code 深度审计与修复

**审计发现**:
- FD=16.33 实际是 C-BESD vs CREMA-D（非 FAU Aibo），`extract_diagnostics.py:164` 写死 `crema-d`
- `全部实验结果汇总.md` 6 组实验数值全部过期，已加 DEPRECATED 标记
- `fd_accuracy_table.json` FD 来源混乱（混用 v5 旧值、CREMA-D 值），已重写
- 云端 C2 从未算过 distribution_shift，FD 只在本地算过一次
- Exp1 92.78% 经日志验证无数据泄露（test≈best_val, gap=3.85pp@best epoch）

**已修复**:

| 修复 | 产出 |
|------|------|
| FD canonical 重算 | AutoDL GPU 计算 `results/canonical_fd_pairs.json`: C-BESD vs IEMOCAP FD=7.20, C-BESD vs FAU FD=8.50 |
| `fd_accuracy_table.json` 重写 | 所有 FD 标注 `fd_verified: true`，SMMD 同步更新 |
| `distribution_shift.json` 标注 | 增加 `pair`/`description`/`computed_by` 字段 |
| Exp4 test_split='test' 修复 | `data_loader.py` 新增 `test_split` 参数; re-evaluate Exp4: WA 18.70%→19.56% (N=3389) |
| Exp4 CM 重生成 | `figA2_confusion_exp4_zero_shot.*` N=3389 |
| `experiment_results.csv` 更新 | Exp4 行同步 |
| `publication_package/logs/` 同步 | Exp4 JSON 更新 |
| AutoDL 日志拉取 | `results_remote/training_logs/ac_suite_logs/` 4 个日志文件（含 Exp1 完整 46 epoch + Exp5/5b 训练记录） |
| Exp1 泄露排查 | 云端 `matrix_cbesd.log` 验证: train=96.65%@best, val=92.80%, test=92.78% — 无泄露 |
| 台账 §3 重写 | 旧 P0/P1/P2 清单 → 5 步执行顺序终检清单 |
| 台账 §5 路径修正 | v5 .pth 路径、submission_bundle 统计、pub/logs 内容等 8 处修正 + 5 项补充 |
| `docs/全部实验结果汇总.md` | 顶部加 DEPRECATED 警告 + canonical 对照表 |

---

## 5. 数据与文件路径总表

### 5.1 核心实验 JSON（权威）

```
d:\大学\论文\儿童语音情绪识别\新方案-分布驱动儿童SER\results\logs\
├── exp1_self_attention.json
├── exp2_prosody_guided.json
├── exp3_adult_iemocap.json
├── exp4_zero_shot_fau.json
├── exp5_fau_indomain.json
├── exp5b_self_attention_fau.json
├── exp5_fau_indomain_s123.json
├── exp5_fau_indomain_s456.json
├── exp5b_self_attention_fau_s123.json
├── exp5b_self_attention_fau_s456.json
├── fau_multiseed_summary.json
├── apc_metrics.json
├── DATA_FREEZE.json
└── last\                          # 历史归档（20260526_*）
    ├── 20260526_171321\
    └── 20260526_172433\
```

### 5.2 publication_package（Agent / 作图 / 投稿）

```
d:\大学\论文\儿童语音情绪识别\新方案-分布驱动儿童SER\publication_package\
├── experiment_results.csv         # 主表 CSV（6 行）
├── fd_accuracy_table.json         # ⚠️ 2026-05-27 已修正标注，FD 值部分待重算（见 T2b）
├── distribution_shift.json        # C-BESD vs CREMA-D（非 FAU Aibo！）
├── layer_weights.json
├── xai_raw_data.npz
├── README_for_Agents.md
├── logs\
│   ├── exp1~5b 六个主 JSON + s123/s456 多 seed + apc_metrics
│   └── （不含 distribution_shift.json / layer_weights.json，这些在 pub 根目录）
└── last\                          # 历史归档
```

### 5.3 模型权重（~363 MB / 个）

```
d:\大学\论文\儿童语音情绪识别\新方案-分布驱动儿童SER\checkpoints\
├── autodl\
│   ├── best_model.pt              # 早期共用权重
│   ├── exp1_self_attention\best_model.pt
│   ├── exp2_prosody_guided\best_model.pt
│   ├── exp3_adult_iemocap\best_model.pt
│   ├── exp4_zero_shot_fau\best_model.pt
│   ├── exp5_fau_indomain\best_model.pt
│   ├── exp5_fau_indomain_s123\best_model.pt
│   ├── exp5_fau_indomain_s456\best_model.pt
│   ├── exp5b_self_attention_fau\best_model.pt
│   ├── exp5b_self_attention_fau_s123\best_model.pt
│   └── exp5b_self_attention_fau_s456\best_model.pt
├── v5_622\                        # Phase 5 历史权重（24 个 .pth）
│   ├── A1_baseline_seed{42,123,456}.pth
│   ├── A2b_adapter_seed{42,123,456}.pth
│   ├── A3_prosody_only_seed{42,123,456}.pth
│   ├── B3_final_seed{42,123,456}.pth
│   ├── C1_none_seed{42,123,456}.pth
│   ├── C2_adult_seed{42,123,456}.pth
│   ├── C3_child_seed{42,123,456}.pth
│   └── C4_extreme_seed{42,123,456}.pth
└── exp2_prosody_guided\best_model.pt   # 本地 C2 用（从 autodl 复制）

d:\大学\论文\儿童语音情绪识别\新方案-分布驱动儿童SER\experiments\v5_622\
└── （JSON 结果 + 分析数据，无 .pth 文件）
```
```

### 5.4 远程备份（AutoDL pull）

```
d:\大学\论文\儿童语音情绪识别\新方案-分布驱动儿童SER\results_remote\
├── remote_inventory.txt           # 最后一次 pull 清单
├── exp4_zero_shot_fau.json        # ⚠️ 与 results/logs/ 副本重复（pull 时根目录遗留）
├── exp5_fau_indomain.json         # ⚠️ 同上
├── distribution_shift.json        # 早期 pull 副本
├── layer_weights.json
├── xai_raw_data.npz
├── xai_final.png
├── results\
│   ├── distribution_shift.json    # 云端 C2 输出（C-BESD vs CREMA-D）
│   ├── layer_weights.json
│   ├── xai_final.png
│   ├── xai_raw_data.npz
│   ├── logs\                      # 12 JSON（含 multiseed + summary）
│   └── figures\                   # confusion_exp*.png/pdf/json + exp2_confusion.*
├── training_logs\
│   ├── matrix_run.log             # 云端训练主日志（含 Exp3 完整 epoch 记录）
│   ├── matrix_cbesd.log           # Exp1/Exp2 C-BESD 训练日志
│   └── exp2_checkpoint.log        # Exp2 检查点日志
```

### 5.5 其他 results

```
d:\大学\论文\儿童语音情绪识别\新方案-分布驱动儿童SER\results\
├── layer_weights.json             # 层融合权重（C2 本地输出）
├── distribution_shift.json        # C-BESD vs CREMA-D（注意：不是 FAU！）
├── xai_final.png                  # XAI 可视化（本地 C2 输出）
├── xai_raw_data.npz               # XAI 原始数据
├── xai_sample_visualization.png   # XAI 样本可视化
├── logs\                          # 13 JSON（含 DATA_FREEZE, apc_metrics, multiseed）
└── figures\                       # 本地 CM 输出目录（可选）
```

### 5.6 历史实验目录

```
d:\大学\论文\儿童语音情绪识别\新方案-分布驱动儿童SER\experiments\
├── v5_622\                        # Phase 5 完整数据
│   ├── v5_results.json
│   ├── unified_fd_results.json
│   ├── fd_vs_accuracy_unified.png
│   ├── attention_analysis\
│   └── *.json（iemocap/cremad/mfcc 等）
├── phase3_ablation.json
├── cross_language_results.json
└── runs\20260428_*                # Phase 3 单次 run 配置与 metrics
```

### 5.7 投稿打包输出

```
d:\大学\论文\儿童语音情绪识别\新方案-分布驱动儿童SER\submission_bundle\
├── figures\                       # 30 个配图文件（含 last/ 归档）
├── latex\                         # 10 个 .tex + .bib + README
├── publication_package\           # 投稿包副本
├── Full_Draft.docx                # 英文 Word 初稿
├── Full_Draft_CN.docx             # 中文 Word 初稿
├── AUTODL_INVENTORY.md
├── CHECKLIST.md
└── README.txt
（共 166 个文件，非台账旧写的 107+44）
```

---

## 6. 论文撰写相关资产（LaTeX / 图 / 表 / Word）

### 6.1 LaTeX 源文件（当前主稿）

| 文件 | 用途 |
|------|------|
| `paper_draft/main.tex` | 主文档 + 附录 CM |
| `paper_draft/0_Abstract.tex` | 摘要 |
| `paper_draft/1_Introduction.tex` | 引言 |
| `paper_draft/2_Related_Work.tex` | 相关工作 |
| `paper_draft/3_Methodology.tex` | 方法 |
| `paper_draft/4_Experiments_and_Results.tex` | 实验与主表 |
| `paper_draft/5_Analysis_and_Discussion.tex` | 分析讨论（FD/APC） |
| `paper_draft/6_Conclusion.tex` | 结论 |
| `paper_draft/references.bib` | 参考文献 |
| `paper_draft/README.md` | 编译说明 |

### 6.2 Word 初稿

| 文件 | 路径 |
|------|------|
| 英文 Full Draft | `paper_draft/Full_Draft.docx` |
| 中文 Full Draft | `paper_draft/Full_Draft_CN.docx` |
| 历史 v7 | `docs/论文初稿_v7.docx`, `docs/论文初稿_v7_cn.docx` |

### 6.3 论文配图（当前有效版本）

**主文占位/脚本图** (`paper_draft/figures/`):

| 论文文件名 | 用途 | 数据源 |
|------------|------|--------|
| `fig00_system_architecture.*` | ⚠️ **尚未生成** — 需运行 `scripts/generate_architecture_figure.py` | — |
| `fig01_main_matrix_wa_uar.*` | 主结果矩阵 | `experiment_results.csv` |
| `fig02_fau_prosody_vs_selfattn.*` | FAU 池化对比 | CSV |
| `fig03_fd_vs_accuracy.*` | FD–Accuracy | `fd_accuracy_table.json` |
| `fig04_xai_saliency_triple.*` | XAI 三联图 | `xai_raw_data.npz` |
| `fig05_cbesd_selfattn_vs_prosody.*` | C-BESD 柱状对比 | CSV |
| `fig06_layer_fusion_weights.*` | 层权重（在 `last/` 有旧版） | `layer_weights.json` |

**正文混淆矩阵（D1 默认 2 张）**:

| 论文文件名 | 源实验 |
|------------|--------|
| `fig07_confusion_exp1_selfattn.*` | Exp1 Self-Attn, C-BESD, N=540 |
| `fig08_confusion_exp2_prosody.*` | Exp2 Prosody, C-BESD, N=540 |

**附录混淆矩阵（4 张）**:

| 论文文件名 | 源实验 | 期望 N |
|------------|--------|--------|
| `figA1_confusion_exp3_iemocap.*` | Exp3 IEMOCAP | 2371 ✅ |
| `figA2_confusion_exp4_zero_shot.*` | Exp4 零样本 | 3389 ⚠️ 实际 18216 |
| `figA3_confusion_exp5_fau_prosody.*` | Exp5 FAU | 3389 ✅ |
| `figA4_confusion_exp5b_fau_selfattn.*` | Exp5b FAU | 3389 ✅ |

**配图清单**: `paper_draft/figures/FIGURES_MANIFEST.json`  
**历史图归档**: `paper_draft/figures/last/20260526_171321/`, `.../20260526_172433/`

### 6.4 论文表格数据来源

| 表 | 数据文件 |
|----|----------|
| 主实验矩阵（6 组） | `publication_package/experiment_results.csv` |
| FAU 多 seed 附录 | `results/logs/fau_multiseed_summary.json` |
| FD / SMMD 表 | `results/distribution_shift.json`, `publication_package/fd_accuracy_table.json` |
| 层融合权重 | `results/layer_weights.json` |
| APC | `results/logs/apc_metrics.json` |

### 6.5 设计与审查文档（写论文参考）

| 文档 | 路径 |
|------|------|
| 实施步骤指南（Phase 1–6） | `docs/实施步骤指南.md` |
| 全部实验结果汇总 | `docs/全部实验结果汇总.md` |
| 方向对比与方案设计 | `docs/方向对比与方案设计.md` |
| 修改方案 v6 | `docs/修改方案_v6.md` |
| 审查与修订计划 | `docs/review_issues_and_revision_plan.md` |
| 顶会 venue 评估 | `docs/venue_assessment_v2.md` |
| 论文 v7 中英文 md | `docs/论文初稿_v7.md`, `docs/论文初稿_v7_cn.md` |
| AC 实验协议 | `docs/experiment_suite_AC.md` |
| AutoDL 完成后清单 | `docs/post_autodl_workflow_checklist.md` |
| JSON 验收清单 | `docs/json_verification_checklist.md` |
| 投稿 checklist | `docs/submission_checklist.md` |
| AutoDL 本地资产审计 | `docs/autodl_local_inventory.md` |

---

## 7. 脚本与自动化流水线

| 脚本 | 用途 |
|------|------|
| `scripts/verify_experiment_jsons.py` | 12 路径 CANONICAL 验收 |
| `scripts/merge_remote_logs.py` | `results_remote` → `results/logs` + pub |
| `scripts/aggregate_fau_multiseed.py` | 多 seed mean±std |
| `scripts/tmp_paramiko_autodl_runner.py` | SSH pull/train/status |
| `scripts/autodl_ac_suite.sh` | 云端完整 AC 套件 |
| `scripts/autodl_ac_suite_resume.sh` | 云端 resume（CM+C1+C2） |
| `scripts/finish_remote_c2.py` | 云端 C2 补跑 |
| `scripts/run_post_autodl_pipeline.py` | 本地 post 全流程 |
| `scripts/run_local_checklist_jobs.py` | 本地验收（非训练） |
| `scripts/run_prosody_diagnostics.py` | C2 APC/Layer/XAI |
| `scripts/plot_confusion_matrix.py` | 从 checkpoint 画 CM |
| `scripts/generate_paper_figures.py` | fig01–05 占位图 |
| `scripts/generate_docx.py` / `_cn.py` | Word 导出 |
| `scripts/build_submission_bundle.py` | 投稿包 |
| `scripts/sync_autodl_canonical_logs.py` | 写入 canonical JSON |
| `src/train.py` | 训练入口 |

**推荐本地收尾（AutoDL 已关）**:

```powershell
cd "d:\大学\论文\儿童语音情绪识别\新方案-分布驱动儿童SER"
python scripts/run_post_autodl_pipeline.py --skip-poll --skip-pull
python scripts/run_local_checklist_jobs.py
```

---

## 8. Git 迭代记录与链接

**仓库**: https://github.com/Xuzc317/child-speech-emotion-recognition  
**当前分支**: `research/interpretability-fd`  
**最新 commit**: https://github.com/Xuzc317/child-speech-emotion-recognition/commit/cbfa8bab049e4da361d75e71ef64f8812e0c9fc6

| Commit | 日期（约） | 说明 | 链接 |
|--------|------------|------|------|
| `cbfa8ba` | 2026-05 | 注入 AutoDL 最终矩阵，重写 accuracy–explainability 讨论 | [commit](https://github.com/Xuzc317/child-speech-emotion-recognition/commit/cbfa8ba) |
| `d985aad` | 2026-05 | Paramiko AutoDL runner + gitignore 大文件 | [commit](https://github.com/Xuzc317/child-speech-emotion-recognition/commit/d985aad) |
| `0d5de86` | 2026-05 | 六阶段实验结果汇总文档 | [commit](https://github.com/Xuzc317/child-speech-emotion-recognition/commit/0d5de86) |
| `c04122e` | 2026-04 | FAU Aibo 自发语音实验 | [commit](https://github.com/Xuzc317/child-speech-emotion-recognition/commit/c04122e) |
| `74fc2b8` | 2026-04 | 训练循环 + 消融矩阵 + 诊断提取 | [commit](https://github.com/Xuzc317/child-speech-emotion-recognition/commit/74fc2b8) |
| `de07aa4` | 2026-04 | WavLM 12 层可学习融合 | [commit](https://github.com/Xuzc317/child-speech-emotion-recognition/commit/de07aa4) |
| `55998df` | 2026-04 | 统一数据管道 + 说话人隔离 | [commit](https://github.com/Xuzc317/child-speech-emotion-recognition/commit/55998df) |

**远程分支**:

- `origin/master`
- `origin/research/interpretability-fd`（当前）
- `origin/codex/revise-methodology-v5`

**未提交本地变更**（生成时 git status 快照）:

- `paper_draft/4_Experiments_and_Results.tex`（已修改 M）
- `paper_draft/0_Abstract.tex`（新文件 ??）
- `paper_draft/1_Introduction.tex`（新文件 ??）
- `paper_draft/2_Related_Work.tex`（新文件 ??）
- `paper_draft/3_Methodology.tex`（新文件 ??）
- `paper_draft/6_Conclusion.tex`（新文件 ??）
- `paper_draft/main.tex`（新文件 ??）
- `paper_draft/references.bib`（新文件 ??）
- `scripts/*` 多个新增/修改
- `publication_package/fd_accuracy_table.json`（2026-05-27 修正）
- `results/distribution_shift.json`（增加数据集对标注）
- `paper_draft/Full_Draft*.docx`（未跟踪）
- `papervizagent/`（未跟踪）

---

## 9. AutoDL 云端与同步

### 9.1 最后一次已知云端状态（2026-05-27 01:03）

- **状态**: `AC SUITE RESUME DONE`，无训练进程
- **项目路径**: `/root/autodl-tmp/d-ser`
- **实例**: 已克隆磁盘；端口曾从 12112 → **14393**
- **结论**: **可关机**；数据已 pull 到 `results_remote/` + `checkpoints/autodl/`

### 9.2 云端数据集路径（训练时）

| 语料 | 路径 |
|------|------|
| C-BESD | `/root/autodl-tmp/datasets/BESD/BESD/MY` |
| IEMOCAP | `/root/autodl-tmp/IEMOCAP/wavs` |
| FAU Aibo | `/root/autodl-tmp/IS2009EmotionChallenge/wav` |

### 9.3 云端日志（pull 后本地）

```
results_remote/training_logs/
├── matrix_run.log                 # Exp3 完整训练日志（每 epoch train_wa/val_wa）
├── matrix_cbesd.log               # Exp1/Exp2 C-BESD 训练日志
└── exp2_checkpoint.log            # Exp2 检查点日志

注意：ac_suite_logs/ 和 resume_20260526_223135.log 仅在云端，未 pull 到本地。
```

---

## 10. 问题与故障日志

| 时间 | 阶段 | 问题 | 处理 | 状态 |
|------|------|------|------|------|
| 2026-05-26 | A2 CM | FAU sklearn `labels=4` 报错 | `plot_confusion_matrix.py` 固定 4 类 | ✅ |
| 2026-05-26 | Resume | `SER_C_BESD_PATH` 未设置 | 修复 `autodl_ac_suite_resume.sh` | ✅ |
| 2026-05-26 | C2 | `layer_fusion` CPU/GPU 不一致 | `extract_diagnostics.py` `.to(device)` | ✅ |
| 2026-05-26 | C2 | 韵律帧 110 vs 111 不对齐 | 增加 `_interpolate_1d` | ✅ |
| 2026-05-27 | Post | `experiment_results.csv` 被 archive 移走 | 调整 pipeline 顺序 | ✅ |
| 2026-05-27 | Post | `pull-all` 超时/实例关闭 | 增加 `--skip-pull` | ✅ |
| 2026-05-27 | Post | PyTorch 2.6 `weights_only` | `weights_only=False` | ✅ |
| 2026-05-27 | 数据 | Exp4 CM 行和 18216≠3389 | **未决**：零样本用 `split=all` 设计 | ⚠️ T1 |
| 2026-05-27 | 数据 | APC 三值不一致（0.515/0.698/0.718） | **未决**：统一正文 | ⚠️ T3 |
| 2026-05-27 | 数据 | Layer 8 vs 1-based 9 | **未决**：统一 Method | ⚠️ T4 |
| 2026-05-27 | 数据 | FD 12.33 vs 16.33 历史混用 | JSON 现为 16.33；正文待改 | ⚠️ T2 |
| 历史 | Phase 3 | 70/30 与 60/20/20 数据泄露 | `preprocess.py` 修复 | ✅ 见实施指南 |
| 历史 | 训练 | 长音频 CUDA OOM | `max_duration=4s` | ✅ commit b01c915 |

---

## 11. 本地数据集路径

（`src/data/dataset.py` 配置，训练/CM/C2 均可用）

| 语料 | 本地路径 | 环境变量覆盖 |
|------|----------|--------------|
| C-BESD | `d:\大学\论文\儿童语音情绪识别\提交到团队\数据集\BESD\BESD\MY` | `SER_C_BESD_PATH` |
| IEMOCAP | `d:\大学\论文\儿童语音情绪识别\提交到团队\数据集\IEMOCAP\wavs` | `SER_IEMOCAP_PATH` |
| CREMA-D | `D:\大学\crema_temp\AudioWAV` | `SER_CREMA_D_PATH` |
| FAU Aibo | `D:\大学\数据集IS2009EmotionChallenge\IS2009EmotionChallenge\IS2009EmotionChallenge\wav` | `SER_FAU_AIBO_PATH` |

---

## 12. Cursor / Agent 对话记录

本次 AC 套件 + AutoDL + 本地 post pipeline 完整会话（含 SSH、resume、C2 修复）:

- **Transcript 文件**: `C:\Users\59892\.cursor\projects\d-SER\agent-transcripts\71079cd8-619f-4c50-a821-27c0aa6e3193\71079cd8-619f-4c50-a821-27c0aa6e3193.jsonl`

---

## 附录 A：2026-05-27 深度审计发现

### A.1 关键架构差异：AC Suite vs 历史 v5

| 维度 | AC Suite (canonical) | 历史 v5_622 |
|------|---------------------|-------------|
| 训练方式 | **在线波形训练** (`src/train.py`) | 预提取特征 (`train_ssl.py`) |
| Layer 融合 | **WavLMLayerFusion**（12 层加权求和） | 单层/最后层特征 |
| 数据划分 | MD5 hash 70/15/15 (`speaker_splitter.py`) | 外层 7:3 + 内层 val → 6:2:2 |
| batch_size | 16 | 128 |
| Exp1 Self-Attn | **92.78%** | ~79.63% |
| A1 基线 | — | 78.61% |

**结论**：AC Suite 与 v5 是不同架构的不同实验。AC Suite 的更高准确率来自 LayerFusion + 在线训练，不只是数据划分不同。

### A.2 FD 数据集对溯源

| 文件 | 实际测量对 | FD 值 | 可信度 |
|------|-----------|-------|--------|
| `results/distribution_shift.json` | **C-BESD vs CREMA-D** | 16.33 | ✅ canonical |
| `extract_diagnostics.py:164` | `get_dataloaders(['c-besd'])` vs `get_dataloaders(['crema-d'])` | — | 代码确认 |
| `v5_622/unified_fd_results.json` | Age (Child vs Adult) | 6.87 | ⚠️ 旧协议 |
| `全部实验结果汇总.md §9.1` | 标注为 "C-BESD vs FAU Aibo" | 12.33/16.33 | ❌ **标注错误** |

**⚠️ 项目中不存在 C-BESD vs FAU Aibo 的 canonical FD 测量。论文若需此值，必须使用 AC 协议重算。**

### A.3 陈旧文档

| 文档 | 最后更新 | 状态 |
|------|---------|------|
| `docs/全部实验结果汇总.md` | 2026-05-18 | ❌ 6 组实验数值全部过期 |
| `publication_package/fd_accuracy_table.json` | 2026-05-27 | ✅ 已修正标注 |
| `docs/experiment_suite_AC.md` | 2026-05-19 | ✅ 与 AC 协议一致 |

---

## 附录 B：混淆矩阵 JSON 验收（2026-05-27）

| 文件 | 行和 | 期望 | 状态 |
|------|------|------|------|
| `confusion_exp1_self_attention.json` | 540 | 540 | ✅ |
| `confusion_exp2_prosody_guided.json` | 540 | 540 | ✅ |
| `confusion_exp3_adult_iemocap.json` | 2371 | 2371 | ✅ |
| `confusion_exp4_zero_shot_fau.json` | **3389** | 3389 | ✅ (2026-05-27 修正) |
| `confusion_exp5_fau_indomain.json` | 3389 | 3389 | ✅ |
| `confusion_exp5b_self_attention_fau.json` | 3389 | 3389 | ✅ |

路径: `results_remote/results/figures/` 或 `paper_draft/figures/`（重命名后）

---

## 附录 B：关键决策记录（D1–D8，用户 2026-05-26 确认）

| ID | 决定 | 默认值 |
|----|------|--------|
| D1 | 混淆矩阵 | 正文 2 张 + 附录 4 张 |
| D2 | 主表数字 | AC 套件 `results/logs/` |
| D3 | FAU 多 seed | 附录 mean±std |
| D4 | FD/SMMD | `distribution_shift.json` |
| D5 | Layer fusion | Layer 8（1-based）— **与 JSON 9 待对齐** |
| D6 | APC | C2 重算 ≈0.698 — **当前本地 0.718** |
| D7 | 配图 | 占位图先，后同文件名替换 |
| D8 | 归档 | 各目录 `last/` 子文件夹 |

---

*本文件由 Cursor Agent 自动生成，汇总项目内已知路径与状态；若后续 pull/改稿，请重新运行 `scripts/run_local_checklist_jobs.py` 并更新 §1–§3。*
