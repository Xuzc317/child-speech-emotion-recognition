# 分布驱动儿童 SER — Canonical 数据手册

> **生成**: 2026-05-27 Claude Code 深度审计  
> **协议**: `ac_suite_2026-05`  
> **验收**: `python scripts/verify_experiment_jsons.py` → 12/12 PASS  
> **权威数据源**: `results/logs/*.json`, `results/canonical_fd_pairs.json`

本文档是投稿唯一数据口径。任何论文、图表、摘要引用的数值必须与本文档一致。

---

## 1. 主实验矩阵（6 组）

| Exp | 训练→测试 | Pooling | Reg | best_epoch | Val WA | Test WA | Test UAR | Test N |
|-----|----------|---------|-----|------------|--------|---------|----------|--------|
| Exp1 | C-BESD→C-BESD | Self-Attention | default | 30 | 92.80% | **92.78%** | **92.79%** | 540 |
| Exp2 | C-BESD→C-BESD | Prosody Guided | default | 10 | 91.26% | **91.30%** | **91.35%** | 540 |
| Exp3 | IEMOCAP→IEMOCAP | Prosody Guided | default | 8 | 57.41% | **58.67%** | **59.11%** | 2371 |
| Exp4 | C-BESD→FAU Aibo | Prosody Guided | default | 10 | 91.26%¹ | **19.56%** | **23.83%** | 3389² |
| Exp5 | FAU→FAU | Prosody Guided | fau | 3 | 64.03% | **66.36%** | **56.35%** | 3389 |
| Exp5b | FAU→FAU | Self-Attention | fau | 2 | 63.42% | **66.18%** | **58.23%** | 3389 |

> ¹ Exp4 best_val 来自 C-BESD 验证集（非 FAU），因为零样本无 FAU 验证集。  
> ² Exp4 2026-05-27 修正：由 `split='all'` (N=18216, WA=18.70%) 改为 `split='test'` (N=3389, WA=19.56%)。

**正则配置**:

| Profile | weight_decay | label_smoothing | pooling_dropout | grad_clip | 适用实验 |
|---------|-------------|-----------------|-----------------|-----------|---------|
| default | 1e-3 | 0.1 | 0.0 | — | Exp1–4 |
| fau | 5e-3 | 0.15 | 0.3 | 1.0 | Exp5, Exp5b |

---

## 2. FAU 多随机种子（C1 稳健性）

| 实验 | Seeds | Test WA mean±std | Test UAR mean±std |
|------|-------|------------------|-------------------|
| Exp5 Prosody | 42, 123, 456 | **65.15% ± 1.12%** | **53.95% ± 2.22%** |
| Exp5b Self-Attn | 42, 123, 456 | **66.46% ± 0.72%** | **55.39% ± 2.05%** |

**明细**:

| Exp | Seed | Test WA | Test UAR | best_epoch |
|-----|------|---------|----------|------------|
| Exp5 | 42 | 66.36% | 56.35% | 3 |
| Exp5 | 123 | 63.65% | 51.00% | 6 |
| Exp5 | 456 | 65.44% | 54.49% | 6 |
| Exp5b | 42 | 66.18% | 58.23% | 2 |
| Exp5b | 123 | 65.76% | 53.50% | 4 |
| Exp5b | 456 | 67.44% | 54.44% | 11 |

---

## 3. 分布偏移诊断（FD & SMMD）

**2026-05-27 AutoDL GPU 重算**。协议：`DistributionShiftProbe`, 在线 WavLM 特征, mean pooling, max_samples=500, seed=42, 70/15/15 hash split。

| 数据集对 | FD | SMMD | 论文用途 |
|----------|-----|------|---------|
| C-BESD vs C-BESD | 0.00 | 0.000 | 同语料基线 (by definition) |
| C-BESD vs IEMOCAP | **7.20** | 0.315 | 年龄偏移 (fig03 row 2) |
| C-BESD vs FAU Aibo | **8.50** | 0.378 | 风格+自发性偏移 (fig03 row 3-4) |
| C-BESD vs CREMA-D | 16.33 | 0.412 | 年龄+语言+录音 (参考，不入图) |

**FD-Accuracy 对应（fig03）**:

| 条件 | FD | 最佳 WA | 实验 |
|------|-----|---------|------|
| 域内表演（同语料） | 0.00 | 92.78% | Exp1 |
| 年龄偏移（成人域内） | 7.20 | 58.67% | Exp3 |
| 风格+自发性（域内） | 8.50 | 66.36% | Exp5 |
| 风格+自发性（零样本） | 8.50 | 19.56% | Exp4 |

> 注：Exp4 和 Exp5 共享同一个 FD=8.50（都是 C-BESD vs FAU），WA 差异来自是否使用域内训练数据（19.56% vs 66.36%），说明域内训练可部分克服分布偏移。

---

## 4. 可解释性诊断（C2）

| 指标 | 值 | 来源 |
|------|-----|------|
| APC_wav | **0.718** | `results/logs/apc_metrics.json` |
| APC_delta | **-0.144** | 同上 |
| Layer argmax (0-based) | **8** | `results/layer_weights.json` |
| Layer argmax (1-based) | **9** | 同上 |
| Layer entropy | **2.484** | 同上 |
| 计算 checkpoint | `checkpoints/exp2_prosody_guided/best_model.pt` | Exp2, epoch 10 |
| 评估样本 | C-BESD test set, seed=42, 首条 utterance | — |

> ⚠️ APC 当前仅基于 1 条样本。论文若写"全 test 平均"，需重跑脚本对所有 540 条 test 样本求 APC 均值。

---

## 5. 混淆矩阵验收

| 图号 | 实验 | Test N | 行和 | 状态 |
|------|------|--------|------|------|
| fig07 | Exp1 Self-Attn C-BESD | 540 | 540 | ✅ |
| fig08 | Exp2 Prosody C-BESD | 540 | 540 | ✅ |
| figA1 | Exp3 IEMOCAP | 2371 | 2371 | ✅ |
| figA2 | Exp4 Zero-Shot FAU | 3389 | 3389 | ✅ (2026-05-27 修正) |
| figA3 | Exp5 FAU Prosody | 3389 | 3389 | ✅ |
| figA4 | Exp5b FAU Self-Attn | 3389 | 3389 | ✅ |

---

## 6. 数据划分统计

**C-BESD** (2780 samples, 237 speakers):
- Train: 1851 (162 spk, 66.6%)
- Val: 389 (36 spk, 14.0%)
- Test: 540 (39 spk, 19.4%)

**IEMOCAP** (8525 samples³, 10 speakers):
- Train: 4313 (5 spk)
- Val: 1841 (2 spk)
- Test: 2371 (3 spk)

**FAU Aibo** (18216 samples, 51 speakers):
- Train: 11577 (35 spk, 63.6%)
- Val: 3250 (7 spk, 17.8%)
- Test: 3389 (9 spk, 18.6%)

> ³ IEMOCAP 有 1269 条因标签不在 {angry, happy, neutral, sad} 中被丢弃。

---

## 7. 训练超参数（AC Suite 通用）

| 参数 | 值 |
|------|-----|
| SSL backbone | WavLM Base+ (microsoft/wavlm-base-sv), frozen |
| Layer fusion | 12-layer learnable weighted sum (WavLMLayerFusion) |
| Pooling | Self-Attention (111,105 params) 或 Prosody Guided (111,105 params) |
| Classifier | SEMLP (593K params) |
| 总可训参数 | ~704K |
| Optimizer | AdamW, lr=3e-4, CosineAnnealing(T_max=100) |
| Early stop | patience=15, monitor=val WA |
| Batch size | 16 |
| Max duration | 4s (200 frames @ 50Hz) |
| Speaker split | MD5 hash, 70/15/15, seed=42, zero-leakage assert |

---

## 8. 权威文件路径索引

### 8.1 Canonical 实验 JSON

```
results/logs/
├── exp1_self_attention.json          # Exp1: Self-Attn C-BESD
├── exp2_prosody_guided.json          # Exp2: Prosody C-BESD
├── exp3_adult_iemocap.json           # Exp3: Prosody IEMOCAP
├── exp4_zero_shot_fau.json           # Exp4: Zero-shot C-BESD→FAU (N=3389)
├── exp5_fau_indomain.json            # Exp5: Prosody FAU
├── exp5b_self_attention_fau.json     # Exp5b: Self-Attn FAU
├── exp5_fau_indomain_s123.json       # Exp5 seed 123
├── exp5_fau_indomain_s456.json       # Exp5 seed 456
├── exp5b_self_attention_fau_s123.json
├── exp5b_self_attention_fau_s456.json
├── fau_multiseed_summary.json        # FAU mean±std
├── apc_metrics.json                  # APC=0.718
└── DATA_FREEZE.json                  # 冻结记录
```

### 8.2 诊断数据

```
results/
├── canonical_fd_pairs.json           # 2026-05-27 AutoDL GPU 重算
├── distribution_shift.json           # C-BESD vs CREMA-D (注意：非 FAU)
├── layer_weights.json                # Layer argmax=9 (1-based)
├── xai_final.png
├── xai_raw_data.npz
└── xai_sample_visualization.png
```

### 8.3 模型权重

```
checkpoints/
├── autodl/                           # AC Suite (10 exp × best_model.pt, ~363 MB/个)
│   ├── exp1_self_attention/
│   ├── exp2_prosody_guided/
│   ├── exp3_adult_iemocap/
│   ├── exp4_zero_shot_fau/
│   ├── exp5_fau_indomain/
│   ├── exp5b_self_attention_fau/
│   └── exp{5,5b}_*_s{123,456}/
└── v5_622/                           # 历史 Phase 5 (24 个 .pth)
```

### 8.4 投稿资产

```
publication_package/
├── experiment_results.csv            # 主表 CSV
├── fd_accuracy_table.json            # fig03 数据源 (2026-05-27 verified)
├── distribution_shift.json
├── layer_weights.json
├── xai_raw_data.npz
├── README_for_Agents.md
└── logs/                             # 同 results/logs/ 同步副本
```

### 8.5 论文源文件

```
paper_draft/
├── main.tex                          # 主文档
├── 0_Abstract.tex
├── 1_Introduction.tex
├── 2_Related_Work.tex
├── 3_Methodology.tex
├── 4_Experiments_and_Results.tex
├── 5_Analysis_and_Discussion.tex
├── 6_Conclusion.tex
├── references.bib
├── Full_Draft.docx
├── Full_Draft_CN.docx
├── figures/
│   ├── fig01–05_*.png/pdf            # 占位图 (generate_paper_figures.py)
│   ├── fig07–08_confusion_*.png/pdf  # 正文 CM
│   ├── figA1–A4_confusion_*.png/pdf  # 附录 CM
│   └── FIGURES_MANIFEST.json
└── README.md
```

### 8.6 云端训练日志

```
results_remote/training_logs/
├── ac_suite_logs/
│   ├── run_20260526_160746.log       # Exp1 46 epoch + Exp2 + Exp4
│   ├── resume_20260526_183253.log
│   ├── resume_20260526_185400.log    # Exp5 s123 + Exp5b s123 训练
│   └── resume_20260526_223135.log    # Exp5+5b s456 + CM + multiseed
├── matrix_run.log                    # Exp3 IEMOCAP 完整 epoch
├── matrix_cbesd.log                  # Exp1/Exp2 C-BESD
└── exp2_checkpoint.log
```

---

## 9. 已知限制与待办

| 项目 | 说明 |
|------|------|
| APC | 仅 1 条样本；论文若写"全 test 平均"需重算 |
| fig06 | Layer fusion weights 图未生成 |
| fig00 | System architecture 图未生成 |
| Exp4 | best_val 来自 C-BESD val（非 FAU），零样本场景无 FAU val |
| APC 旧值 | 0.698 (v5 叙事) 和 0.515 (云端 bug) 已废弃，勿引用 |
| FD 旧值 | 6.87 (v5)、12.33 (旧本地) 已废弃，以本文档 §3 为准 |
| SMMD 旧值 | 0.369 (旧本地)、0.41 (文中小数) 已废弃，以本文档 §3 为准 |

---

*本文档为投稿唯一数据口径。所有数值均经 `verify_experiment_jsons.py` 验收通过。*
