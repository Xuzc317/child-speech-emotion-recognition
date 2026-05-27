# AutoDL 完成后 · 本地收尾任务核对清单

> **协议 ID**: `ac_suite_2026-05`  
> **创建**: 2026-05-26  
> **用途**: 云端跑完后按序执行；请你逐项 **☐/☑ 核对** 后再进入投稿。  
> **数字权威源（冻结后）**: `results/logs/*.json` + `fau_multiseed_summary.json` + C2 诊断 JSON  
> **用户确认 (2026-05-26)**: D1–D8 全部同意默认；云端完成后自动执行 `scripts/run_post_autodl_pipeline.py`

---

## 0. 当前云端状态（2026-05-27 更新）

| 阶段 | 状态 | 说明 |
|------|------|------|
| **A1 训练** 六组主实验 | ✅ 完成 | JSON 与 CANONICAL 一致；权重在 `checkpoints/autodl/` |
| **A2 混淆矩阵** | ✅ 完成 | 6 组 CM 已拉取并重命名为 fig07/08、figA1–A4 |
| **C1 FAU 多种子** | ✅ 完成 | seeds 42/123/456；`fau_multiseed_summary.json` |
| **C2 APC/Layer/XAI** | ✅ 完成 | 云端 + 本地交叉验证；本地 APC_wav≈0.718 |
| **本地 post pipeline** | ⚠️ 部分 | verify/merge/docx/bundle 已跑；Exp4 CM 行和待统一（见 §2.3） |

**云端恢复命令**（A1 已完成时只需 resume）：

```powershell
python scripts/tmp_paramiko_autodl_runner.py --run-ac-suite-resume
python scripts/tmp_paramiko_autodl_runner.py --ac-suite-status
```

---

## 1. 需你确认的决定项（默认推荐已标注 ⭐）

请在每项后写 **同意默认 / 修改意见**：

| ID | 决定项 | ⭐ 默认推荐 | 影响路径 |
|----|--------|-------------|----------|
| **D1** | **混淆矩阵：正文几张？** | ⭐ **正文 2 张**（Exp1 Self-Attn + Exp2 Prosody，C-BESD 核心对比）；**附录 4 张**（Exp3/4/5/5b） | `paper_draft/figures/fig07_*`, `fig08_*`；附录 `figA1–A4` 或 `appendix_figures/` |
| **D2** | **主表数字以哪为准？** | ⭐ **AC 套件 `results/logs/`**（已与 CANONICAL 一致）；不再引用 `experiments/v5_622/` | 表 1、摘要、讨论全文 |
| **D3** | **FAU 多种子如何写？** | ⭐ 正文一句 + **附录小表** mean±std（`fau_multiseed_summary.json`）；主表仍用 seed=42 | `4_Experiments_and_Results.tex` 新小节 |
| **D4** | **FD/SMMD 用哪套？** | ⭐ 以 **`results/distribution_shift.json`**（本地 FD=12.33 等）为准；文中 16.33/0.41 若与 JSON 不符则 **统一改 JSON 值或重算** | `5_Analysis_and_Discussion.tex` FD 表 |
| **D5** | **Layer fusion 层号** | ⭐ **Layer 8**（1-based），来自 AC C2 重导出的 `layer_weights.json` | `fig06_layer_fusion_weights`, `3_Methodology.tex` |
| **D6** | **APC 值** | ⭐ C2 重算后写入；预期 ≈ **0.698** | 摘要、讨论、fig04 |
| **D7** | **配图策略** | ⭐ **先用脚本生成占位图**（`generate_paper_figures.py` + 云端 confusion）；你后续替换同文件名 | `paper_draft/figures/` |
| **D8** | **旧版归档** | ⭐ 更新前复制到各目录下 **`last/`** 子文件夹（见 §3） | 全项目 |

---

## 2. 数据核对清单（冻结前必做）

### 2.1 主实验矩阵（P0）

| 文件 | 核对项 | 期望（CANONICAL / AC） |
|------|--------|------------------------|
| `results/logs/exp1_self_attention.json` | test_wa / uar, seed=42, reg=default | 92.78% / 92.79% |
| `results/logs/exp2_prosody_guided.json` | 同上 | 91.30% / 91.35% |
| `results/logs/exp3_adult_iemocap.json` | 同上 | 58.67% / 59.11% |
| `results/logs/exp4_zero_shot_fau.json` | train c-besd, test fau | 18.70% / 22.75% |
| `results/logs/exp5_fau_indomain.json` | reg=fau | 66.36% / 56.35% |
| `results/logs/exp5b_self_attention_fau.json` | reg=fau | 66.18% / 58.23% |

**命令**: `python scripts/verify_experiment_jsons.py` → 须 **12/12 PASS**

### 2.2 衍生指标（P1）

| 文件 | 核对项 |
|------|--------|
| `results/logs/fau_multiseed_summary.json` | Exp5/5b 各 3 seed，WA/UAR mean±std |
| `results/logs/apc_metrics.json` | `apc_wav`, `checkpoint` 指向 Exp2 |
| `results/layer_weights.json` | `argmax_layer_1based` = 8，entropy ≈ 2.43 |
| `results/distribution_shift.json` | FD/SMMD 与正文 FD 表一致 |
| `publication_package/xai_raw_data.npz` | 与 fig04 可重绘 |

### 2.3 混淆矩阵（P1，resume 完成后）

| 远程 → 本地 | 行和 = test N |
|-------------|----------------|
| `results/figures/confusion_exp1_self_attention.json` | C-BESD test **540** |
| `confusion_exp2_prosody_guided.json` | 540 |
| `confusion_exp3_adult_iemocap.json` | IEMOCAP test **2371** |
| `confusion_exp4_zero_shot_fau.json` | FAU test **3389**（跨语料） |
| `confusion_exp5_fau_indomain.json` | 3389 |
| `confusion_exp5b_self_attention_fau.json` | 3389 |

### 2.4 权重与日志（P2）

| 路径 | 说明 |
|------|------|
| `checkpoints/autodl/exp{1,2,3,4,5,5b}_*/best_model.pt` | 各 ~363MB |
| `results_remote/training_logs/*.log` | epoch 过程 |
| `results/logs/ac_suite_manifest.json` | 产物清单 |

### 2.5 已知不一致项（须在 Phase 4 统一）

| 位置 | 现写 | 待核对源 |
|------|------|----------|
| FD 表 age shift | 6.87 | `distribution_shift.json` / 旧 v5_622 |
| FD zero-shot | 16.33 | 同上 |
| SMMD | 0.41 | 同上 vs 本地 0.369 |
| `Methods_and_Results_Skeleton.tex` | 旧 92.04% 等 | **不更新主稿**，仅归档 |

---

## 3. 路径地图（同步 / 归档 / 更新）

### 3.1 云端 → 本地同步

```powershell
python scripts/tmp_paramiko_autodl_runner.py --pull-all
python scripts/merge_remote_logs.py
python scripts/aggregate_fau_multiseed.py
python scripts/verify_experiment_jsons.py
```

| 云端 | 本地落点 |
|------|----------|
| `results/logs/` | `results_remote/results/logs/` → merge → `results/logs/` + `publication_package/logs/` |
| `checkpoints/` | `checkpoints/autodl/` |
| `results/figures/confusion_*` | `results_remote/results/figures/` → 复制/重命名到 `paper_draft/figures/` |
| `ac_suite_logs/` | `results_remote/training_logs/`（或 `ac_suite_logs/`） |
| `publication_package/` | `publication_package/` |

### 3.2 归档规则（更新前执行）

在 **每个被覆盖的目录** 下创建 `last/`，移动**当前**文件（非 last）进去：

| 目录 | 归档内容 | 归档目标 |
|------|----------|----------|
| `results/logs/` | 现有 `*.json`（除 smoke） | `results/logs/last/` |
| `publication_package/` | csv, json, npz, md | `publication_package/last/` |
| `paper_draft/figures/` | 将被替换的 fig | `paper_draft/figures/last/` |
| `paper_draft/` | `Full_Draft*.docx` | `paper_draft/last/` |
| `checkpoints/autodl/` | 旧 best_model.pt（若覆盖） | `checkpoints/autodl/last/` |

**不移动**: `experiments/v5_622/`（已是历史）；`docs/` 旧稿单独 `docs/last/` 可选。

### 3.3 配图命名（⭐ D1 默认）

| 用途 | 源文件（脚本/云端） | 论文文件名 |
|------|---------------------|------------|
| 正文 C-BESD 对比 | `confusion_exp1_*` | `fig07_confusion_exp1_selfattn.{png,pdf}` |
| 正文 C-BESD 对比 | `confusion_exp2_*` | `fig08_confusion_exp2_prosody.{png,pdf}` |
| 附录 | `confusion_exp3_*` | `figA1_confusion_exp3_iemocap.*` |
| 附录 | `confusion_exp4_*` | `figA2_confusion_exp4_zero_shot.*` |
| 附录 | `confusion_exp5_*` | `figA3_confusion_exp5_fau_prosody.*` |
| 附录 | `confusion_exp5b_*` | `figA4_confusion_exp5b_fau_selfattn.*` |
| 占位主图 | 已有 | `fig00–fig06`（generate_paper_figures.py） |
| XAI | C2 npz | `fig04_xai_saliency_triple.*` |

### 3.4 需更新数字/图的文档（仅「最新档」）

| 优先级 | 文件 |
|--------|------|
| P0 | `paper_draft/0_Abstract.tex` |
| P0 | `paper_draft/4_Experiments_and_Results.tex`（主表 + FAU 多种子句） |
| P0 | `paper_draft/5_Analysis_and_Discussion.tex`（FD 表、APC、+1.48pp） |
| P0 | `paper_draft/1_Introduction.tex`, `6_Conclusion.tex` |
| P1 | `paper_draft/main.tex`（figure 环境、附录） |
| P1 | `paper_draft/3_Methodology.tex`（Layer 8 一句） |
| P1 | `publication_package/experiment_results.csv` |
| P1 | `paper_draft/figures/FIGURES_MANIFEST.json` |
| P2 | `scripts/generate_docx.py` → `Full_Draft.docx` / `Full_Draft_CN.docx` |
| P2 | `docs/json_verification_checklist.md` |
| P2 | `docs/submission_checklist.md` |
| 不更新 | `Methods_and_Results_Skeleton.tex` → 移 `paper_draft/last/` |

---

## 4. 按序执行任务清单

### Phase 0 — 云端收尾（AutoDL 上）

- [ ] **0.1** 确认 A1 六组 JSON + checkpoint 齐全（已完成可跳过）
- [ ] **0.2** 重跑 ` --run-ac-suite-resume`（已修复路径 + CM bug）
- [ ] **0.3** `--ac-suite-status` 直到出现 `AC SUITE RESUME DONE`
- [ ] **0.4** `--pull-all` + merge + aggregate + verify

### Phase 1 — 本地验收（数据冻结）

- [ ] **1.1** 跑通 `verify_experiment_jsons.py`（12/12）
- [ ] **1.2** 核对 §2.2 衍生指标文件存在且合理
- [ ] **1.3** 核对 §2.3 六组混淆矩阵 JSON 行和
- [ ] **1.4** 本地可选复跑 C2：`run_prosody_diagnostics.py`（与云端交叉验证 APC/Layer8）
- [ ] **1.5** 填写 **数据冻结记录**（日期 + git commit hash）到 `results/logs/DATA_FREEZE.json`

### Phase 2 — 归档 + 占位配图

- [ ] **2.1** 按 §3.2 创建各 `last/` 并移旧文件
- [ ] **2.2** 云端 confusion → 重命名为 §3.3 文件名
- [ ] **2.3** `python scripts/generate_paper_figures.py` 刷新 fig00–fig06
- [ ] **2.4** 更新 `FIGURES_MANIFEST.json`

### Phase 3 — 文稿更新（LaTeX 为源）

- [ ] **3.1** 更新 `4_Experiments_and_Results.tex` 主表（若冻结数有变）
- [ ] **3.2** 新增 FAU multi-seed 附录表/句
- [ ] **3.3** 插入 fig07/08（正文）；figA1–A4（附录）— 按 **D1**
- [ ] **3.4** 统一 `5_Analysis_and_Discussion.tex` FD/SMMD（**D4**）
- [ ] **3.5** 同步 Abstract / Intro / Conclusion 数字
- [ ] **3.6** `main.tex`：figure 引用 + 可选 appendix 节

### Phase 4 — 衍生文档与打包

- [ ] **4.1** `merge_remote_logs` → `publication_package/experiment_results.csv`
- [ ] **4.2** `generate_docx.py` / `generate_docx_cn.py`
- [ ] **4.3** `build_submission_bundle.py`
- [ ] **4.4** 更新 `docs/submission_checklist.md` 勾选状态
- [ ] **4.5** INTERSPEECH 官方模板替换（若截止前需要）

### Phase 5 — 终检（投稿前）

- [ ] **5.1** 全文 grep 旧数字（92.04、60.14、6.87 等 v5 残留）
- [ ] **5.2** 表图编号与 `\ref{}` 无 broken
- [ ] **5.3** 匿名化作者/致谢/路径
- [ ] **5.4** 你替换 `paper_draft/figures/` 中占位图为终稿（同文件名覆盖）

---

## 5. 给你核对的「最小确认包」

请直接回复格式示例：

```
D1: 同意默认（正文2+附录4）
D2: 同意
D3: 同意
D4: 统一用 distribution_shift.json，改正文 FD 表
D5–D8: 同意
Phase 0 完成后通知我开始 Phase 1
```

若 **D1** 要改：例如「正文只放 fig05 柱状图，6 张 CM 全附录」也请写明。

---

## 6. 立即动作（清单编写后）

1. 已修复 `autodl_ac_suite_resume.sh` 的 `SER_*` 路径 → **建议现在重跑 resume**  
2. 云端 DONE 后执行 Phase 0.4 → 按 Phase 1–5 推进  
3. 收到你对 §5 的确认后，从 Phase 2 起自动改 tex/csv/fig

---

*关联文档*: `docs/experiment_suite_AC.md`, `docs/submission_checklist.md`, `docs/json_verification_checklist.md`
