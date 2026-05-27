# 实验 JSON 对照检查清单 & 任务执行顺序

> **论文矩阵来源**：AutoDL 最终跑批（2026-05-19），你已批准写入 `paper_draft/4_*.tex`。  
> **本地缺口**：`results_remote/` 未保留；`results/logs/` 与 `publication_package/` 曾为旧一轮数值。  
> **原则**：论文表中的 WA/UAR/Epoch 必须与 JSON 一致；`best_val_wa` 等未从远端恢复字段在 manifest 中标注。

---

## 一、应对照检查的 JSON 文件（6 个核心实验）

| 优先级 | 文件路径 | 实验 | 训练→测试 | 池化 | reg_profile | 论文 WA | 论文 UAR | best_epoch | 状态 |
|:------:|----------|------|-----------|------|-------------|---------|----------|------------|------|
| P0 | `results/logs/exp1_self_attention.json` | Exp1 | C-BESD → C-BESD | self_attention | default | 92.78% | 92.79% | 30 | ⬜ 待验收 |
| P0 | `results/logs/exp2_prosody_guided.json` | Exp2 | C-BESD → C-BESD | prosody_guided | default | 91.30% | 91.35% | 10 | ⬜ 待验收 |
| P0 | `results/logs/exp3_adult_iemocap.json` | Exp3 | IEMOCAP → IEMOCAP | prosody_guided | default | 58.67% | 59.11% | 8 | ⬜ 待验收 |
| P0 | `results/logs/exp4_zero_shot_fau.json` | Exp4 | C-BESD → FAU | prosody_guided | default | 18.70% | 22.75% | 10 | ⬜ 待验收 |
| P0 | `results/logs/exp5_fau_indomain.json` | Exp5 | FAU → FAU | prosody_guided | **fau** | 66.36% | 56.35% | 3 | ⬜ 待验收 |
| P0 | `results/logs/exp5b_self_attention_fau.json` | Exp5b | FAU → FAU | self_attention | **fau** | 66.18% | 58.23% | 2 | ⬜ 待验收 |

**同步副本（与上表数值必须一致）**

| 文件路径 | 状态 |
|----------|------|
| `publication_package/logs/exp1_self_attention.json` | ⬜ |
| `publication_package/logs/exp2_prosody_guided.json` | ⬜ |
| `publication_package/logs/exp3_adult_iemocap.json` | ⬜ |
| `publication_package/logs/exp4_zero_shot_fau.json` | ⬜ |
| `publication_package/logs/exp5_fau_indomain.json` | ⬜ |
| `publication_package/logs/exp5b_self_attention_fau.json` | ⬜ |
| `publication_package/experiment_results.csv`（含 Exp5b 行） | ⬜ |

---

## 二、辅助 JSON / 数据（非 6 实验矩阵，但论文会引用）

| 优先级 | 文件 | 论文引用 | 期望/已知值 | 状态 |
|:------:|------|----------|-------------|------|
| P1 | `results/logs/apc_metrics.json` | §APC / 可解释性 | APC_wav ≈ 0.698, APC_δ ≈ -0.147 | ⬜ 核对 |
| P1 | `publication_package/logs/apc_metrics.json` | 同上 | 同上 | ⬜ 同步副本 |
| P1 | `results/layer_weights.json` | 层融合 | argmax layer 8, entropy ≈ 2.43 | ⬜ 核对 |
| P1 | `publication_package/layer_weights.json` | 同上 | 同上 | ⬜ |
| P1 | `results/distribution_shift.json` | FD 表 | FD=16.33, SMMD=0.41（C-BESD vs FAU） | ⬜ 核对 |
| P1 | `publication_package/distribution_shift.json` | 同上 | 同上 | ⬜ |
| P2 | `publication_package/xai_raw_data.npz` | XAI 图 | 已存在 | ✅ 已有 |

---

## 三、JSON 字段验收标准（`src/train.py` 输出格式）

每个核心实验 JSON 应包含：

| 字段 | 说明 | 验收 |
|------|------|------|
| `exp_name` | 与文件名一致 | 必填 |
| `pooling_type` | `self_attention` 或 `prosody_guided` | 必填 |
| `train_data` / `test_data` | 语料列表 | 必填 |
| `test_wa` / `test_uar` | 0–1 浮点；×100 与论文表一致（±0.01pp） | **与论文矩阵对齐** |
| `best_epoch` | 整数 | **与论文矩阵对齐** |
| `reg_profile` | `default` 或 `fau` | Exp5/5b 必须为 `fau` |
| `weight_decay` / `label_smoothing` / `pooling_dropout` / `grad_clip` | 与 reg_profile 一致 | 见下表 |
| `best_val_wa` | 验证集最佳 WA | 若缺失，manifest 标注，不强行编造 |
| `source` / `matrix_version` | 溯源元数据 | 建议 `autodl_final_2026-05-19` |

**reg_profile 与超参对应关系（代码 `train.py`）**

| profile | weight_decay | label_smoothing | pooling_dropout | grad_clip |
|---------|--------------|-----------------|-----------------|-----------|
| default | 1e-3 | 0.1 | 0.0 | null |
| fau | 5e-3 | 0.15 | 0.3 | 1.0 |

---

## 四、任务执行顺序（推荐）

### 阶段 A — 本地即可完成（**当前从这里开始**）

| 步骤 | 任务 | 需要 AutoDL？ | 产出 |
|:----:|------|:-------------:|------|
| **A1** | 本清单 + `scripts/verify_experiment_jsons.py` | 否 | 可重复验收 |
| **A2** | 将批准矩阵写入 `results/logs/` + `publication_package/logs/` | 否 | 6×JSON 对齐论文 |
| **A3** | 更新 `experiment_results.csv`、`README_for_Agents` Δ 表 | 否 | 作图/Agent 可读 |
| **A4** | 运行 `python scripts/verify_experiment_jsons.py` 全部 PASS | 否 | 验收记录 |
| **A5** | 核对 APC / layer_weights / distribution_shift | 否 | P1 辅助文件 |
| **A6** | PaperVizAgent 生成架构图、层权重、APC、FD 图 | 否 | `figures/` |
| **A7** | 补写摘要、引言、结论；与矩阵叙事一致 | 否 | LaTeX/docx |

### 阶段 B — 需要 AutoDL 或新数据（**先告知你再开机器**）

| 步骤 | 任务 | 需要 AutoDL？ | 说明 |
|:----:|------|:-------------:|------|
| **B1** | 从远端重新拉取**完整** JSON（含 `best_val_wa`、checkpoint） | **是** | 实例关机后：`python scripts/tmp_paramiko_autodl_runner.py --auto-resume`；若远端已清空则需重跑 |
| **B2** | FAU 多随机种子（3–5 seeds） | **是** | 论文 limitation 中已写单 seed |
| **B3** | MyST / KidsTALC / EmoReact 扩展实验 | **是** + 数据 | 需先按 `docs/dataset_instructions.md` 放置语料 |

---

## 五、一键命令

```bash
# 写入 canonical JSON + 同步 publication_package + 更新 CSV
python scripts/sync_autodl_canonical_logs.py

# 对照论文矩阵验收（应全部 PASS）
python scripts/verify_experiment_jsons.py
```

---

## 六、当前进度（随执行更新）

- [x] A1 清单与验收脚本（`docs/json_verification_checklist.md`, `scripts/verify_experiment_jsons.py`）
- [x] A2 JSON 同步（`scripts/sync_autodl_canonical_logs.py` → `results/logs/` + `publication_package/logs/`）
- [x] A3 CSV / README 更新（`experiment_results.csv`, `README_for_Agents.md`）
- [x] A4 verify 全 PASS（12/12 路径）
- [x] A5 辅助 JSON 核对（APC/FD 表已整理；layer_weights 阻塞见 `publication_package/AUXILIARY_METRICS_STATUS.md`）
- [x] A6 配图（`paper_draft/figures/fig01–fig05`，见 `FIGURES_MANIFEST.json`）
- [x] A7 论文章节补全（`0_Abstract`–`6_Conclusion` LaTeX + docx 重生成）
- [x] 投稿骨架：`main.tex` + `references.bib` + `paper_draft/README.md`
- [x] 架构图 `fig00_system_architecture`（本地无 pdflatex 时需自行编译 PDF）
- [ ] B1 远端完整拉回（可选）
- [ ] B2 多 seed FAU
- [ ] B3 扩展语料

### A5 初检（2026-05-19）

| 文件 | 论文/叙事 | 本地 JSON | 结论 |
|------|-----------|-----------|------|
| `apc_metrics.json` | APC_wav=0.698 | 0.6979 | ✅ 一致 |
| `layer_weights.json` | argmax **layer 8** | argmax_layer=**3**，含负权重、entropy=NaN | ⚠️ 需从 checkpoint 重导出 softmax 权重 |
| `distribution_shift.json` | FD=**16.33**（零样本行） | FD=**12.33**（C-BESD vs FAU 单次对比） | ⚠️ 论文表含多条件 FD；需统一 `statistics.py` 输出或改表注 |
