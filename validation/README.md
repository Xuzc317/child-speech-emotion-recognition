# validation/ — 数据校验审计追踪

> 协议: `ac_suite_2026-06-validated` | 最后更新: 2026-06-22
> 审计结论: **RELIABLE**（见 `audit_00_SUMMARY.md`）
> 权威数据源: `docs/current/权威数据手册.md` | 原始日志: `results/logs/` (192 runs)

本目录包含对 192 实验全量审计的校验产物：自动生成的数据清单、数值验证报告、以及 8 份独立复核报告。全部文件可一键重建（见 `PROGRESS.md`）。

---

## 从哪里读起

1. **`audit_00_SUMMARY.md`** — 入口总表，汇总全部校验结论、修正清单、已知问题
2. **`audit_review_1_phases1-3.md`** → **`audit_review_2_phase4.md`** → **`audit_review_3_phase5_sealing.md`** → **`audit_review_4_final_reliability.md`** — 按验证阶段依次深入
3. **`review_conditions_of_record.md`** — 如果关心清单对脚本的忠实度
4. **`review_design_vs_actual.md`** — 如果关心设计方案 vs 实际执行的偏差
5. **`wa_uar_by_corpus.md`** — 如果关心逐语料 WA-UAR 差距分析
6. **`discrepancy_report.md`** — 如果需要逐字段查看旧手册与原始日志的差异
7. **`CHANGELOG.md`** — 如果需要完整的修正历史

---

## 文件清单

### 审计报告（人工撰写）

| 文件 | 核心结论 |
|------|---------|
| `audit_00_SUMMARY.md` | 7 个 Phase 全部 VERIFIED；3 个 INVALID 实验；6 项修正；整体 RELIABLE |
| `audit_review_1_phases1-3.md` | 92.92%/67.81% 天花板被 REFUTED（实为单 seed，非 3-seed mean）；其余 5 项 CONFIRMED；部分可信但有重要保留 |
| `audit_review_2_phase4.md` | 8 行全部 CONFIRMED；3 个 INVALID 实验；B7 溯源依赖 launch_b7.sh；5 项无法独立证明 |
| `audit_review_2_wa_uar_detail.md` | 逐语料 WA-UAR 差距表：C-BESD 0.35pp / FAU 21.15pp / IEMOCAP 4.93pp / GLOBAL 9.11pp |
| `audit_review_3_phase5_sealing.md` | 初评 NOT-SEALED（5 个阻塞问题），未做任何 commit；问题后被 audit_review_4 验证已修复 |
| `audit_review_4_final_reliability.md` | 7 项全部 PASS；整体 RELIABLE |
| `review_conditions_of_record.md` | provenance_manifest.csv 对 Launch 脚本 + 代码默认值的忠实度独立复核 |
| `review_design_vs_actual.md` | design_vs_actual_diff.md 的独立复核报告 |

### 自动生成 — 流水线产物（一键重建）

| 文件 | 产生脚本 | 用途 |
|------|---------|------|
| `provenance_manifest.csv` | `scripts/gen_manifest.py` / `rebuild_manifest.py` | 全量溯源清单（含 ddof=1、INVALID 标记） |
| `unverifiable.md` | `scripts/gen_manifest.py` | 无法验证的实验清单 |
| `discrepancy_report.md` | `scripts/regen_handbook.py` | 旧手册 vs 日志 136 字段逐格 diff（分类: COPY_ERR / WRONG_CELL / PHANTOM） |
| `handbook_snapshot_2026-06-22.md` | `scripts/regen_handbook.py` | 重建前的旧手册快照 |
| `metrics_inventory.csv` | `scripts/check_metrics.py` | 指标完整性清单 |
| `missing_metrics.md` | `scripts/check_metrics.py` | 缺失指标分析 |
| `reproducibility_report.md` | `scripts/phase4_audit.py` | 可复现性验证报告（跨 seed 一致性、配置可信度、B7 溯源） |
| `wa_uar_by_corpus.md` | `scripts/phase4_audit.py` | 逐语料 WA-UAR 差距分析 |
| `rerun_candidates.md` | `scripts/phase4_audit.py` | 建议补跑实验清单（6 个） |
| `manuscript_correction_ledger.md` | 手动扫描 `paper_draft/current/v10_*.tex` | LaTeX SAFE-SWAP / CLAIM-AFFECTED 账本 |
| `PROGRESS.md` | 手动更新 | 验证进度日志、一键重建命令、已知残余问题 |

### 自动生成 — 辅助脚本

| 文件 | 产生脚本 | 用途 |
|------|---------|------|
| `ledger_full.csv` | `scripts/gen_ledger.py` | 全量实验明细台账 |
| `design_vs_actual_diff.md` | `scripts/gen_master_reference.py` | 设计方案 vs 实际执行分歧记录 |

### 命令输出快照（手动捕获）

| 文件 | 来源 | 备注 |
|------|------|------|
| `phase0_verify_BARE.txt` | 早期流水线输出 | 旧路径 `results_remote/` 未发现日志（已过期，仅供参考） |
| `phase0_verify_RESULTS.txt` | Phase 0 全量校验输出 | 192/192 验证通过（使用旧单 seed 天花板值） |
| `git_status.txt` | `git status` 快照 | 重命名前的状态记录 |
| `all_tracked_files.txt` | `git ls-files` | 全仓跟踪文件清单 |
| `tracked_binaries.txt` | `git ls-files` 过滤 | 二进制文件清单（当前为空） |

### 其他

| 文件 | 类型 | 用途 |
|------|------|------|
| `CHANGELOG.md` | 人工撰写 | 修正史与审计轨迹 |
| `independent_verify.py` | 人工撰写 | 独立验证脚本 |

---

## 已知边界

1. **WA/UAR 无法独立重算** — 实验日志不存储帧级预测结果，WA/UAR 值取信于训练代码的 `sklearn.metrics` 计算，仅能做交叉核对而非从零重算
2. **B7 源域依赖启动脚本** — 实验日志未记录源 checkpoint 路径，迁移源的确认依赖 `launch_b7.sh` 的正确执行，无法从日志独立验证
3. **数据划分 / 说话人泄露未验证** — 当前校验未覆盖 `data_split_seed=42` 划分策略下是否存在说话人跨集泄露
