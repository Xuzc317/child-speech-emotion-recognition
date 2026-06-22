# 验证进度日志

> 启动: 2026-06-22 | 协议: ac_suite_2026-06-validated

## 总览

| Phase | 内容 | 状态 |
|-------|------|------|
| Phase 0 | 文件盘点（预检） | ✅ 完成 |
| Phase 1 | 溯源清单与索引 | ✅ 完成 |
| Phase 2 | 数值验证（手册 vs 日志） | ✅ 完成 |
| Phase 3 | 指标完整性（UAR） | ✅ 完成 |
| Phase 4 | 步骤/可复现性验证 + 独立 review 强制修正 | ✅ 完成 |
| Phase 5 | 整理与版本化 | ⏳ 待开始 |
| Phase 6 | 收尾 | ⏳ 待开始 |

## Phase 4 完成报告

### 产出文件

| 文件 | 说明 |
|------|------|
| `validation/reproducibility_report.md` | 完整可复现性审计报告 (9 节) |
| `validation/wa_uar_by_corpus.md` | 逐语料 WA-UAR 表 |
| `validation/rerun_candidates.md` | 6 个建议补跑实验 |
| `validation/provenance_manifest.csv` | 已更新 (ddof=1, 3 个 INVALID 标记) |
| `validation/ledger_full.csv` | 191 字段完整账本 |
| `scripts/phase4_audit.py` | 可重跑审计脚本 |
| `scripts/rebuild_manifest.py` | 可重跑 manifest 重建 |

### Part 0 强制修正（全部完成）

| 修正项 | 结果 |
|--------|------|
| E4-04 配置不一致 | CONFIRMED: s42 vs s123/s456 配置不同 → INVALID |
| std 改为 ddof=1 | DONE: 全部表重算，差异约 1.225x |
| 天花板数字修正 | DONE: CLAUDE.md 已更新 91.87%/67.05% |
| WA-UAR 逐语料表 | DONE: C-BESD 0.31pp / FAU 21.15pp / IEMOCAP 4.98pp |

### Part 1 关键发现

| 检查项 | 裁决 |
|--------|------|
| 跨 seed 配置一致性 | 43 OK, **3 INVALID** (E1-08, E4-04, E4-10) |
| 配置字段可信度 | 14 TRUSTED, **6 UNTRUSTED** (代码默认值) |
| B7 源域可追溯性 | **CANNOT-VERIFY** — 日志无源 checkpoint 记录 |
| Seed 完整性 | All clear — 无缺 seed |
| UAR 信任边界 | 192/192 存在但无法独立重算 |

### 无效实验详情

| 实验 | 问题 | 影响 |
|------|------|------|
| E1-08 | s42 使用旧代码 (aug/fusion/adapter=None), s123/s456 使用新代码 | B1 IEMOCAP SA 均值不可靠 |
| E4-04 | s42 单语料 C-BESD, s123/s456 双语料 C-BESD+IEMOCAP | B3 C4 增强均值不可用 |
| E4-10 | s42/s123 单语料 IEMOCAP, s456 双语料 IEMOCAP+FAU | B3 C2 prosody 均值不可用 |

## 下一步

Phase 5 — 整理与版本化：提出目标目录结构 + 手册改为生成物 + git tag。
