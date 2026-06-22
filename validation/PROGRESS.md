# 验证进度日志

> 启动: 2026-06-22 | 协议: ac_suite_2026-06-validated | Git tag: `ac_suite_2026-06-validated`

## 总览 — VALIDATION COMPLETE

| Phase | 内容 | 状态 |
|-------|------|------|
| Phase 0 | 文件盘点（预检） | ✅ |
| Phase 1 | 溯源清单与索引 | ✅ |
| Phase 2 | 数值验证（手册 vs 日志） | ✅ |
| Phase 3 | 指标完整性（UAR） | ✅ |
| Phase 4 | 步骤/可复现性验证 | ✅ |
| Phase 5 | 整理与版本化 | ✅ |

## 最终交付清单

### 权威数据
| 文件 | 说明 |
|------|------|
| `docs/current/权威数据手册.md` | **AUTO-GENERATED** — 从 192 日志一键生成，唯一真相源 |
| `results/logs/E*-*.json` | 192 原始日志（不可变） |
| `validation/provenance_manifest.csv` | ddof=1, INVALID 已标记 |

### 校验报告
| 文件 | 内容 |
|------|------|
| `validation/audit_00_SUMMARY.md` | 可信度总表、修正清单、rerun 候选 |
| `validation/CHANGELOG.md` | 完整修正史 + 审计轨迹 |
| `validation/discrepancy_report.md` | 手册 vs 日志 136 字段逐格 diff |
| `validation/reproducibility_report.md` | 跨 seed 一致性、配置可信度、B7 溯源 |
| `validation/manuscript_correction_ledger.md` | LaTeX SAFE-SWAP / CLAIM-AFFECTED 账本 |
| `validation/wa_uar_by_corpus.md` | 逐语料 WA-UAR 分析 |
| `validation/missing_metrics.md` | 指标缺失分析 |
| `validation/rerun_candidates.md` | 6 个建议补跑实验 |

### 脚本
| 脚本 | 功能 |
|------|------|
| `scripts/regen_handbook.py` | 从日志生成权威手册 |
| `scripts/gen_manifest.py` | 生成溯源清单 |
| `scripts/phase4_audit.py` | 可复现性审计 |
| `scripts/check_metrics.py` | 指标完整性检查 |
| `scripts/verify_all.py` | **一键全量验证** |

### 版本管理
- Git commit: `bb3b2b7` — 42 files changed
- Git tag: `ac_suite_2026-06-validated`
- 旧手册归档: `docs/archive/权威数据手册_v1_污染版.md` + README

## 一键重建命令

```
python scripts/verify_all.py
```

或单独运行：

```
python scripts/regen_handbook.py    # → docs/current/权威数据手册.md
python scripts/gen_manifest.py      # → validation/provenance_manifest.csv
```

## 已知残余问题

1. **0/192 无预测数据** — WA/UAR 只能取信训练代码的 sklearn 计算
2. **B7 源域不可独立确认** — 依赖 launch_b7.sh 正确执行
3. **6 个配置字段不可信** — 代码默认值，非实验条件
4. **3 个 INVALID 实验** — 跨 seed 配置不一致，建议补跑
5. **LaTeX CLAIM-AFFECTED** — 9 处结论级错误待叙事修订（SAFE-SWAP 已完成）
