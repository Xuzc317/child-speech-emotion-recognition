# Validation Changelog — 修正史与审计轨迹

> Protocol: ac_suite_2026-06 → ac_suite_2026-06-validated
> Period: 2026-06-22

## Summary

| Category | Count | Description |
|----------|-------|-------------|
| PHANTOM values removed | 5 | 41.05%, 35.47%, 76.02%, 19.17% (as B2 min), 30.24% |
| Ceilings corrected | 2 | C-BESD 92.92→91.87%, FAU 67.81→67.05% |
| Conclusion reversed | 1 | FAU unfreeze: +8.2pp → -0.67pp |
| INVALID experiments | 3 | E1-08, E4-04, E4-10 excluded from aggregation |
| std formula unified | 1 | ddof=0 → ddof=1 (sample std) |
| Config fields degraded | 6 | augment_condition et al. marked UNTRUSTED |
| B7 source untraceable | 6 | All B7 source domains CANNOT-VERIFY from logs alone |

## Detailed Correction Log

### 2026-06-22 — Phase 2: Handbook vs Logs Diff

| # | Location | Old Value | Corrected Value | Evidence | Classification |
|---|----------|-----------|----------------|----------|---------------|
| 1 | B2 E3-17 | 41.05% | 33.71% | `E3-17.json test_wa` | PHANTOM — not in any log |
| 2 | B2 text claim | best=41.05% | best=34.68% (E3-14) | Full B2 scan | PHANTOM |
| 3 | B2 E3-10 | 35.47% | 26.20% | `E3-10.json test_wa` | PHANTOM |
| 4 | B2 E3-05 | 19.17% (as range min) | 27.26% | `E3-05.json test_wa` | WRONG_CELL — E3-05 is not the min |
| 5 | B2 range claim | 19.17%-35.47% | 20.51%-34.68% | Full B2 scan | Mix of phantom + wrong cell |
| 6 | B5 E2-02 mean | 76.02% | 66.37% | `E2-02_s*.json` | PHANTOM |
| 7 | B5 FAU delta | +8.2pp | -0.67pp | Frozen SA 67.05% vs unfreeze 66.37% | CONCLUSION REVERSED |
| 8 | B5 C-BESD delta | +4.0pp | +5.04pp | Frozen SA 91.87% vs unfreeze 96.91% | UNDERSTATED |
| 9 | B5 IEMOCAP delta | +1.2pp | +1.99pp | Frozen prosody 64.38% vs unfreeze 66.37% | POOLING MISMATCH |
| 10 | B1 E1-02 ceiling | 92.92% (s42) | 91.87% (3-seed mean) | `E1-02_s*.json` | SINGLE-SEED→MEAN |
| 11 | B1 E1-05 ceiling | 67.81% (s42) | 67.05% (3-seed mean) | `E1-05_s*.json` | SINGLE-SEED→MEAN |
| 12 | B1 E1-03 | 87.38% | 91.86% | `E1-03_s*.json` | WRONG_CELL — matched nothing |
| 13 | B1 E1-09 | 56.22% | 64.38% | `E1-09_s*.json` | WRONG_CELL — matched nothing |

### 2026-06-22 — Phase 4: Reproducibility Audit

| # | Finding | Detail | Impact |
|---|---------|--------|--------|
| 14 | E1-08 INVALID | s42 uses old protocol (aug/fusion/adapter=None) | IEMOCAP SA mean unreliable |
| 15 | E4-04 INVALID | s42 train=[c-besd], s123/s456 train=[c-besd,iemocap] | B3 C4 aggregation invalid |
| 16 | E4-10 INVALID | s42/s123 train=[iemocap], s456 train=[iemocap,fau-aibo] | B3 C2 prosody aggregation invalid |
| 17 | std: ddof=1 | All mean±std recalculated with sample std | ~1.225× larger std for n=3 |
| 18 | 6 config fields UNTRUSTED | augment_condition, fusion_mode, use_adapter, unfreeze_ssl, reg_profile, fusion_best_layer | Code defaults, not experimental |
| 19 | B7 source CANNOT-VERIFY | Logs record target domain only; source relies on launch_b7.sh | Target-Domain Dominance premise weakened |
| 20 | 0/192 predictions | No raw predictions or confusion matrices saved | WA/UAR trusted as-is from sklearn |

### Artifacts Updated

| File | Change |
|------|--------|
| `CLAUDE.md` | Ceilings corrected, validation section added, protocol updated |
| `docs/current/权威数据手册.md` | Replaced with AUTO-GENERATED version from regen_handbook.py |
| `docs/archive/权威数据手册_v1_污染版.md` | Old contaminated version archived with README |
| `validation/provenance_manifest.csv` | ddof=1, INVALID marks, aug_trusted column |
| `paper_draft/current/v10_*.tex` | SAFE-SWAP only (ceiling 92.92→91.87, 67.81→67.05) |
| `validation/manuscript_correction_ledger.md` | Full SAFE-SWAP / CLAIM-AFFECTED ledger |

### 2026-06-23 — r2 Re-Seal: B3 4-seed Re-run + std ddof=1 + INVALID→0

| # | Finding | Detail | Impact |
|---|---------|--------|--------|
| 21 | E4-04_s42 重跑 | 旧: single train_data=[c-besd], WA=68.02%, best_epoch=8 → 新: dual [c-besd,iemocap], WA=61.21%, best_epoch=29 | B3 C4 聚合由 INVALID 转正 |
| 22 | E4-10_s42 重跑 | 旧: single train_data=[iemocap], WA=63.60% → 新: dual [iemocap,fau-aibo], WA=65.25%, best_epoch=14 | B3 C2 prosody 聚合由 INVALID 转正 |
| 23 | E4-10_s123 重跑 | 旧: best_epoch=0, WA=56.87%, train_data=[iemocap] → 新: dual, WA=65.74%, best_epoch=17 | B3 C2 第二坏 seed 转正 |
| 24 | E4-12_s123 重跑 | 旧: best_epoch=1, WA=56.35% → 新: dual [iemocap,fau-aibo], WA=60.96%, best_epoch=24 | B3 C4 第二坏 seed 转正 |
| 25 | INVALID 归零 | INVALID_AGGREGATION 从 {E1-08,E4-04,E4-10} 清空为 set() | 0 INVALID, 192/192 全部有效 |
| 26 | std: ddof=0→1 全链确认 | rebuild_manifest.py 使用 n-1；provenance_manifest.csv 使用 ASCII +- | 全链统一 ddof=1 |
| 27 | gen_manifest.py 删除 | 旧版使用 ddof=0 且缺 aggregation_valid/aug_trusted/seed_validity 列 | 仓库仅剩一个权威 manifest 生成器 |
| 28 | 确定性重生验证 | 连续两次全量重生 (manifest→handbook→总表) 产出零 diff | 脚本确定性成立 |
| 29 | 说话人零泄漏独立验证 | scripts/verify_speaker_split.py 复现切分，三数据集三对交集全0 | 说话人独立性可复现证实 |

### 数值变动对照 (E4-04 / E4-10 / E4-12 3-seed 聚合, ddof=1)

| 实验 | 旧 WA (INVALID, 含坏 seed) | 新 WA (全 seed 有效) | 旧 UAR | 新 UAR |
|------|---------------------------|---------------------|--------|--------|
| E4-04 | 64.83±2.82% (旧 s42 WA=68.02%, 单元素 train_data) | 62.56±1.30% | 60.64±6.54% | 56.42±1.21% |
| E4-10 | 61.93±4.46% (旧 s42=63.60%, s123=56.87%, 单元素) | 65.44±0.26% | 55.57±5.39% | 58.88±0.40% |
| E4-12 | 59.62±2.91% (旧 s123 WA=56.35%, best_epoch=1) | 61.16±0.67% | 47.27±6.47% | 50.53±0.86% |

### Tag

- **新 tag**: `ac_suite_2026-06-validated-r2`
- **上一 tag**: `ac_suite_2026-06-validated` (2026-06-22)

### 2026-06-24 — r3 Re-Seal: gen_master_reference 改造为核对工具, 总表退出自动生成链

| # | Finding | Detail | Impact |
|---|---------|--------|--------|
| 30 | gen_master_reference.py 改造 | 默认输出 stdout；`--output` 写文件；指向手工总表需 `--force`；B5 key_finding 硬编码旧值 (+4pp/+8pp) 修正为 +5.04pp/−0.68pp/+1.99pp | 裸跑脚本无副作用 |
| 31 | verify_all.py 切除调用 | 删除 gen_master_reference.py 步骤，防未来全量重生覆写手工内容 | 总表.md 不再被自动覆盖 |
| 32 | 权威数据手册 gap 不一致根除 | regen_handbook.py 新增 `gaps_by_test` 字典，正文从硬编码 "0.3pp" 改为 f-string 引用同源 gap 变量；两次重生 git diff 为零 | 表格与正文数字永不再漂 |
| 33 | B4 概览表口径对齐 | "54 exps (18+36)" → "42 configs (54 total runs: 18 multi-seed + 36 grid single-seed)"，与同列其他行统一用配置数 | 实验矩阵语义一致 |
| 34 | ⚠️ 近距离擦肩 — 手工总表曾意外覆写 | `gen_master_reference.py` 改造期间（旧 main() 仍直接覆写 OUTPUT_MD），`verify_all.py` 链式调用导致 `实验方案与数据_总表.md` 的 Phase 1+3 手工内容被纯自动版本全覆盖。凭 Claude Code 文件历史快照 (`7eba699d3451e153@v3`, mtime 2026-06-24 02:21:26) 逐字节恢复。教训：手工维护文档无独立备份机制，依赖工具内部缓存属侥幸。后续每个会话结束前手工文档变更应显式 `git add` + commit 保护 | 零数据损失，但暴露了单点风险 |
| 35 | 总表头部去 AUTO-GENERATED | 改为手工维护说明，标注数据变更时需独立复核确认与 `results/logs/` 一致 | 维护语义清晰 |
| 36 | 新增 C2/C4 val/test 混合边界说明 | 总表 §6 + 报告 §6 记录：`get_dataloaders()` 对 train/val/test 三 split 均用混合 dataset 构造，C2/C4 评估口径与 C1/C3 不完全一致 | 论文写作需明确此限制 |

### Tag

- **新 tag**: `ac_suite_2026-06-validated-r3`
- **上一 tag**: `ac_suite_2026-06-validated-r2` (2026-06-23)
