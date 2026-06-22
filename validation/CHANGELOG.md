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
