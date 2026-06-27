# Validation Report — 分布驱动儿童 SER 全量实验验证

> Protocol: ac_suite_2026-06-validated | Date: 2026-06-22
> 192 experiments across B1–B7, verified Phase 0–5

## Trustworthiness Summary

| Phase | Content | Verdict | Publishable? |
|-------|---------|---------|-------------|
| B1 (E1) | Pooling baseline | VERIFIED, 1 INVALID (E1-08) | Yes, with E1-08 caveat |
| B2 (E3) | Zero-shot transfer | VERIFIED, E3 ID mapping differs from old handbook | Yes, from logs |
| B3 (E4) | Augmentation | VERIFIED, 2 INVALID (E4-04, E4-10) | Yes, with caveats |
| B4 (E5) | LayerFusion ablation | VERIFIED | Yes |
| B5 (E2) | Unfreeze | VERIFIED, FAU conclusion reversed | Yes, corrected conclusion |
| B6 (E6) | Module ablation | VERIFIED | Yes |
| B7 (E7) | Model transfer | VERIFIED, source CANNOT-VERIFY | Yes, with caveat |

## What's Trustworthy

1. **All 192 WA/UAR values** — directly from `results/logs/E*-*.json`, sklearn-computed
2. **B6/B7 tables** — perfectly consistent between handbook and logs
3. **3-seed sample means** — recalculated with ddof=1, verified by independent re-extraction
4. **Per-seed values** — all verified from original JSON files with full file paths
5. **C-BESD ceiling = 91.87% (3-seed mean)**, FAU = 67.05%, unfreeze = 96.91%

## What's NOT Trustworthy (and Why)

1. **Old handbook B2 table** — 14/18 values are PHANTOM (not in any log). Replaced.
2. **Old handbook B5 FAU unfreeze = 76.02%** — PHANTOM. Actual = 66.37%.
3. **augment_condition, fusion_mode, use_adapter, unfreeze_ssl** — code defaults, not experimental conditions
4. **B7 source domain** — not independently verifiable from logs (rely on launch_b7.sh)
5. **E1-08, E4-04, E4-10 3-seed aggregation** — INVALID (config divergence across seeds)
6. **WA/UAR cannot be recomputed** — 0/192 have raw predictions or confusion matrices

## INVALID Experiments

| ID | Reason | Per-seed data preserved? |
|----|--------|------------------------|
| E1-08 | s42 old protocol (aug/fusion/adapter=None) | Yes — in manifest |
| E4-04 | s42 train=['c-besd'], s123/s456 train=['c-besd-4cl','iemocap'] | Yes — in manifest |
| E4-10 | s42/s123 train=['iemocap'], s456 train=['iemocap','fau-aibo'] | Yes — in manifest |

## Rerun Candidates (if reviewer demands confusion matrices)

1. E1-02 (C-BESD frozen SA) — in-domain ceiling
2. E1-05 (FAU frozen SA) — class imbalance benchmark
3. E6-04 (C-BESD SA+WF) — best B6 config
4. E7-03 (FAU→C-BESD) — best transfer
5. E7-05 (IEMOCAP→C-BESD) — cross-age transfer
6. E3-14 (IEMOCAP→C-BESD zero-shot) — best zero-shot

## Key Corrections Applied

| Correction | Old | New |
|-----------|-----|-----|
| C-BESD ceiling | 92.92% (s42) | 91.87% (3-seed mean) |
| FAU ceiling | 67.81% (s42) | 67.05% (3-seed mean) |
| Best zero-shot | 41.05% (phantom) | 34.68% (E3-14) |
| FAU unfreeze delta | +8.2pp (phantom) | -0.67pp |
| std formula | ddof=0 | ddof=1 |
| Handbook | Hand-edited | AUTO-GENERATED from logs |

## Regeneration Command

To rebuild all validation artifacts from raw logs:

```
python scripts/regen_handbook.py          # → docs/current/权威数据手册.md
python scripts/gen_manifest.py            # → validation/provenance_manifest.csv
python scripts/check_metrics.py           # → validation/metrics_inventory.csv
python scripts/phase4_audit.py            # → validation/reproducibility_report.md
```

Or use the verify-all script:
```
python scripts/verify_all.py
```
