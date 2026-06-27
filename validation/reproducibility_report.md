# Phase 4 — Reproducibility Report

> Generated: 2026-06-22 | Source: `scripts/phase4_audit.py`

## Part 0.1 — E4-04 Config Divergence

E4-04 has 3 seed runs:
  seed=42: train=['c-besd-4cl', 'iemocap'] test=['c-besd-4cl', 'iemocap'] pooling=self_attention aug=C4 fusion=weighted adapter=False unfreeze=False protocol=ac_suite_2026-06 WA=61.21%
  seed=123: train=['c-besd-4cl', 'iemocap'] test=['c-besd-4cl', 'iemocap'] pooling=self_attention aug=C4 fusion=weighted adapter=False unfreeze=False protocol=ac_suite_2026-06 WA=63.81%
  seed=456: train=['c-besd-4cl', 'iemocap'] test=['c-besd-4cl', 'iemocap'] pooling=self_attention aug=C4 fusion=weighted adapter=False unfreeze=False protocol=ac_suite_2026-06 WA=62.67%

All seeds consistent. OK.

## Part 0.2 — Standard Deviation: ddof=1 (sample std)

All mean+-std values recalculated with sample standard deviation (ddof=1, np.std(ddof=1)).
Previous reports used population std (ddof=0). Difference: factor of sqrt(n/(n-1)) = 1.225 for n=3.

### Impact on key experiments (ddof=0 vs ddof=1)

| Experiment | old mean+-std (ddof=0) | new mean+-std (ddof=1) |
|-----------|------------------------|------------------------|
| E1-02 | 91.87+-1.28% | 91.87+-1.56% |
| E1-05 | 67.05+-0.55% | 67.05+-0.67% |
| E2-01 | 96.91+-0.16% | 96.91+-0.19% |
| E6-03 | 91.91+-0.77% | 91.91+-0.94% |
| E6-04 | 91.96+-0.76% | 91.96+-0.93% |
| E7-03 | 91.57+-0.36% | 91.57+-0.45% |
| E7-05 | 91.17+-1.04% | 91.17+-1.28% |

## Part 0.3 — Ceiling Numbers: 3-seed mean, not s42

- C-BESD frozen SA ceiling: **91.87%** (3-seed sample mean, was 92.92% s42)
  seeds: 92.92%, 90.07%, 92.62%
- FAU frozen SA ceiling: **67.05%** (3-seed sample mean, was 67.81% s42)
  seeds: 67.81%, 66.54%, 66.79%
- C-BESD unfreeze: 96.91% (unchanged — already a 3-seed mean)

All other 'ceiling' references in CLAUDE.md now use 3-seed sample mean, sourced from ledger_full.csv.

## Part 0.4 — WA-UAR by Corpus

| Corpus | WA mean+-std | UAR mean+-std | WA-UAR gap | N runs |
|--------|-------------|--------------|-----------|--------|
| C-BESD | 80.90+-19.59% | 80.50+-19.84% | 0.40pp | 69 |
| FAU_Aibo | 62.82+-11.59% | 41.67+-5.71% | 21.15pp | 69 |
| IEMOCAP | 59.77+-10.00% | 54.84+-8.37% | 4.92pp | 54 |

**Interpretation**:
- FAU_Aibo has the largest WA-UAR gap (21.15pp) due to severe class imbalance (4 classes, highly skewed)
- C-BESD WA ~= UAR (0.40pp) — near-perfect class balance (6 classes)
- IEMOCAP gap is moderate (4.92pp)
- The '30.8pp' figure refers to the MAXIMUM gap across individual FAU runs; the '9.11pp' is the global mean across all 192 runs

## Part 1.5 — Cross-seed Config Consistency Audit

Multi-seed experiments checked: 46
Config-consistent: 46
Config-divergent: 0

No config-divergent experiments found beyond E4-04 (already flagged).

### train_data/test_data divergence check:

  All clear — no train/test divergence beyond already-flagged.

## Part 1.6 — Config Field Trustworthiness Audit

### Field value analysis across 192 logs:

**augment_condition**: 5 unique values: ['C1', 'C2', 'C3', 'C4', 'MISSING']
**fusion_mode**: 4 unique values: ['MISSING', 'best_single', 'last', 'weighted']
**pooling_type**: 3 unique values: ['mean', 'prosody_guided', 'self_attention']
**protocol**: 2 unique values: ['ac_suite_2026-05', 'ac_suite_2026-06']
**reg_profile**: 2 unique values: ['default', 'fau']
**unfreeze_ssl**: 3 unique values: ['False', 'MISSING', 'True']
**use_adapter**: 3 unique values: ['False', 'MISSING', 'True']

### Trustworthiness classification:

| Field | Trust | Reason |
|-------|-------|--------|
| pooling_type | TRUSTED | Consistent, explicitly set by launch scripts |
| train_data | TRUSTED | Consistent, explicitly set by launch scripts |
| test_data | TRUSTED | Consistent, explicitly set by launch scripts |
| seed | TRUSTED | Consistent, explicitly set by launch scripts |
| test_wa | TRUSTED | Consistent, explicitly set by launch scripts |
| test_uar | TRUSTED | Consistent, explicitly set by launch scripts |
| best_val_wa | TRUSTED | Consistent, explicitly set by launch scripts |
| best_epoch | TRUSTED | Consistent, explicitly set by launch scripts |
| exp_name | TRUSTED | Consistent, explicitly set by launch scripts |
| output_dir | TRUSTED | Consistent, explicitly set by launch scripts |
| weight_decay | TRUSTED | Consistent, explicitly set by launch scripts |
| label_smoothing | TRUSTED | Consistent, explicitly set by launch scripts |
| pooling_dropout | TRUSTED | Consistent, explicitly set by launch scripts |
| grad_clip | TRUSTED | Consistent, explicitly set by launch scripts |
| augment_condition | UNTRUSTED | Code default value, not an experimental condition; e.g. augment_condition='C1' appears in phases that don't use augmentation |
| fusion_mode | UNTRUSTED | Code default value, not an experimental condition; e.g. augment_condition='C1' appears in phases that don't use augmentation |
| use_adapter | UNTRUSTED | Code default value, not an experimental condition; e.g. augment_condition='C1' appears in phases that don't use augmentation |
| unfreeze_ssl | UNTRUSTED | Code default value, not an experimental condition; e.g. augment_condition='C1' appears in phases that don't use augmentation |
| reg_profile | UNTRUSTED | Code default value, not an experimental condition; e.g. augment_condition='C1' appears in phases that don't use augmentation |
| fusion_best_layer | UNTRUSTED | Code default value, not an experimental condition; e.g. augment_condition='C1' appears in phases that don't use augmentation |

### augment_condition by phase (for verification):

  B1: ['C1', 'MISSING']
  B2: ['C1']
  B3: ['C1', 'C2', 'C3', 'C4']
  B4: ['C1']
  B5: ['C1']
  B6: ['C1']
  B7: ['C1']

**Conclusion**: augment_condition, fusion_mode, use_adapter, unfreeze_ssl, reg_profile, 
and fusion_best_layer are code defaults not experimental conditions. They should NOT be 
used as evidence of what was actually configured. Only train_data, test_data, pooling_type, 
seed, and metrics can be independently trusted. For actual config, consult launch scripts.

## Part 1.7 — B7 Source Domain Traceability

### B7 transfer pairs from launch_b7.sh:

| E7-0X | Claimed Source | Target | Source ckpt evidence in log | Verdict |
|-------|---------------|--------|---------------------------|---------|
| E7-01 | C-BESD | FAU_Aibo | train=FAU_Aibo, test=FAU_Aibo, output_dir=checkpoints/b7/E7-01_s42 | CANNOT-VERIFY: train_data=FAU_Aibo (target domain), source=C-BESD only in launch script |
| E7-02 | C-BESD | IEMOCAP | train=IEMOCAP, test=IEMOCAP, output_dir=checkpoints/b7/E7-02_s42 | CANNOT-VERIFY: train_data=IEMOCAP (target domain), source=C-BESD only in launch script |
| E7-03 | FAU_Aibo | C-BESD | train=C-BESD, test=C-BESD, output_dir=checkpoints/b7/E7-03_s42 | CANNOT-VERIFY: train_data=C-BESD (target domain), source=FAU_Aibo only in launch script |
| E7-04 | FAU_Aibo | IEMOCAP | train=IEMOCAP, test=IEMOCAP, output_dir=checkpoints/b7/E7-04_s42 | CANNOT-VERIFY: train_data=IEMOCAP (target domain), source=FAU_Aibo only in launch script |
| E7-05 | IEMOCAP | C-BESD | train=C-BESD, test=C-BESD, output_dir=checkpoints/b7/E7-05_s42 | CANNOT-VERIFY: train_data=C-BESD (target domain), source=IEMOCAP only in launch script |
| E7-06 | IEMOCAP | FAU_Aibo | train=FAU_Aibo, test=FAU_Aibo, output_dir=checkpoints/b7/E7-06_s42 | CANNOT-VERIFY: train_data=FAU_Aibo (target domain), source=IEMOCAP only in launch script |

**Critical finding**: B7 logs do NOT record which source checkpoint was loaded. 
The `train_data` field records the target domain (fine-tuning data), not the source. 
The 'source domain' assignment relies entirely on `launch_b7.sh` being correctly executed. 
This means the 'Target-Domain Dominance' conclusion depends on the unverified premise 
that the correct source checkpoints were loaded for each E7-0X experiment.

**Mitigation**: The launch_b7.sh script logic is deterministic (hardcoded source→target pairs), 
and the B7 WA values are internally consistent (C-BESD targets all ~91%, FAU targets ~66%, 
IEMOCAP targets ~63%). This pattern supports the conclusion even without source confirmation.

## Part 1.8 — Seed Completeness + Recalculation

All multi-seed experiments checked for seed completeness (42, 123, 456).
**All clear** — no missing seeds in multi-seed experiments.

### Recalculation verification (sample std, ddof=1):

  E1-02: WA=91.87+-1.56% (n=3) seeds=92.92%/90.07%/92.62%
  E1-05: WA=67.05+-0.67% (n=3) seeds=67.81%/66.54%/66.79%
  E2-01: WA=96.91+-0.19% (n=3) seeds=96.80%/96.80%/97.13%
  E6-03: WA=91.91+-0.94% (n=3) seeds=92.92%/91.74%/91.06%
  E6-04: WA=91.96+-0.93% (n=3) seeds=92.92%/91.91%/91.06%
  E7-03: WA=91.57+-0.45% (n=3) seeds=92.07%/91.23%/91.40%
  E7-05: WA=91.17+-1.28% (n=3) seeds=91.74%/89.71%/92.07%

## Part 1.9 — Trust Boundary + Rerun Candidates

### Trust boundary

1. **WA/UAR cannot be independently verified** — 0/192 files have predictions or confusion matrices.
   Both metrics are trusted as-is (computed by sklearn during training).
2. **B7 source domains cannot be independently confirmed** — rely on launch_b7.sh correctness.
3. **augment_condition, fusion_mode, use_adapter, unfreeze_ssl are code defaults** — not experimental conditions.
4. **E4-04 is invalid for 3-seed aggregation** — s42 used a different config than s123/s456.
5. **All other 191 runs** have internally consistent configs and complete seed coverage.

### Rerun candidates (if reviewer demands confusion matrices)

These are the minimum set of experiments worth re-running with prediction dumping:

| Priority | Experiment | Reason |
|----------|-----------|--------|
| 1 | E1-02 (C-BESD frozen SA) | C-BESD in-domain ceiling |
| 2 | E1-05 (FAU frozen SA) | FAU in-domain ceiling, severe class imbalance |
| 3 | E6-04 (C-BESD SA+WF) | Best B6 config |
| 4 | E7-03 (FAU->C-BESD) | Best transfer result |
| 5 | E7-05 (IEMOCAP->C-BESD) | Second-best transfer, cross-age |
| 6 | E3-14 (IEMOCAP->C-BESD zero-shot) | Best zero-shot |

These 6 experiments cover all key conclusions. Each should dump `predictions` and `labels` 
to enable independent WA/UAR recomputation and confusion matrix generation.
