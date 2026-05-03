# v5 6/2/2 Experiment Archive — Final Results

Source: cloud server `root@connect.cqa1.seetacloud.com:12112`, `/root/autodl-tmp/v5_data/`.
Split protocol: outer 8:2 + inner val → effective 6:2:2 by sample count.
**Training completed: 2026-05-04 01:05 CST**.

## Dataset

| split | samples | class distribution |
| --- | ---: | --- |
| train | 2468 | c0=410, c1=414, c2=415, c3=410, c4=409, c5=410 |
| val | 860 | c0=145, c1=145, c2=145, c3=144, c4=136, c5=145 |
| test | 851 | c0=140, c1=140, c2=140, c3=145, c4=146, c5=140 |

## Run 1: Pipeline (no model saving, 2026-05-03)

| experiment | val mean | val std | test mean | test std |
| --- | ---: | ---: | ---: | ---: |
| A1_baseline | 0.8190 | 0.0073 | 0.7791 | 0.0094 |
| A2b_adapter | 0.8225 | 0.0043 | 0.7787 | 0.0084 |
| B3_final | 0.8581 | 0.0057 | 0.8124 | 0.0111 |
| C1_none | 0.8504 | 0.0073 | 0.8034 | 0.0015 |
| C2_adult | 0.5376 | 0.0043 | 0.5229 | 0.0053 |
| C3_child | 0.5992 | 0.0070 | 0.5875 | 0.0092 |
| C4_extreme | 0.4682 | 0.0113 | 0.4685 | 0.0129 |

## Run 2: Save Models (with .pth, includes A3, 2026-05-04) — FINAL

| experiment | val mean | val std | test mean | test std |
| --- | ---: | ---: | ---: | ---: |
| A1_baseline | 0.8202 | 0.0038 | 0.7861 | 0.0048 |
| A2b_adapter | 0.8306 | 0.0058 | 0.7881 | 0.0078 |
| **A3_prosody_only** | **0.8473** | **0.0052** | **0.8085** | **0.0060** |
| B3_final | 0.8620 | 0.0091 | 0.8147 | 0.0108 |
| C1_none | 0.8519 | 0.0079 | 0.8124 | 0.0058 |
| C2_adult | 0.5415 | 0.0147 | 0.5061 | 0.0055 |
| C3_child | 0.6097 | 0.0077 | 0.5989 | 0.0116 |
| C4_extreme | 0.4721 | 0.0038 | 0.4649 | 0.0188 |

## Supplementary A3 (standalone, 2026-05-03)

| seed | val | test |
| --- | ---: | ---: |
| 42 | 0.8674 | 0.8237 |
| 123 | 0.8523 | 0.7897 |
| 456 | 0.8465 | 0.8085 |
| **mean/std** | **0.8554 / 0.0088** | **0.8073 / 0.0139** |

## Module Contribution Analysis (6:2:2 Run 2)

| comparison | delta test | significance |
| --- | ---: | --- |
| A1 → A2b (add Adapter, mean pool) | +0.20pp | negligible |
| A1 → A3 (add Prosody, no Adapter) | +2.24pp | **primary driver** |
| A3 → B3 (add Adapter on top) | +0.62pp | marginal |
| A1 → B3 (full stack) | +2.86pp | — |

**Conclusion**: Prosody Pooling accounts for ~80% of the total module gain. Adapter contributes at most 0.6pp. Final recommended model: **A3** (WavLM + Prosody Pooling + DrseNet, no Adapter).

## Augmentation Sensitivity (Run 2)

All augmentation remains strongly harmful. C2 (adult params) loses ~28pp, C4 (extreme) loses ~34pp vs C1.

## Files

| file | description |
| --- | --- |
| `v5_results.json` | Run 1 results |
| `v5_save_results.json` | Run 2 results (final, with A3 and full C1-C4) |
| `A3_results.json` | Supplementary standalone A3 |
| `run_v5.log` | Run 1 stdout |
| `run_v5_save.log` | Run 2 stdout (complete) |
| `../checkpoints/v5_622/*.pth` | 24 model checkpoints (8 configs × 3 seeds, ~90 MB total) |

## Model Weights

All 24 .pth files downloaded to `checkpoints/v5_622/`:

| config | seeds | size each |
| --- | --- | ---: |
| A1_baseline (no adapter, no prosody) | 42, 123, 456 | ~2.3 MB |
| A2b_adapter (adapter, mean pool) | 42, 123, 456 | ~3.5 MB |
| A3_prosody_only (no adapter, prosody) | 42, 123, 456 | ~2.8 MB |
| B3_final (adapter, prosody) | 42, 123, 456 | ~3.9 MB |
| C1_none (clean train) | 42, 123, 456 | ~3.9 MB |
| C2_adult | 42, 123, 456 | ~3.9 MB |
| C3_child | 42, 123, 456 | ~3.9 MB |
| C4_extreme | 42, 123, 456 | ~3.9 MB |
