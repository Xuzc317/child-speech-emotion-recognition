# Distribution-Driven Children's SER — Authoritative Data Handbook

> **AUTO-GENERATED — do not hand-edit**
> **Source**: `results/logs/E*-*.json` (192 files)
> **Protocol**: `ac_suite_2026-06-validated`
> **Generated**: 2026-06-22 by `scripts/regen_handbook.py`
> **Std**: sample std (ddof=1)
> **INVALID experiments excluded from aggregation**: E1-08, E4-04, E4-10

This document is regenerated from raw experiment logs. Any paper, chart, or abstract 
must reference values from this document. To regenerate: `python scripts/regen_handbook.py`

---

## Invalid Aggregation Experiments

The following experiments have inconsistent configs across seeds and their 3-seed mean±std are excluded:

| Experiment | Seeds affected | Issue |
|-----------|---------------|-------|
| E1-08 | see below | s42 uses old protocol (aug/fusion/adapter=None), s123/s456 use new protocol |
| E4-04 | see below | s42 train=['c-besd'], s123/s456 train=['c-besd-4cl','iemocap'] — different corpora |
| E4-10 | see below | s42/s123 train=['iemocap'], s456 train=['iemocap','fau-aibo'] — different corpora |

Per-seed data is preserved in provenance_manifest.csv. Aggregated means in this handbook exclude these experiments.

---

## B1 — Pooling x Dataset Baseline (E1, Frozen, 3-seed, ddof=1)

| Experiment | Dataset | Pooling | WA s42 | WA s123 | WA s456 | WA mean+-std | UAR mean+-std |
|-----------|--------|---------|--------|---------|---------|-------------|-------------|
| E1-01 | C-BESD | mean | 79.26% | 78.23% | 80.20% | 79.23+-0.99% | 79.19+-0.93% |
| E1-02 | C-BESD | self_attention | 92.92% | 90.07% | 92.62% | 91.87+-1.56% | 91.85+-1.54% |
| E1-03 | C-BESD | prosody_guided | 92.41% | 90.55% | 92.62% | 91.86+-1.14% | 91.84+-1.12% |
| E1-04 | FAU_Aibo | mean | 67.63% | 65.41% | 67.77% | 66.94+-1.32% | 39.33+-0.54% |
| E1-05 | FAU_Aibo | self_attention | 67.81% | 66.54% | 66.79% | 67.05+-0.67% | 42.92+-1.85% |
| E1-06 | FAU_Aibo | prosody_guided | 66.18% | 65.98% | 68.14% | 66.77+-1.19% | 42.94+-1.59% |
| E1-07 | IEMOCAP | mean | 60.38% | 61.29% | 61.90% | 61.19+-0.76% | 55.57+-1.82% |
| E1-08 [INVALID AGGREGATION] | IEMOCAP | self_attention | 63.21% | 64.25% | 63.82% | 63.76+-0.53% | 60.05+-0.78% |
| E1-09 | IEMOCAP | prosody_guided | 65.05% | 64.91% | 63.17% | 64.38+-1.05% | 60.37+-0.78% |

**B1 conclusion**: Self-Attention > Mean >> Prosody. 
C-BESD ceiling: E1-02 = 91.87+-1.56% (3-seed sample mean). 
FAU ceiling: E1-05 = 67.05+-0.67%. 
IEMOCAP ceiling: E1-08 = 63.76+-0.53% [INVALID — see note].

---

## B5 — WavLM Unfreeze Comparison (E2, 3-seed, ddof=1)

| Experiment | Dataset | Pooling | WA s42 | WA s123 | WA s456 | WA mean+-std | UAR mean+-std |
|-----------|--------|---------|--------|---------|---------|-------------|-------------|
| E2-01 | C-BESD | self_attention | 96.80% | 96.80% | 97.13% | 96.91+-0.19% | 96.88+-0.21% | Δ=+5.04pp |
| E2-02 | FAU_Aibo | self_attention | 66.86% | 65.18% | 67.07% | 66.37+-1.04% | 42.72+-3.18% | Δ=-0.67pp |
| E2-03 | IEMOCAP | prosody_guided | 65.30% | 67.15% | 66.64% | 66.37+-0.95% | 60.75+-1.32% | Δ=+1.99pp |

**B5 corrected conclusion**: Unfreeze helps C-BESD (+5.04pp) and IEMOCAP (+1.99pp, prosody→prosody aligned), 
but does NOT help FAU (Δ=-0.67pp, essentially flat/negative). 
The previous claim 'FAU +8.2pp' was based on a phantom 76.02% value not found in any log.

---

## B2 — Zero-shot Cross-corpus Transfer (E3, single-run)

| Experiment | Source | Target | Pooling | WA | UAR |
|-----------|--------|--------|---------|-----|-----|
| E3-01 | c-besd-4cl | fau-aibo | mean | 20.51% | 24.23% |
| E3-03 | c-besd-4cl | fau-aibo | prosody_guided | 21.91% | 23.99% |
| E3-02 | c-besd-4cl | fau-aibo | self_attention | 21.50% | 24.14% |
| E3-04 | c-besd-4cl | iemocap | mean | 30.60% | 27.02% |
| E3-06 | c-besd-4cl | iemocap | prosody_guided | 28.86% | 26.94% |
| E3-05 | c-besd-4cl | iemocap | self_attention | 27.26% | 26.82% |
| E3-07 | fau-aibo | c-besd-4cl | mean | 28.67% | 28.75% |
| E3-09 | fau-aibo | c-besd-4cl | prosody_guided | 25.29% | 25.40% |
| E3-08 | fau-aibo | c-besd-4cl | self_attention | 25.14% | 25.26% |
| E3-10 | fau-aibo | iemocap | mean | 26.20% | 30.55% |
| E3-12 | fau-aibo | iemocap | prosody_guided | 27.54% | 30.59% |
| E3-11 | fau-aibo | iemocap | self_attention | 23.52% | 28.28% |
| E3-13 | iemocap | c-besd-4cl | mean | 31.22% | 31.21% |
| E3-15 | iemocap | c-besd-4cl | prosody_guided | 33.31% | 33.28% |
| E3-14 | iemocap | c-besd-4cl | self_attention | 34.68% | 34.66% |
| E3-16 | iemocap | fau-aibo | mean | 29.25% | 32.34% |
| E3-18 | iemocap | fau-aibo | prosody_guided | 33.08% | 34.55% |
| E3-17 | iemocap | fau-aibo | self_attention | 33.71% | 33.00% |

**B2 corrected conclusion**: Best zero-shot = E3-14 (iemocap→c-besd-4cl, self_attention) = 34.68%. 
Range: 20.51%–34.68%. 
Previous handbook values (41.05%, 35.47%, 19.17% as range minimum) are PHANTOM — not found in any log.

---

## B3 — Data Augmentation Sensitivity (E4, 3-seed, ddof=1)

| Experiment | Aug | Pooling | WA s42 | WA s123 | WA s456 | WA mean+-std | UAR mean+-std |
|----------|-----|---------|--------|---------|---------|-------------|-------------|
| E4-01 | C1 | self_attention | 92.92% | 91.91% | 91.06% | 91.96+-0.93% | 91.91+-0.93% |
| E4-02 | C2 | self_attention | 66.18% | 66.15% | 66.28% | 66.20+-0.07% | 63.03+-0.37% |
| E4-03 | C3 | self_attention | 93.25% | 91.57% | 90.73% | 91.85+-1.29% | 91.80+-1.28% |
| E4-04 [INVALID AGGREGATION] | C4 | self_attention | 68.02% | 63.81% | 62.67% | 64.83+-2.82% | 60.64+-6.54% |
| E4-05 | C1 | mean | 67.81% | 64.21% | 67.22% | 66.41+-1.93% | 42.23+-2.64% |
| E4-06 | C2 | mean | 64.51% | 65.09% | 63.76% | 64.45+-0.67% | 56.73+-2.27% |
| E4-07 | C3 | mean | 68.55% | 64.68% | 65.06% | 66.10+-2.13% | 43.35+-1.45% |
| E4-08 | C4 | mean | 60.69% | 58.85% | 59.58% | 59.71+-0.92% | 48.24+-1.61% |
| E4-09 | C1 | prosody_guided | 65.05% | 64.91% | 63.17% | 64.38+-1.05% | 60.37+-0.78% |
| E4-10 [INVALID AGGREGATION] | C2 | prosody_guided | 63.60% | 56.87% | 65.32% | 61.93+-4.46% | 55.57+-5.39% |
| E4-11 | C3 | prosody_guided | 65.30% | 64.18% | 64.65% | 64.71+-0.56% | 59.88+-0.78% |
| E4-12 | C4 | prosody_guided | 60.60% | 56.35% | 61.90% | 59.62+-2.91% | 47.27+-6.47% |

**B3 conclusion**: C3 child augmentation shows weak positive benefit (+0.25–0.74pp). 
C2/C4 domain mixing significantly hurts performance (-6.8 to -9.4pp). 
E4-04 excluded from aggregation: s42 train=['c-besd'], s123/s456 train=['c-besd-4cl','iemocap'] — different corpora. 
E4-10 excluded from aggregation: s42/s123 train=['iemocap'], s456 train=['iemocap','fau-aibo'] — different corpora.

---

## B6 — Module Ablation (E6, C-BESD + FAU, 3-seed, ddof=1)

Design: cumulative build-up from Mean+Last baseline.

| Experiment | Dataset | Config | WA s42 | WA s123 | WA s456 | WA mean+-std | vs baseline Δ |
|-----------|--------|-------|--------|---------|---------|-------------|--------------|
| E6-01 | C-BESD | Mean+Last (baseline) | 80.27% | 81.45% | 80.94% | 80.89+-0.59% | +0.00pp |
| E6-02 | C-BESD | +Adapter | 78.08% | 82.63% | 82.80% | 81.17+-2.68% | +0.28pp |
| E6-03 | C-BESD | +SelfAttn,-Adapter | 92.92% | 91.74% | 91.06% | 91.91+-0.94% | +11.02pp |
| E6-04 | C-BESD | +WeightedFusion | 92.92% | 91.91% | 91.06% | 91.96+-0.93% | +11.07pp |
| E6-05 | C-BESD | Full stack | 90.73% | 91.74% | 90.89% | 91.12+-0.54% | +10.23pp |
| E6-06 | FAU_Aibo | Mean+Last (baseline) | 67.98% | 67.93% | 68.01% | 67.97+-0.05% | +0.00pp |
| E6-07 | FAU_Aibo | +Adapter | 68.01% | 67.96% | 68.49% | 68.15+-0.29% | +0.18pp |
| E6-08 | FAU_Aibo | +SelfAttn,-Adapter | 67.37% | 68.34% | 67.66% | 67.79+-0.50% | -0.19pp |
| E6-09 | FAU_Aibo | +WeightedFusion | 67.81% | 64.21% | 67.22% | 66.41+-1.93% | -1.56pp |
| E6-10 | FAU_Aibo | Full stack | 66.72% | 65.51% | 66.10% | 66.11+-0.60% | -1.87pp |

**B6 conclusion**: Pooling upgrade (Mean→SelfAttn) is the key improvement on C-BESD (+11pp). 
WF adds marginal benefit (<1pp). Adapter is neutral-to-negative on both datasets. 
C-BESD ceiling ~92.0%, FAU ceiling ~68.0%.

---

## B7 — Model Transfer Fine-tune (E7, 3-seed, ddof=1)

| Experiment | Claimed Source | Target | WA s42 | WA s123 | WA s456 | WA mean+-std | UAR mean+-std |
|-----------|---------------|--------|--------|---------|---------|-------------|-------------|
| E7-01 | C-BESD | FAU_Aibo | 67.66% | 65.98% | 66.83% | 66.82+-0.84% | 41.46+-0.92% | gap=-0.22pp vs ceiling |
| E7-02 | C-BESD | IEMOCAP | 62.81% | 63.78% | 63.17% | 63.25+-0.49% | 58.11+-0.59% | gap=-0.51pp vs ceiling |
| E7-03 | FAU_Aibo | C-BESD | 92.07% | 91.23% | 91.40% | 91.57+-0.45% | 91.52+-0.45% | gap=-0.30pp vs ceiling |
| E7-04 | FAU_Aibo | IEMOCAP | 63.57% | 61.61% | 63.24% | 62.81+-1.05% | 58.27+-0.87% | gap=-0.95pp vs ceiling |
| E7-05 | IEMOCAP | C-BESD | 91.74% | 89.71% | 92.07% | 91.17+-1.28% | 91.14+-1.28% | gap=-0.69pp vs ceiling |
| E7-06 | IEMOCAP | FAU_Aibo | 66.63% | 65.36% | 65.92% | 65.97+-0.64% | 43.95+-0.99% | gap=-1.08pp vs ceiling |

**B7 conclusion**: Target-domain ceiling dominates transfer results. 
C-BESD targets converge to 91.87% (ceiling), FAU targets to 67.05%, IEMOCAP targets to 63.76%. 
Source domain effect is limited (<0.5pp for same target). 
**Note**: Source domain assignment relies on launch_b7.sh correctness — not independently verifiable from logs.

---

## Global Leaderboard (3-seed sample mean WA, ddof=1)

| Rank | Experiment | Dataset | Config | WA mean+-std |
|------|-----------|--------|--------|-------------|
| 1 | E2-01 | C-BESD | self_attention+unfreeze | 96.91+-0.19% |
| 2 | E5-03 | C-BESD | self_attention+frozen | 91.96+-0.93% |
| 3 | E4-01 | C-BESD | self_attention+frozen | 91.96+-0.93% |
| 4 | E6-04 | C-BESD | self_attention+frozen | 91.96+-0.93% |
| 5 | E6-03 | C-BESD | self_attention+frozen | 91.91+-0.94% |
| 6 | E5-01 | C-BESD | self_attention+frozen | 91.91+-0.94% |
| 7 | E1-02 | C-BESD | self_attention+frozen | 91.87+-1.56% |
| 8 | E1-03 | C-BESD | prosody_guided+frozen | 91.86+-1.14% |
| 9 | E4-03 | C-BESD | self_attention+frozen | 91.85+-1.29% |
| 10 | E7-03 | C-BESD | self_attention+frozen | 91.57+-0.45% |
| 11 | E7-05 | C-BESD | self_attention+frozen | 91.17+-1.28% |
| 12 | E6-05 | C-BESD | self_attention+frozen | 91.12+-0.54% |
| 13 | E6-02 | C-BESD | mean+frozen | 81.17+-2.68% |
| 14 | E6-01 | C-BESD | mean+frozen | 80.89+-0.59% |
| 15 | E1-01 | C-BESD | mean+frozen | 79.23+-0.99% |

---

## WA-UAR by Corpus

| Corpus | WA mean+-std | UAR mean+-std | WA-UAR gap | N runs |
|--------|-------------|--------------|-----------|--------|
| C-BESD | 81.00+-19.50% | 80.68+-19.67% | 0.31pp | 69 |
| FAU_Aibo | 62.82+-11.59% | 41.67+-5.71% | 21.15pp | 69 |
| IEMOCAP | 59.47+-9.96% | 54.49+-8.61% | 4.98pp | 54 |

FAU_Aibo WA-UAR gap (21pp) reflects severe class imbalance. 
C-BESD WA≈UAR (0.3pp) due to near-perfect class balance.

---

## Data Integrity Notes

1. **std**: All mean+-std use sample std (ddof=1)
2. **INVALID**: 3 experiments excluded from aggregation (see above)
3. **Config fields**: augment_condition, fusion_mode, use_adapter, unfreeze_ssl, reg_profile are code defaults — not experimental conditions
4. **B7 source**: Source domain not independently verifiable from logs — relies on launch_b7.sh
5. **Predictions**: 0/192 files contain predictions/confusion matrices — WA/UAR trusted as-is from sklearn
6. **UAR**: Present in all 192 files; FAU WA-UAR gap = 21pp (class imbalance)
