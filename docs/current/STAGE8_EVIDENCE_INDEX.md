# STAGE8_EVIDENCE_INDEX

## 1. Purpose

This file is an evidence index for Stage 8D Results / Discussion outlining. It is not paper prose, not a Discussion section, and not an old-conclusion audit. It records table-ready evidence, source boundaries, and allowed wording for the current clean project state.

## 2. Source Files and Trust Boundaries

| Source | Use | Trust Boundary |
| --- | --- | --- |
| `results/logs/E*-*.json` | Recorded metrics and run fields | Current count is 210; result fields trusted, config fields not sufficient alone |
| `scripts/launch_b*.sh` | Actual CLI configuration | Required for `data_split_seed`, `batch_size`, `ssl_lr`, `load_checkpoint`, and B7 source checkpoints |
| `docs/current/权威数据手册.md` | Aggregated statistics | Header says `192 files`; stale/inconsistent; lacks independent B4/E5 table; current generator lacks B7-ext table |
| `docs/current/实验设计方案_v3_含学习笔记.md` | Experiment design | Design intent, not result proof |
| `README.md`, `CLEAN_PROJECT_STATUS.md`, `AGENT_QA_MEMORY.md` | Clean boundaries and onboarding | Do not use as raw result evidence |

## 3. Main Conclusions from Stage 8B

| Conclusion | Stage-8B Strength | Short Safe Form |
| --- | --- | --- |
| Zero-shot weak, fine-tune recovers | Strong | Direct cross-corpus zero-shot transfer is weak; target fine-tuning substantially recovers performance. |
| Unfreeze target-dependent | Medium-strong | Unfreezing helps C-BESD/IEMOCAP but not FAU under current settings. |
| Pooling dominates C-BESD build-up | Medium | E6 build-up suggests the largest C-BESD gain comes from the pooling change. |
| FAU must use UAR | Strong boundary | FAU interpretations must report UAR, not WA alone. |
| 2x2 design is bounded | Strong boundary | The corpus design covers three cells and lacks adult naturalistic speech. |
| Target-side constraint | Discussion-only | E7 frozen results suggest target-side constraints, but not a causal law. |

## 4. Numeric Scope and Counting Rules

| Item | Current Rule |
| --- | --- |
| Current JSON count | 210 |
| Series distribution | E1=27, E2=9, E3=18, E4=36, E5=54, E6=30, E7=36 |
| E5 split | 18 regular 3-seed files + 36 L1-L12 single-layer scan files |
| 3-seed mean+-std | E1, E2, E4, E5 regular, E6, E7, E7-ext |
| Single-seed results | E3 all directions; E5 single-layer scans |
| Std convention | sample std, ddof=1 |
| FAU metric rule | Always report UAR with WA; prefer in-domain FAU for FAU difficulty and by-test-data for all FAU-target evaluations |
| Leaderboard rule | WA index only; not a scientific conclusion |

## 5. Table A: E3 Zero-shot vs E7 Frozen Fine-tune Paired Comparison

Purpose: support the claim that zero-shot transfer is weak while target fine-tuning recovers performance. E3 is single seed; E7 is 3-seed mean+-std. This is a task comparison, not a single-variable ablation.

| Source | Target | Best E3 Exp | Best E3 Pooling | E3 WA | E3 UAR | E7 Exp | E7 WA mean+-std | E7 UAR mean+-std | WA Delta | UAR Delta | Caveat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C-BESD | fau-aibo | E3-03 | prosody_guided | 21.91% | 23.99% | E7-01 | 66.82% +- 0.84% | 41.46% +- 0.92% | 44.91 pp | 17.47 pp | E3 single seed; E7 3-seed fine-tune; task comparison, not single-variable ablation |
| C-BESD | iemocap | E3-04 | mean | 30.60% | 27.02% | E7-02 | 63.25% +- 0.49% | 58.11% +- 0.59% | 32.65 pp | 31.10 pp | E3 single seed; E7 3-seed fine-tune; task comparison, not single-variable ablation |
| FAU Aibo | c-besd | E3-07 | mean | 28.67% | 28.75% | E7-03 | 91.57% +- 0.45% | 91.52% +- 0.45% | 62.90 pp | 62.77 pp | E3 single seed; E7 3-seed fine-tune; task comparison, not single-variable ablation |
| FAU Aibo | iemocap | E3-12 | prosody_guided | 27.54% | 30.59% | E7-04 | 62.81% +- 1.05% | 58.27% +- 0.87% | 35.27 pp | 27.68 pp | E3 single seed; E7 3-seed fine-tune; task comparison, not single-variable ablation |
| IEMOCAP | c-besd | E3-14 | self_attention | 34.68% | 34.66% | E7-05 | 91.17% +- 1.28% | 91.14% +- 1.28% | 56.50 pp | 56.48 pp | E3 single seed; E7 3-seed fine-tune; task comparison, not single-variable ablation |
| IEMOCAP | fau-aibo | E3-17 | self_attention | 33.71% | 33.00% | E7-06 | 65.97% +- 0.64% | 43.95% +- 0.99% | 32.26 pp | 10.95 pp | E3 single seed; E7 3-seed fine-tune; task comparison, not single-variable ablation |

## 6. Table B: Frozen vs Unfrozen Adaptation Delta

Purpose: support target-dependent unfreeze behavior. B7-ext uses `--unfreeze_ssl --ssl_lr 1e-5` and batch_size=8; E7 frozen uses batch_size=16, so transfer pairs are not pure single-variable comparisons.

| Setting | Source | Target | Frozen Exp | Frozen WA mean+-std | Frozen UAR mean+-std | Unfrozen Exp | Unfrozen WA mean+-std | Unfrozen UAR mean+-std | WA Delta | UAR Delta | Batch Size Caveat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| In-domain B5 vs B1 | none | C-BESD | E1-02 | 91.87% +- 1.56% | 91.85% +- 1.54% | E2-01 | 96.91% +- 0.19% | 96.88% +- 0.21% | 5.04 pp | 5.03 pp | B1 batch_size=32; B5 unfreeze uses batch_size=8, ssl_lr=1e-5 |
| In-domain B5 vs B1 | none | FAU Aibo | E1-05 | 67.05% +- 0.67% | 42.92% +- 1.85% | E2-02 | 66.37% +- 1.04% | 42.72% +- 3.18% | -0.67 pp | -0.19 pp | B1 batch_size=32; B5 unfreeze uses batch_size=8, ssl_lr=1e-5 |
| In-domain B5 vs B1 | none | IEMOCAP | E1-09 | 64.38% +- 1.05% | 60.37% +- 0.78% | E2-03 | 66.37% +- 0.95% | 60.75% +- 1.32% | 1.99 pp | 0.38 pp | B1 batch_size=32; B5 unfreeze uses batch_size=8, ssl_lr=1e-5 |
| Transfer B7-ext vs B7 | C-BESD | FAU Aibo | E7-01 | 66.82% +- 0.84% | 41.46% +- 0.92% | E7-07 | 65.95% +- 2.16% | 41.37% +- 4.06% | -0.88 pp | -0.09 pp | E7 frozen batch_size=16; E7-ext unfreeze + ssl_lr=1e-5 + batch_size=8 |
| Transfer B7-ext vs B7 | C-BESD | IEMOCAP | E7-02 | 63.25% +- 0.49% | 58.11% +- 0.59% | E7-08 | 66.85% +- 0.58% | 61.35% +- 0.67% | 3.59 pp | 3.24 pp | E7 frozen batch_size=16; E7-ext unfreeze + ssl_lr=1e-5 + batch_size=8 |
| Transfer B7-ext vs B7 | FAU Aibo | C-BESD | E7-03 | 91.57% +- 0.45% | 91.52% +- 0.45% | E7-09 | 96.96% +- 0.61% | 96.93% +- 0.61% | 5.40 pp | 5.41 pp | E7 frozen batch_size=16; E7-ext unfreeze + ssl_lr=1e-5 + batch_size=8 |
| Transfer B7-ext vs B7 | FAU Aibo | IEMOCAP | E7-04 | 62.81% +- 1.05% | 58.27% +- 0.87% | E7-10 | 67.19% +- 0.25% | 62.10% +- 0.99% | 4.38 pp | 3.82 pp | E7 frozen batch_size=16; E7-ext unfreeze + ssl_lr=1e-5 + batch_size=8 |
| Transfer B7-ext vs B7 | IEMOCAP | C-BESD | E7-05 | 91.17% +- 1.28% | 91.14% +- 1.28% | E7-11 | 96.40% +- 0.59% | 96.38% +- 0.60% | 5.23 pp | 5.24 pp | E7 frozen batch_size=16; E7-ext unfreeze + ssl_lr=1e-5 + batch_size=8 |
| Transfer B7-ext vs B7 | IEMOCAP | FAU Aibo | E7-06 | 65.97% +- 0.64% | 43.95% +- 0.99% | E7-12 | 65.30% +- 1.09% | 42.31% +- 1.93% | -0.67 pp | -1.64 pp | E7 frozen batch_size=16; E7-ext unfreeze + ssl_lr=1e-5 + batch_size=8 |

## 7. Table C: E6 Build-up Ablation

Purpose: support that E6 is build-up, not remove-down, and that C-BESD has its largest step at the pooling change.

| Dataset | E6 Exp | Build-up Step | Config Summary | WA mean+-std | UAR mean+-std | WA Delta vs Minimal | UAR Delta vs Minimal | Interpretation Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C-BESD | E6-01 | Minimal | Mean + Last, no adapter | 80.89% +- 0.59% | 80.85% +- 0.58% | 0.00 pp | 0.00 pp | Build-up step; not remove-down.  |
| C-BESD | E6-02 | +Adapter | Mean + Last + adapter | 81.17% +- 2.68% | 81.16% +- 2.69% | 0.28 pp | 0.30 pp | Build-up step; not remove-down.  |
| C-BESD | E6-03 | +Pooling | Self-attention + Last, no adapter | 91.91% +- 0.94% | 91.86% +- 0.92% | 11.02 pp | 11.01 pp | Build-up step; not remove-down. Pooling-change row. |
| C-BESD | E6-04 | +LayerFusion | Self-attention + Weighted, no adapter | 91.96% +- 0.93% | 91.91% +- 0.93% | 11.07 pp | 11.06 pp | Build-up step; not remove-down.  |
| C-BESD | E6-05 | Full | Adapter + Self-attention + Weighted | 91.12% +- 0.54% | 91.07% +- 0.54% | 10.23 pp | 10.22 pp | Build-up step; not remove-down.  |
| FAU Aibo | E6-06 | Minimal | Mean + Last, no adapter | 67.97% +- 0.05% | 38.44% +- 0.71% | 0.00 pp | 0.00 pp | Build-up step; not remove-down.  |
| FAU Aibo | E6-07 | +Adapter | Mean + Last + adapter | 68.15% +- 0.29% | 38.89% +- 1.54% | 0.18 pp | 0.45 pp | Build-up step; not remove-down.  |
| FAU Aibo | E6-08 | +Pooling | Self-attention + Last, no adapter | 67.79% +- 0.50% | 41.31% +- 0.69% | -0.19 pp | 2.88 pp | Build-up step; not remove-down. Pooling-change row. |
| FAU Aibo | E6-09 | +LayerFusion | Self-attention + Weighted, no adapter | 66.41% +- 1.93% | 42.23% +- 2.64% | -1.56 pp | 3.80 pp | Build-up step; not remove-down.  |
| FAU Aibo | E6-10 | Full | Adapter + Self-attention + Weighted | 66.11% +- 0.60% | 44.10% +- 1.39% | -1.87 pp | 5.66 pp | Build-up step; not remove-down.  |

## 8. Table D: FAU WA-UAR Gap Evidence

Purpose: show why FAU Aibo cannot be interpreted with WA alone. FAU has severe class imbalance; UAR must stay visible.

| Experiment | Setting | Source | Target | WA mean+-std | UAR mean+-std | WA-UAR Gap | Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E1-04 | B1 mean in-domain | none | FAU Aibo | 66.94% +- 1.32% | 39.33% +- 0.54% | 27.60 pp | FAU is class-imbalanced; UAR must accompany WA |
| E1-05 | B1 self-attn in-domain | none | FAU Aibo | 67.05% +- 0.67% | 42.92% +- 1.85% | 24.13 pp | FAU is class-imbalanced; UAR must accompany WA |
| E1-06 | B1 prosody in-domain | none | FAU Aibo | 66.77% +- 1.19% | 42.94% +- 1.59% | 23.83 pp | FAU is class-imbalanced; UAR must accompany WA |
| E2-02 | B5 unfrozen in-domain | none | FAU Aibo | 66.37% +- 1.04% | 42.72% +- 3.18% | 23.65 pp | FAU is class-imbalanced; UAR must accompany WA |
| E6-06 | E6 minimal | none | FAU Aibo | 67.97% +- 0.05% | 38.44% +- 0.71% | 29.54 pp | FAU is class-imbalanced; UAR must accompany WA |
| E6-07 | E6 +Adapter | none | FAU Aibo | 68.15% +- 0.29% | 38.89% +- 1.54% | 29.27 pp | FAU is class-imbalanced; UAR must accompany WA |
| E6-08 | E6 +Pooling | none | FAU Aibo | 67.79% +- 0.50% | 41.31% +- 0.69% | 26.48 pp | FAU is class-imbalanced; UAR must accompany WA |
| E6-09 | E6 +Fusion | none | FAU Aibo | 66.41% +- 1.93% | 42.23% +- 2.64% | 24.18 pp | FAU is class-imbalanced; UAR must accompany WA |
| E6-10 | E6 full | none | FAU Aibo | 66.11% +- 0.60% | 44.10% +- 1.39% | 22.01 pp | FAU is class-imbalanced; UAR must accompany WA |
| E7-01 | E7 frozen transfer | C-BESD | FAU Aibo | 66.82% +- 0.84% | 41.46% +- 0.92% | 25.36 pp | FAU is class-imbalanced; UAR must accompany WA |
| E7-06 | E7 frozen transfer | IEMOCAP | FAU Aibo | 65.97% +- 0.64% | 43.95% +- 0.99% | 22.01 pp | FAU is class-imbalanced; UAR must accompany WA |
| E7-07 | E7-ext unfrozen transfer | C-BESD | FAU Aibo | 65.95% +- 2.16% | 41.37% +- 4.06% | 24.58 pp | FAU is class-imbalanced; UAR must accompany WA |
| E7-12 | E7-ext unfrozen transfer | IEMOCAP | FAU Aibo | 65.30% +- 1.09% | 42.31% +- 1.93% | 22.99 pp | FAU is class-imbalanced; UAR must accompany WA |

## 9. Table E: E7 Same-target Source Comparison

Purpose: support a Level 3 target-side constraint interpretation using E7 frozen only. Do not mix E7-ext into this table.

| Target | Source | E7 Exp | WA mean+-std | UAR mean+-std | Within-target WA Range | Within-target UAR Range | Interpretation Boundary |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C-BESD | FAU Aibo | E7-03 | 91.57% +- 0.45% | 91.52% +- 0.45% | 0.39 pp | 0.38 pp | Level 3 plausible target-side constraint only; not causal proof |
| C-BESD | IEMOCAP | E7-05 | 91.17% +- 1.28% | 91.14% +- 1.28% | 0.39 pp | 0.38 pp | Level 3 plausible target-side constraint only; not causal proof |
| FAU Aibo | C-BESD | E7-01 | 66.82% +- 0.84% | 41.46% +- 0.92% | 0.86 pp | 2.49 pp | Level 3 plausible target-side constraint only; not causal proof |
| FAU Aibo | IEMOCAP | E7-06 | 65.97% +- 0.64% | 43.95% +- 0.99% | 0.86 pp | 2.49 pp | Level 3 plausible target-side constraint only; not causal proof |
| IEMOCAP | C-BESD | E7-02 | 63.25% +- 0.49% | 58.11% +- 0.59% | 0.45 pp | 0.16 pp | Level 3 plausible target-side constraint only; not causal proof |
| IEMOCAP | FAU Aibo | E7-04 | 62.81% +- 1.05% | 58.27% +- 0.87% | 0.45 pp | 0.16 pp | Level 3 plausible target-side constraint only; not causal proof |

## 10. Table F: E5 LayerFusion / Last / Weighted / Single-layer Summary

Purpose: constrain LayerFusion claims. The handbook lacks an independent B4/E5 detail table; single-layer scan is seed42 only and cannot establish stability.

| Dataset | Last WA mean+-std | Last UAR mean+-std | Weighted WA mean+-std | Weighted UAR mean+-std | Best Single Layer | Best Single WA | Best Single UAR | Seed Scope | Interpretation Boundary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C-BESD | 91.91% +- 0.94% | 91.86% +- 0.92% | 91.96% +- 0.93% | 91.91% +- 0.93% | L12 (E5-02_L12_s42) | 92.92% | 92.85% | Last/weighted are 3-seed; best single is seed42 only | Handbook lacks independent B4/E5 detail; do not overclaim LayerFusion superiority |
| FAU Aibo | 67.79% +- 0.50% | 41.31% +- 0.69% | 66.41% +- 1.93% | 42.23% +- 2.64% | L9 (E5-05_L9_s42) | 67.87% | 41.14% | Last/weighted are 3-seed; best single is seed42 only | Handbook lacks independent B4/E5 detail; do not overclaim LayerFusion superiority |
| IEMOCAP | 63.64% +- 0.46% | 56.61% +- 1.86% | 64.38% +- 1.05% | 60.37% +- 0.78% | L11 (E5-08_L11_s42) | 65.16% | 60.01% | Last/weighted are 3-seed; best single is seed42 only | Handbook lacks independent B4/E5 detail; do not overclaim LayerFusion superiority |

## 11. Conclusion Writability Matrix

| Conclusion | Status | Evidence Tables | Strength | Allowed Wording | Forbidden Wording | Need More Work? |
| --- | --- | --- | --- | --- | --- | --- |
| Zero-shot weak, fine-tune recovers | 可直接写 | Table A | Strong | Direct zero-shot transfer is weak, while target fine-tuning substantially recovers performance. | Fine-tuning proves a causal mechanism. | No; table should still be cited. |
| Unfreeze target-dependent | 需补表后写 | Table B | Medium-strong | Unfreezing improves C-BESD/IEMOCAP but not FAU under current settings. | Unfreeze is universally beneficial. | Needs batch-size caveat in text. |
| Pooling dominates C-BESD build-up | 需补表后写 | Table C | Medium | E6 build-up suggests pooling is the largest C-BESD architecture step. | E6 proves full stack is best. | Needs E6 build-up table. |
| FAU must be interpreted by UAR | 可直接写 | Table D | Strong | FAU results must report UAR because WA is inflated by imbalance. | FAU WA alone shows strong generalization. | No. |
| 2x2 design is bounded | 可直接写 | Design matrix, README, AGENT_QA_MEMORY | Strong | The three corpora support a bounded age/style contrast with one missing cell. | The study fully proves age x style causal effects. | No. |
| Target-side constraint | 只能 Discussion 弱写 | Table E | Medium/weak | E7 frozen results suggest target-side constraints may shape fine-tuned performance. | Target-domain ceiling is a proven causal law. | Needs careful Discussion wording. |
| LayerFusion conclusion | 不建议写主结论 | Table F | Weak/medium | Layer-fusion effects are dataset-dependent and should be interpreted cautiously. | Weighted LayerFusion is globally superior. | Needs E5 detail table and single-seed caveat. |
| Data augmentation conclusion | 需补表后写 | E4 table needed | Medium/weak | Augmentation effects are condition- and dataset-dependent. | Augmentation is stably beneficial. | Need E4 C1-C4 WA/UAR table. |
| Leaderboard conclusion | 禁止写成结论 | Leaderboard appendix only | Weak | Use leaderboard only as WA index with rank boundary noted. | Leaderboard first is scientific best. | No main-text conclusion. |

## 12. Results Section Evidence Map

| Section | Purpose | Evidence Tables | Core Message | Caveats | Do Not Say |
| --- | --- | --- | --- | --- | --- |
| Dataset design and in-domain baselines | Establish corpora and baseline difficulty | Design matrix, E1, Table D | C-BESD is high-performing; FAU requires UAR; IEMOCAP is adult acted control. | Adult naturalistic cell missing; do not claim full causal decomposition. | Do not say the design fully proves age/style causal effects. |
| Zero-shot transfer exposes cross-corpus generalization difficulty | Show direct transfer weakness | Table A E3 side | Zero-shot transfer remains weak across directions. | E3 is single seed. | Do not treat E3 as fine-tune. |
| Target fine-tuning recovers cross-corpus performance | Compare E3 vs E7 | Table A | Target fine-tuning substantially recovers performance. | Task comparison, not single-variable ablation. | Do not claim causal mechanism. |
| Unfreezing provides dataset-dependent adaptation gains | Show frozen/unfrozen differences | Table B | Unfreeze helps C-BESD/IEMOCAP but not FAU. | B7-ext batch_size differs; FAU UAR required. | Do not say unfreeze universally helps. |
| Ablations identify dataset-specific architectural contributions | Summarize E6/E5 | Table C, Table F | C-BESD build-up has largest pooling step; fusion evidence is cautious. | E6 excludes IEMOCAP; E5 single-layer seed42 only. | Do not overclaim LayerFusion. |
| Metric caveats and reliability boundaries | Make interpretation safe | Table D, caveat list | WA-only ranking is insufficient for FAU and leaderboard is not science. | Handbook 192/210, E5 missing detail. | Do not use leaderboard as main conclusion. |

## 13. Discussion Section Evidence Map

| Section | Purpose | Evidence Tables | Safe Interpretation | Limitations | Do Not Say |
| --- | --- | --- | --- | --- | --- |
| What is robust across the evidence | Identify strongest claims | Table A, Table D | Zero-shot weak, fine-tune recovers, FAU UAR caveat are robust. | E3 single seed and missing 2x2 cell remain. | Do not claim final causal proof. |
| What appears dataset-dependent | Explain adaptation variation | Table B, Table C, Table F | Unfreeze and architecture contributions vary by dataset. | FAU-specific causes are not proven. | Do not say one module is globally best. |
| Why FAU must be interpreted separately | Prevent WA-only interpretation | Table D | FAU WA-UAR gap is large and persistent. | UAR does not alone diagnose why. | Do not call high WA strong generalization. |
| Target-side constraints as an interpretation, not a law | Safely handle old target-ceiling idea | Table E | E7 frozen same-target clustering suggests target-side constraints. | Not causal; E7-ext has extra variables. | Do not write target-domain ceiling dominates as a law. |
| Limitations and future work | State boundaries | Design matrix, Table F, caveats | Future work should add adult naturalistic corpus, stronger E3 seeds, reconciled handbook/E5 detail. | Limitations are not failures. | Do not hide metadata caveats. |

## 14. Caveats That Must Stay Visible

- The current JSON count is 210, not 192.
- The handbook header `192 files` is stale or inconsistent metadata.
- Current `regen_handbook.py` cannot fully reproduce the current handbook because it does not generate B7-ext table.
- The handbook lacks an independent B4/E5 detail table.
- E3 is single seed.
- E5 single-layer scans are seed42 only.
- E7 and E7-ext source checkpoints must be checked in launch scripts.
- B7-ext changes unfreeze and batch_size relative to E7.
- FAU requires UAR because WA can be misleading under class imbalance.
- The 2x2 design lacks adult naturalistic speech.


## 15. Forbidden Overclaims

- Do not write that target-domain ceiling dominance is a proven causal law.
- Do not treat leaderboard rank 1 as the scientific best result.
- Do not claim unfreeze universally helps.
- Do not claim LayerFusion is globally superior.
- Do not claim augmentation is stably beneficial without an E4 table.
- Do not describe E3 as fine-tuning or E7 as zero-shot.
- Do not describe E6 as remove-down ablation.
- Do not interpret FAU using WA alone.
- Do not cite old-agent conclusions as evidence.


## 16. What Still Needs Manual Verification Before Paper Writing

- Decide whether FAU difficulty text uses in-domain UAR or by-test-data UAR, and state that scope explicitly.
- For Table A, decide whether E3 rows should use best pooling per direction or matched pooling; this index uses best E3 pooling.
- For Table B, explicitly explain B7-ext batch_size=8 in any paper text.
- For Table F, consider adding full L1-L12 appendix rows if LayerFusion becomes a major discussion point.
- Create a separate E4 C1-C4 table before writing any data-augmentation conclusion.
- Reconcile or footnote handbook metadata issues before final paper packaging.
