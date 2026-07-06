# STAGE8_RESULTS_DISCUSSION_OUTLINE

## 1. Purpose

This file records the Stage 8D Results / Discussion outline in a traceable form.
It is intended for Stage 8E, where the outline may be expanded into editable
Results and Discussion paragraph drafts.

This file is not complete paper prose, not a final Discussion, and not a final
Conclusion. It does not revive or inherit old-agent conclusions. Claims here are
bounded by the clean project evidence index and must remain tied to source files
before paper writing.

## 2. Source and Evidence Boundaries

The evidence map for this outline comes primarily from
`docs/current/STAGE8_EVIDENCE_INDEX.md`, with clean boundary context from
`README.md`, `docs/current/CLEAN_PROJECT_STATUS.md`, and
`docs/current/AGENT_QA_MEMORY.md`.

Required boundaries:

- Current `results/logs/E*-*.json` count is 210.
- E5 contains 54 JSON files: 18 regular seed files and 36 L1-L12 single-layer scan files.
- E3 is single seed zero-shot transfer.
- E7 and E7-ext are reported as 3-seed mean+-std.
- B7-ext uses `batch_size=8`; E7 frozen uses `batch_size=16`.
- FAU Aibo must be interpreted with UAR, not WA alone.
- The Global Leaderboard is a WA index and does not participate in main conclusions.
- Target-side constraint is only a Level 3 plausible interpretation, not a proven causal law.
- The handbook header `192 files` is stale or inconsistent metadata.
- The current `scripts/regen_handbook.py` cannot fully reproduce the current handbook because it does not generate the B7-ext table.

## 3. Results Section Outline

### 4.1 Dataset design and in-domain baselines

**Section title**: Dataset design and in-domain baselines

**Purpose**: Establish what the three-corpus design can and cannot support before presenting performance comparisons.

**Core claim**: C-BESD, FAU Aibo, and IEMOCAP provide a bounded age/style comparison, but they cover only three cells of the 2x2 matrix.

**Evidence tables**: Design matrix; E1 in-domain baseline summaries; Table D for FAU WA-UAR gap.

**Key numbers to mention**:

- C-BESD: child acted corpus.
- FAU Aibo: child naturalistic child-robot interaction corpus.
- IEMOCAP: adult acted corpus.
- Adult naturalistic speech is not covered.
- FAU WA-UAR gaps in Table D are typically about 22-30 pp.

**Safe interpretation**: The design supports a bounded comparison across age and expression style, while preserving the missing adult-naturalistic cell as a limitation.

**Caveats**:

- Do not infer the missing adult-naturalistic cell.
- Do not claim independent causal separation of age and style.
- FAU difficulty must be discussed with UAR.

**Do not say**: The design fully proves age effects and style effects as independent causal mechanisms.

**Suggested paragraph skeleton**:

1. Introduce the three corpora and the two design axes.
2. State which 2x2 cells are covered and which cell is absent.
3. Summarize in-domain baselines as dataset context rather than final method ranking.
4. Add the FAU metric caveat immediately because WA alone is misleading there.

### 4.2 Cross-corpus zero-shot transfer

**Section title**: Cross-corpus zero-shot transfer

**Purpose**: Present direct source-to-target transfer without target training.

**Core claim**: E3 zero-shot transfer is weak across evaluated directions.

**Evidence tables**: Table A, E3 side.

**Key numbers to mention**:

- E3 is zero-shot and single seed.
- Best E3 entry in Table A is E3-14, IEMOCAP to C-BESD, WA 34.68%, UAR 34.66%.
- Other E3 best-direction values in Table A are roughly 21-34% WA.

**Safe interpretation**: Direct transfer across corpora remains difficult under the current setup.

**Caveats**:

- E3 is single seed, so avoid stability claims.
- E3 is not fine-tuning.
- E3 and E7 compare different task settings.

**Do not say**: E3 is a fine-tune experiment, or E3 proves the full mechanism of distribution shift.

**Suggested paragraph skeleton**:

1. Define E3 as source training followed by direct target testing.
2. Report the range and best observed zero-shot result from Table A.
3. Emphasize that all directions remain low relative to target fine-tuning.
4. Close with the single-seed caveat.

### 4.3 Target fine-tuning recovers cross-corpus performance

**Section title**: Target fine-tuning recovers cross-corpus performance

**Purpose**: Compare E3 zero-shot outcomes with E7 frozen target-domain fine-tuning outcomes.

**Core claim**: Target-domain fine-tuning substantially improves cross-corpus performance relative to zero-shot transfer.

**Evidence tables**: Table A.

**Key numbers to mention**:

- E7 is target-domain fine-tune transfer.
- C-BESD to FAU: E3 WA 21.91% to E7-01 WA 66.82%.
- IEMOCAP to C-BESD: E3 WA 34.68% to E7-05 WA 91.17%.
- E7 values are 3-seed mean+-std.

**Safe interpretation**: Fine-tuning on the target training split recovers much of the performance lost in direct zero-shot transfer.

**Caveats**:

- This is a task comparison, not a single-variable ablation.
- B7 source checkpoints must be verified from launch scripts.
- Do not infer a causal mechanism from the recovery alone.

**Do not say**: Fine-tuning proves why transfer succeeds or establishes a causal law.

**Suggested paragraph skeleton**:

1. Remind the reader that E7 loads a source checkpoint and trains on the target split.
2. Use Table A to contrast paired E3 and E7 directions.
3. Give two representative numeric examples.
4. State the recovery claim with the task-comparison caveat.

### 4.4 Frozen vs unfrozen adaptation

**Section title**: Frozen vs unfrozen adaptation

**Purpose**: Evaluate whether unfreezing WavLM gives uniform gains.

**Core claim**: Unfreeze benefits are dataset-dependent: C-BESD and IEMOCAP improve under current settings, while FAU does not show stable gains.

**Evidence tables**: Table B.

**Key numbers to mention**:

- In-domain C-BESD: E2-01 vs E1-02 gives +5.04 pp WA.
- In-domain FAU: E2-02 vs E1-05 gives -0.67 pp WA and -0.19 pp UAR.
- E7-ext FAU targets are negative in WA: about -0.67 to -0.88 pp.
- E7-ext uses `batch_size=8`; E7 frozen uses `batch_size=16`.

**Safe interpretation**: Unfreezing appears helpful for some targets but should not be treated as universally beneficial.

**Caveats**:

- B7-ext changes both unfreeze state and batch size.
- FAU must be evaluated with UAR.
- Differences should be described as current-setting evidence, not general laws.

**Do not say**: Unfreeze universally improves speech emotion recognition.

**Suggested paragraph skeleton**:

1. Separate in-domain B5 vs B1 and transfer B7-ext vs B7 evidence.
2. Report positive C-BESD/IEMOCAP examples and negative FAU examples.
3. Explicitly add the batch-size caveat for B7-ext.
4. Conclude that adaptation behavior is target-dependent under this design.

### 4.5 Ablation: pooling, fusion, and build-up

**Section title**: Ablation: pooling, fusion, and build-up

**Purpose**: Describe architectural contribution evidence without overclaiming global module superiority.

**Core claim**: E6 build-up suggests the largest C-BESD architectural step is the pooling change, while LayerFusion evidence is more limited and dataset-dependent.

**Evidence tables**: Table C for E6 build-up; Table F for E5 LayerFusion / last / weighted / single-layer summary.

**Key numbers to mention**:

- E6 is build-up, not remove-down.
- C-BESD E6-01 Mean+Last minimal baseline: WA 80.89%.
- C-BESD E6-03 +Pooling: WA 91.91%.
- E5 C-BESD last: WA 91.91%; weighted: WA 91.96%.
- E5 single-layer scans are seed42 only.

**Safe interpretation**: Pooling is the clearest C-BESD build-up step; fusion-related differences should be stated cautiously.

**Caveats**:

- E6 does not include IEMOCAP.
- The handbook lacks an independent B4/E5 detail table.
- E5 single-layer evidence is seed42 only.

**Do not say**: LayerFusion is globally or stably optimal.

**Suggested paragraph skeleton**:

1. Define E6 as a cumulative build-up design.
2. Present C-BESD and FAU build-up patterns from Table C.
3. Use Table F to constrain LayerFusion claims.
4. Close by saying architectural contributions are dataset-specific.

### 4.6 Metric caveats and reliability boundaries

**Section title**: Metric caveats and reliability boundaries

**Purpose**: Make explicit which metrics and metadata boundaries control interpretation.

**Core claim**: FAU requires UAR, and project-level leaderboard or metadata inconsistencies must not be treated as scientific conclusions.

**Evidence tables**: Table D, Table F, and the caveats list in the evidence index.

**Key numbers to mention**:

- FAU WA-UAR gaps are typically about 22-30 pp in Table D.
- Current JSON count is 210.
- Handbook header `192 files` is stale or inconsistent.
- E5 contains 18 regular files and 36 single-layer scan files.

**Safe interpretation**: Metric and metadata boundaries are part of the result interpretation, not housekeeping details.

**Caveats**:

- Leaderboard is WA-only and does not enter main conclusions.
- Handbook metadata and generator reproducibility boundaries must remain visible.
- JSON configuration fields cannot replace launch scripts.

**Do not say**: Leaderboard rank 1 is the scientific best result.

**Suggested paragraph skeleton**:

1. Explain why FAU is reported with both WA and UAR.
2. State the 210-log scope and E5 split.
3. Note the 192-file handbook metadata boundary.
4. Exclude leaderboard ranking from main scientific claims.

## 4. Discussion Section Outline

### 5.1 What is robust across the evidence

**Section title**: What is robust across the evidence

**Purpose**: Identify claims that are supported across the current evidence index.

**Main interpretation**: The strongest evidence supports weak zero-shot transfer, substantial recovery with target fine-tuning, and mandatory UAR-based FAU interpretation.

**Evidence tables**: Table A and Table D.

**Allowed wording**: Direct zero-shot transfer is weak, target fine-tuning substantially recovers performance, and FAU requires UAR because WA is misleading under class imbalance.

**Forbidden wording**: These results prove a final causal mechanism for cross-corpus transfer.

**Limitations to include**:

- E3 is single seed.
- The 2x2 design lacks adult naturalistic speech.
- Fine-tune recovery is not a single-variable causal proof.

**Suggested paragraph skeleton**:

1. Start with the most stable empirical pattern.
2. Link zero-shot weakness and fine-tune recovery through Table A.
3. Add FAU UAR as a reliability constraint.
4. State remaining design limitations.

### 5.2 Dataset-dependent adaptation behavior

**Section title**: Dataset-dependent adaptation behavior

**Purpose**: Discuss why adaptation and module effects should be framed as dataset-dependent.

**Main interpretation**: Unfreezing and architectural changes do not behave uniformly across C-BESD, FAU, and IEMOCAP.

**Evidence tables**: Table B, Table C, and Table F.

**Allowed wording**: Under current settings, unfreezing benefits C-BESD/IEMOCAP more than FAU, and module contributions differ by dataset.

**Forbidden wording**: Unfreeze universally improves SER, or a single architectural module is globally best.

**Limitations to include**:

- B7-ext changes batch size as well as unfreezing.
- FAU-specific causes are not proven.
- E6 excludes IEMOCAP.

**Suggested paragraph skeleton**:

1. Summarize target-dependent unfreeze behavior.
2. Connect this to dataset-specific E6/E5 ablation patterns.
3. Explain why this supports cautious, dataset-aware interpretation.
4. State unresolved causes as future work rather than conclusions.

### 5.3 Why FAU Aibo must be interpreted separately

**Section title**: Why FAU Aibo must be interpreted separately

**Purpose**: Prevent WA-only interpretation of FAU results.

**Main interpretation**: FAU's large WA-UAR gap indicates that overall accuracy can overstate performance under class imbalance.

**Evidence tables**: Table D.

**Allowed wording**: FAU Aibo should be interpreted with UAR foregrounded, because WA and UAR diverge substantially.

**Forbidden wording**: High WA on FAU indicates strong generalization.

**Limitations to include**:

- UAR highlights the difficulty but does not by itself diagnose all causes.
- FAU-specific explanation should avoid unsupported claims about noise or annotation mechanisms unless separately verified.

**Suggested paragraph skeleton**:

1. Point to the persistent WA-UAR gap.
2. Explain why this changes interpretation of FAU-target and FAU in-domain results.
3. Separate metric evidence from causal diagnosis.
4. State that later analyses should keep FAU as a special reliability boundary.

### 5.4 Target-side constraints as an interpretation, not a causal law

**Section title**: Target-side constraints as an interpretation, not a causal law

**Purpose**: Safely handle the old target-domain ceiling idea without overstating it.

**Main interpretation**: E7 frozen same-target clustering is consistent with target-side constraints shaping fine-tuned outcomes, but it does not prove target-domain ceiling dominance.

**Evidence tables**: Table E.

**Allowed wording**: E7 frozen results suggest that target-side constraints may shape fine-tuned performance.

**Forbidden wording**: Target-domain ceiling dominance is proven, or target-side constraints are a causal law.

**Limitations to include**:

- This is Level 3 plausible interpretation.
- E7-ext should not be mixed into this argument because it introduces unfreeze and batch-size differences.
- Domain-internal performance differences do not automatically prove the mechanism.

**Suggested paragraph skeleton**:

1. Describe same-target clustering from Table E.
2. Present target-side constraint as an interpretation.
3. Explicitly reject causal-law wording.
4. List the variables that prevent stronger claims.

### 5.5 Architectural implications

**Section title**: Architectural implications

**Purpose**: Translate ablation evidence into cautious design implications.

**Main interpretation**: Pooling appears important for C-BESD in E6, but fusion and adapter conclusions are weaker and dataset-specific.

**Evidence tables**: Table C and Table F.

**Allowed wording**: E6 build-up suggests pooling is the largest C-BESD step, while LayerFusion effects should be interpreted cautiously.

**Forbidden wording**: LayerFusion is globally optimal, adapters are universally useless, or the full stack is always best.

**Limitations to include**:

- E6 does not include IEMOCAP.
- E5 single-layer scans are seed42 only.
- The handbook lacks an independent B4/E5 detail table.

**Suggested paragraph skeleton**:

1. State the C-BESD pooling observation.
2. Compare it with FAU build-up behavior.
3. Use E5 to limit claims about LayerFusion.
4. Present implications as architecture guidance, not universal rules.

### 5.6 Limitations and future work

**Section title**: Limitations and future work

**Purpose**: Collect the design, statistical, and reproducibility boundaries that must stay visible.

**Main interpretation**: The current study provides a bounded, source-grounded analysis but still requires additional evidence before stronger causal or universal claims.

**Evidence tables**: Design matrix; Table A; Table B; Table F; caveats list.

**Allowed wording**: Future work should add an adult naturalistic corpus, repeat E3 with multiple seeds, expand E5/E4 reporting, and reconcile handbook generation boundaries.

**Forbidden wording**: The current evidence fully resolves age/style causality or all adaptation mechanisms.

**Limitations to include**:

- Missing adult naturalistic cell.
- E3 single seed.
- E5 single-layer seed42 scans.
- B7-ext batch-size difference.
- Handbook `192 files` metadata issue.
- `regen_handbook.py` reproducibility boundary.

**Suggested paragraph skeleton**:

1. Start with design limitations.
2. Move to seed/statistical limitations.
3. State configuration and handbook reproducibility boundaries.
4. End with concrete future work items.

## 5. Main Conclusion to Section Mapping

| Main Conclusion | Results Section | Discussion Section | Evidence Tables | Strength | Caveat | Paper-safe Wording |
| --- | --- | --- | --- | --- | --- | --- |
| Zero-shot weak, fine-tune recovers | 4.2, 4.3 | 5.1 | Table A | Strong | E3 is single seed; E7 is target fine-tune, not a single-variable ablation | Direct cross-corpus zero-shot transfer is weak, while target-domain fine-tuning substantially recovers performance. |
| Unfreeze target-dependent | 4.4 | 5.2 | Table B | Medium-strong | B7-ext changes unfreeze and batch size; FAU requires UAR | Unfreezing helps C-BESD/IEMOCAP under current settings but does not produce stable gains on FAU. |
| Pooling dominates C-BESD build-up | 4.5 | 5.5 | Table C | Medium | E6 excludes IEMOCAP | E6 build-up suggests the pooling change is the largest C-BESD architecture step. |
| FAU must be interpreted by UAR | 4.1, 4.6 | 5.1, 5.3 | Table D | Strong boundary | UAR indicates metric difficulty but not the full cause | FAU Aibo results must report UAR because WA can overstate performance under class imbalance. |
| 2x2 design is bounded | 4.1 | 5.6 | Design matrix | Strong boundary | Adult naturalistic cell missing | The three-corpus design supports a bounded age/style comparison, not a complete causal decomposition. |
| Target-side constraint | 4.3, 4.6 | 5.4 | Table E | Medium/weak | Level 3 only; E7-ext excluded from this interpretation | E7 frozen same-target clustering suggests target-side constraints may shape fine-tuned outcomes. |
| LayerFusion conclusion | 4.5 | 5.5 | Table F | Weak/medium | Handbook lacks independent E5 detail; single-layer scan seed42 only | LayerFusion effects are dataset-dependent and should be interpreted cautiously. |
| Data augmentation conclusion | 4.6 or future E4 subsection | 5.2, 5.6 | E4 table still needed | Weak until supplemented | Needs E4 C1-C4 WA/UAR table | Augmentation effects should be treated as condition- and dataset-dependent until fully tabulated. |
| Leaderboard caveat | 4.6 | 5.6 | Caveats list | Boundary only | WA-only ranking; rank 1/2 sorting boundary | The leaderboard is an index for navigation, not a scientific conclusion. |

## 6. Abstract-safe Sentences

1. We evaluate child speech emotion recognition across three corpora spanning child acted, child naturalistic, and adult acted speech, while explicitly preserving the missing adult-naturalistic cell as a design limitation.
2. Direct cross-corpus zero-shot transfer remains weak, whereas target-domain fine-tuning substantially recovers performance across evaluated directions.
3. Adaptation gains are dataset-dependent, and FAU Aibo must be interpreted with UAR rather than WA alone because of its large WA-UAR gap.

## 7. Introduction Contribution Bullets

- We provide a bounded three-corpus design for studying child SER across age and expression-style differences.
- We separate zero-shot transfer, target fine-tuning, unfreezing, and architectural build-up ablations using launch-verified configurations.
- We show that metric choice is central for FAU Aibo, where WA alone can misrepresent task difficulty.

## 8. Conclusion-safe Sentences

1. The experiments indicate that target-domain fine-tuning is critical for recovering cross-corpus SER performance after weak zero-shot transfer.
2. Unfreezing and architectural changes do not yield uniform benefits across datasets, highlighting the need for dataset-specific interpretation.
3. The study supports a bounded age/style comparison, but the missing adult-naturalistic corpus and FAU's WA-UAR gap must remain explicit limitations.

## 9. Old-conclusion Risk Warnings

| Old Wording | Risk | Safer Replacement | Evidence Source |
| --- | --- | --- | --- |
| 目标域天花板主导迁移结果 | Turns a Level 3 interpretation into a causal law | E7 frozen same-target clustering suggests target-side constraints may shape fine-tuned outcomes. | Table E |
| leaderboard 第一就是科学最优 | Treats a WA index as scientific proof | Leaderboard ranking is navigational only and does not define the main conclusion. | Evidence-index caveats |
| LayerFusion 稳定最优 | Overstates weak and dataset-dependent fusion evidence | LayerFusion effects are dataset-dependent and require cautious interpretation. | Table F |
| 数据增强稳定有效 | Claims stability without a complete E4 evidence table | Augmentation effects require an E4 C1-C4 table before strong wording. | Conclusion writability matrix |
| FAU WA 高说明泛化好 | Ignores class imbalance and UAR | FAU must be interpreted with UAR because WA can be misleading. | Table D |
| E3/E7 都是 fine-tune | Confuses zero-shot and fine-tune protocols | E3 is zero-shot; E7 is target-domain fine-tune transfer. | Design docs and launch scripts |
| E6 是 remove-down ablation | Reverses the experiment direction | E6 is build-up from a minimal Mean+Last baseline. | Table C and launch_b6.sh |
| unfreeze 普遍有效 | Ignores FAU negative/unstable deltas and batch-size caveat | Unfreeze benefits are dataset-dependent under current settings. | Table B |
| E7-09 证明优于 E2-01 | Converts a tiny leaderboard difference into a scientific claim | E7-09 vs E2-01 belongs to a WA ranking boundary, not a main conclusion. | Evidence-index leaderboard caveat |

## 10. Required Caveats for Paper Writing

- E3 is single seed.
- E5 single-layer scans are seed42 only.
- E7-ext uses `batch_size=8`, while E7 frozen uses `batch_size=16`.
- FAU must use UAR alongside WA.
- The 2x2 design lacks the adult-naturalistic cell.
- The leaderboard is a WA-only index.
- The handbook header `192 files` is stale or inconsistent metadata.
- The current `regen_handbook.py` cannot fully reproduce the current handbook.
- JSON configuration fields cannot replace launch scripts.
- Source checkpoints for B7/B7-ext must be verified from launch scripts.

## 11. What This Outline Is Not

This outline is not:

- complete paper prose;
- a final Discussion;
- a final Conclusion;
- a revival of old-agent conclusions;
- the final version to send to a teacher;
- a replacement for the evidence index, raw JSON, launch scripts, or handbook.

Stage 8E may expand this outline into editable paragraph drafts. Stage 9 should
perform old-conclusion sentence-by-sentence audit before any legacy wording is
reused.

## 12. Recommended Next Step

1. Stage 8D-3: review and selectively commit `docs/current/STAGE8_RESULTS_DISCUSSION_OUTLINE.md`.
2. Stage 8E: use this outline to write editable Results / Discussion paragraph drafts.
3. Stage 9: audit old conclusions line by line against the clean evidence base.
