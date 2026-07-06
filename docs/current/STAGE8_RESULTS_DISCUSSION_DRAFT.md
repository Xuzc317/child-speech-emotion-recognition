# STAGE8_RESULTS_DISCUSSION_DRAFT

## 1. Purpose and Status

This file is an editable paragraph draft derived from `docs/current/STAGE8_RESULTS_DISCUSSION_OUTLINE.md` and `docs/current/STAGE8_EVIDENCE_INDEX.md`. It is intended to support later manuscript writing, especially Stage 8F. It is not final paper prose, not a final Discussion, and not a final Conclusion.

The draft uses cautious English manuscript language. All numerical claims are taken from `STAGE8_EVIDENCE_INDEX.md`. Claims that still require manual verification before final paper writing are listed explicitly below rather than strengthened in the prose.

One-sentence working argument: In child speech emotion recognition, we show that direct cross-corpus zero-shot transfer remains weak, target-domain fine-tuning recovers performance, and adaptation behavior must be interpreted with dataset-specific metric and design boundaries.

## 2. Results Draft

### 4.1 Dataset Design and In-domain Baselines

We evaluated speech emotion recognition across three corpora selected to form a bounded age and expression-style comparison. C-BESD represents child acted speech, FAU Aibo represents child naturalistic child-robot interaction, and IEMOCAP represents adult acted speech. This design covers three cells of a child/adult by acted/naturalistic matrix, while the adult naturalistic cell remains absent. The corpus design therefore supports a controlled comparative framing, but it does not cover the full SER space and cannot by itself separate age and expression-style effects as independent causal factors.

The in-domain baselines provide the context for interpreting later transfer and adaptation results. They should be read as dataset difficulty anchors rather than as a global ranking of methods. This distinction is especially important for FAU Aibo, where WA and UAR diverge substantially because of class imbalance. For FAU, later sections therefore report UAR alongside WA and avoid using WA alone as evidence of strong generalization.

**Editing notes**:

- Depends on: design matrix, E1 baseline summaries, Table D.
- Keep caveats: missing adult naturalistic cell; FAU requires UAR; in-domain baselines are context, not final method ranking.
- Do not strengthen: do not claim complete age-style causal separation or complete SER coverage.

### 4.2 Cross-corpus Zero-shot Transfer

E3 evaluated cross-corpus zero-shot transfer: models were trained on a source corpus and tested directly on a target corpus without target-domain fine-tuning. Across the evaluated directions, zero-shot transfer remained weak. The best E3 entry in Table A was E3-14 for IEMOCAP to C-BESD, with WA 34.68% and UAR 34.66%. Other best-direction E3 results in Table A were in the approximate WA range of 21-34%.

These results indicate that direct cross-corpus generalization was limited under the current setup. Because E3 was run as single-seed zero-shot transfer, the result should be used as evidence of transfer difficulty rather than as a strong stability claim. The text should also keep E3 clearly separate from E7, which uses target-domain fine-tuning.

**Editing notes**:

- Depends on: Table A, E3 side.
- Keep caveats: E3 is zero-shot; E3 is single seed; E3 is not fine-tune.
- Do not strengthen: do not claim stable multi-seed behavior from E3; do not infer a complete mechanism from E3 alone.

### 4.3 Target Fine-tuning Recovers Cross-corpus Performance

E7 evaluated transfer fine-tuning by loading a source checkpoint and then training on the target-domain training split before testing on the target test split. Compared with E3 zero-shot transfer, E7 frozen fine-tuning produced substantially higher target-domain performance across the paired directions in Table A. For example, C-BESD to FAU increased from 21.91% WA in E3 to 66.82% WA in E7-01, and IEMOCAP to C-BESD increased from 34.68% WA in E3 to 91.17% WA in E7-05.

This comparison supports the result-level claim that target-domain fine-tuning recovers much of the performance lost in direct zero-shot transfer. However, the comparison is between two task settings rather than a single-variable ablation. The source checkpoint mapping for B7 must remain tied to the launch scripts, and the recovery pattern should not be described as proof of a causal transfer mechanism.

**Editing notes**:

- Depends on: Table A and launch-verified B7 checkpoint mapping.
- Keep caveats: E7 is target-domain fine-tune; E7 values are 3-seed mean+-std; E3 vs E7 is a task comparison.
- Do not strengthen: do not say fine-tuning proves why transfer succeeds; do not merge E3 and E7 protocols.

### 4.4 Frozen vs Unfrozen Adaptation

The frozen and unfrozen comparisons suggest that unfreezing WavLM is not uniformly beneficial across datasets. In the in-domain comparison, C-BESD improved from E1-02 to E2-01 by +5.04 pp WA, whereas FAU changed from E1-05 to E2-02 by -0.67 pp WA and -0.19 pp UAR. In transfer fine-tuning, the FAU-target E7-ext comparisons were also negative in WA, with deltas of about -0.67 to -0.88 pp.

These observations support a dataset-dependent interpretation of unfreezing. Under the current settings, unfreezing is associated with gains for C-BESD and IEMOCAP but does not yield stable gains for FAU. The transfer comparison must retain the configuration caveat that B7-ext uses `batch_size=8`, whereas E7 frozen uses `batch_size=16`; therefore, E7-ext versus E7 is not a pure single-variable comparison. FAU-related interpretations must also include UAR.

**Editing notes**:

- Depends on: Table B.
- Keep caveats: B7-ext changes unfreeze and batch size; FAU must be read with UAR; FAU non-gain cause is not established.
- Do not strengthen: do not say unfreezing is universally effective or ineffective; do not assign FAU behavior to one unverified cause.

### 4.5 Ablation: Pooling, Fusion, and Build-up

E6 was designed as a build-up ablation, starting from a minimal Mean+Last baseline and progressively adding or changing modules. In C-BESD, the largest observed step occurred at the pooling change: E6-01 achieved 80.89% WA, whereas E6-03 reached 91.91% WA. This pattern supports the claim that pooling is the clearest C-BESD architectural contribution in the current build-up table.

The LayerFusion evidence should be interpreted more cautiously. Table F shows that C-BESD last-layer and weighted-fusion settings were very close, with 91.91% WA for last and 91.96% WA for weighted. The E5 single-layer scan is seed42 only, and the handbook lacks an independent B4/E5 detail table. Because E6 does not include IEMOCAP, the ablation evidence should be framed as dataset-specific rather than as proof that one architectural configuration is globally optimal.

**Editing notes**:

- Depends on: Table C for E6, Table F for E5.
- Keep caveats: E6 is build-up; E6 excludes IEMOCAP; E5 single-layer scan is seed42 only; handbook lacks independent B4/E5 detail.
- Do not strengthen: do not say LayerFusion is stably best; do not say E6 proves the full stack is optimal.

### 4.6 Metric Caveats and Reliability Boundaries

Several reliability boundaries must be stated alongside the main results. First, FAU Aibo cannot be interpreted with WA alone. Table D shows FAU WA-UAR gaps that are typically about 22-30 pp, indicating that overall accuracy can substantially overstate performance under class imbalance. Second, the current result scope is 210 JSON logs, not the stale `192 files` metadata shown in the handbook header.

The evidence base also contains reporting boundaries that should not be hidden in final paper writing. E5 consists of 18 regular files and 36 single-layer scan files, and JSON configuration fields cannot replace launch-script confirmation for real experimental settings. The Global Leaderboard is a WA-only navigation index and should not enter the main conclusions as evidence of scientific optimality.

**Editing notes**:

- Depends on: Table D, Table F, caveats list in the evidence index.
- Keep caveats: 210 JSON count; `192 files` stale; leaderboard not a conclusion; JSON config fields do not replace launch scripts.
- Do not strengthen: do not treat metadata repair issues as results; do not claim the leaderboard defines the best scientific method.

## 3. Discussion Draft

### 5.1 What Is Robust Across the Evidence

The most robust pattern in the current evidence is the contrast between weak zero-shot transfer and substantial recovery after target-domain fine-tuning. Table A shows low E3 zero-shot performance across directions, with the best zero-shot entry reaching 34.68% WA and 34.66% UAR, while paired E7 fine-tuning results are much higher. This supports a cautious conclusion that target-domain training is critical for recovering cross-corpus SER performance under the evaluated protocols.

A second robust boundary is metric reliability for FAU Aibo. Across multiple FAU settings, WA and UAR diverge enough that WA-only interpretation would be misleading. These two points, fine-tuning recovery and FAU UAR interpretation, are stronger than more mechanism-oriented explanations. They still remain bounded by E3 single-seed scope and by the incomplete 2x2 design, which lacks adult naturalistic speech.

**Editing notes**:

- Depends on: Table A and Table D.
- Keep caveats: E3 single seed; missing adult naturalistic cell; recovery is not causal proof.
- Do not strengthen: do not call these final mechanism proofs or complete age/style conclusions.

### 5.2 Dataset-dependent Adaptation Behavior

The adaptation results suggest that neither unfreezing nor architectural modification should be described as globally beneficial. Unfreezing was associated with gains for C-BESD and IEMOCAP under current settings, but FAU did not show the same pattern in either in-domain or transfer comparisons. Similarly, E6 and E5 indicate that module contributions vary by dataset rather than following a single universal ordering.

This dataset dependence should be interpreted cautiously. The FAU exception should not be attributed to label noise, acoustic noise, or any single dataset property without additional evidence. The B7-ext comparison also changes batch size relative to E7 frozen, so it supports a bounded adaptation claim rather than a pure unfreeze-only conclusion.

**Editing notes**:

- Depends on: Table B, Table C, Table F.
- Keep caveats: B7-ext batch_size differs; FAU-specific cause is not established; E6 excludes IEMOCAP.
- Do not strengthen: do not say unfreeze universally improves SER; do not claim one module is globally best.

### 5.3 Why FAU Aibo Must Be Interpreted Separately

FAU Aibo requires separate interpretation because its WA and UAR are consistently separated. Table D reports WA-UAR gaps that are often about 22-30 pp across in-domain, ablation, and transfer settings. This pattern means that high or stable WA on FAU can coexist with weaker class-balanced recall, so WA alone is insufficient for assessing task difficulty or generalization.

UAR helps expose class-level recall limitations that are hidden by overall accuracy. At the same time, UAR does not by itself diagnose the source of FAU difficulty. The current evidence supports a metric caveat, not a complete causal explanation of why FAU behaves differently from C-BESD or IEMOCAP.

**Editing notes**:

- Depends on: Table D.
- Keep caveats: UAR is required but not a full causal diagnosis; FAU explanations need additional evidence.
- Do not strengthen: do not say high FAU WA indicates strong generalization; do not claim the cause of the WA-UAR gap is proven.

### 5.4 Target-side Constraints as an Interpretation, Not a Causal Law

The E7 frozen same-target comparisons are consistent with a target-side constraint interpretation. In Table E, different source checkpoints fine-tuned to the same target produce clustered target performance, with small within-target ranges in the frozen E7 setting. This pattern suggests that target-domain properties may shape fine-tuned outcomes after adaptation.

This interpretation must remain at Level 3. The evidence does not prove a causal law, and it should not be written as target-domain ceiling dominance. E7-ext should not be mixed into this specific interpretation because it introduces both unfreezing and batch-size differences. Likewise, domain-internal performance levels provide useful context but do not automatically establish the mechanism behind the clustering.

**Editing notes**:

- Depends on: Table E only for the target-side interpretation.
- Keep caveats: Level 3 plausible interpretation; E7 frozen only; E7-ext excluded; no causal-law wording.
- Do not strengthen: do not write that target-domain ceiling dominance is proven or that target-side constraints dominate transfer as a rule.

### 5.5 Architectural Implications

The architecture results point to dataset-specific design implications. For C-BESD, the E6 build-up table makes the pooling change the most visible architectural step, moving from 80.89% WA in the Mean+Last minimal baseline to 91.91% WA after the pooling change. This supports prioritizing pooling behavior when discussing C-BESD architecture effects.

Fusion and adapter claims should remain more conservative. Table F does not support a strong global claim for LayerFusion, and the single-layer E5 scan is limited to seed42. Because E6 excludes IEMOCAP, the architecture discussion should emphasize observed dataset-specific patterns rather than a universal architecture prescription.

**Editing notes**:

- Depends on: Table C and Table F.
- Keep caveats: E6 excludes IEMOCAP; E5 single-layer seed42 only; handbook lacks independent B4/E5 detail table.
- Do not strengthen: do not say LayerFusion is globally optimal, adapters are universally useless, or the full stack is always best.

### 5.6 Limitations and Future Work

The current evidence supports a bounded analysis rather than a complete causal account of child and adult SER transfer. The 2x2 design lacks an adult naturalistic corpus, so the study cannot fully decompose age and expression-style effects. E3 is single seed, E5 single-layer scans are seed42 only, and B7-ext changes batch size relative to E7 frozen. These constraints should remain visible in any manuscript version derived from this draft.

Future work should extend the corpus design and strengthen the statistical basis for transfer claims. Priority directions include adding an adult naturalistic corpus, repeating E3 with multiple seeds, preparing independent E4 and E5 tables, running more complete statistical checks, and analyzing FAU class-level behavior in greater detail. The handbook `192 files` versus 210 JSON boundary and the `regen_handbook.py` reproducibility boundary should also be reconciled before final paper packaging.

**Editing notes**:

- Depends on: design matrix, Table A, Table B, Table F, caveats list.
- Keep caveats: missing adult naturalistic corpus; E3 single seed; E5 seed42 scan; B7-ext batch size; handbook 192/210 boundary.
- Do not strengthen: do not frame future work as optional polish if it is required for stronger causal or stability claims.

## 4. Claims That Need Final Paper Verification

- Confirm final table numbering before inserting manuscript references.
- Recheck every exact number against `docs/current/STAGE8_EVIDENCE_INDEX.md` and, where necessary, raw JSON.
- Confirm whether Table A should use best E3 pooling per direction or matched-pooling E3 rows.
- Reconfirm all E3/E7 pairings before final manuscript writing.
- Reconfirm B7 and B7-ext checkpoint mapping from launch scripts.
- Keep the E7-ext `batch_size=8` versus E7 frozen `batch_size=16` caveat visible.
- Keep the E5 single-layer seed42-only scope visible.
- Create an independent E4 data augmentation table before writing any strong augmentation conclusion.
- Keep the leaderboard outside the main conclusions.
- Do not inherit old conclusions without sentence-by-sentence rechecking.

## 5. Forbidden Overclaims Still Excluded

- Excluded: “目标域天花板主导迁移结果” as a proven causal conclusion.
- Excluded: “leaderboard 第一就是科学最优”.
- Excluded: “LayerFusion 稳定最优”.
- Excluded: “数据增强稳定有效”.
- Excluded: “FAU WA 高说明泛化好”.
- Excluded: “E3/E7 都是 fine-tune”.
- Excluded: “E6 是 remove-down ablation”.
- Excluded: “unfreeze 普遍有效”.

## 6. Recommended Next Step

1. Stage 8E-2: review and selectively commit `docs/current/STAGE8_RESULTS_DISCUSSION_DRAFT.md`.
2. Stage 8F: use this draft to prepare a more formal Results / Discussion manuscript version while preserving caveats.
3. Stage 9: audit old conclusion wording line by line against the clean evidence base.
