# STAGE8O_MANUSCRIPT_RESULTS_DISCUSSION_INTEGRATION

## 1. Purpose and Scope

This document is a manuscript-integration draft for the Results and Discussion sections. It is not the final manuscript. It integrates the clean evidence chain, the paper-integration review, and the Table A / B7 wording decisions into paper-style prose that can be further edited for a target venue.

The sections below are written closer to manuscript form than prior planning documents. The Results draft prioritizes observations, comparisons, and table-linked evidence. The Discussion draft interprets those observations while preserving the main limitations. Internal project-management language should not be carried from this document into a final paper version.

## 2. Integrated Results Draft

### 2.1 Experimental Overview

We evaluated speech emotion recognition across three corpora selected to form a bounded age and speaking-style comparison: C-BESD for child acted speech, FAU Aibo for child naturalistic child-robot interaction speech, and IEMOCAP for adult acted speech. This design covers three cells of an age-by-style matrix and intentionally leaves the adult naturalistic condition absent. The resulting comparison therefore supports bounded cross-corpus analysis rather than a complete factorial decomposition of age and speaking style.

The experiments combined in-domain baselines, direct cross-corpus zero-shot transfer, target-domain fine-tuning, frozen versus unfrozen adaptation, and architecture ablations. Most multi-run experiments were summarized as 3-seed mean and sample standard deviation. E3 zero-shot transfer was a single-seed protocol and is reported accordingly. The main result tables separate protocol-level comparisons from architecture ablations and metric reliability checks.

### 2.2 In-Domain Baselines

The in-domain baselines established the performance range of each corpus before cross-corpus transfer was evaluated. C-BESD reached the highest frozen in-domain baseline among the three corpora, with E1-02 reporting 91.87% WA and 91.85% UAR. IEMOCAP reached a lower but relatively balanced baseline, with E1-09 reporting 64.38% WA and 60.37% UAR. FAU Aibo showed a different metric profile: E1-05 reported 67.05% WA but only 42.92% UAR.

This baseline pattern motivates two constraints on the remaining Results. First, FAU Aibo cannot be interpreted from WA alone because WA and UAR diverge substantially. Second, the in-domain baselines provide reference levels for transfer analysis but do not by themselves prove independent age or speaking-style effects.

### 2.3 Cross-Corpus Zero-Shot Transfer and Target-Domain Fine-Tuning

Direct cross-corpus zero-shot transfer was weak across the evaluated transfer directions. In E3, models were trained on a source corpus and tested directly on a target corpus without target-domain training. Using the best observed single-seed E3 zero-shot configuration per transfer direction, the strongest E3 result was IEMOCAP to C-BESD, with 34.68% WA and 34.66% UAR. Other best-observed E3 directions remained in a low range, approximately 21% to 34% WA.

Target-domain fine-tuning substantially increased performance relative to direct zero-shot transfer. In E7, a source checkpoint was loaded and then fine-tuned on the target-domain training split before evaluation on the target test split. For example, C-BESD to FAU increased from 21.91% WA in E3 to 66.82% WA in E7-01, and IEMOCAP to C-BESD increased from 34.68% WA in E3 to 91.17% WA in E7-05. UAR showed the same broad recovery pattern, although FAU-target results remained lower in UAR than WA.

Table A should be read as a protocol-level comparison. It compares each transfer direction using the best observed single-seed E3 zero-shot configuration and the corresponding 3-seed E7 target fine-tuning result. This comparison evaluates recovery after target-domain fine-tuning rather than a single-variable ablation. A matched self-attention sensitivity analysis can be reported separately using E3-02, E3-05, E3-08, E3-11, E3-14, and E3-17.

### 2.4 Frozen and Unfrozen Transfer Adaptation

Backbone unfreezing showed target-dependent behavior rather than uniform improvement. In the in-domain frozen-versus-unfrozen comparison, C-BESD improved from E1-02 to E2-01 by +5.04 percentage points WA and +5.03 percentage points UAR. IEMOCAP showed a smaller positive change, while FAU did not show a stable gain: E2-02 was lower than E1-05 by -0.67 percentage points WA and -0.19 percentage points UAR.

The transfer adaptation comparison followed a similar target-dependent pattern. For C-BESD targets, E7-ext improved over the corresponding frozen E7 results by approximately +5.23 to +5.40 percentage points WA. For IEMOCAP targets, the gains were approximately +3.59 to +4.38 percentage points WA. For FAU targets, the WA changes were negative, approximately -0.88 and -0.67 percentage points, and UAR remained essential for interpretation.

The E7-ext comparison has an important protocol caveat. B7-ext follows the same source-checkpoint and target-domain fine-tuning directions as E7, but differs by enabling backbone unfreezing and using a smaller batch size required for unfrozen training. E7 frozen uses batch size 16, whereas B7-ext uses batch size 8 and passes `--unfreeze_ssl` with a differential SSL learning rate. Therefore, E7-ext versus E7 should be interpreted as an unfrozen adaptation comparison rather than a strict single-variable ablation.

### 2.5 Architectural Ablations

The build-up ablation identified pooling as the clearest architecture-related change on C-BESD. E6 began from a minimal Mean plus Last-layer baseline. On C-BESD, E6-01 reported 80.89% WA and 80.85% UAR. Replacing mean pooling with self-attention in E6-03 increased performance to 91.91% WA and 91.86% UAR. This was the largest step in the C-BESD build-up sequence.

The FAU build-up sequence did not show the same WA gain. FAU E6-06 started at 67.97% WA and 38.44% UAR, and later build-up variants did not produce a comparable increase in WA. Some variants increased UAR while reducing WA, which further supports reporting both metrics for FAU. E6 did not include IEMOCAP, so this ablation should not be generalized to all three corpora.

LayerFusion results should be interpreted cautiously. E5 showed close last-layer and weighted-fusion results on C-BESD, and dataset-dependent behavior across the other corpora. The single-layer scan was seed42 only. These results constrain broad architecture claims: LayerFusion and single-layer selection can be discussed as ablation evidence, but they do not establish stable global superiority.

### 2.6 FAU Aibo Metric-Sensitivity

FAU Aibo required separate metric treatment because WA and UAR diverged across multiple settings. In-domain, transfer, and ablation results often placed FAU WA in the mid-to-high 60% range while UAR remained in the low-to-mid 40% range. For example, E1-05 reported 67.05% WA and 42.92% UAR, E7-01 reported 66.82% WA and 41.46% UAR, and E7-06 reported 65.97% WA and 43.95% UAR.

This gap prevents WA-only interpretation. FAU results should always report UAR alongside WA, and any comparison involving FAU should avoid treating high WA as evidence of strong balanced recognition across emotion classes.

### 2.7 Leaderboard Interpretation

The leaderboard can be used as a navigational index of high-WA entries, but it should not be treated as a scientific ranking of model quality. The top entries combine different training regimes, including in-domain unfreezing and transfer unfreezing. They also differ in target-domain access, batch size, and training protocol. Therefore, the main paper claims should be based on paired comparisons and ablations rather than leaderboard rank.

In the manuscript, leaderboard information should be reported only as a bookkeeping or supplementary note if needed. It should not be used to claim that one top-ranked entry is scientifically superior to another.

## 3. Integrated Discussion Draft

### 3.1 Zero-Shot Transfer and the Role of Target-Domain Fine-Tuning

The strongest pattern across the evidence is the separation between direct cross-corpus zero-shot transfer and target-domain fine-tuning. Direct transfer remained weak even when each direction used its best observed E3 zero-shot configuration. In contrast, E7 target-domain fine-tuning substantially recovered performance across all paired directions. This supports the practical conclusion that target-domain supervision is critical for cross-corpus SER adaptation under the evaluated protocols.

This finding should not be overinterpreted as a mechanism claim. E3 and E7 differ in target-domain training access, so the comparison demonstrates protocol-level recovery rather than isolating a single architectural variable. The single-seed nature of E3 also limits claims about numerical stability, even though the direction of the E3-versus-E7 contrast is large and consistent.

### 3.2 Dataset-Dependent Effects of Backbone Unfreezing

Backbone unfreezing did not behave as a universal improvement strategy. C-BESD and IEMOCAP benefited under several unfrozen settings, while FAU did not show stable gains and sometimes declined in WA. This pattern indicates that the value of updating the SSL backbone depends on the target dataset and training protocol.

The FAU pattern should remain bounded. The current evidence shows metric divergence and lack of stable unfrozen gains, but it does not identify the cause. Additional class-level recall, confusion matrices, or error analyses would be required before attributing FAU behavior to noise, label uncertainty, or any specific corpus property.

### 3.3 Architecture Contributions and Pooling Effects

The architecture ablations suggest that component contributions are dataset-specific. The clearest contribution was the pooling change in the C-BESD build-up experiment, where self-attention pooling produced the largest observed step from the minimal baseline. This supports a C-BESD-specific architecture claim, not a global claim across all corpora.

The evidence for LayerFusion is weaker. Last-layer and weighted-fusion results were close in some settings, and the single-layer scan was seed42 only. LayerFusion should therefore be discussed as a bounded ablation finding rather than as a stable optimal strategy.

### 3.4 Target-Side Constraints as a Bounded Interpretation

The frozen E7 same-target comparisons are consistent with a target-side constraint interpretation. Different source checkpoints fine-tuned to the same target often converged to similar target-specific levels. This pattern suggests that target-domain properties may shape fine-tuned outcomes after adaptation.

This interpretation must remain bounded. The current evidence does not establish a causal law, and E7-ext should not be mixed into this specific interpretation because it changes both unfreezing and batch size. The target-side interpretation is most appropriate as a Discussion-level explanation tied to the frozen E7 comparisons, not as a main causal conclusion.

### 3.5 Implications for Child Speech Emotion Recognition

The results suggest three implications for child SER studies. First, direct cross-corpus transfer is insufficient under the tested protocol, and target-domain fine-tuning is central to performance recovery. Second, adaptation strategy should be chosen with the target dataset in mind rather than assumed to generalize across corpora. Third, metric choice is part of the experimental design: FAU Aibo requires UAR-aware interpretation because WA alone can obscure class-level difficulty.

The three-corpus design also provides a useful but bounded framework for studying child SER. It contrasts child acted speech, child naturalistic speech, and adult acted speech, but it does not include an adult naturalistic corpus. The results can therefore support bounded age and style comparisons, but not a complete age-by-style causal decomposition.

### 3.6 Limitations

Several limitations should remain explicit in the final manuscript. E3 zero-shot transfer was single seed. Table A uses best observed E3 pooling in the main comparison, and a matched self-attention appendix or sensitivity check should be added if space allows. B7-ext differs from E7 by both backbone unfreezing and batch size, so it is not a strict single-variable unfreezing ablation.

The architecture evidence also has scope limits. E6 did not include IEMOCAP, and E5 single-layer scans were seed42 only. Data augmentation should not be claimed as stable or beneficial without an independent E4 table. FAU difficulty should not be given a causal explanation without class-level evidence. Finally, the missing adult naturalistic corpus prevents a full causal decomposition of age and speaking-style effects.

## 4. Required Table Captions / Notes

### 4.1 Table A Caption Note

Recommended caption note:

“Table A compares each transfer direction using the best observed single-seed E3 zero-shot configuration and the corresponding 3-seed E7 target fine-tuning result. This comparison evaluates protocol-level recovery after target-domain fine-tuning rather than a single-variable ablation.”

Appendix / sensitivity note:

“A matched self-attention sensitivity check can be reported with E3-02, E3-05, E3-08, E3-11, E3-14, and E3-17 to verify that the E3-versus-E7 direction of the result is not driven solely by best-pooling selection.”

### 4.2 Table B Caption Note

Recommended caption note:

“Frozen and unfrozen comparisons are shown for in-domain and transfer settings. B7-ext follows the same source-checkpoint and target-domain fine-tuning directions as E7, but enables backbone unfreezing and uses a smaller batch size required for unfrozen training. Thus, E7-ext versus E7 is an unfrozen adaptation comparison rather than a strict single-variable ablation.”

### 4.3 Table C Caption Note

Recommended caption note:

“E6 is a build-up ablation from a minimal Mean plus Last-layer baseline. It covers C-BESD and FAU Aibo but does not include IEMOCAP, so architecture conclusions should remain dataset-scoped.”

### 4.4 Table D Caption Note

Recommended caption note:

“FAU Aibo results report both WA and UAR because class imbalance creates a persistent WA-UAR gap. FAU should not be interpreted from WA alone.”

### 4.5 Table E Caption Note

Recommended caption note:

“Same-target E7 frozen comparisons are used only to support a bounded target-side constraint interpretation. They do not establish a causal mechanism, and E7-ext is excluded from this interpretation because its training protocol differs from E7.”

### 4.6 Table F Caption Note

Recommended caption note:

“LayerFusion, last-layer, weighted-fusion, and single-layer results are summarized as architecture ablation evidence. Single-layer scans are seed42 only, and the table should not be used to claim stable global LayerFusion superiority.”

## 5. Manuscript-Safe Claim Set

The following claims are safe for manuscript use when paired with the stated caveats:

- Direct cross-corpus zero-shot transfer is weak under the tested protocol.
- Target-domain fine-tuning substantially recovers performance across the evaluated transfer directions.
- Backbone unfreezing shows target-dependent behavior rather than universal benefit.
- FAU Aibo requires UAR-aware interpretation due to persistent WA-UAR divergence.
- Pooling appears to be the dominant build-up change for C-BESD under E6, with dataset-scope limitations.
- The three-corpus design supports bounded age and speaking-style comparisons but not a complete age-by-style causal decomposition.
- Frozen E7 same-target clustering is consistent with target-side constraints as a bounded interpretation.

## 6. Remaining Verification Items Before Final Manuscript

The following items should be resolved before final manuscript submission:

- Generate an independent E4 augmentation table before making any augmentation claim.
- Add or reference complete E5 details before making fine-grained LayerFusion claims.
- Add FAU class recall, confusion matrix, or related evidence before explaining FAU difficulty mechanisms.
- Keep B7 and B7-ext checkpoint wording tied to launch-script evidence.
- Add the matched self-attention Table A appendix or sensitivity check if space allows.
- Verify all final table numbers against the evidence index and raw JSON result fields.
- Keep E3 labeled as single seed and E7/E7-ext labeled as 3-seed summaries.

## 7. What Must Still Be Excluded

The following claims should remain excluded from manuscript prose:

- Leaderboard rank 1 equals the scientifically best model.
- E7-09 significantly proves superiority over E2-01.
- Target-side constraint is a causal law.
- Unfreezing is universally beneficial.
- LayerFusion is stably optimal.
- Augmentation is stably effective.
- FAU WA alone shows strong generalization.
- FAU non-benefit is proven to be caused by noise or label contamination.
- The three datasets fully prove age and speaking-style effects.
- E3, E7, and E6 protocols can be merged or described interchangeably.
- The 192 JSON count is the current complete result fact.
- JSON configuration defaults alone establish true experimental configuration.

## 8. Recommended Next Step

Recommended next step: review and selectively commit this manuscript-integration draft if it passes structure, evidence-boundary, and safety checks. After that, a subsequent manuscript editing stage can adapt this draft to the target venue's length, table format, and section structure.
