# STAGE8F_RESULTS_DISCUSSION_FORMAL_DRAFT

## 1. Purpose and Source Boundary

This document is a formal writing draft for the Results and Discussion sections of the child speech emotion recognition study. It is derived from the clean Stage 8 evidence chain and the Stage 9 old-claim review. It is closer to manuscript prose than the Stage 8D outline and Stage 8E editable draft, but it is still not the final paper text.

The draft uses the following sources as its evidence boundary:

- `docs/current/STAGE8_EVIDENCE_INDEX.md`
- `docs/current/STAGE8_RESULTS_DISCUSSION_OUTLINE.md`
- `docs/current/STAGE8_RESULTS_DISCUSSION_DRAFT.md`
- `docs/current/STAGE9_OLD_CLAIMS_REVIEW.md`
- `README.md`
- `docs/current/AGENT_QA_MEMORY.md`
- `docs/current/CLEAN_PROJECT_STATUS.md`

No old paper draft, old report, validation file, or local operations note is used as a factual source. Legacy claims are treated only as reviewed risks through `STAGE9_OLD_CLAIMS_REVIEW.md`.

The numeric boundary is also fixed. The current clean project contains 210 `results/logs/E*-*.json` files. E5 contains 54 JSON files, including 18 regular seed files and 36 L1-L12 single-layer scan files. The handbook header that reports `192 files` is stale or inconsistent with the current log set. The current `regen_handbook.py` does not fully reproduce the present handbook because it does not generate the B7-ext table. JSON result fields are usable for reported scores, while configuration-critical fields must be verified from launch scripts.

The main writing constraints are:

- E3 is zero-shot transfer and is single seed.
- E7 is target-domain fine-tuning and is summarized as 3-seed mean and sample standard deviation.
- B7-ext uses unfrozen adaptation with `batch_size=8`, whereas E7 frozen uses `batch_size=16`.
- E6 is a build-up ablation and does not include IEMOCAP.
- E5 single-layer scans are seed42 only.
- FAU Aibo must be interpreted with UAR, not WA alone.
- The 2x2 design lacks an adult naturalistic corpus, so age and expression style cannot be fully separated as independent causal factors.
- Target-side constraints are a bounded interpretation, not a demonstrated mechanism.
- Leaderboard rank, LayerFusion, data augmentation, and universal unfreezing are not strong main conclusions.
- Table A still needs a final paper-level decision on whether E3 comparisons should use each direction's best pooling or a strictly matched pooling protocol.

## 2. Results

### 2.1 Overview of Experimental Evidence

The experimental evidence is organized around a three-corpus design and a staged sequence of within-domain, cross-corpus, adaptation, and ablation experiments. The corpora cover child acted speech (C-BESD), child naturalistic child-robot interaction speech (FAU Aibo), and adult acted speech (IEMOCAP). This design supports a bounded comparison across age and expression style, but it does not complete the full age-by-style matrix because the adult naturalistic cell is absent.

The clean result set contains 210 JSON logs. The principal result families include in-domain baselines, zero-shot transfer, data augmentation sensitivity, LayerFusion ablations, frozen versus unfrozen backbone comparisons, build-up ablations, frozen transfer fine-tuning, and unfrozen transfer fine-tuning. Most multi-seed results are reported as mean and sample standard deviation across three model seeds. E3 zero-shot transfer is the main exception and should be described as single-seed evidence.

The Results section should therefore be read in two layers. First, the staged experiments provide the empirical pattern: zero-shot transfer is weak, target-domain fine-tuning substantially recovers performance, unfreezing behaves differently across target datasets, and architectural changes do not have a single global effect. Second, several interpretation boundaries are essential: FAU Aibo requires UAR, E3 is single seed, E5 single-layer scans are seed42 only, and the 2x2 dataset design is incomplete.

### 2.2 In-Domain Baselines

The in-domain baselines establish the reference performance for each corpus before cross-corpus transfer is considered. Under frozen WavLM with self-attention or dataset-appropriate pooling, C-BESD reaches a substantially higher in-domain level than FAU Aibo and IEMOCAP. The C-BESD frozen self-attention baseline E1-02 is reported at 91.87% WA and 91.85% UAR. The FAU self-attention baseline E1-05 reaches 67.05% WA but only 42.92% UAR, giving an early indication that WA alone overstates apparent performance on FAU. The IEMOCAP frozen prosody-guided baseline E1-09 reaches 64.38% WA and 60.37% UAR.

These baselines should be used as empirical reference points rather than as evidence for a complete causal decomposition of age and expression style. C-BESD and IEMOCAP both involve acted speech but differ in age. C-BESD and FAU both involve child speech but differ in expression style and collection setting. Because the adult naturalistic cell is missing, the design supports a bounded contrast, not a full factorial claim.

The baseline results also define the metric boundary for FAU. Across FAU-related entries, the WA-UAR gap is large, commonly around 22 to 30 percentage points. This makes UAR necessary for judging FAU difficulty and model behavior. FAU WA can be reported, but it should not be interpreted alone.

### 2.3 Direct Cross-Corpus Zero-Shot Transfer

E3 evaluates direct cross-corpus zero-shot transfer. In this setting, a model is trained on a source corpus and evaluated directly on a target corpus without target-domain training. The E3 results are single-seed results and should be framed accordingly.

Across the six transfer directions summarized in Table A, zero-shot transfer remains weak. The strongest E3 entry is IEMOCAP to C-BESD, E3-14, with 34.68% WA and 34.66% UAR. Other directions remain in a low range, with WA values approximately between 21% and 34%. For FAU targets, UAR remains especially important: C-BESD to FAU reports 21.91% WA and 23.99% UAR, while IEMOCAP to FAU reports 33.71% WA and 33.00% UAR.

The safe result statement is that direct zero-shot transfer is limited under the evaluated settings. This observation is strong as a pattern across directions, but its numerical precision is bounded by the single-seed nature of E3. The paper should not describe E3 as fine-tuning, and it should not use E3 alone to make a stable ranking claim about source corpora.

A final paper version should also decide how Table A will handle pooling comparability. The current evidence index uses the available E3 best or reported direction entries, while a stricter paper table may need to separate best-pooling comparisons from matched-pooling comparisons.

### 2.4 Target-Domain Fine-Tuning

E7 evaluates transfer followed by target-domain fine-tuning. This is a different task setting from E3: E3 tests zero-shot generalization, whereas E7 loads a source checkpoint and continues training on the target training set before testing on the target corpus. Source checkpoint relationships must be verified from launch scripts rather than inferred from JSON alone.

The contrast between E3 and E7 is large across all six transfer directions in Table A. For C-BESD to FAU, WA increases from 21.91% in E3 to 66.82% in E7, while UAR increases from 23.99% to 41.46%. For IEMOCAP to C-BESD, WA increases from 34.68% to 91.17%, and UAR increases from 34.66% to 91.14%. Similar recovery patterns appear for the IEMOCAP target directions and the other C-BESD and FAU target directions.

The safe result statement is that target-domain fine-tuning substantially recovers performance relative to direct zero-shot transfer. This is one of the strongest empirical findings in the clean evidence set. However, this comparison should be written as a contrast between two transfer protocols rather than as a single-variable ablation, because E3 and E7 differ in whether target-domain training is allowed.

### 2.5 Frozen versus Unfrozen Adaptation

The frozen versus unfrozen evidence combines two related comparisons. B5/E2 compares in-domain frozen and unfrozen training, while B7-ext compares frozen and unfrozen transfer fine-tuning for the same source-target directions. The launch-verified configuration boundary matters here: B7-ext introduces backbone unfreezing with a differential SSL learning rate and uses `batch_size=8`, whereas the frozen E7 transfer uses `batch_size=16`.

In the in-domain comparison, C-BESD benefits from unfreezing: E2-01 improves over E1-02 by +5.04 percentage points WA and +5.03 percentage points UAR. FAU does not show the same pattern: E2-02 is -0.67 percentage points WA and -0.19 percentage points UAR relative to E1-05. IEMOCAP shows a smaller positive WA change, with E2-03 improving over E1-09 by +1.99 percentage points WA and +0.38 percentage points UAR.

The transfer setting shows a similar target-dependent pattern. For C-BESD targets, unfrozen transfer improves over frozen transfer by approximately +5.23 to +5.40 percentage points WA. For IEMOCAP targets, the improvements are approximately +3.59 to +4.38 percentage points WA. For FAU targets, the changes are negative, approximately -0.88 and -0.67 percentage points WA, and the corresponding UAR values must be reported because FAU remains class-imbalanced.

The safe result statement is that unfreezing is target-dependent rather than universally beneficial. The evidence is medium-strong because the direction of the pattern is consistent across in-domain and transfer comparisons, but the B7-ext comparison is not a perfectly isolated single-variable test due to the batch size difference.

### 2.6 Architectural Ablations and Pooling Effects

E6 provides a build-up ablation, not a remove-down ablation. It begins from a minimal Mean plus Last-layer baseline and progressively changes or adds components. The strongest architectural observation appears on C-BESD. C-BESD E6-01 starts at 80.89% WA and 80.85% UAR. Replacing mean pooling with self-attention in E6-03 raises performance to 91.91% WA and 91.86% UAR. This is an approximately +11 percentage point WA change and is the clearest module-related effect in the clean evidence set.

The FAU build-up pattern is different. FAU E6-06 starts at 67.97% WA and 38.44% UAR, while later build-up variants do not produce a comparable WA gain. Some variants raise UAR while lowering WA, again reinforcing that FAU must be read through both metrics. Because E6 does not include IEMOCAP, the architectural finding should be described as dataset-specific and strongest for C-BESD, not as a universal architecture result.

E5 constrains the LayerFusion interpretation. On C-BESD, last-layer and weighted-fusion summaries are very close: 91.91% WA for last-layer and 91.96% WA for weighted fusion. On FAU, last-layer reaches 67.79% WA while weighted fusion reaches 66.41% WA. On IEMOCAP, weighted fusion is modestly higher than last-layer in the regular 3-seed summary. However, the E5 single-layer scans are seed42 only, and the current handbook lacks an independent B4/E5 detail table. The safe statement is therefore that LayerFusion does not support a strong global superiority claim in the current evidence.

### 2.7 FAU Aibo and the WA-UAR Interpretation Boundary

FAU Aibo is the main metric boundary in the study. Because of class imbalance, WA and UAR diverge substantially. Several FAU entries show WA in the mid-to-high 60% range while UAR remains in the low-to-mid 40% range. For example, E1-05 reports 67.05% WA and 42.92% UAR. E7-01 reports 66.82% WA and 41.46% UAR. E7-06 reports 65.97% WA and 43.95% UAR.

This gap changes how FAU results should be written. A high WA on FAU does not by itself indicate strong balanced recognition across emotion classes. UAR should be reported alongside WA whenever FAU appears in a table, a comparison, or a discussion claim. This requirement applies to in-domain baselines, transfer fine-tuning, unfrozen transfer, and any claim involving FAU adaptation.

The safe result statement is that FAU is empirically harder to interpret through WA alone. The paper can report the WA-UAR gap as a reliability boundary and should avoid turning FAU WA into a broad generalization claim.

### 2.8 Leaderboard Results and Their Interpretation Limits

The global leaderboard is useful as an index of high WA entries, but it is not a scientific conclusion by itself. The clean status documents a sorting boundary in the current handbook leaderboard: read-only recomputation from the current JSON set places E7-09 first and E2-01 second. This corrects the rank ordering as a WA list, but it does not make the top-ranked entry a general scientific optimum.

Leaderboard values combine different experimental regimes, including in-domain unfreezing and transfer unfreezing. These regimes differ in training protocol, target-domain exposure, batch size, and transfer setting. Therefore, leaderboard rank should not be used as the main basis for method comparison. The scientific comparisons should instead be made through the paired evidence tables: Table A for E3 versus E7, Table B for frozen versus unfrozen adaptation, Table C for E6 build-up, Table D for FAU WA-UAR gaps, Table E for same-target E7 comparisons, and Table F for E5 fusion summaries.

The safe result statement is that leaderboard ordering can be reported as a bookkeeping note, but it should not enter the main claims.

## 3. Discussion

### 3.1 Main Finding 1: Zero-Shot Transfer Is Weak, Target Fine-Tuning Recovers Performance

The most robust finding is the separation between direct zero-shot transfer and target-domain fine-tuning. Under E3, direct cross-corpus transfer remains weak across all evaluated directions. Under E7, target-domain fine-tuning substantially recovers performance. This pattern is supported by the paired comparisons in Table A and is visible for child-to-child, child-to-adult, adult-to-child, and adult-to-child naturalistic directions.

This finding should be interpreted as evidence that target-domain supervision is critical for recovering performance after cross-corpus mismatch. It should not be written as a mechanism claim about why the recovery occurs. E3 and E7 differ in target-domain training access, so the contrast demonstrates the practical importance of fine-tuning rather than isolating a single architectural or distributional variable.

The strength of this finding is high, with one important caveat: E3 is single seed. The overall magnitude and consistency of the E3-to-E7 recovery supports a strong manuscript claim, but exact E3 numbers should be presented as single-seed results.

### 3.2 Main Finding 2: Backbone Unfreezing Is Target-Dependent

The second main finding is that backbone unfreezing is not uniformly beneficial. C-BESD benefits clearly from unfreezing in both in-domain and transfer settings. IEMOCAP also shows positive gains, especially in the transfer setting. FAU does not show stable improvement and can decline in WA under unfrozen variants, while UAR remains essential for interpretation.

This pattern suggests that the value of updating the SSL backbone depends on the target dataset. It may reflect differences in corpus size, label structure, acoustic regularity, class balance, or mismatch between the pretrained representation and the downstream target. However, those mechanisms are not directly established by the current experiments. The paper should keep the interpretation at the level of target-dependent adaptation behavior.

The strength of this finding is medium-strong. It is supported by both E2 versus E1 and E7-ext versus E7, but the B7-ext comparison contains a batch size caveat: unfrozen transfer used `batch_size=8`, whereas frozen E7 used `batch_size=16`. This prevents an overstrong single-variable statement.

### 3.3 Main Finding 3: Architectural Contributions Are Dataset-Specific

The architectural evidence indicates that component contributions are dataset-specific. The clearest module effect is the C-BESD E6 build-up change from mean pooling to self-attention pooling, which raises performance by approximately 11 percentage points WA. This supports a medium-strength claim that pooling choice is important for C-BESD under the evaluated architecture.

The same conclusion should not be generalized to all datasets. FAU does not show a comparable WA gain across the E6 build-up sequence, and E6 does not include IEMOCAP. E5 also limits any strong LayerFusion conclusion: last-layer and weighted-fusion summaries are close on C-BESD, differ by target dataset, and the single-layer scan is seed42 only.

The safe interpretation is that architectural choices matter, but their effects are conditional on the dataset and metric. Pooling is the strongest architectural factor in the current C-BESD build-up evidence. LayerFusion and single-layer selection are useful ablation dimensions, but they should not be framed as globally superior choices.

### 3.4 Target-Side Constraints as a Bounded Interpretation

E7 frozen same-target comparisons provide a plausible basis for discussing target-side constraints. In Table E, different sources fine-tuned to the same target often cluster near similar target-specific levels. For C-BESD targets, the E7 frozen WA range across sources is 0.39 percentage points. For IEMOCAP targets, the corresponding range is 0.45 percentage points. For FAU targets, WA is also close, although UAR varies more and must remain visible.

This pattern is consistent with the idea that the target dataset places strong constraints on the final fine-tuned performance. However, it is only a bounded interpretation. It does not establish a causal mechanism, and it should not be used as the title-level main finding of the paper. Domain-internal baseline differences can provide context, but they do not by themselves identify the mechanism behind the fine-tuned outcomes.

E7-ext should not be mixed directly into this argument because it changes the adaptation protocol by unfreezing the backbone and changing batch size. The target-side constraint discussion should therefore be placed in Discussion, tied to Table E, and labeled as a plausible interpretation rather than a demonstrated law.

### 3.5 Implications for Child Speech Emotion Recognition

The results have three implications for child speech emotion recognition. First, direct transfer from another corpus is not sufficient under the evaluated setup; target-domain supervision remains central. Second, adaptation strategies should be selected with the target dataset in mind. C-BESD benefits from unfreezing and from the pooling change in the build-up experiment, while FAU requires more cautious interpretation because WA and UAR diverge. Third, evaluation design matters as much as model design: a child SER study that includes FAU must prioritize UAR, and a cross-age or cross-style claim must preserve the missing adult naturalistic cell as a limitation.

These implications support a bounded research narrative. The study can present a three-corpus framework for comparing child acted, child naturalistic, and adult acted speech. It can show that zero-shot transfer is weak and that fine-tuning recovers performance. It can show that unfreezing and architectural changes are not universally beneficial. It should not claim a complete age-by-style causal decomposition or a single best architecture for all target domains.

### 3.6 Limitations

Several limitations must remain explicit in any manuscript version derived from this draft.

First, the dataset design lacks an adult naturalistic corpus. The design therefore supports a bounded comparison across three cells of the age-by-style matrix, not a full factorial analysis.

Second, E3 zero-shot transfer is single seed. Its direction-level scores should be interpreted as protocol evidence rather than stable multi-seed estimates.

Third, E5 single-layer scans are seed42 only, and the current handbook does not include an independent B4/E5 detail table. Layer-level statements should therefore remain cautious.

Fourth, B7-ext differs from E7 not only by backbone unfreezing but also by batch size. This is necessary for feasible unfrozen training but limits the strength of direct single-variable claims.

Fifth, E6 excludes IEMOCAP. The build-up ablation can support C-BESD and FAU observations, but it cannot directly establish the same architectural behavior for IEMOCAP.

Sixth, FAU Aibo is class-imbalanced. Any FAU result must report UAR alongside WA.

Seventh, the handbook contains known consistency boundaries: the header reports `192 files`, whereas the current log count is 210, and the current regeneration script does not fully reproduce the current handbook. These issues do not invalidate the JSON result fields used in the evidence tables, but they must be documented for reproducibility.

### 3.7 What Should Not Be Claimed

The following claims should not be used in the paper without major qualification or additional evidence:

- Do not claim that target-side constraints are an established causal mechanism. They are a plausible interpretation of E7 frozen same-target clustering.
- Do not claim that the leaderboard winner is the scientifically best model. The leaderboard is a WA index across heterogeneous regimes.
- Do not claim that LayerFusion is globally optimal. E5 shows dataset-dependent and often small differences.
- Do not claim that data augmentation is stably effective. The current Stage 8 evidence does not support it as a main conclusion.
- Do not claim that unfreezing is universally effective. The clean evidence supports target-dependent effects.
- Do not interpret FAU performance using WA alone. UAR is required.
- Do not describe E3 as fine-tuning or E7 as zero-shot transfer.
- Do not describe E6 as a remove-down ablation. E6 is build-up.
- Do not present the 192-file handbook header as the current experiment count.
- Do not claim a complete age-by-style causal decomposition because the adult naturalistic corpus is missing.

## 4. Paper-Safe Summary Sentences

The following sentences are safe candidates for later Abstract, Results, Discussion, or Conclusion drafting, subject to final table verification.

1. We evaluate child speech emotion recognition across a bounded three-corpus design spanning child acted, child naturalistic, and adult acted speech, while preserving the missing adult naturalistic condition as a design limitation.
2. Direct cross-corpus zero-shot transfer remains weak under the evaluated protocol, whereas target-domain fine-tuning substantially recovers performance across all evaluated transfer directions.
3. Backbone unfreezing provides target-dependent gains: it improves C-BESD and IEMOCAP targets but does not provide stable improvement for FAU Aibo.
4. FAU Aibo requires UAR-based interpretation because WA and UAR diverge substantially under class imbalance.
5. Architectural ablations indicate that pooling is a major contributor on C-BESD, while LayerFusion and single-layer choices do not support a strong global superiority claim.
6. Same-target E7 clustering supports target-side constraints as a plausible interpretation, but not as a demonstrated causal mechanism.
7. Leaderboard rank should be treated as a bookkeeping summary of WA, not as the basis for scientific conclusions.

## 5. Claims Requiring Final Verification Before Paper Submission

Before this draft is incorporated into a paper, the following checks should be completed.

- Verify every number in the final tables against `STAGE8_EVIDENCE_INDEX.md` and the underlying JSON patterns.
- Decide whether E3 versus E7 comparisons will use best-pooling E3 entries or a stricter matched-pooling protocol.
- Confirm B7 and B7-ext source checkpoint paths from launch scripts for any transfer table or methods description.
- Keep E3 labeled as single seed in all tables and captions.
- Keep E7 and E7-ext labeled as 3-seed mean plus sample standard deviation.
- Keep the B7-ext batch size caveat visible wherever E7-ext is compared with E7.
- Keep E5 single-layer scans labeled as seed42 only.
- Keep E6 labeled as build-up and state that it excludes IEMOCAP.
- Report UAR for every FAU entry, especially when WA appears high.
- Keep the missing adult naturalistic corpus as a design limitation.
- Keep the handbook `192 files` header and 210 JSON count boundary documented in reproducibility notes.
- Do not use old drafts, reports, or reviewed legacy claims as factual evidence.

## 6. Recommended Next Step

The next step should be Stage 8G: review this formal draft and selectively commit only `docs/current/STAGE8F_RESULTS_DISCUSSION_FORMAL_DRAFT.md` if it passes structure, evidence, caveat, and safety checks.

After Stage 8G, a later writing stage can convert this formal draft into manuscript-specific Results and Discussion text. That later stage should still preserve the evidence boundaries in this file and should not remove the FAU UAR caveat, E3 single-seed caveat, E7-ext batch size caveat, E5 seed42-only caveat, E6 scope caveat, or the missing adult naturalistic design limitation.
