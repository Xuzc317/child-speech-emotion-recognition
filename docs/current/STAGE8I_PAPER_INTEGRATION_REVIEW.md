# STAGE8I_PAPER_INTEGRATION_REVIEW

## 1. Purpose

This file records the paper-integration review after Stage 8F. Its role is to convert the Stage 8F read-only audit into a concrete revision plan before any manuscript-level Results or Discussion integration is attempted.

This file is a planning document only. It does not rewrite `docs/current/STAGE8F_RESULTS_DISCUSSION_FORMAL_DRAFT.md`, does not create final manuscript prose, and does not modify any result, launch script, JSON log, handbook, or source file.

The one-sentence integration argument is: the paper can use Stage 8F as the main Results and Discussion basis, because the evidence chain supports weak zero-shot transfer, strong recovery after target fine-tuning, target-dependent unfreezing, FAU UAR-aware interpretation, and dataset-specific architectural effects, while preserving clear limits around target-side interpretation, leaderboard ranking, and the incomplete 2x2 design.

## 2. Current Status After Stage 8F

The current clean branch has completed a traceable writing chain:

- Clean project entrypoint and status documents are already committed.
- `AGENT_QA_MEMORY.md` is available for safe Q&A and onboarding.
- `STAGE8_EVIDENCE_INDEX.md` provides Table A-F and the evidence map.
- `STAGE8_RESULTS_DISCUSSION_OUTLINE.md` provides the structured Results and Discussion outline.
- `STAGE8_RESULTS_DISCUSSION_DRAFT.md` provides editable paragraph drafts.
- `STAGE9_OLD_CLAIMS_REVIEW.md` reviews legacy claims and blocks unsafe old conclusions.
- `STAGE8F_RESULTS_DISCUSSION_FORMAL_DRAFT.md` provides a formal English Results and Discussion draft.

Stage 8F is suitable as the main basis for the later manuscript Results and Discussion integration. It is not yet the manuscript version because it still contains internal project traces and intentionally dense caveats.

## 3. Stage 8F Readiness Assessment

Stage 8F readiness status:

| Check Item | Assessment | Action Before Paper Integration |
|---|---|---|
| Main writing basis | Ready | Use Stage 8F as the primary draft source. |
| Blocking factual issues | None found in Stage 8H | Continue with planned revisions rather than restarting. |
| Sensitive information | None found | Keep manuscript free of local operations details. |
| Old-claim contamination | None found | Continue enforcing Stage 9 exclusions. |
| Caveat coverage | Sufficient, slightly dense | Consolidate caveats in Results and Discussion. |
| Results / Discussion boundary | Mostly clear | Keep observations in Results and explanations in Discussion. |
| Internal process traces | Present | Remove Stage labels, file paths, and project-management wording from manuscript prose. |

The main readiness conclusion is that Stage 8F can be integrated into a paper, but only after removing project-internal language and deciding several evidence presentation details.

## 4. Required Changes Before Paper Integration

Required manuscript-preparation changes:

1. Remove or rewrite Stage numbering, including phrases such as Stage 8, Stage 9, Stage 8F, and Stage 8G.
2. Remove project-document phrases such as clean evidence chain, old-claim review, clean project, and reviewed legacy claims.
3. Remove file-path lists and replace them with manuscript-style references to experiments, tables, and methods.
4. Remove `Recommended Next Step` wording from any manuscript draft.
5. Convert project-document tone into paper-body tone.
6. Reduce caveat density in Results, keeping only protocol-critical caveats near the relevant observation.
7. Move broader interpretation and clustered limitations into Discussion and Limitations.
8. Avoid turning Discussion interpretations into Results claims.
9. Keep all numerical statements tied to Table A-F or the final manuscript tables derived from them.
10. Preserve all launch-verification boundaries for B7 and B7-ext checkpoint descriptions.

These changes are editorial and structural. They do not require changing Stage 8F before a separate manuscript integration pass.

## 5. Recommended Compression Plan for Results

The Results section should be compressed around the evidence ladder: design and baselines, zero-shot transfer, target fine-tuning, frozen versus unfrozen adaptation, architectural ablation, and metric reliability.

Recommended compression:

| Stage 8F Section | Compression Action | Reason |
|---|---|---|
| 2.1 Overview of Experimental Evidence | Compress substantially | Keep only the three-corpus design, 210-log scope, and main experiment families. |
| 2.2 In-Domain Baselines | Keep but tighten | Report baseline anchors and FAU UAR boundary without over-explaining design logic. |
| 2.3 Direct Cross-Corpus Zero-Shot Transfer | Keep | This supports the first main result; retain E3 single-seed caveat. |
| 2.4 Target-Domain Fine-Tuning | Keep | This supports the strongest paired result with Table A. |
| 2.5 Frozen versus Unfrozen Adaptation | Keep but caveat cleanly | Retain target-dependent pattern and B7-ext batch-size boundary. |
| 2.6 Architectural Ablations and Pooling Effects | Keep but shorten | Focus on C-BESD pooling effect and E5 caution. |
| 2.7 FAU Aibo and WA-UAR Boundary | Keep | This is a strong metric boundary and should remain visible. |
| 2.8 Leaderboard Results | Compress heavily or move to note / supplement | Leaderboard is bookkeeping only, not a main scientific result. |

Results should primarily report observations: values, deltas, table references, and experimental conditions. Avoid mechanistic language such as may reflect, likely due to, or causal mechanism unless clearly moved to Discussion.

Protocol caveats to keep in Results:

- E3 is single seed.
- E7 is target fine-tuning, not zero-shot.
- B7-ext changes unfreezing and batch size relative to E7.
- FAU entries require UAR.
- E6 is build-up and excludes IEMOCAP.
- E5 single-layer scan is seed42 only if single-layer results are mentioned.

## 6. Recommended Compression Plan for Discussion

The Discussion should interpret the results without repeating every table. Its structure should move from strongest supported claims to bounded interpretation and limitations.

Recommended compression:

1. Lead with the robust contrast: direct zero-shot transfer is weak, while target-domain fine-tuning recovers performance.
2. Treat unfreezing as target-dependent, not universally beneficial.
3. Recast Section 3.5 from repeated results into higher-level implications for child SER evaluation and adaptation strategy.
4. Consolidate caveats into a focused Limitations paragraph or subsection.
5. Discuss target-side constraints only as a bounded interpretation supported by E7 frozen same-target clustering.
6. Do not treat FAU difficulty causes as established. UAR shows the metric and class-balance boundary, not the cause.
7. Avoid making LayerFusion, augmentation, or leaderboard rank part of the main Discussion claims.

Discussion should answer what the evidence means and where it stops. It should not retell every Results number.

## 7. Table Reference and Evidence Alignment Plan

Table alignment plan:

| Table | Manuscript Role | Required Alignment |
|---|---|---|
| Table A: E3 vs E7 | Main evidence for zero-shot weak and fine-tune recovery | Decide whether E3 uses best pooling or matched pooling before final manuscript integration. Label E3 as single seed and E7 as 3-seed mean plus sample standard deviation. |
| Table B: Frozen vs unfrozen adaptation | Main evidence for target-dependent unfreezing | State that B7-ext changes both unfreezing and `batch_size=8`, while E7 frozen uses `batch_size=16`. Do not present it as a pure single-variable test. |
| Table C: E6 build-up | Main evidence for C-BESD pooling effect | State that E6 is build-up from Mean + Last and does not include IEMOCAP. |
| Table D: FAU WA-UAR gap | Main evidence for FAU metric boundary | Always present UAR with WA for FAU. Use this table to block WA-only interpretations. |
| Table E: E7 same-target source comparison | Discussion evidence for target-side constraints | Use only for bounded interpretation. Do not mix E7-ext into this argument. |
| Table F: E5 LayerFusion summary | Constraint on LayerFusion claims | Do not claim stable LayerFusion superiority. Label single-layer scans as seed42 only. |

All manuscript paragraphs should map to at least one of these tables or to the design matrix. If a paragraph cannot be mapped, it should be removed, softened, or marked for additional evidence.

## 8. Open Evidence Decisions Before Manuscript Integration

Open decisions before generating a manuscript-integrated version:

1. Table A pooling scope: decide whether E3 rows use best pooling per direction or strictly matched pooling.
2. E4 augmentation table: decide whether to create an independent C1-C4 WA/UAR table before making any augmentation claim.
3. E5 detail: decide whether to add full E5 details or keep LayerFusion as a cautious secondary observation only.
4. FAU class-level evidence: decide whether FAU interpretation needs class-level recall, confusion matrix, or error analysis before discussing causes.
5. B7 and B7-ext checkpoints: reconfirm source checkpoint paths from launch scripts before final Methods and Results text.
6. Leaderboard placement: decide whether leaderboard appears only in supplementary / appendix / project note rather than the main Results.
7. Target journal constraints: decide target venue length and whether Results and Discussion must be combined or separated.

No final manuscript integration should proceed without at least resolving the Table A pooling scope and B7/B7-ext checkpoint verification wording.

## 9. Claims Allowed in the Paper

The following claims are allowed if they remain tied to the evidence tables and caveats:

| Claim | Strength | Evidence Basis | Required Caveat |
|---|---|---|---|
| Direct cross-corpus zero-shot transfer is weak, while target-domain fine-tuning recovers performance. | Strong | Table A | E3 is single seed; E3 vs E7 is a protocol comparison, not a pure ablation. |
| Unfreezing is target-dependent. | Medium-strong | Table B | B7-ext changes both unfreezing and batch size; FAU requires UAR. |
| FAU requires UAR-aware interpretation. | Strong boundary | Table D | WA alone can misrepresent class-balanced difficulty. |
| Pooling has the clearest architectural effect on C-BESD build-up. | Medium | Table C | E6 excludes IEMOCAP; do not globalize to all datasets. |
| The 2x2 design is bounded because the adult naturalistic cell is missing. | Strong design boundary | Design matrix | Do not claim full age-style causal decomposition. |
| Target-side constraints may shape fine-tuned outcomes. | Bounded interpretation | Table E | Level 3 only; do not write as a causal law. |
| LayerFusion effects are dataset-dependent and cautious. | Weak / medium | Table F | Not a main conclusion; single-layer scans are seed42 only. |

These claims should be expressed with calibrated verbs such as show, indicate, suggest, or support depending on the evidence strength.

## 10. Claims That Must Remain Excluded

The following claims must remain excluded from the paper unless new evidence is produced and reviewed:

- Leaderboard rank 1 equals the scientifically best model.
- E7-09 is meaningfully or significantly superior to E2-01 as a scientific conclusion.
- Target-side constraints are a causal law.
- Unfreezing is universally effective.
- LayerFusion is stably or globally optimal.
- Data augmentation is stably effective.
- High FAU WA proves strong generalization.
- FAU non-gain is established to be caused by noise or label contamination.
- The three-dataset design fully proves independent age and expression-style effects.
- 192 JSON files is the current complete result count.
- E3 and E7 are both fine-tuning experiments.
- E7 is zero-shot.
- E6 is remove-down ablation.
- JSON configuration defaults alone are sufficient to define the true experimental configuration.

These exclusions come from the Stage 9 old-claim review and must be enforced during manuscript integration.

## 11. Caveat Placement Plan

Recommended caveat placement:

| Caveat | Best Placement | Rationale |
|---|---|---|
| E3 single seed | Results near Table A and table caption | Needed to interpret E3 numerical stability. |
| E3 best pooling vs matched pooling | Table A caption or Methods note | Must be resolved before final table. |
| B7/B7-ext checkpoint source | Methods and Table A/B notes | Configuration source is launch scripts, not JSON. |
| B7-ext batch_size=8 vs E7 batch_size=16 | Results near Table B and table caption | Prevents pure single-variable claim. |
| FAU UAR requirement | Results and Discussion | Central metric boundary. |
| E6 excludes IEMOCAP | Results near Table C | Limits architecture claim scope. |
| E5 seed42-only single-layer scan | Results near Table F | Limits LayerFusion / layer-selection claims. |
| Missing adult naturalistic cell | Introduction, Results design paragraph, Limitations | Limits age-style causal framing. |
| Handbook 192 vs 210 issue | Reproducibility / data availability note, not main Results | Important for provenance but not a scientific result. |
| Leaderboard boundary | Optional appendix or brief Results note | Prevents rank overinterpretation. |

The manuscript should avoid scattering every caveat in every paragraph. Use table captions and a consolidated Limitations subsection to keep the prose readable.

## 12. Internal Project Traces to Remove

The manuscript-integrated version should remove or translate the following internal traces:

- Stage numbers and labels, including Stage 8F and Stage 9.
- File names and paths, unless they appear in reproducibility materials rather than paper prose.
- Phrases such as clean project, clean evidence chain, old-claim review, legacy claims, and project status.
- Recommended next step language.
- Instructions to future agents or maintainers.
- Git, branch, commit, or worktree references.
- Handbook-generation caveats from the main scientific narrative, except in reproducibility notes.
- Any local operations wording.

The scientific content should remain, but the document-management scaffolding should not appear in the paper body.

## 13. Proposed Stage 8J Scope

Recommended Stage 8J scope: create a manuscript-integration plan or produce a first paper-integrated Results and Discussion version only after confirming the Table A pooling policy.

Two possible Stage 8J routes:

1. Conservative route, recommended: create `STAGE8J_MANUSCRIPT_INTEGRATION_PLAN.md` that resolves table scope, paragraph order, compression targets, and caveat placement before drafting final prose.
2. Direct writing route: generate a paper-integrated Results and Discussion draft using Stage 8F, but only if the user confirms the Table A pooling policy and whether E4/E5 supplemental tables will be added.

The conservative route is safer because the current remaining issues are not prose-only issues; they affect table policy and claim strength.

## 14. What This File Does Not Do

This file does not:

- modify Stage 8F;
- create final paper prose;
- rewrite Results or Discussion;
- update any evidence table;
- verify launch scripts again;
- rerun or regenerate any experiment;
- modify JSON logs, launch scripts, handbook files, source code, README, Memory, or Stage 8/9 documents;
- resolve the Table A pooling policy;
- create or submit a paper-ready manuscript section.

It only records the paper-integration review and the recommended revision plan.
