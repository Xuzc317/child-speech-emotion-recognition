# STAGE9_OLD_CLAIMS_REVIEW

## 1. Purpose

This file records the Stage 9 review of legacy claims, old-agent conclusions,
old paper drafts, old reports, and validation-era statements.

Legacy claims are treated only as text to be reviewed. They are not evidence
sources. The review baseline comes from the clean Stage 8 evidence chain,
especially `docs/current/STAGE8_EVIDENCE_INDEX.md`,
`docs/current/STAGE8_RESULTS_DISCUSSION_OUTLINE.md`, and
`docs/current/STAGE8_RESULTS_DISCUSSION_DRAFT.md`.

The purpose of this file is to prevent Stage 8F writing from reintroducing old
strong claims that are not supported by the clean evidence. This file is not
paper prose, not a final Discussion, and does not modify or clean any legacy
file.

## 2. Review Baseline

| Clean Claim | Strength | Evidence |
| --- | --- | --- |
| Zero-shot weak, target fine-tune recovers | Strong | `STAGE8_EVIDENCE_INDEX.md` Table A |
| Unfreeze target-dependent | Medium-strong | Table B with batch_size caveat |
| FAU must be interpreted by UAR | Strong boundary | Table D |
| Pooling dominates C-BESD build-up | Medium | Table C; E6 excludes IEMOCAP |
| 2x2 design is bounded | Strong boundary | README / Memory / outline |
| Target-side constraint | Level 3 interpretation | Table E; not causal law |
| LayerFusion / augmentation / leaderboard | Not strong main conclusions | Table F / E4 not yet indexed / caveats |

## 3. Scanned Legacy Sources

| Path | Status | Risk Summary |
| --- | --- | --- |
| `儿童语音情感识别顶会论文重塑报告.md` | exists | Contains “核心发现”, strong mechanism claims, old paper reshaping suggestions |
| `docs/current/AI项目理解提示词.md` | exists | Contains 192-entry and old AI entrypoint framing |
| `docs/current/实验方案与数据_总表.md` | exists | Contains target-ceiling framing, E7-09 strong interpretation, 192 framing |
| `docs/current/实验方案与数据_报告_给老师.md` | exists | Contains target-ceiling and noise-dominance style strong explanations |
| `paper_draft/current` | 28 files | Current paper drafts; require sentence-level review |
| `paper_draft/archive` | 4 files | Historical drafts; cannot be reused directly |
| `docs/archive` | 40 files | Mixed old design, handbook, technical docs |
| `validation` | 30 files | Mostly based on 192 old scope; historical audit only |

## 4. Clean Conclusions Allowed After Stage 8

- E3 is zero-shot, and direct cross-corpus transfer is weak; must keep the single-seed caveat.
- E7 is target fine-tune transfer, and it substantially outperforms E3; must note task comparison, not single-variable ablation.
- FAU must be reported with both WA and UAR.
- E6 is build-up; remove any mixed remove-down wording.
- Unfreeze is target-dependent, not universally beneficial.
- Target-side constraints may shape E7 frozen outcomes, but only as Level 3 plausible interpretation.

## 5. Old Claims Review Table

| Old Claim / Old Wording | Source File | Risk Type | Current Evidence Status | Decision | Safer Replacement | Evidence Basis | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| “目标域天花板主导迁移结果” | `docs/current/实验方案与数据_总表.md`; `docs/current/实验方案与数据_报告_给老师.md`; `docs/archive/权威数据手册_v1_污染版.md` | 强因果过度 | E7 frozen same-target clustering supports only a Level 3 interpretation | 保留但降级 | E7 frozen suggests target-side constraints may shape fine-tuned outcomes. | Table E | 不作主标题，不写因果律。 |
| “目标域决定迁移表现” | `docs/current/实验方案与数据_总表.md`; `docs/current/实验方案与数据_报告_给老师.md` | 强因果过度 | Evidence supports target-dependent wording, not deterministic mechanism | 改写 | Fine-tuned outcomes appear target-dependent under current evidence. | Table B / Table E | Avoid “decides” or “dominates” wording. |
| “leaderboard 第一就是最优模型” | Legacy leaderboard-style interpretations | leaderboard 误读 | Leaderboard is WA-only and not a scientific conclusion | 删除 | Leaderboard is a navigational / WA-only index. | Caveats | Do not put leaderboard rank in the main conclusion. |
| “E7-09 证明优于 E2-01” | `docs/current/实验方案与数据_总表.md` | leaderboard 误读 | Current JSON rank boundary exists, but it is not proof of scientific superiority | 改写 | E7-09 / E2-01 is a WA ranking boundary, not a main conclusion. | Clean status / evidence caveat | Tiny WA difference and different setting; do not overclaim. |
| “LayerFusion 稳定最优” | Old report and draft-style mechanism claims | LayerFusion 过度 | Table F shows close last/weighted results and seed-scope caveats | 删除 | LayerFusion effects are dataset-dependent and require caution. | Table F | Not a main conclusion. |
| “Weighted fusion 稳定优于 last layer” | `docs/current/实验方案与数据_总表.md`; E5-related legacy text | LayerFusion 过度 | Weighted and last are close; single-layer scan is seed42 only | 改写 | Weighted and last are close under current evidence. | Table F | Needs full E5 detail if emphasized. |
| “数据增强稳定有效” | `docs/archive/实施步骤指南.md`; legacy meeting material | augmentation 过度 | E4 independent C1-C4 table is not yet indexed | 暂不使用 | Augmentation effects require E4 C1-C4 table. | Writability matrix | Do not write a strong augmentation conclusion yet. |
| “FAU WA 高说明泛化强” | Legacy FAU interpretations and old reports | WA-only 误读 | FAU has large WA-UAR gap | 删除 | FAU must be interpreted with UAR. | Table D | WA-only FAU interpretation is prohibited. |
| “E3 和 E7 都是 fine-tune transfer” | Potential legacy confusion | E3/E7 混淆 | E3 is zero-shot; E7 is fine-tune | 删除 | E3 is zero-shot; E7 is target fine-tune. | Design + launch boundary | Keep protocols separated. |
| “E7 是 zero-shot” | Potential legacy confusion | E3/E7 混淆 | E7 loads source checkpoint and trains on target | 删除 | E7 loads source checkpoint and fine-tunes on target. | Launch + Memory | Do not mix with E3. |
| “E6 是 remove-down 消融” | `docs/archive/superpowers/specs/2026-06-10-encyclopedic-experiment-design.md` and mixed legacy wording | E6 误读 | E6 is build-up from Mean+Last | 改写 | E6 is build-up from Mean+Last baseline. | Table C / launch_b6 | Remove “remove-down” wording. |
| “unfreeze 普遍有效” | Legacy unfreeze summaries | unfreeze 过度 | Benefits differ by target/dataset; FAU does not show stable gains | 改写 | Unfreeze is target-dependent. | Table B | Must keep batch_size caveat for E7-ext. |
| “FAU 不收益已证明是噪声 / 标签污染导致” | `docs/current/实验方案与数据_报告_给老师.md`; old mechanism-style text | 强因果过度 | UAR shows metric difficulty, not causal diagnosis | 删除 | FAU behavior requires UAR and further class-level analysis. | Table D | Need class-level recall, confusion matrix, or further analysis. |
| “三数据集完整证明年龄效应和风格效应” | Old paper reshaping report and broad narrative drafts | 设计边界忽略 | 2x2 lacks adult naturalistic cell | 改写 | Three corpora support a bounded age/style contrast. | Design matrix | Do not claim full causal decomposition. |
| “192 JSON 是当前事实” | `docs/current/AI项目理解提示词.md`; `docs/current/实验方案与数据_总表.md`; `validation/*` | 旧口径 / 来源过时 | Current log count is 210 | 改写 | Current logs contain 210 JSON; 192 is stale/partial B1-B7 scope. | Clean status | Validation can remain historical audit only. |

## 6. File and Directory Handling Recommendations

| File or Directory | Observed Risk | Recommended Action | Reason |
| --- | --- | --- | --- |
| `docs/current/实验方案与数据_总表.md` | Old strong conclusions, target-ceiling wording, 192 scope | 改写后可用 | Contains useful tables, but conclusion layer needs Stage 8/9 rewrite. |
| `docs/current/实验方案与数据_报告_给老师.md` | Strong target-ceiling and noise-dominance explanations | 改写后可用 | Teacher-facing format is useful, but claims must be softened. |
| `docs/current/AI项目理解提示词.md` | Old AI entrypoint framing and partial 192 wording | 保留但不作为权威入口 | New agents should use README, clean status, Memory, and Stage 8/9 docs first. |
| `儿童语音情感识别顶会论文重塑报告.md` | Strong mechanism claims and old paper reshaping advice | 移动到 archive 候选 | Historical strategy reference only; not safe as current entrypoint. |
| `paper_draft/current` | Existing paper prose may contain old conclusions | 需要人工审阅 | Any reuse requires sentence-level review against Stage 8/9. |
| `paper_draft/archive` | Historical drafts and outdated claims | 保留但不作为权威入口 | May inspire structure, not evidence or claims. |
| `docs/archive` | Mixed old designs, handbooks, and technical notes | 保留历史参考 | Useful for history, unsafe for current conclusions. |
| `validation` | Mostly 192 old scope and historical audit outputs | 保留但不作为当前结果入口 | Can document prior audits, but current result scope is 210. |

No file should be moved, deleted, archived, or edited during this stage.

## 7. Old Claims That Can Be Kept

- E3 is zero-shot and direct transfer is weak, with the single-seed caveat.
- E7 is fine-tune transfer and outperforms E3, with the task-comparison caveat.
- FAU needs both WA and UAR.
- E6 is build-up, but mixed remove-down wording must be removed.

## 8. Old Claims That Must Be Downgraded

- “目标域天花板主导迁移” -> Level 3 target-side constraint.
- “目标域决定迁移表现” -> target-dependent / target-side constraint.
- “Pooling 是关键贡献” -> only C-BESD, medium strength.
- “unfreeze 带来收益” -> dataset / target-dependent.

## 9. Old Claims That Must Be Deleted or Not Used

- leaderboard 第一就是科学最优。
- E7-09 证明优于 E2-01。
- LayerFusion 稳定最优 / weighted fusion 稳定优于 last。
- 数据增强稳定有效。
- FAU WA 高说明泛化强。
- FAU 不收益已证明由噪声 / 标签污染导致。
- 三数据集完整证明年龄效应和风格效应。
- 192 JSON 是当前全量事实。

## 10. Old Claims That Need More Evidence

- E4 数据增强结论：需要独立 C1-C4 WA/UAR 表。
- LayerFusion 细粒度结论：需要完整 E5 明细和 single-layer seed scope。
- FAU 困难原因：需要类别级召回、混淆矩阵或更细分析。
- B7/B7-ext checkpoint 相关表述：写论文前仍需 launch 再确认。
- Table A E3 best pooling vs matched pooling：最终表格前需定口径。

## 11. Impact on Current Clean Paper Line

The current clean paper line is not materially affected by legacy conclusions.
Stage 8 has already isolated old strong-causal claims, WA-only FAU readings, the
192-scope framing, leaderboard overinterpretation, and LayerFusion overclaims.

Legacy texts remain useful only as materials to be reviewed. They must not be
used as fact sources. Stage 8F writing must cite or trace claims to Stage 8/9
files and trusted project sources, not to old reports or old drafts.

## 12. Rules for Stage 8F Writing

- Do not use old paper drafts as fact sources.
- Do not revive the strong “目标域天花板主导迁移结果” wording.
- Do not put leaderboard rank into the main conclusions.
- Do not use FAU WA-only interpretation.
- Do not write unfreeze, LayerFusion, or augmentation as globally effective.
- Every Results / Discussion paragraph must trace back to `STAGE8_EVIDENCE_INDEX.md`.
- Legacy text may provide expression style or structural inspiration only; conclusion sentences cannot be inherited directly.

## 13. What This File Does Not Do

This file does not:

- modify old files;
- move files into archive;
- delete content;
- generate final paper prose;
- replace `STAGE8_EVIDENCE_INDEX.md`;
- replace final human review before manuscript integration.

## 14. Recommended Next Step

1. Stage 9C: review and selectively commit `docs/current/STAGE9_OLD_CLAIMS_REVIEW.md`.
2. Stage 8F: use the Stage 8E draft plus this Stage 9 old-claim review to prepare a more formal Results / Discussion writing version.
3. Before final manuscript integration, manually check that legacy text has not been reintroduced as evidence or unreviewed conclusion wording.
