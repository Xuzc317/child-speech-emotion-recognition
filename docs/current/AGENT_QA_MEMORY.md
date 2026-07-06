# AGENT_QA_MEMORY

## 1. Purpose of This Memory

This memory is a compact Q&A support file for future agents, sub-agents, and
teacher-facing discussions. It is not a README, not a paper Discussion, and not
a final conclusion document.

Use it to:

- orient a new agent quickly;
- prepare safe answers to teacher questions;
- separate facts, aggregated statistics, explanations, and hypotheses;
- avoid repeating old-agent conclusions without rechecking sources;
- prepare Stage 7 Q&A work and Stage 8 conclusion re-derivation.

This file does not contain re-derived final paper conclusions.

## 2. Project One-line Summary

This is a speech emotion recognition experiment project focused on child SER,
cross-corpus design, WavLM-based modeling, and clean handoff for later
source-grounded Q&A and paper-level conclusion re-derivation.

## 3. What a New Agent Must Know First

| Item | Safe Memory |
|---|---|
| Current posture | Do not inherit old-agent conclusions. Re-read trusted files. |
| Current clean branch | `clean-agent-ready-stage6` in the clean worktree. |
| Current log count | 210 `results/logs/E*-*.json` files. |
| Current README role | Clean entrypoint and boundary map, not final paper conclusion. |
| Current status file role | `docs/current/CLEAN_PROJECT_STATUS.md` records caveats and known boundaries. |
| Main risk | Mixing experiment design, metric facts, interpretation, and hypothesis. |
| Strongest guardrail | Check launch scripts for real configuration; JSON config fields are not enough. |

## 4. Core Research Design

The project studies speech emotion recognition across three corpora that differ
by speaker age and expression style.

Core design axes:

- age axis: child vs adult;
- expression style axis: acted vs naturalistic;
- method axis: WavLM Base representations with pooling, layer fusion,
  unfreezing, module build-up, zero-shot transfer, and fine-tune transfer.

Do not treat the design itself as proof of any result. Design intent comes from
`docs/current/实验设计方案_v3_含学习笔记.md`; result summaries come from
`docs/current/权威数据手册.md` and raw JSON logs, with caveats.

## 5. Dataset Cheat Sheet

| Dataset | Age / Group | Style | Classes Used | Source Type |
|---|---|---|---|---|
| C-BESD | Children | Acted | 6 in-domain; 4-class subset for cross-corpus alignment | Child acted corpus |
| FAU Aibo | Children | Naturalistic child-robot interaction | 4-class mapping | Child naturalistic corpus |
| IEMOCAP | Adults | Acted | 4 classes | Adult acted control |

FAU Aibo is class-imbalanced. For FAU-facing answers, mention UAR and avoid WA-only claims.

## 6. 2x2 Design Matrix

| Age / Style | Acted | Naturalistic |
|---|---|---|
| Child | C-BESD | FAU Aibo |
| Adult | IEMOCAP | Not covered |

The adult x naturalistic cell is currently blank. Do not fill it by inference
or use it to claim a verified result.

## 7. Model Architecture Cheat Sheet

Core model family:

```text
WavLM Base -> 12-layer LayerFusion -> Pooling -> SEMLP classifier
```

Components:

- WavLM Base backbone, using WavLM speech representations.
- 12-layer LayerFusion for weighted or ablated layer use.
- Pooling: mean, self-attention, or prosody-guided depending on stage.
- SEMLP classifier head.

Architecture descriptions are design facts. Performance claims require source
checks in handbook or JSON and configuration checks in launch scripts.

## 8. Preprocessing and Split Rules

| Rule | Memory |
|---|---|
| Input | Raw WAV audio. |
| Audio format | 16 kHz mono. |
| Normalization | Peak normalization. |
| Duration | 4 s / about 200 frames after WavLM frame extraction. |
| Split design | Speaker-independent MD5 split. |
| Data split seed | `data_split_seed=42` is fixed. |
| Model seed | `--seed` controls model initialization and training randomness. |
| Seed separation | `data_split_seed` and model seed must remain conceptually separate. |
| Feature extraction | WavLM online extraction, 768-dim frame-level features at about 50 Hz. |

If a question depends on split reproducibility, check launch scripts and data
pipeline docs before answering.

## 9. Experiment Stage Map

| B Stage | E Series | Safe Description |
|---|---|---|
| B1 | E1 | Three datasets x three pooling methods, frozen in-domain baselines. |
| B2 | E3 | Zero-shot cross-corpus transfer: train source, test target directly. |
| B3 | E4 | Data augmentation sensitivity across C1-C4. |
| B4 | E5 | LayerFusion ablation: last layer, weighted sum, and single-layer scans. |
| B5 | E2 | WavLM frozen vs unfreeze comparison. |
| B6 | E6 | Module build-up ablation from minimal Mean+Last baseline. |
| B7 | E7 | Frozen transfer fine-tune: load source checkpoint, train on target split. |
| B7-ext | E7-07 to E7-12 | Unfrozen transfer counterpart to E7. |

Critical distinctions:

- E3 is zero-shot, not fine-tune.
- E7 is fine-tune, not zero-shot.
- E6 is build-up, not remove-down from a full stack.

## 10. Metrics Cheat Sheet

| Metric | Safe Use |
|---|---|
| WA | Weighted accuracy / overall accuracy. Useful but can be misleading under imbalance. |
| UAR | Unweighted average recall. Important for class-imbalanced FAU Aibo. |
| mean+-std | Aggregated across seeds where applicable. |
| std | Handbook uses sample standard deviation, `ddof=1`. |
| Global Leaderboard | WA ranking only; not a scientific conclusion by itself. |

For FAU, never answer using WA alone. Say that UAR is required because class
imbalance makes WA look easier than the task really is.

## 11. Trusted Source Hierarchy

Use this order:

1. `scripts/launch_b*.sh` for actual CLI parameters, checkpoint paths, batch
   size, unfreeze flags, learning rates, and transfer source checkpoints.
2. `results/logs/E*-*.json` for recorded scalar result fields.
3. `docs/current/权威数据手册.md` for aggregated statistics, while remembering
   the 192/210, E5, B7-ext, and leaderboard caveats.
4. `docs/current/实验设计方案_v3_含学习笔记.md` for experiment design intent.
5. `README.md` and `docs/current/CLEAN_PROJECT_STATUS.md` for clean handoff
   boundaries and onboarding order.
6. Module docs for implementation background.

Do not use old paper drafts, old reports, old-agent memory, `project_lore.md`,
or untracked local operation notes as authority.

## 12. Most Important Numbers to Remember

These are memory anchors, not final conclusions.

| Number | Meaning | Source Level |
|---|---|---|
| 210 | Current `results/logs/E*-*.json` count. | Level 1 / clean status |
| 10 | Current `scripts/launch_b*.sh` count in clean worktree. | Level 1 |
| 54 | E5 total JSON count. | Level 1 / clean status |
| 18 + 36 | E5 = 18 regular seed files + 36 L1-L12 single-layer scan files. | Level 1 / clean status |
| `192 files` | Stale or inconsistent handbook metadata, not current log count. | Level 1 boundary |
| `ddof=1` | Handbook mean+-std uses sample standard deviation. | Level 2 |
| E7-09 vs E2-01 | Current JSON recomputation ranks E7-09 first and E2-01 second by WA. | Level 2 boundary |
| FAU UAR emphasis | FAU class imbalance means UAR must be foregrounded. | Level 2 / design caveat |

Avoid turning these anchors into causal claims before Stage 8 re-derivation.

## 13. Known Boundaries and Caveats

- Current log count is 210 JSON files.
- E5 has 54 JSON files: 18 regular seed files and 36 L1-L12 single-layer scan files.
- `docs/current/权威数据手册.md` header says `192 files`; this is stale or inconsistent metadata.
- Current `scripts/regen_handbook.py` cannot fully reproduce the current handbook because it does not generate the B7-ext table.
- Current handbook lacks an independent B4/E5 detail table.
- Global Leaderboard rank 1/2 has a sorting boundary: read-only recomputation from current JSON should rank E7-09 first and E2-01 second.
- JSON result fields are usable as recorded facts.
- JSON config fields cannot replace launch scripts.
- `data_split_seed`, `batch_size`, `ssl_lr`, and `load_checkpoint` are not in sampled JSON and must be checked in launch scripts.
- B7 / E7-ext source checkpoint mapping must be checked in launch scripts.
- Current clean project state has not yet re-derived final paper conclusions.

## 14. Common Misreadings to Avoid

| Unsafe Reading | Correct Guardrail |
|---|---|
| E3 is fine-tune. | E3 is zero-shot. |
| E7 is zero-shot. | E7 is target-domain fine-tune transfer. |
| E6 removes modules from a full stack. | E6 is build-up from a minimal Mean+Last baseline. |
| FAU high WA means strong generalization. | FAU requires UAR due to class imbalance. |
| Leaderboard rank equals scientific best method. | Leaderboard is WA sorting only. |
| JSON config fields prove real configuration. | Launch scripts are required for true CLI configuration. |
| `192 files` is current fact. | Current JSON count is 210; `192 files` is stale/inconsistent metadata. |
| README contains final conclusions. | README is a clean entrypoint and boundary map. |
| Old-agent conclusions can be reused directly. | They must be rechecked from trusted sources. |

## 15. What Must Be Checked in Files Before Answering

Before answering a substantive question, check:

- `README.md` for clean boundary and onboarding framing.
- `docs/current/CLEAN_PROJECT_STATUS.md` for current caveats.
- `docs/current/实验设计方案_v3_含学习笔记.md` for experiment design intent.
- `scripts/launch_b*.sh` for actual CLI configuration.
- `docs/current/权威数据手册.md` for aggregated numbers, with caveats.
- Raw JSON only when the question needs a specific run metric or field.

For B7 or B7-ext source direction, always check launch scripts. For FAU
difficulty, always check UAR context. For E5, remember the missing independent
B4/E5 handbook detail table.

## 16. How to Answer Teacher Questions

Use this structure:

A. What can be stated as fact.

B. Which file or source supports it.

C. A concise answer suitable for the teacher.

D. What wording must be avoided.

E. If the teacher asks follow-up, what source should be checked next.

Keep answers short unless the teacher explicitly asks for detailed evidence.

## 17. Safe Answer Templates

### Template: E3 vs E7

A. Fact: E3 is zero-shot; E7 is fine-tune transfer.

B. Evidence: design document and `scripts/launch_b2.sh` / `scripts/launch_b7.sh`.

C. Answer: "E3 trains on the source corpus and tests directly on the target
corpus without target-domain training. E7 loads a source checkpoint, then trains
on the target training split before testing on the target test split."

D. Avoid: "E3 and E7 are both model transfer fine-tuning."

E. Follow-up: check launch scripts for `--load_checkpoint` in E7 and its absence in E3.

### Template: FAU Metric

A. Fact: FAU Aibo is class-imbalanced, so UAR is essential.

B. Evidence: handbook WA-UAR section and dataset description.

C. Answer: "For FAU, WA alone can overstate performance because class imbalance
is severe. We should report UAR alongside WA, and foreground UAR when discussing
FAU difficulty."

D. Avoid: "FAU works well because WA is high."

E. Follow-up: check the handbook WA-UAR table.

### Template: Leaderboard

A. Fact: Global Leaderboard is WA sorting and has a known rank 1/2 boundary.

B. Evidence: `CLEAN_PROJECT_STATUS.md` and handbook leaderboard section.

C. Answer: "The leaderboard is useful as a WA index, but it should not be read
as a scientific conclusion. There is also a known rank 1/2 sorting boundary:
current JSON recomputation places E7-09 before E2-01."

D. Avoid: "Leaderboard number one is scientifically best."

E. Follow-up: recompute from JSON if a ranking is central to the answer.

### Template: B7 Checkpoint Direction

A. Fact: B7/E7-ext source checkpoint is not recoverable from JSON alone.

B. Evidence: clean status boundary and launch scripts.

C. Answer: "For B7 and B7-ext, JSON records target train/test metadata, but the
source model comes from `--load_checkpoint`. The transfer direction must be
confirmed from `launch_b7.sh` or `launch_b7_unfrozen.sh`."

D. Avoid: "The JSON alone proves the B7 source domain."

E. Follow-up: inspect the exact `--load_checkpoint` mapping.

## 18. Unsafe Claims to Avoid

Do not say:

- "Target-domain ceiling dominates transfer results" as a proved final conclusion.
- "E3 is fine-tune."
- "E7 is zero-shot."
- "E6 is ordinary remove-down ablation."
- "FAU only needs WA."
- "Leaderboard first place is the scientifically best method."
- "192 JSON files is the current project fact."
- "The current README contains final paper conclusions."
- "Old-agent conclusions can be inherited directly."
- "The current handbook is fully reproduced by `regen_handbook.py`."
- "The handbook contains complete independent B4/E5 details."

## 19. Open Questions Not Yet Re-derived

These are not settled in this memory:

- Final paper-level causal or explanatory claims.
- Whether any method should be called "best" under a specific scientific framing.
- How to phrase the final Discussion.
- Whether FD-WA should be the main final thesis after fresh review.
- Whether B7-ext changes the paper's main narrative.
- How to reconcile handbook generation with B7-ext and E5 detail tables.

Mark such issues as `not yet re-derived` until Stage 8.

## 20. File Reading Order for a Fresh Agent

1. `README.md`
2. `docs/current/CLEAN_PROJECT_STATUS.md`
3. `docs/current/AGENT_QA_MEMORY.md`
4. `docs/current/实验设计方案_v3_含学习笔记.md`
5. `scripts/launch_b*.sh`
6. `docs/current/权威数据手册.md`
7. Targeted `results/logs/E*-*.json` samples only when needed
8. Module docs for implementation details

Do not start from old paper drafts or old-agent summaries.

## 21. Stage 7 Usage Instructions

Stage 7 Q&A agents should:

- answer from this memory only as an orientation layer;
- cite the underlying file when the answer matters;
- classify statements as fact, aggregated statistic, reasonable explanation, or hypothesis;
- keep E3/E7/E6 distinctions visible;
- foreground UAR for FAU;
- refuse to present final paper conclusions as already settled;
- check launch scripts before answering configuration questions.

## 22. Stage 8 Preparation Notes

Stage 8 should re-derive paper-level conclusions from trusted sources. It
should not import old-agent conclusions.

Before Stage 8 conclusion writing:

- verify the 210 JSON count;
- verify E5 regular vs single-layer scan split;
- verify B7 and B7-ext checkpoint mappings from launch scripts;
- handle the handbook `192 files` metadata boundary;
- handle the missing independent B4/E5 handbook detail table;
- handle the Global Leaderboard rank 1/2 sorting boundary;
- separate facts, statistics, interpretations, and hypotheses;
- decide what claims are supported strongly enough for paper writing.

