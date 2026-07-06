# Distribution-Driven Child Speech Emotion Recognition

This repository contains a child speech emotion recognition (SER) experiment
suite built around three speech corpora, WavLM-based acoustic representations,
and controlled cross-corpus experiment stages.

This README is a clean project entry point. It describes project structure,
experiment design, data locations, trusted sources, and known boundaries. It is
not a paper conclusion and does not replace the raw logs, launch scripts, or the
clean status note in `docs/current/CLEAN_PROJECT_STATUS.md`.

## Project Overview

The project studies child SER under dataset and distribution differences. It
uses three corpora, a WavLM Base backbone, multiple pooling and fusion settings,
and a staged experiment matrix covering in-domain baselines, zero-shot transfer,
augmentation sensitivity, layer-fusion ablation, unfreezing, module ablation,
and transfer fine-tuning.

Current local experiment result logs are stored in `results/logs/`. The current
read-only count of matching result files is 210 JSON files.

## Research Goal

The research goal is to understand how child speech SER behaves across datasets
that differ in speaker age and expression style, and to keep the relationship
between experiment design, actual launch configuration, and recorded metrics
auditable.

At this clean stage, the repository does not contain a re-derived final paper
argument. Future writing should first separate direct facts, aggregated
statistics, interpretations, and hypotheses.

## Current Clean Version Status

- Clean status boundary file: `docs/current/CLEAN_PROJECT_STATUS.md`.
- Original JSON logs are preserved under `results/logs/`.
- Launch scripts are preserved under `scripts/launch_b*.sh`.
- The experiment design document and authoritative data handbook are preserved.
- Known handbook and generator consistency issues are documented rather than
  silently fixed.
- No final paper conclusions should be inferred from this README.

## Datasets

| Dataset | Samples | Classes | Speakers | Age / Group | Style |
|---|---:|---:|---:|---|---|
| C-BESD (MY) | 4,179 | 6 in-domain; 4-class subset for cross-corpus alignment | 70 children | 6-12y | Acted, English + Telugu |
| FAU Aibo | 18,216 | 4 | 51 children | 10-13y | Naturalistic child-robot interaction |
| IEMOCAP | ~9,794 | 4 | 10 adults | Adult control | Acted |

FAU Aibo is class-imbalanced. Any future interpretation involving FAU must
consider UAR, not WA alone.

## 2x2 Design Matrix

| Age / Style | Acted | Naturalistic |
|---|---|---|
| Child | C-BESD | FAU Aibo |
| Adult | IEMOCAP | Not covered by the current three-corpus design |

The empty adult-naturalistic cell is a design boundary, not something to fill in
from inference.

## Model Architecture

The core model family follows this structure:

```text
WavLM Base -> 12-layer LayerFusion -> Pooling -> SEMLP classifier
```

Main components:

- WavLM Base backbone: `microsoft/wavlm-base-sv`.
- LayerFusion: 12 WavLM layers combined by learnable weights or ablated through
  last-layer / single-layer settings.
- Pooling: mean, self-attention, or prosody-guided pooling depending on stage.
- Classifier: SEMLP head.

Configuration details must be confirmed from launch scripts, not inferred from
JSON defaults alone.

## Experiment Stages

| Stage | Series | Purpose | Count |
|---|---|---|---:|
| B1 | E1 | Pooling x dataset in-domain frozen baselines | 27 |
| B2 | E3 | Zero-shot cross-corpus transfer | 18 |
| B3 | E4 | Augmentation sensitivity across C1-C4 | 36 |
| B4 | E5 | LayerFusion ablation and single-layer scans | 54 |
| B5 | E2 | WavLM unfreeze comparison | 9 |
| B6 | E6 | Module ablation | 30 |
| B7 | E7 | Frozen transfer fine-tuning | 18 |
| B7-ext | E7-07 to E7-12 | Unfrozen transfer fine-tuning counterpart | 18 |

Important design boundaries:

- E3 is zero-shot: train on source, test directly on target, with no target
  fine-tuning.
- E7 is fine-tune transfer: load a source checkpoint, train on the target
  training split, and test on the target test split.
- E6 is a build-up ablation from a minimal Mean+Last baseline, not a remove-down
  ablation from a full stack.

## Key Files for New Agents

| Need | File |
|---|---|
| Clean status and known boundaries | `docs/current/CLEAN_PROJECT_STATUS.md` |
| Experiment design intent | `docs/current/实验设计方案_v3_含学习笔记.md` |
| Aggregated handbook values, with caveats | `docs/current/权威数据手册.md` |
| Actual launch configuration | `scripts/launch_b*.sh` |
| Handbook generator | `scripts/regen_handbook.py` |
| Data pipeline documentation | `docs/current/模块1_数据管道.md` |
| WavLM backbone documentation | `docs/current/模块2_WavLM主干网络.md` |
| Pooling documentation | `docs/current/模块3_注意力池化.md` |
| XAI documentation | `docs/current/模块5_可解释性可视化.md` |
| Distribution-shift diagnostics documentation | `docs/current/模块6_分布偏移诊断.md` |

If a local `AGENTS.md` file exists, it may be useful as an agent operation
manual. It is currently not assumed to be a tracked, public-safe source, and its
inclusion in Git requires human review.

## Data and Result Files

| Path | Role |
|---|---|
| `results/logs/` | Raw scalar experiment JSON logs for E1-E7-ext |
| `results/analysis/` | Analysis artifacts such as FD/XAI/layer-weight outputs |
| `paper_draft/figures/` | Figure assets and figure subdirectories |
| `checkpoints/` | Checkpoint index or locally synced weights, depending on local state |
| `scripts/` | Launch, validation, synchronization, and plotting utilities |

Current `results/logs/E*-*.json` distribution:

| Series | Count | Notes |
|---|---:|---|
| E1 | 27 | 9 configs x 3 seeds |
| E2 | 9 | 3 configs x 3 seeds |
| E3 | 18 | single-run zero-shot experiments |
| E4 | 36 | 12 configs x 3 seeds |
| E5 regular seed files | 18 | last/weighted regular seed files |
| E5 single-layer scans | 36 | L1-L12 scans, seed 42 only |
| E5 total | 54 | 18 regular + 36 scan files |
| E6 | 30 | 10 configs x 3 seeds |
| E7-01 to E7-06 | 18 | frozen transfer fine-tuning |
| E7-07 to E7-12 | 18 | unfrozen transfer fine-tuning |
| Total | 210 | current JSON count |

## Trusted Sources

Use sources in this order when reconstructing facts:

1. `scripts/launch_b*.sh` for actual CLI parameters and checkpoint paths.
2. `results/logs/E*-*.json` for recorded scalar metrics and run metadata.
3. `docs/current/权威数据手册.md` for aggregated statistics, with the caveats
   below.
4. `docs/current/实验设计方案_v3_含学习笔记.md` for design intent.
5. Module documentation for implementation context.

Do not treat old paper drafts, historical memory, or discussion prose as primary
evidence.

## Known Boundaries and Caveats

These boundaries are part of the clean project state and should be checked
before writing summaries, papers, or new agent memory:

- Current `results/logs/E*-*.json` count is 210.
- E5 has 54 JSON files: 18 regular seed files and 36 L1-L12 single-layer scan
  files.
- `docs/current/权威数据手册.md` reports `192 files` in its header; that is stale
  or inconsistent metadata relative to the current log directory.
- The current `scripts/regen_handbook.py` reads JSON logs but cannot fully
  reproduce the current handbook because it does not generate the B7-ext table.
- The current handbook does not include an independent B4/E5 detail table.
- The current handbook Global Leaderboard has a rank 1/2 ordering boundary:
  read-only recomputation from current JSON ranks E7-09 first and E2-01 second.
- JSON result fields such as `exp_name`, `train_data`, `test_data`, `seed`,
  `best_val_wa`, `test_wa`, `test_uar`, and `best_epoch` are usable as recorded
  run facts.
- JSON configuration fields cannot replace launch scripts.
- `data_split_seed`, `batch_size`, `ssl_lr`, and `load_checkpoint` are not
  present in sampled JSON logs and must be checked in launch scripts.
- B7 and B7-ext source checkpoints must be confirmed from
  `scripts/launch_b7.sh` and `scripts/launch_b7_unfrozen.sh`.
- FAU Aibo interpretation must emphasize UAR because of class imbalance.
- The clean project state has not yet re-derived final paper conclusions.

## What This Clean Version Does Not Include

This clean README does not include:

- Final paper conclusions.
- A new Discussion narrative.
- Claims that any method is definitively best.
- Claims that the model proves a causal mechanism.
- Sensitive local operation details, credentials, SSH endpoints, or cloud
  account information.
- Automatic acceptance of existing deletion changes in the dirty Git worktree.
- Any commit, push, or staging decision.

## Suggested Onboarding Order for Future Agents

1. Read this README.
2. Read `docs/current/CLEAN_PROJECT_STATUS.md`.
3. Read `docs/current/实验设计方案_v3_含学习笔记.md` for design intent.
4. Read `scripts/launch_b*.sh` before trusting configuration fields.
5. Read `docs/current/权威数据手册.md` with the metadata and B7-ext caveats in
   mind.
6. Sample a few JSON logs only to understand field structure.
7. Review module documentation if code-level implementation details are needed.
8. Only after those steps, decide whether a result is a direct fact, aggregated
   statistic, interpretation, or hypothesis.

## Git / Reproducibility Notes

- Clean branch target: `clean-agent-ready`.
- Do not stage or commit unrelated dirty-worktree changes without human review.
- Do not automatically include untracked `AGENTS.md`, `cleanup.sh`, or
  `project_lore.md`.
- Do not automatically accept existing deletion entries in `git status`.
- Do not run `scripts/regen_handbook.py` unless explicitly requested; it writes
  the handbook and currently has known reproducibility boundaries.
- Do not use 192-file wording as the current log count.

## Next Steps

Recommended next actions:

1. Human review of this README and `docs/current/CLEAN_PROJECT_STATUS.md`.
2. Decide whether `AGENTS.md` should be sanitized and tracked, kept local only,
   or replaced by a public-safe agent guide.
3. Decide how to handle existing dirty-worktree deletion entries.
4. If committing later, stage only reviewed files rather than using broad add
   commands.
5. In a later stage, re-derive any paper conclusions from the trusted sources
   and boundaries above.
