# Clean Project Status

This file records the clean handoff state of the project after staged review.
It is not a paper conclusion, not a discussion section, and not a replacement
for the raw logs, launch scripts, experiment design, or authoritative handbook.

## What This Clean Version Contains

- Original source code and model/data pipeline files.
- Original experiment logs under `results/logs/`.
- Launch scripts under `scripts/launch_b*.sh`.
- The current `scripts/regen_handbook.py` script.
- The experiment design document:
  `docs/current/实验设计方案_v3_含学习笔记.md`.
- The current authoritative handbook:
  `docs/current/权威数据手册.md`.
- Module documentation and data-processing notes.
- This clean status boundary document.

## What Has Been Removed or Archived

No files were removed, renamed, or archived during this clean-status pass.

Potential old-agent or old-model outputs were only identified for review. Any
movement to `archive/old_agent_outputs/` requires explicit user confirmation.

## Trusted Sources

Use sources in this order when reconstructing facts:

1. `scripts/launch_b*.sh` for actual CLI configuration.
2. `results/logs/E*-*.json` for recorded scalar metrics and run metadata.
3. `docs/current/权威数据手册.md` for aggregated statistics, with the
   consistency issues below kept in view.
4. `docs/current/实验设计方案_v3_含学习笔记.md` for experiment design intent.
5. `AGENTS.md` for project context and operational notes.

Historical summaries, old memory, draft discussion text, and paper narratives
must not be treated as primary evidence.

## Known Data / Handbook Consistency Issues

- Current `results/logs/E*-*.json` count is 210.
- `docs/current/权威数据手册.md` reports `192 files` in its metadata. This is
  stale or inconsistent metadata relative to the current log directory.
- The current `scripts/regen_handbook.py` reads JSON files from
  `results/logs/`, but it cannot fully reproduce the current handbook because
  it does not generate the B7-ext table.
- The current handbook lacks an independent B4/E5 detail table.
- The current handbook Global Leaderboard rank 1/2 order is inconsistent with
  a read-only recomputation from the current JSON files.

## JSON Count and Distribution

Current read-only file count:

| Series | Count | Notes |
|---|---:|---|
| E1 | 27 | 9 configs x 3 seeds |
| E2 | 9 | 3 configs x 3 seeds |
| E3 | 18 | single-run zero-shot experiments |
| E4 | 36 | 12 configs x 3 seeds |
| E5 regular seed files | 18 | last/weighted regular 3-seed runs |
| E5 single-layer scans | 36 | L1-L12 scans, seed 42 only |
| E5 total | 54 | 18 regular + 36 layer-scan files |
| E6 | 30 | 10 configs x 3 seeds |
| E7-01~06 | 18 | frozen fine-tune transfer |
| E7-07~12 / E7-ext | 18 | unfrozen fine-tune transfer |
| Total | 210 | current `results/logs/E*-*.json` count |

## Handbook Metadata Issue: 192 vs 210

`docs/current/权威数据手册.md` currently states that its source is
`results/logs/E*-*.json` with `192 files`. The current directory contains 210
matching JSON files.

This must be treated as a metadata inconsistency until the handbook generator
and handbook content are reconciled. Do not cite `192 files` as the current
project-wide log count.

## `regen_handbook.py` Reproducibility Boundary

The current `scripts/regen_handbook.py` loads all `.json` files in
`results/logs/`, but its emitted metadata still hard-codes `192 files` and
`0/192` prediction files. It also does not generate the B7-ext table found in
the current handbook.

Therefore, the current handbook should not be described as fully reproducible
from the current `regen_handbook.py` without additional reconciliation.

## E5 / B4 Detail Boundary

E5 has 54 current JSON files:

- 18 regular seed files for last/weighted fusion runs.
- 36 single-layer scan files for L1-L12, seed 42 only.

The current handbook does not contain a standalone B4/E5 detail table. E5 values
may appear in the leaderboard, but that is not a complete E5 accounting.

## Leaderboard Sorting Boundary

The current handbook lists E2-01 before E7-09 in the Global Leaderboard, even
though the current JSON values show:

- E7-09: 96.96 +- 0.61% WA.
- E2-01: 96.91 +- 0.19% WA.

Read-only recomputation ranks E7-09 first and E2-01 second. Do not copy the
current handbook leaderboard rank order without noting this sorting boundary.

## JSON vs Launch Configuration Boundary

JSON scalar result fields are usable as recorded run facts:

- `exp_name`
- `train_data`
- `test_data`
- `seed`
- `best_val_wa`
- `test_wa`
- `test_uar`
- `best_epoch`

JSON configuration fields cannot replace launch-script confirmation. In
particular, the following may be absent or default-derived in JSON:

- `augment_condition`
- `fusion_mode`
- `fusion_best_layer`
- `use_adapter`
- `unfreeze_ssl`
- `reg_profile`

The following are not present in the sampled JSON structure and must be checked
from launch scripts:

- `data_split_seed`
- `batch_size`
- `ssl_lr`
- `load_checkpoint`

## B7 / E7-ext Checkpoint Boundary

B7 and B7-ext source domains cannot be independently recovered from JSON logs.
The JSON logs record the target train/test domain, while the source checkpoint
is provided through `--load_checkpoint` in the launch scripts.

Use `scripts/launch_b7.sh` and `scripts/launch_b7_unfrozen.sh` to confirm source
checkpoint paths and transfer directions.

## E3 / E7 Boundary

E3 is zero-shot:

- Train on source.
- Test directly on target.
- No target-domain fine-tuning.

E7 is fine-tune transfer:

- Load a source checkpoint.
- Train/fine-tune on the target training split.
- Test on the target test split.

Do not merge E3 and E7 when describing experiment design or interpreting
numbers.

## E6 Build-up Boundary

E6 is a cumulative build-up ablation. It starts from a minimal Mean+Last
baseline and progressively adds or changes modules. It is not a remove-down
ablation from a full stack.

E6-01 and E6-06 are Mean+Last minimal baselines, not full-stack baselines.

## What Must Not Be Treated as Final Conclusions

Do not treat the following as final conclusions without a fresh, source-grounded
review:

- Old paper discussion or conclusion paragraphs.
- Claims that a method is "best" without source and aggregation scope.
- Claims that the model "proved" a causal mechanism.
- FAU claims based only on WA without UAR.
- B7 source-domain statements derived only from JSON.
- Any leaderboard rank copied without addressing the rank 1/2 sorting issue.
- Any claim that the handbook is fully reproducible from the current
  `regen_handbook.py`.
- Any claim that the current handbook contains complete B4/E5 details.

## Required Checks Before Writing README, Memory, or Paper Conclusions

Before writing a README, project memory, paper results, or paper conclusions:

1. Reconfirm the 210 JSON count.
2. Reconfirm E5 = 18 regular seed files + 36 L1-L12 scan files.
3. Reconfirm B7 and B7-ext checkpoint mappings from launch scripts.
4. Recompute or verify leaderboard ordering from current JSON.
5. Treat handbook `192 files` metadata as stale until fixed.
6. Avoid using JSON default configuration fields as the sole configuration
   source.
7. Keep E3 zero-shot and E7 fine-tune transfer separate.
8. Keep E6 described as build-up.
9. Report FAU with UAR, not WA alone.
10. Separate direct facts, aggregated statistics, interpretations, and
    hypotheses.

## Next Step: README and Git Branch

The next recommended step is to write a clean README and then prepare a Git
branch for the cleaned project state. Do not do this until the user explicitly
requests stage 5.
