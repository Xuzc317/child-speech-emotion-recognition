# STAGE8K_TABLEA_B7_DECISION

## 1. Purpose

This document records the Stage 8K decision on two manuscript-integration issues:

1. How Table A should compare E3 zero-shot transfer with E7 target fine-tuning.
2. How B7 and B7-ext should be described so that the paper does not overstate the unfreezing comparison.

This file is a decision record. It does not modify Stage 8F, Stage 8I, launch scripts, JSON logs, the handbook, or any source code.

## 2. Decision Summary

The two decisions are:

| Topic | Decision | Required Caveat |
|---|---|---|
| Table A main-text policy | Use best observed E3 pooling per transfer direction. | Describe it as best observed single-seed E3 zero-shot configuration, not a matched-pooling-only comparison. |
| Table A appendix policy | Put matched self-attention E3 rows in appendix or sensitivity check. | Use this only to show the direction of the E3-vs-E7 conclusion is robust to matched pooling. |
| B7/B7-ext wording | Describe B7-ext as an unfrozen adaptation counterpart / comparison. | Do not call it a strict single-variable unfreeze ablation because batch size also changes. |

## 3. Table A Current Evidence Status

The current evidence index already uses the best-pooling E3 policy. In `STAGE8_EVIDENCE_INDEX.md`, Table A has the columns `Best E3 Exp` and `Best E3 Pooling`, and the caveat states that the comparison is a task comparison, not a single-variable ablation.

The best E3 rows currently used are:

| Direction | Best E3 row | Best E3 pooling | Purpose in Table A |
|---|---|---|---|
| C-BESD -> FAU | E3-03 | prosody_guided | Best observed zero-shot result for the direction. |
| C-BESD -> IEMOCAP | E3-04 | mean | Best observed zero-shot result for the direction. |
| FAU -> C-BESD | E3-07 | mean | Best observed zero-shot result for the direction. |
| FAU -> IEMOCAP | E3-12 | prosody_guided | Best observed zero-shot result for the direction. |
| IEMOCAP -> C-BESD | E3-14 | self_attention | Best observed zero-shot result for the direction. |
| IEMOCAP -> FAU | E3-17 | self_attention | Best observed zero-shot result for the direction. |

The current Stage 8F and Stage 8I documents correctly leave this as a final manuscript decision. This file resolves that decision for the manuscript integration stage.

## 4. Table A Main-Text Decision

The main-text Table A should use the best observed E3 zero-shot configuration per transfer direction.

Recommended table label:

`Best observed E3 zero-shot configuration per transfer direction versus corresponding E7 target fine-tuning result.`

Reasoning:

- The main scientific purpose is to show the best observed E3 zero-shot performance and compare it with E7 target-domain fine-tuning recovery.
- Using best observed E3 pooling is conservative for the paper's main claim because it gives E3 its strongest observed configuration before comparing it with E7.
- E7 still substantially outperforms E3 under this best observed E3 policy.
- This policy avoids under-reporting E3 by forcing a matched-pooling-only view in the main table.

This is not a matched-pooling-only comparison and must not be described as a strict single-variable ablation.

## 5. Table A Appendix / Sensitivity Recommendation

A matched-pooling appendix or sensitivity check is recommended. The matched-pooling version should use the self-attention E3 rows because E7 uses self-attention during target fine-tuning.

Recommended matched self-attention E3 rows:

| Direction | Matched self-attention E3 row |
|---|---|
| C-BESD -> FAU | E3-02 |
| C-BESD -> IEMOCAP | E3-05 |
| FAU -> C-BESD | E3-08 |
| FAU -> IEMOCAP | E3-11 |
| IEMOCAP -> C-BESD | E3-14 |
| IEMOCAP -> FAU | E3-17 |

Purpose of appendix / sensitivity check:

- Control pooling architecture more strictly.
- Show that the direction of the conclusion remains the same: E3 zero-shot remains much lower than E7 target fine-tuning.
- Avoid using matched pooling to replace the main best-observed zero-shot comparison.

## 6. Safe Table A Caption and Results Wording

Recommended caption sentence:

“Table A compares each transfer direction using the best observed single-seed E3 zero-shot configuration and the corresponding 3-seed E7 target fine-tuning result. This comparison evaluates protocol-level recovery after target-domain fine-tuning rather than a single-variable ablation.”

Recommended Results wording:

“Using the best observed E3 zero-shot configuration for each transfer direction, direct cross-corpus transfer remained weak, whereas the corresponding E7 target-domain fine-tuning results were substantially higher. Because E3 and E7 differ in target-domain training access, this comparison should be interpreted as a protocol-level recovery comparison rather than a single-variable ablation.”

Recommended appendix wording:

“A matched self-attention sensitivity check can be used to confirm that the E3-versus-E7 direction of the result is not driven solely by the best-pooling selection used in the main Table A.”

## 7. E3/E7 Caveats That Must Remain

The manuscript must keep these caveats visible:

- E3 is single-seed zero-shot transfer.
- E7 is target-domain fine-tuning reported as 3-seed mean plus sample standard deviation.
- E3 versus E7 is a protocol-level comparison, not a strict single-variable ablation.
- JSON result fields support reported scores such as `test_wa`, `test_uar`, `seed`, and `best_epoch`.
- Configuration-critical fields such as source checkpoint, batch size, SSL learning rate, and transfer direction must be verified from launch scripts.
- E3 and E7 must not be merged into a single transfer category.

## 8. B7/B7-ext Launch Evidence

Launch-script evidence:

| Script | Evidence |
|---|---|
| `scripts/launch_b7.sh` | B7 / E7 is frozen transfer fine-tuning. It loads a source checkpoint and trains on the target domain. |
| `scripts/launch_b7.sh` | Source checkpoints are provided through `CBESD_CKPT`, `FAU_CKPT`, and `IEMO_CKPT`. |
| `scripts/launch_b7.sh` | Seeds are 42, 123, and 456. |
| `scripts/launch_b7.sh` | Batch size is 16. |
| `scripts/launch_b7.sh` | The script does not pass `--unfreeze_ssl`. |
| `scripts/launch_b7_unfrozen.sh` | B7-ext / E7-07 to E7-12 uses the same source checkpoints and transfer directions as B7 / E7. |
| `scripts/launch_b7_unfrozen.sh` | It passes `--unfreeze_ssl`, `--ssl_lr 1e-5`, and `--lr 3e-4`. |
| `scripts/launch_b7_unfrozen.sh` | Batch size is 8. |
| `scripts/launch_b7_unfrozen.sh` | Seeds are 42, 123, and 456. |

JSON evidence boundary:

- JSON files can support result fields such as `test_wa`, `test_uar`, `best_val_wa`, `best_epoch`, and `seed`.
- JSON files do not contain `load_checkpoint`, `batch_size`, or `ssl_lr` in the sampled E7 and E7-ext records.
- Therefore, source checkpoint mapping and batch-size interpretation must come from launch scripts, not JSON alone.

## 9. B7/B7-ext Decision

B7-ext should be described as an unfrozen adaptation counterpart or unfrozen adaptation comparison.

B7-ext should not be described as a strict single-variable unfreeze ablation.

Reasoning:

- B7-ext uses the same source checkpoints and transfer directions as E7.
- B7-ext enables backbone unfreezing.
- B7-ext also changes batch size from 16 to 8.
- B7-ext uses differential learning rates for the backbone and head.
- Because more than one training-protocol feature differs, the comparison is not strictly unfreeze-only.

## 10. Safe B7/B7-ext Paper Wording

Recommended Methods / Results wording:

“B7-ext follows the same source-checkpoint and target-domain fine-tuning directions as E7, but differs by enabling backbone unfreezing and using a smaller batch size required for unfrozen training; therefore, E7-ext versus E7 should be interpreted as an unfrozen adaptation comparison rather than a strict single-variable ablation.”

Recommended shorter wording:

“E7-ext is the unfrozen counterpart to E7, with the same source-checkpoint directions but a different training protocol that includes backbone unfreezing and a smaller batch size.”

Recommended forbidden replacement:

Do not write: “Unfreezing alone improves performance.”

Write instead: “Unfrozen adaptation improves some target-domain settings under the B7-ext training protocol, but the comparison includes a batch-size caveat.”

## 11. Claims Explicitly Not Supported

The current evidence does not support the following claims:

- E3 versus E7 is a strict single-variable ablation.
- Unfreezing alone improves performance.
- B7-ext proves unfreeze is universally beneficial.
- B7-ext is a pure unfreeze-only experiment.
- Target-side constraint is a causal law.
- Leaderboard rank 1 is the scientifically best model.
- E7-09 is scientifically superior to E2-01 because of leaderboard rank.
- LayerFusion is stably optimal.
- Augmentation is stably effective.
- FAU can be interpreted by WA alone.
- JSON configuration defaults alone prove the true experiment configuration.

## 12. Implications for Stage 8F / Manuscript Integration

This decision affects later manuscript integration as follows:

- Stage 8F should not be directly modified in this stage.
- Later manuscript integration should update Table A wording to explicitly state best observed E3 pooling.
- Later manuscript integration should add a matched self-attention appendix or sensitivity note if space allows.
- Results text should describe E3 versus E7 as protocol-level recovery.
- Methods text should describe B7 and B7-ext source checkpoints using launch-verified wording.
- Results and Discussion should describe E7-ext as an unfrozen adaptation comparison with a batch-size caveat.
- Any claim about unfreezing should remain target-dependent and should not be written as universal.

## 13. What This File Does Not Do

This file does not:

- modify Stage 8F;
- modify Stage 8I;
- modify any launch script;
- modify any JSON result file;
- rerun or regenerate results;
- create a final manuscript section;
- create a new evidence table;
- decide journal-specific word limits;
- replace the need to verify final manuscript numbers against the evidence index.

## 14. Recommended Next Step

Recommended next step: Stage 8M should review and selectively commit `docs/current/STAGE8K_TABLEA_B7_DECISION.md` if it passes structure, safety, and claim-boundary checks.

After Stage 8M, manuscript integration can proceed using these fixed decisions:

1. Main Table A uses best observed E3 pooling per transfer direction.
2. Matched self-attention E3 rows are appendix / sensitivity material.
3. B7-ext is an unfrozen adaptation comparison with a batch-size caveat, not a strict single-variable unfreeze ablation.
