# Legacy Checkpoints (DO NOT USE)

These checkpoints were trained with **data leakage** — the same speaker's
augmented variants appeared in both train and test sets. Their reported
accuracies (~86% for DrseCNN) are inflated and not valid for the
speaker-independent evaluation claimed in the paper.

## To use

Re-train from scratch on the corrected speaker-independent split:
```bash
python src/training/train.py
```
