# Missing Metrics Report

> Generated: 2026-06-22 | Source: `scripts/check_metrics.py` | 192 files checked

## Summary

- **test_wa**: 192/192 present
- **test_uar**: 192/192 present
- **confusion_matrix**: 0/192 present
- **predictions**: 0/192 present
- **per_class_recall**: 0/192 present

## UAR: CANNOT be recomputed offline

**Reason**: None of confusion_matrix, predictions, or per_class_recall exist in any log file.

All 192 files contain `test_uar` as an aggregate metric computed by the training script, but the raw per-class data needed for independent verification was not saved.

## What exists

- `test_uar` is present in **all 192 files** — computed by `sklearn.recall_score(average='macro')` during training
- `test_wa` is present in all 192 files — computed by `sklearn.accuracy_score` during training
- `best_val_wa` is present in all 192 files

## What's missing (needed for offline UAR recomputation)

| Field | Present | Missing | Required for |
|-------|---------|---------|-------------|
| confusion_matrix | 0 | 192 | offline UAR recomputation |
| predictions | 0 | 192 | offline UAR recomputation |
| per_class_recall | 0 | 192 | offline UAR recomputation |
| per_class_precision | 0 | 192 | offline UAR recomputation |
| per_class_f1 | 0 | 192 | offline UAR recomputation |

## Recommendation

To enable independent UAR verification, future training runs should save:
1. `confusion_matrix` (NxN numpy array or list of lists) — covers all per-class metrics
2. OR `predictions` + `labels` (for recomputation)

For the 192 existing runs, the `test_uar` values are trusted as-is (computed consistently by the same sklearn function). They cannot be independently verified without re-running inference with saved model checkpoints.
