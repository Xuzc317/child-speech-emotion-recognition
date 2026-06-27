# Rerun Candidates

> Generated: 2026-06-22

If reviewers demand confusion matrices or independent WA/UAR verification, these experiments should be re-run with prediction dumping enabled.

| Priority | Experiment | Config | Reason |
|----------|-----------|--------|--------|
| 1 | E1-02 | C-BESD, frozen, self_attn | C-BESD in-domain ceiling |
| 2 | E1-05 | FAU_Aibo, frozen, self_attn | FAU ceiling, severe class imbalance |
| 3 | E6-04 | C-BESD, self_attn, weighted fusion | Best B6 config |
| 4 | E7-03 | FAU->C-BESD transfer | Best transfer result |
| 5 | E7-05 | IEMOCAP->C-BESD transfer | Cross-age transfer |
| 6 | E3-14 | IEMOCAP->C-BESD zero-shot | Best zero-shot |

Each re-run should save `predictions` (list) and `labels` (list) in the JSON log to enable independent WA/UAR recomputation and confusion matrix generation.
