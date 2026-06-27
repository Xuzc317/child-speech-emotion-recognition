# Manuscript Correction Ledger — v10 LaTeX

> Generated: 2026-06-22 | Source: `paper_draft/current/v10_*.tex` scan

## Classification

| Tag | Meaning | Action |
|-----|---------|--------|
| SAFE-SWAP | Number changes, claim stands | Mechanical replace |
| CLAIM-AFFECTED | Conclusion itself changes | Do NOT touch — flag for narrative revision |

## SAFE-SWAP Entries

| # | File | Old Value | New Value | Context |
|---|------|-----------|-----------|---------|
| 1 | v10_0_Abstract.tex | 92.92% | 91.87% | C-BESD ceiling |
| 2 | v10_0_Abstract.tex | 67.81% | 67.05% | FAU ceiling |
| 3 | v10_4_Experiments.tex | 92.92$\pm$2.15% (E1-02 mean) | 91.87$\pm$1.56% | B1 table, ddof=1 |
| 4 | v10_4_Experiments.tex | 67.81$\pm$2.15% (E1-05 mean) | 67.05$\pm$0.67% | B1 table, ddof=1 |
| 5 | v10_4_Experiments.tex | 92.92% (C-BESD frozen baseline for E2-01) | 91.87% | B5 table frozen ref |
| 6 | v10_4_Experiments.tex | 67.81% (FAU frozen baseline for E2-02) | 67.05% | B5 table frozen ref |
| 7 | v10_5_Analysis.tex | 92.92% (25pp gap ref) | 91.87% | Gap becomes ~24.8pp |
| 8 | v10_5_Analysis.tex | 67.81% (25pp gap ref) | 67.05% | Gap becomes ~24.8pp |
| 9 | v10_6_Conclusion.tex | 92.92% | 91.87% | Ceiling reference |
| 10 | v10_6_Conclusion.tex | 67.81% | 67.05% | Ceiling reference |
| 11 | v10_standalone.tex | 92.92$\pm$2.15% | 91.87$\pm$1.56% | B1 table |
| 12 | v10_standalone.tex | 67.81$\pm$2.15% | 67.05$\pm$0.67% | B1 table |
| 13 | v10_standalone.tex | 92.92% (25pp gap) | 91.87% | Gap context |
| 14 | v10_standalone.tex | 67.81% (25pp gap) | 67.05% | Gap context |

## CLAIM-AFFECTED Entries (DO NOT MECHANICALLY REPLACE)

| # | File | Problematic Value | Evidence | What Changed |
|---|------|-------------------|----------|-------------|
| A | v10_4 | 35.47% as "best zero-shot" | Actual best = 34.68% (E3-14). 35.47% is PHANTOM | Direction + value both wrong |
| B | v10_4, v10_standalone | E3-10 = 35.47% | Log E3-10 = 26.20%. 35.47% not in any log | Table cell wrong |
| C | v10_4 | 76.02$\pm$0.68% (E2-02 unfreeze) | Log E2-02 = 66.37$\pm$0.85%. 76.02% PHANTOM | FAU unfreeze ~10pp lower than claimed |
| D | v10_4, v10_standalone | +8.2pp FAU unfreeze gain | Actual FAU Δ = -0.67pp (flat/negative) | "Largest gain on FAU" is FALSE |
| E | v10_4, v10_standalone | +4.0pp C-BESD unfreeze | Actual Δ = +5.04pp | Understated |
| F | v10_4, v10_standalone | +1.2pp IEMOCAP unfreeze (mixed pooling) | Actual Δ = +1.99pp (prosody→prosody aligned) | Understated, pooling mismatch |
| G | v10_4, v10_standalone | E3-05 = 19.17% (as range minimum) | Log E3-05 = 27.26%. Actual min = 20.51% (E3-01) | Wrong experiment cited |
| H | v10_4, v10_standalone | 41.05% (any reference) | PHANTOM — not in any of 192 logs | Delete entirely |
| I | v10_standalone | 35.47% in zero-shot collapse sentence | Actual zero-shot range: 20.51%–34.68% | Rewrite range claim |

## Summary

- **SAFE-SWAP**: 14 occurrences across 5 files — mechanical replace only
- **CLAIM-AFFECTED**: 9 distinct claim-level errors — flagged for narrative revision, NOT touched mechanically
- **Most dangerous claims**:
  1. "FAU unfreeze +8.2pp" is FALSE (actual: -0.67pp)
  2. "Best zero-shot 35.47%/41.05%" are PHANTOM values
  3. "25pp gap" becomes 24.8pp (SAFE-SWAP, claim unchanged)
