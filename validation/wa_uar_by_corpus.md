# WA-UAR by Corpus

> Generated: 2026-06-22 | Source: `scripts/phase4_audit.py` | 192 files

| Corpus | WA mean+-std | UAR mean+-std | WA-UAR gap | N runs |
|--------|-------------|--------------|-----------|--------|
| C-BESD | 81.00+-19.50% | 80.68+-19.67% | 0.31pp | 69 |
| FAU_Aibo | 62.82+-11.59% | 41.67+-5.71% | 21.15pp | 69 |
| IEMOCAP | 59.47+-9.96% | 54.49+-8.61% | 4.98pp | 54 |

## Interpretation

- FAU_Aibo: WA-UAR gap = 21.15pp — severe class imbalance (4 classes, highly skewed)
- C-BESD: WA-UAR gap = 0.31pp — near-perfect class balance (6 classes)
- IEMOCAP: WA-UAR gap = 4.98pp — moderate imbalance

The '30.8pp' figure = maximum individual FAU run gap.
The '9.11pp' figure = global mean across all 192 runs.
