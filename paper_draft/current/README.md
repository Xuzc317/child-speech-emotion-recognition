# Paper Draft — Build Guide

> v9 版本 (ac_suite_2026-06)

## File map

| File | Section |
|------|---------|
| `main.tex` | Master document (compile entry) |
| `0_Abstract.tex` … `6_Conclusion.tex` | Section sources |
| `references.bib` | Bibliography |

## Build PDF

```bash
cd paper_draft/current
latexmk -pdf main.tex
```

## Verify experiment data

```bash
python scripts/verify_all_192.py    # All 192 experiments
```
