# Paper Draft — Build Guide

## File map

| File | Section |
|------|---------|
| `main.tex` | Master document (compile entry) |
| `0_Abstract.tex` … `6_Conclusion.tex` | Section sources |
| `references.bib` | Bibliography |
| `figures/` | Generated figures (`scripts/generate_paper_figures.py`, `generate_architecture_figure.py`) |

## Build PDF (local)

```bash
cd paper_draft
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or with `latexmk`:

```bash
latexmk -pdf main.tex
```

## Regenerate figures & Word

From repo root:

```bash
python scripts/generate_architecture_figure.py
python scripts/generate_paper_figures.py
python scripts/sync_autodl_canonical_logs.py   # if JSON matrix updated
python scripts/verify_experiment_jsons.py
python scripts/generate_docx.py
python scripts/generate_docx_cn.py
```

## INTERSPEECH template

Replace `\documentclass{article}` in `main.tex` with the official INTERSPEECH style file when available; section `\input{}` paths can remain unchanged.

## Known gaps

- `layer_weights.json`: invalid until checkpoint export (`scripts/export_layer_weights.py`)
- AutoDL checkpoints not synced locally — see `docs/autodl_local_inventory.md`
