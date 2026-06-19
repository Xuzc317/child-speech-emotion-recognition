# hotfix — repository correction scripts

Generated fixes for the SER repo. **Run from the repo root.**

## Run order

```bash
# 1) text fixes — Chunks A, C, D, E (creates *.bak backups first, then verifies)
bash hotfix/run_text_fixes.sh

# 2) Chunk B — detect the E5-03 / E1-02 duplicate
python hotfix/check_chunk_b.py

# 3) Chunk B — recompute genuine layer-weight entropy (needs torch)
python hotfix/extract_e5_01.py
```

## What each chunk does

- **A** (`docs/current/补充实验方案_v1.md`): line 15 `97.13% → 97.22%`; line 17 `67.15% → 63.98%`.
- **E** (same file): typo `SelfAtten → SelfAttn`. The legitimate word `SelfAttention`
  is masked during substitution so it is **never** corrupted into `SelfAttntion`.
- **C** (`CLAUDE.md`): B6 reframed from "removal" to **cumulative build-up (累加式)**;
  table verbs `去→加` and `换MeanPool→换SelfAttn`; conclusion `SA→MeanPool` → `Mean→SelfAttn`.
- **D** (`CLAUDE.md`): B2 zero-shot range `34-41% → 19.17%-35.47%`.
- **B**: `check_chunk_b.py` reports that `layer_weights_E5-03` and `layer_weights_E1-02`
  are identical — which is **expected**: E5-03 (weighted-fusion baseline) and E1-02
  (default self_attn/frozen/C-BESD model) are the *same trained model* (identical
  `layer_fusion.layer_weights`, identical `val_wa=0.95896`). The real defect is
  `entropy=NaN`, computed on the raw signed logits instead of `softmax(logits)`.
  `extract_e5_01.py` re-reads `layer_fusion.layer_weights` from the **E5-03** and
  **E1-02** checkpoints (`checkpoints/autodl/{b4/E5-03_s42,b1/E1-02_s42}/best_model.pt`),
  recomputes entropy on the softmax distribution (nats), and rewrites both JSONs
  (backed up to `.bak`) preserving the original keys plus a new `softmax_weights`.
  Note: **E5-01/E5-04 are `fusion_mode=last`** (single best layer) and carry no
  learnable fusion weights — there is nothing to extract from them.

## Backups & restore

`run_text_fixes.sh` copies each target to `<file>.bak` **only if no backup exists**
(safe to re-run). Restore with:

```bash
mv "docs/current/补充实验方案_v1.md.bak" "docs/current/补充实验方案_v1.md"
mv CLAUDE.md.bak CLAUDE.md
```

The script prints an `[OK]/[WARN]` report. Any `[WARN]` means a target string was not
found exactly as expected — inspect that file before committing.

## Notes

- Requires GNU `sed`, `grep`, `bash` (Ubuntu/Linux); `python` + `torch` for Chunk B.
- If you see `bad interpreter: ...^M`, strip CRLF: `sed -i 's/\r$//' hotfix/*.sh`.
