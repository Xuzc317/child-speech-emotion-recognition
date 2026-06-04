# Publication Package for AI Visualization Agents

This directory contains all raw experimental data needed to render
professional paper figures. Designed for agents like PaperVizAgent or K-Dense.

## File Inventory

### 1. xai_raw_data.npz
**Description:** Temporally aligned 1D arrays for XAI saliency visualization.

**Keys and shapes:**
| Key | Shape | Type | Meaning |
|-----|-------|------|---------|
| `waveform` | (T_wav,) | float32 | Raw 16kHz mono audio, peak-normalized to [-1, 1] |
| `f0_contour` | (T_attn,) | float32 | F0 contour (Hz), interpolated to attention frame grid |
| `rms_contour` | (T_attn,) | float32 | RMS energy contour, interpolated to attention frame grid |
| `attention_weights` | (T_attn,) | float32 | Softmax attention weights from prosody-guided pooling |
| `time_wav` | (T_wav,) | float32 | Time axis for waveform (seconds) |
| `time_attn` | (T_attn,) | float32 | Time axis for attention/prosody (seconds) |

**Frame rate:** 50 Hz (WavLM output), hop = 320 samples at 16kHz

**Plotting guide:**
- Top subplot: `time_wav` vs `waveform` (grey line)
- Middle subplot: `time_attn` vs normalized `f0_contour` (blue) + normalized `rms_contour` (orange)
- Bottom subplot: `time_attn` vs min-max scaled `attention_weights` (red area)
- All subplots share X-axis labeled "Time (s)"

### 2. experiment_results.csv
**Description:** Aggregated results from the 6-experiment matrix (AutoDL final, `autodl_final_2026-05-19`).

**Columns:**
| Column | Meaning |
|--------|---------|
| `experiment` | Exp identifier (exp1–exp5b) |
| `train_data` | Training corpus name |
| `test_data` | Test corpus name |
| `pooling_type` | self_attention or prosody_guided |
| `reg_profile` | default (acted) or fau (spontaneous FAU) |
| `test_wa` | Test Weighted Accuracy (0-1) |
| `test_uar` | Test Unweighted Average Recall (0-1) |
| `best_val_wa` | Best validation WA (null if not synced from remote) |
| `best_epoch` | Epoch of best validation WA |
| `speech_type` | acted or spontaneous |
| `matrix_version` | Provenance tag |

### 3. layer_weights.json
**Description:** Learned WavLM layer fusion weights after softmax.

**Keys:**
- `layer_weights`: float array of 12 values (sum = 1.0), index 0 = layer 1, index 11 = layer 12
- `argmax_layer`: 0-indexed layer with maximum weight
- `entropy`: Shannon entropy of weight distribution (higher = more uniform)

### 4. distribution_shift.json
**Description:** FD and SMMD metrics for child-vs-adult distribution shift.

**Keys:**
- `fd`: Frechet Distance (Wasserstein-2 under Gaussian)
- `smmd`: Speech Maximum Mean Discrepancy (non-parametric, RBF kernel)

### 5. Epoch-by-epoch JSON logs
Located in `logs/` subdirectory. Each file contains:
- `exp_name`: experiment identifier
- `pooling_type`: attention mechanism used
- `train_data` / `test_data`: corpus names
- `best_val_wa`: best validation accuracy
- `test_wa` / `test_uar`: final test metrics
- `best_epoch`: best epoch number

## Expected Delta Metrics (for figure captions)

| Comparison | ΔWA | Meaning |
|------------|-----|---------|
| Exp1 - Exp2 | +1.48pp | Self-Attn higher on acted C-BESD (memorization) |
| Exp2 - Exp3 | +32.63pp | Child acted vs adult spontaneous (prosody path) |
| Exp1 in-domain vs Exp4 zero-shot | −74.08pp | Acted→spontaneous cross-domain collapse |
| Exp5 - Exp5b | +0.18pp | Prosody WA lead on FAU (fau reg) |
| Exp5b UAR - Exp5 UAR | +1.88pp | Self-Attn UAR lead on FAU |

## APC Metrics (for XAI figure annotation)

- `apc_wav`: Pearson r(attention, RMS energy) — attention tracks vocal energy
- `apc_delta`: Pearson r(attention, |dF0/dt|) — attention tracks pitch inflections
