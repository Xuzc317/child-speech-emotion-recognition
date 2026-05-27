# Module 5: Physical Mechanism Explanation & Visualization (XAI)

## Overview

Built an Explainable AI (XAI) visualization and statistical analysis pipeline to prove that Prosody-Guided Temporal Importance Pooling successfully forces the model to attend to temporal frames with high acoustic density (F0 mutations, RMS peaks).

## Architecture

```
Waveform (T_wav)  +  F0 contour (T_f0)  +  RMS contour (T_rms)  +  Attention weights (T_attn)
        │                    │                      │                        │
        └────────────────────┴──────────────────────┴────────────────────────┘
                                         │
                           Temporal Alignment (1D interpolation)
                           → all signals mapped to T_attn frames
                                         │
                        ┌────────────────┼────────────────┐
                        │                │                │
                   APC_wav (ρ)     APC_delta (ρ)    3-Stack Plot
                   attn vs RMS     attn vs |dF0|    waveform/F0+RMS/attn
                        │                │                │
                        └────────────────┴────────────────┘
                                         │
                              results/xai_sample_visualization.png
```

## Attention-Prosody Correlation (APC) Metrics

### APC_wav — Energy-Guided Attention

Measures whether attention weights track vocal energy (RMS amplitude).

$$\text{APC}_{\text{wav}} = \rho(\mathbf{a}, \mathbf{e})$$

where:
- $\mathbf{a} = [a_1, ..., a_T]$ are the attention weights (softmax-normalized)
- $\mathbf{e} = [e_1, ..., e_T]$ is the RMS energy contour
- $\rho$ is the Pearson correlation coefficient

**Physical interpretation:** High APC_wav indicates the model attends to louder frames — which typically carry higher emotional intensity in children's speech.

### APC_delta — F0 Mutation Tracking

Measures whether attention weights track pitch inflection points.

$$\text{APC}_{\delta} = \rho(\mathbf{a}, |\Delta\mathbf{f}|)$$

where:
- $|\Delta f_t| = |f_t - f_{t-1}|$ for $t \geq 2$, with $|\Delta f_1| = 0$
- $f_t$ is the F0 value at frame t (in Hz)

**Physical interpretation:** High APC_delta indicates the model attends to frames where pitch changes rapidly — F0 inflections are key emotional markers in children's speech (e.g., rising F0 for excitement, falling for sadness).

### Normalization

Both a and the prosody signal are z-score normalized before correlation:

$$\tilde{x} = \frac{x - \mu_x}{\sigma_x}$$

## Temporal Alignment

Different components operate at different temporal resolutions:

| Signal | Source | Frame Rate | Hop Length @ 16kHz |
|--------|--------|------------|---------------------|
| Waveform | Raw WAV | 16,000 Hz | 1 sample |
| F0 / RMS | librosa (hop=320) | 50 Hz | 320 samples |
| Attention | WavLM output | 50 Hz | 320 samples |

Alignment uses `scipy.interpolate.interp1d` with linear interpolation to map all signals to the attention temporal grid (T_attn frames).

## 3-Stack Saliency Map

Single figure with vertically stacked subplots sharing X-axis (Time in seconds):

1. **Top — Waveform**: Raw audio amplitude (grey), shows speech/non-speech regions
2. **Middle — Prosody**: Normalized F0 contour (blue) + RMS energy (orange)
3. **Bottom — Attention**: Attention weights as filled area plot (red), with APC scores annotated

## Code Location

| File | Component | Description |
|------|-----------|-------------|
| `src/evaluation/__init__.py` | export | Module exports |
| `src/evaluation/xai_visualizer.py:24` | `AttentionProsodyExplainer` | Main XAI class |
| `src/evaluation/xai_visualizer.py:55` | `align_all()` | Temporal interpolation |
| `src/evaluation/xai_visualizer.py:74` | `compute_apc()` | APC metric computation |
| `src/evaluation/xai_visualizer.py:124` | `plot_saliency()` | 3-stack matplotlib visualization |

## Usage

```python
from src.evaluation import AttentionProsodyExplainer

explainer = AttentionProsodyExplainer(sr=16000, ssl_frame_rate=50)

# Compute APC scores
apc = explainer.compute_apc(attn_weights, f0_contour, rms_contour)
# → {'apc_wav': 0.623, 'apc_delta': 0.487}

# Generate saliency map
explainer.plot_saliency(
    waveform, f0_contour, rms_contour, attn_weights,
    save_path='results/xai_sample_visualization.png'
)
```

## Dry-Run Results

```
Dummy data: 3-second utterance, 150 attention frames
APC_wav   = 0.2223  (positive: attention tracks energy)
APC_delta = 0.0342  (near-zero: dummy F0 has no correlated inflection pattern)
Output: results/xai_sample_visualization.png (379 KB)
Status: PASS
```

Note: The low APC_delta on dummy data is expected — random F0 has no structured inflection-attention correlation. Real child speech with emotional F0 mutations should show significantly higher APC_delta for the prosody-guided pooler vs. the self-attention baseline.
