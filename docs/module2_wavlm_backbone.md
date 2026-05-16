# Module 2: WavLM Backbone & Layer-wise Analysis

## Overview

Refactored the WavLM feature extraction module to extract **all 12 hidden layers** and introduced a **Learnable Weighted Sum** mechanism (`WavLMLayerFusion`). The model can dynamically learn which transformer layers are most important for children's vs. adult speech emotion recognition.

## Architecture

```
Waveform (B, T_wav)
    │
    ▼
WavLM Base+ (frozen, 94M params, output_hidden_states=True)
    │
    ├── hidden_states[0]  : input embeddings      (B, T, 768)
    ├── hidden_states[1]  : transformer layer  1  (B, T, 768)
    ├── hidden_states[2]  : transformer layer  2  (B, T, 768)
    ├── ...
    └── hidden_states[12] : transformer layer 12  (B, T, 768)
    │
    ▼
WavLMLayerFusion (12 learnable params)
    ├── layer_weights: nn.Parameter(torch.ones(12) / 12)
    ├── F.softmax(weights)
    └── Σ(weights[i] * hidden_states[i+1])  for i in 0..11
    │
    ▼
Fused Features (B, T, 768)
    │
    ▼
Temporal Pooling → Classifier
```

## Code Location

| File | Component | Description |
|------|-----------|-------------|
| `src/models/layer_fusion.py` | `WavLMLayerFusion` | Learnable weighted sum over 12 hidden layers |
| `src/models/ssl_backbone.py` | `SSLBackbone.forward(return_all_layers=True)` | Updated to return all hidden states |
| `src/models/__init__.py` | export | Added WavLMLayerFusion |

## WavLMLayerFusion

- **Parameters**: 12 learnable floats, initialized to 1/12 each
- **Softmax**: Ensures weights sum to 1
- **Forward**: `hidden_states[1:]` (12 transformer layers) → stacked → weighted sum → (B, T, 768)
- **Gradient isolation**: Only the 12 layer weights are trainable; WavLM backbone is fully frozen

## Dry-Run Results

```
Batch shape: (4, T_wav)
WavLM output: 13 hidden states (1 embedding + 12 layers), each (4, T_frames, 768)
Fused output: (4, 187, 768)
Layer weights (init): 0.0833 each, sum = 1.0
Backbone trainable params: 0
Fusion trainable params: 12
Shape match with last_hidden_state: OK
Gradient isolation: OK (backbone frozen, only layer_weights require grad)
```

## Usage

```python
from src.models import SSLBackbone, WavLMLayerFusion

backbone = SSLBackbone(model_name='wavlm', frozen=True, device='cuda')
fusion = WavLMLayerFusion(num_layers=12)

# Extract all hidden states
last_hidden, all_hidden = backbone(waveforms, return_all_layers=True)
# Fuse layers
fused = fusion(all_hidden)  # (B, T, 768)

# Inspect learned weights
weights = fusion.get_layer_weights()  # (12,) numpy array
```

## Integration with Full Pipeline

The fused output `(B, T, 768)` has the same shape as the previous single-layer output, ensuring seamless integration with downstream Temporal Importance Pooling and SEMLP classifier.

## Backward Compatibility

- `SSLBackbone.forward(waveforms)` without `return_all_layers` returns only `last_hidden_state` as before
- Existing training scripts continue to work unchanged
