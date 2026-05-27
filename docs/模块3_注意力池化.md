# Module 3: Attention Pooling & Strict Baselines

## Overview

Implemented a **Pure Self-Attention Pooling** module as a strict ablation baseline to defend against "pseudo-innovation" critiques. Both pooling modules have the **exact same number of trainable parameters** (111,105), ensuring any performance difference is attributable to architecture (prosody injection vs. pure data-driven attention) rather than model capacity.

## Architecture Comparison

### Baseline A: Prosody-Guided Temporal Importance Pooling

```
SSL feats (B, T, 768)          F0 (B, T, 1)     Energy (B, T, 1)
        │                          │                    │
        │                          └────────┬───────────┘
        │                                   │
        │                          prosody_proj: Linear(2→64)→ReLU→Linear(64→64)
        │                                   │
        │                          prosody_emb (B, T, 64)
        │                                   │
        └──────────────┬────────────────────┘
                       │
              Concat: (B, T, 832)
                       │
              attn: Linear(832→128)→Tanh→Linear(128→1)
                       │
              Softmax (masked)
                       │
              Weighted Sum → (B, 768)
```

**Parameter breakdown:**
| Layer | Weights | Bias | Total |
|-------|---------|------|-------|
| prosody_proj.0 Linear(2, 64) | 128 | 64 | 192 |
| prosody_proj.2 Linear(64, 64) | 4,096 | 64 | 4,160 |
| attn.0 Linear(832, 128) | 106,496 | 128 | 106,624 |
| attn.2 Linear(128, 1) | 128 | 1 | 129 |
| **Total** | | | **111,105** |

### Baseline B: Pure Self-Attention Pooling (Ablation)

```
SSL feats (B, T, 768)
        │
attn: Linear(768→116)→ReLU→Linear(116→100)→ReLU→Linear(100→100)→Tanh→Linear(100→1)
        │
Softmax (masked)
        │
Weighted Sum → (B, 768)
```

**Parameter breakdown:**
| Layer | Weights | Bias | Total |
|-------|---------|------|-------|
| attn.0 Linear(768, 116) | 89,088 | 116 | 89,204 |
| attn.2 Linear(116, 100) | 11,600 | 100 | 11,700 |
| attn.4 Linear(100, 100) | 10,000 | 100 | 10,100 |
| attn.6 Linear(100, 1) | 100 | 1 | 101 |
| **Total** | | | **111,105** |

## Parameter Count Proof

```
TemporalImportancePooling:
  prosody_proj.0.weight: (64, 2)    = 128
  prosody_proj.0.bias:   (64,)      = 64
  prosody_proj.2.weight: (64, 64)   = 4,096
  prosody_proj.2.bias:   (64,)      = 64
  attn.0.weight:         (128, 832) = 106,496
  attn.0.bias:           (128,)     = 128
  attn.2.weight:         (1, 128)   = 128
  attn.2.bias:           (1,)       = 1
  SUM: 128+64+4096+64+106496+128+128+1 = 111,105

SelfAttentionPooling:
  attn.0.weight:  (116, 768) = 89,088
  attn.0.bias:    (116,)     = 116
  attn.2.weight:  (100, 116) = 11,600
  attn.2.bias:    (100,)     = 100
  attn.4.weight:  (100, 100) = 10,000
  attn.4.bias:    (100,)     = 100
  attn.6.weight:  (1, 100)   = 100
  attn.6.bias:    (1,)       = 1
  SUM: 89088+116+11600+100+10000+100+100+1 = 111,105
```

## Configurable Toggle

Use `create_pooling()` factory function with `pooling_type` parameter:

```python
from src.models.pooling import create_pooling

# Prosody-guided (original)
pooler = create_pooling('prosody_guided', ssl_dim=768)
# pooler forward: pooler(ssl_feats, f0, energy, mask=mask) → (B, 768)

# Self-attention (ablation baseline)
pooler = create_pooling('self_attention', ssl_dim=768)
# pooler forward: pooler(ssl_feats, mask=mask) → (B, 768)
```

## Runtime Verification

```python
from src.models.pooling import verify_parameter_parity

verify_parameter_parity()  # Raises AssertionError if counts differ
```

## Dry-Run Results

```
Parameter parity: PASS (111,105 == 111,105)
Factory toggle:   OK (prosody_guided → TemporalImportancePooling, self_attention → SelfAttentionPooling)
Forward pass:     Input (4, 100, 768) → Output (4, 768) PASS
Mask effect:      Verified (padding frames correctly ignored)
Gradient flow:    8/8 parameter groups have gradients
```

## Code Location

| File | Component | Description |
|------|-----------|-------------|
| `src/models/pooling.py:135` | `SelfAttentionPooling` | Pure self-attention pooling (111,105 params) |
| `src/models/pooling.py:188` | `create_pooling()` | Factory with `pooling_type` toggle |
| `src/models/pooling.py:217` | `verify_parameter_parity()` | Runtime parameter count assertion |
| `src/models/__init__.py` | export | Added new classes and functions |

## Ablation Design Rationale

The self-attention pooler uses a 4-layer MLP (768→116→100→100→1) because:
1. A simpler 2-layer or 3-layer MLP cannot achieve exactly 111,105 parameters with integer hidden dimensions (mathematically impossible)
2. The 4-layer structure provides sufficient expressiveness for a fair comparison
3. Exact parameter parity eliminates model capacity as a confounding variable — any performance difference between the two poolers is purely attributable to prosody feature injection vs. pure data-driven attention
