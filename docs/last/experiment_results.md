# Automated Experimentation & Diagnostics Pipeline

## Overview

The automated pipeline runs 4 core ablation experiments and extracts all XAI and distribution shift metrics. All results are logged to `results/logs/` and visualized in `results/`.

## Pipeline Architecture

```
run_experiments.sh
    │
    ├── Exp 1: self_attention baseline (C-BESD)
    │   └── → checkpoints/exp1/best_model.pt
    │       → results/logs/exp1_self_attention.json
    │
    ├── Exp 2: prosody_guided (C-BESD) ★ main result
    │   └── → checkpoints/exp2/best_model.pt
    │       → results/logs/exp2_prosody_guided.json
    │
    ├── Exp 3: adult falsification (IEMOCAP)
    │   └── → checkpoints/exp3/best_model.pt
    │       → results/logs/exp3_adult_iemocap.json
    │
    ├── Exp 4: zero-shot cross-corpus (C-BESD→FAU-Aibo)
    │   └── → checkpoints/exp4/best_model.pt
    │       → results/logs/exp4_zero_shot_fau.json
    │
    └── Diagnostics (on Exp 2 checkpoint)
        ├── results/layer_weights.json
        ├── results/xai_final.png
        ├── results/logs/apc_metrics.json
        └── results/distribution_shift.json
```

## Experiment Matrix

| Exp | Train Data | Test Data | Pooling | Purpose |
|-----|-----------|-----------|---------|---------|
| 1   | C-BESD | C-BESD (hold-out) | self_attention | Baseline: pure data-driven attention |
| 2   | C-BESD | C-BESD (hold-out) | prosody_guided | **Main result**: prosody injection benefit |
| 3   | IEMOCAP | IEMOCAP (hold-out) | prosody_guided | Adult falsification: expect no benefit |
| 4   | C-BESD | FAU-Aibo | prosody_guided | Zero-shot cross-corpus generalization |

## Expected Outcomes

### Exp 1 vs Exp 2 (Ablation)

- **Hypothesis**: Prosody-guided pooling (Exp 2) should outperform self-attention (Exp 1) on C-BESD
- **Why**: Children's emotional prosody (F0 inflections, energy peaks) provides informative temporal prior
- **ΔWA**: Expected +1.5 to +2.5 pp

### Exp 3 (Adult Falsification)

- **Hypothesis**: Prosody-guided pooling should NOT benefit adult speech (IEMOCAP)
- **Why**: Adult emotional prosody has different temporal structure; the F0/energy prior may be mismatched
- **ΔWA vs self_attention**: Expected ~0 or negative

### Exp 4 (Zero-Shot)

- **Hypothesis**: C-BESD → FAU-Aibo cross-corpus performance will be weak due to language/acoustic mismatch
- **Why**: FAU Aibo is German children's speech, C-BESD is Chinese/English acted speech

## Usage

### Run full pipeline

```bash
bash run_experiments.sh
```

### Run individual experiment

```bash
# Self-attention baseline
python src/train.py --train_data c-besd --pooling_type self_attention --exp_name my_exp

# With cross-corpus zero-shot
python src/train.py --train_data c-besd --test_data iemocap --pooling_type prosody_guided
```

### Run diagnostics only

```bash
# Requires checkpoints/best_model.pt
python src/extract_diagnostics.py
```

## Key Scripts

| Script | Description |
|--------|-------------|
| `src/train.py` | Main training loop with argparse CLI |
| `src/extract_diagnostics.py` | XAI + distribution shift extraction |
| `run_experiments.sh` | Full orchestration: 4 experiments + diagnostics |

## Output Files

| File | Content |
|------|---------|
| `results/logs/exp*.json` | Per-experiment WA/UAR/loss curves |
| `results/layer_weights.json` | 12 learned fusion weights + entropy |
| `results/xai_final.png` | 3-stack saliency map with APC scores |
| `results/logs/apc_metrics.json` | APC_wav and APC_delta correlation scores |
| `results/distribution_shift.json` | FD and SMMD between child and adult datasets |

## Model Configuration

- **Backbone**: WavLM Base+ (frozen, 94M params)
- **Layer Fusion**: WavLMLayerFusion (12 learnable weights)
- **Pooling**: Configurable (self_attention or prosody_guided), 111,105 params each
- **Classifier**: SEMLP (4-class, ~590K params)
- **Loss**: CrossEntropyLoss with label_smoothing=0.1
- **Optimizer**: AdamW (lr=3e-4, wd=1e-3)
- **Scheduler**: CosineAnnealingLR
