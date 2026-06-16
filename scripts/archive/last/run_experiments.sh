#!/bin/bash
# ============================================================
# Automated Ablation Matrix & Diagnostics Extraction
# ============================================================
#
# Experiment Matrix:
#   Exp1: C-BESD + self_attention (baseline)
#   Exp2: C-BESD + prosody_guided (ours, main result)
#   Exp3: IEMOCAP + prosody_guided (adult falsification)
#   Exp4: C-BESD→FAU-Aibo + prosody_guided (zero-shot cross-corpus)
#
# After training, runs diagnostics extraction on Exp2 checkpoint.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p checkpoints results/logs

echo "============================================"
echo "Experiment Matrix: SER Ablation Studies"
echo "============================================"

# ── Exp 1: Self-Attention Baseline ─────────────────────────
echo ""
echo "=== Exp 1: Self-Attention Baseline (C-BESD) ==="
python -u src/train.py \
    --train_data c-besd \
    --pooling_type self_attention \
    --epochs 100 \
    --batch_size 16 \
    --lr 3e-4 \
    --patience 15 \
    --exp_name exp1_self_attention \
    --output_dir checkpoints/exp1

# ── Exp 2: Prosody-Guided (Ours) ───────────────────────────
echo ""
echo "=== Exp 2: Prosody-Guided (C-BESD) ==="
python -u src/train.py \
    --train_data c-besd \
    --pooling_type prosody_guided \
    --epochs 100 \
    --batch_size 16 \
    --lr 3e-4 \
    --patience 15 \
    --exp_name exp2_prosody_guided \
    --output_dir checkpoints/exp2

# ── Exp 3: Adult Falsification (IEMOCAP) ───────────────────
echo ""
echo "=== Exp 3: Adult Falsification (IEMOCAP) ==="
python -u src/train.py \
    --train_data iemocap \
    --pooling_type prosody_guided \
    --epochs 100 \
    --batch_size 16 \
    --lr 3e-4 \
    --patience 15 \
    --exp_name exp3_adult_iemocap \
    --output_dir checkpoints/exp3

# ── Exp 4: Zero-Shot Cross-Corpus ──────────────────────────
echo ""
echo "=== Exp 4: Zero-Shot C-BESD → FAU-Aibo ==="
python -u src/train.py \
    --train_data c-besd \
    --test_data fau-aibo \
    --pooling_type prosody_guided \
    --epochs 100 \
    --batch_size 16 \
    --lr 3e-4 \
    --patience 15 \
    --exp_name exp4_zero_shot_fau \
    --output_dir checkpoints/exp4

# ── Diagnostics Extraction ─────────────────────────────────
echo ""
echo "=== Diagnostics Extraction ==="
# Copy best model from Exp2 for diagnostics
cp checkpoints/exp2/best_model.pt checkpoints/best_model.pt
python -u src/extract_diagnostics.py

echo ""
echo "============================================"
echo "Pipeline complete. Results in results/"
echo "============================================"
