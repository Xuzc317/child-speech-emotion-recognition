#!/bin/bash
# B4 RESUME: Complete missing IEMOCAP experiments
# - E5-08 L12 (layer scan, seed=42 only)
# - E5-07 (IEMOCAP last fusion, 3 seeds)
# - E5-09 (IEMOCAP weighted fusion, 3 seeds)
# Then auto-chain to B5 -> B6 -> B7
set -uo pipefail
export SER_C_BESD_PATH=/root/autodl-tmp/datasets/BESD/BESD/MY
export SER_IEMOCAP_PATH=/root/autodl-tmp/IEMOCAP/wavs
export SER_FAU_AIBO_PATH=/root/autodl-tmp/IS2009EmotionChallenge/IS2009EmotionChallenge/wav
export PYTHONPATH=/root/autodl-tmp/d-ser
cd /root/autodl-tmp/d-ser
PYTHON=/root/miniconda3/bin/python
find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
mkdir -p checkpoints/b4 results/b4

IEMO_POOL="prosody_guided"  # E1-09 best

echo "========================================="
echo " B4 RESUME: IEMOCAP experiments - $(date)"
echo " E5-08 L12 + E5-07 (last x3) + E5-09 (weighted x3)"
echo " GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "========================================="

# === E5-08 L12: IEMOCAP best_single layer 12 ===
echo ""
echo "  [B4-resume] E5-08_L12: IEMOCAP best_single layer=12"
mkdir -p "checkpoints/b4/E5-08_L12_s42"
$PYTHON -m src.train \
    --train_data "iemocap" --pooling_type "$IEMO_POOL" \
    --num_classes 4 --reg_profile "default" \
    --fusion_mode "best_single" --fusion_best_layer 12 \
    --seed 42 --data_split_seed 42 --epochs 100 --batch_size 16 --patience 15 \
    --exp_name "E5-08_L12_s42" \
    --output_dir "checkpoints/b4/E5-08_L12_s42"
echo "  E5-08_L12 DONE"

# === E5-07: IEMOCAP last fusion, 3 seeds ===
echo ""
echo "  [B4-resume] E5-07: IEMOCAP last fusion (3 seeds)"
for seed in 42 123 456; do
    echo "  --- seed=$seed ---"
    mkdir -p "checkpoints/b4/E5-07_s${seed}"
    $PYTHON -m src.train \
        --train_data "iemocap" --pooling_type "$IEMO_POOL" \
        --num_classes 4 --reg_profile "default" \
        --fusion_mode "last" \
        --seed "$seed" --data_split_seed 42 --epochs 100 --batch_size 16 --patience 15 \
        --exp_name "E5-07_s${seed}" \
        --output_dir "checkpoints/b4/E5-07_s${seed}"
    echo "  E5-07 seed=$seed DONE"
done

# === E5-09: IEMOCAP weighted fusion, 3 seeds ===
echo ""
echo "  [B4-resume] E5-09: IEMOCAP weighted fusion (3 seeds)"
for seed in 42 123 456; do
    echo "  --- seed=$seed ---"
    mkdir -p "checkpoints/b4/E5-09_s${seed}"
    $PYTHON -m src.train \
        --train_data "iemocap" --pooling_type "$IEMO_POOL" \
        --num_classes 4 --reg_profile "default" \
        --fusion_mode "weighted" \
        --seed "$seed" --data_split_seed 42 --epochs 100 --batch_size 16 --patience 15 \
        --exp_name "E5-09_s${seed}" \
        --output_dir "checkpoints/b4/E5-09_s${seed}"
    echo "  E5-09 seed=$seed DONE"
done

echo ""
echo "========================================="
echo " B4 RESUME COMPLETED: $(date)"
echo "========================================="
echo ""
echo "=== AUTO-CHAINING TO B5 ==="
nohup bash scripts/launch_b5.sh > b5_output.log 2>&1 &
echo "B5 launched in background (PID: $!)"
