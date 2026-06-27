#!/bin/bash
# B6 Fill: E6-10_s123 + E6-10_s456 (2 remaining runs)
# Run AFTER B7 completes.
set -uo pipefail
export SER_C_BESD_PATH=/root/autodl-tmp/datasets/BESD/BESD/MY
export SER_IEMOCAP_PATH=/root/autodl-tmp/IEMOCAP/wavs
export SER_FAU_AIBO_PATH=/root/autodl-tmp/IS2009EmotionChallenge/IS2009EmotionChallenge/wav
export PYTHONPATH=/root/autodl-tmp/d-ser
cd /root/autodl-tmp/d-ser
PYTHON=/root/miniconda3/bin/python
find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
mkdir -p checkpoints/b6

echo "========================================="
echo " B6 FILL: E6-10_s123 + s456 - $(date)"
echo " GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "========================================="

for seed in 123 456; do
    echo ""
    echo "  E6-10_s${seed}: fau-aibo | FULL (adapter, self_attention, weighted)"
    echo "--- seed=$seed ---"
    mkdir -p "checkpoints/b6/E6-10_s${seed}"
    $PYTHON -m src.train \
        --train_data "fau-aibo" --pooling_type "self_attention" \
        --num_classes 4 --reg_profile "fau" \
        --fusion_mode "weighted" --fusion_best_layer 8 \
        --use_adapter \
        --seed "$seed" --data_split_seed 42 --epochs 100 --batch_size 16 --patience 15 \
        --exp_name "E6-10_s${seed}" \
        --output_dir "checkpoints/b6/E6-10_s${seed}"
    echo "  seed=$seed DONE"
done

echo ""
echo "========================================="
echo " B6 FILL COMPLETED: $(date)"
echo "========================================="
