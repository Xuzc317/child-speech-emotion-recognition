#!/bin/bash
# B5: E2 WavLM Unfreeze Comparison (9 runs)
# 3 datasets x 3 seeds — unfreeze WavLM backbone (~95M params)
# Uses E1 best pooling per dataset.
set -uo pipefail
export SER_C_BESD_PATH=/root/autodl-tmp/datasets/BESD/BESD/MY
export SER_IEMOCAP_PATH=/root/autodl-tmp/IEMOCAP/wavs
export SER_FAU_AIBO_PATH=/root/autodl-tmp/IS2009EmotionChallenge/IS2009EmotionChallenge/wav
export PYTHONPATH=/root/autodl-tmp/d-ser
cd /root/autodl-tmp/d-ser
PYTHON=/root/miniconda3/bin/python
find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
mkdir -p checkpoints/b5 results/b5

# E1 best pooling per dataset (UPDATE AFTER B1 COMPLETES)
CBESD_POOL="self_attention"
FAU_POOL="self_attention"
IEMO_POOL="self_attention"

echo "========================================="
echo " B5: WAVLM UNFREEZE - $(date)"
echo " 3 datasets x 3 seeds = 9 runs"
echo " WARNING: ~95M params, may need smaller batch_size"
echo " GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "========================================="

run_unfrozen() {
    local exp_id=$1 dataset=$2 pooling=$3 reg=$4 ncls=$5
    echo ""
    echo "  $exp_id: $dataset | $pooling | UNFROZEN"
    for seed in 42 123 456; do
        echo "--- seed=$seed ---"
        mkdir -p "checkpoints/b5/${exp_id}_s${seed}"
        $PYTHON -m src.train \
            --train_data "$dataset" --pooling_type "$pooling" \
            --num_classes "$ncls" --reg_profile "$reg" \
            --unfreeze_ssl --ssl_lr 1e-5 --lr 3e-4 \
            --seed "$seed" --epochs 100 --batch_size 8 --patience 15 \
            --exp_name "${exp_id}_s${seed}" \
            --output_dir "checkpoints/b5/${exp_id}_s${seed}"
        echo "  seed=$seed DONE"
    done
}

# ============================================================
run_unfrozen "E2-01" "c-besd" "$CBESD_POOL" "default" 6
run_unfrozen "E2-02" "fau-aibo" "$FAU_POOL" "fau" 4
run_unfrozen "E2-03" "iemocap" "$IEMO_POOL" "default" 4

echo ""
echo "========================================="
echo " B5 COMPLETED: $(date)"
echo "========================================="
