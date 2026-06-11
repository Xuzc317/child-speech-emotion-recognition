#!/bin/bash
# B6: E6 Module Ablation (30 runs)
# C-BESD x5 + FAU x5 = 10 groups x 3 seeds
# Gradually add Adapter, Best Pooling, LayerFusion.
#
# NOTE: Requires B4 results for best fusion config.
#   If weighted > best_single, use weighted. If best_single wins, use that layer.
set -uo pipefail
export SER_C_BESD_PATH=/root/autodl-tmp/datasets/BESD/BESD/MY
export SER_IEMOCAP_PATH=/root/autodl-tmp/IEMOCAP/wavs
export SER_FAU_AIBO_PATH=/root/autodl-tmp/IS2009EmotionChallenge/IS2009EmotionChallenge/wav
export PYTHONPATH=/root/autodl-tmp/d-ser
cd /root/autodl-tmp/d-ser
PYTHON=/root/miniconda3/bin/python
find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
mkdir -p checkpoints/b6 results/b6

# E1 best pooling + E5 best fusion (UPDATE AFTER B1/B4 COMPLETE)
CBESD_POOL="self_attention"
FAU_POOL="self_attention"
FUSION_MODE="weighted"   # or "best_single" if E5 shows it's better
BEST_LAYER=8

echo "========================================="
echo " B6: MODULE ABLATION - $(date)"
echo " C-BESD x5 + FAU x5 = 10 groups x 3 seeds = 30 runs"
echo " GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "========================================="

# Minimal: no adapter, mean pooling, last layer only
run_minimal() {
    local exp_id=$1 dataset=$2 reg=$3 ncls=$4
    echo ""
    echo "  $exp_id: $dataset | MINIMAL (no adapter, mean, last layer)"
    for seed in 42 123 456; do
        echo "--- seed=$seed ---"
        mkdir -p "checkpoints/b6/${exp_id}_s${seed}"
        $PYTHON -m src.train \
            --train_data "$dataset" --pooling_type "mean" \
            --num_classes "$ncls" --reg_profile "$reg" \
            --fusion_mode "last" \
            --seed "$seed" --epochs 100 --batch_size 16 --patience 15 \
            --exp_name "${exp_id}_s${seed}" \
            --output_dir "checkpoints/b6/${exp_id}_s${seed}"
        echo "  seed=$seed DONE"
    done
}

# +Adapter only
run_adapter_only() {
    local exp_id=$1 dataset=$2 reg=$3 ncls=$4
    echo ""
    echo "  $exp_id: $dataset | +ADAPTER (adapter, mean, last layer)"
    for seed in 42 123 456; do
        echo "--- seed=$seed ---"
        mkdir -p "checkpoints/b6/${exp_id}_s${seed}"
        $PYTHON -m src.train \
            --train_data "$dataset" --pooling_type "mean" \
            --num_classes "$ncls" --reg_profile "$reg" \
            --fusion_mode "last" --use_adapter \
            --seed "$seed" --epochs 100 --batch_size 16 --patience 15 \
            --exp_name "${exp_id}_s${seed}" \
            --output_dir "checkpoints/b6/${exp_id}_s${seed}"
        echo "  seed=$seed DONE"
    done
}

# +Pooling only (best from E1, no adapter, last layer)
run_pooling_only() {
    local exp_id=$1 dataset=$2 pooling=$3 reg=$4 ncls=$5
    echo ""
    echo "  $exp_id: $dataset | +POOLING ($pooling, no adapter, last layer)"
    for seed in 42 123 456; do
        echo "--- seed=$seed ---"
        mkdir -p "checkpoints/b6/${exp_id}_s${seed}"
        $PYTHON -m src.train \
            --train_data "$dataset" --pooling_type "$pooling" \
            --num_classes "$ncls" --reg_profile "$reg" \
            --fusion_mode "last" \
            --seed "$seed" --epochs 100 --batch_size 16 --patience 15 \
            --exp_name "${exp_id}_s${seed}" \
            --output_dir "checkpoints/b6/${exp_id}_s${seed}"
        echo "  seed=$seed DONE"
    done
}

# +LayerFusion (best pooling, no adapter, best fusion)
run_fusion_only() {
    local exp_id=$1 dataset=$2 pooling=$3 reg=$4 ncls=$5
    echo ""
    echo "  $exp_id: $dataset | +FUSION ($pooling, no adapter, $FUSION_MODE)"
    for seed in 42 123 456; do
        echo "--- seed=$seed ---"
        mkdir -p "checkpoints/b6/${exp_id}_s${seed}"
        $PYTHON -m src.train \
            --train_data "$dataset" --pooling_type "$pooling" \
            --num_classes "$ncls" --reg_profile "$reg" \
            --fusion_mode "$FUSION_MODE" --fusion_best_layer "$BEST_LAYER" \
            --seed "$seed" --epochs 100 --batch_size 16 --patience 15 \
            --exp_name "${exp_id}_s${seed}" \
            --output_dir "checkpoints/b6/${exp_id}_s${seed}"
        echo "  seed=$seed DONE"
    done
}

# Full stack (all modules)
run_full() {
    local exp_id=$1 dataset=$2 pooling=$3 reg=$4 ncls=$5
    echo ""
    echo "  $exp_id: $dataset | FULL (adapter, $pooling, $FUSION_MODE)"
    for seed in 42 123 456; do
        echo "--- seed=$seed ---"
        mkdir -p "checkpoints/b6/${exp_id}_s${seed}"
        $PYTHON -m src.train \
            --train_data "$dataset" --pooling_type "$pooling" \
            --num_classes "$ncls" --reg_profile "$reg" \
            --fusion_mode "$FUSION_MODE" --fusion_best_layer "$BEST_LAYER" \
            --use_adapter \
            --seed "$seed" --epochs 100 --batch_size 16 --patience 15 \
            --exp_name "${exp_id}_s${seed}" \
            --output_dir "checkpoints/b6/${exp_id}_s${seed}"
        echo "  seed=$seed DONE"
    done
}

# ============================================================
# C-BESD (6-class, reg=default)
# ============================================================
run_minimal      "E6-01" "c-besd" "default" 6
run_adapter_only "E6-02" "c-besd" "default" 6
run_pooling_only "E6-03" "c-besd" "$CBESD_POOL" "default" 6
run_fusion_only  "E6-04" "c-besd" "$CBESD_POOL" "default" 6
run_full         "E6-05" "c-besd" "$CBESD_POOL" "default" 6

# ============================================================
# FAU (4-class, reg=fau)
# ============================================================
run_minimal      "E6-06" "fau-aibo" "fau" 4
run_adapter_only "E6-07" "fau-aibo" "fau" 4
run_pooling_only "E6-08" "fau-aibo" "$FAU_POOL" "fau" 4
run_fusion_only  "E6-09" "fau-aibo" "$FAU_POOL" "fau" 4
run_full         "E6-10" "fau-aibo" "$FAU_POOL" "fau" 4

echo ""
echo "========================================="
echo " B6 COMPLETED: $(date)"
echo "========================================="
