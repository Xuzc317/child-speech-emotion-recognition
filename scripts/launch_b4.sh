#!/bin/bash
# B4: E5 Layer Fusion Ablation (27 runs + best_single grid search)
# 3 datasets x 3 fusion modes (last/best_single/weighted) x 3 seeds
# best_single: grid search L1-L12 (seed=42 only), pick best layer
#
# Uses E1 best pooling per dataset.
set -uo pipefail
export SER_C_BESD_PATH=/root/autodl-tmp/datasets/BESD/BESD/MY
export SER_IEMOCAP_PATH=/root/autodl-tmp/IEMOCAP/wavs
export SER_FAU_AIBO_PATH=/root/autodl-tmp/IS2009EmotionChallenge/IS2009EmotionChallenge/wav
export PYTHONPATH=/root/autodl-tmp/d-ser
cd /root/autodl-tmp/d-ser
PYTHON=/root/miniconda3/bin/python
find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
mkdir -p checkpoints/b4 results/b4

# E1 best pooling per dataset (UPDATE AFTER B1 COMPLETES)
CBESD_POOL="self_attention"
FAU_POOL="self_attention"
IEMO_POOL="prosody_guided"  # E1-09 best (WA=0.6505)

echo "========================================="
echo " B4: LAYER FUSION ABLATION - $(date)"
echo " 3 datasets x 3 fusion x 3 seeds = 27 runs"
echo " + best_single grid search (L1-L12 x seed=42)"
echo " GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "========================================="

run_fusion() {
    local exp_id=$1 dataset=$2 pooling=$3 reg=$4 ncls=$5 mode=$6 layer=${7:-8}
    echo ""
    echo "  $exp_id: $dataset | $pooling | fusion=$mode"
    for seed in 42 123 456; do
        echo "--- seed=$seed ---"
        mkdir -p "checkpoints/b4/${exp_id}_s${seed}"
        $PYTHON -m src.train \
            --train_data "$dataset" --pooling_type "$pooling" \
            --num_classes "$ncls" --reg_profile "$reg" \
            --fusion_mode "$mode" --fusion_best_layer "$layer" \
            --seed "$seed" --data_split_seed 42 --epochs 100 --batch_size 16 --patience 15 \
            --exp_name "${exp_id}_s${seed}" \
            --output_dir "checkpoints/b4/${exp_id}_s${seed}"
        echo "  seed=$seed DONE"
    done
}

# Grid search: find best single layer (seed=42 only, 12 quick runs)
run_layer_scan() {
    local prefix=$1 dataset=$2 pooling=$3 reg=$4 ncls=$5
    echo ""
    echo "  === Layer grid search: $dataset L1-L12 ==="
    local best_layer=8 best_wa=0
    for layer in $(seq 1 12); do
        local eid="${prefix}_L${layer}"
        mkdir -p "checkpoints/b4/${eid}_s42"
        $PYTHON -m src.train \
            --train_data "$dataset" --pooling_type "$pooling" \
            --num_classes "$ncls" --reg_profile "$reg" \
            --fusion_mode "best_single" --fusion_best_layer "$layer" \
            --seed 42 --data_split_seed 42 --epochs 100 --batch_size 16 --patience 15 \
            --exp_name "${eid}_s42" \
            --output_dir "checkpoints/b4/${eid}_s42"
        echo "  Layer $layer DONE"
    done
}

# ============================================================
# C-BESD (6-class)
# ============================================================
run_layer_scan "E5-02" "c-besd" "$CBESD_POOL" "default" 6
run_fusion "E5-01" "c-besd" "$CBESD_POOL" "default" 6 "last"
run_fusion "E5-03" "c-besd" "$CBESD_POOL" "default" 6 "weighted"

# ============================================================
# FAU (4-class, reg=fau)
# ============================================================
run_layer_scan "E5-05" "fau-aibo" "$FAU_POOL" "fau" 4
run_fusion "E5-04" "fau-aibo" "$FAU_POOL" "fau" 4 "last"
run_fusion "E5-06" "fau-aibo" "$FAU_POOL" "fau" 4 "weighted"

# ============================================================
# IEMOCAP (4-class)
# ============================================================
run_layer_scan "E5-08" "iemocap" "$IEMO_POOL" "default" 4
run_fusion "E5-07" "iemocap" "$IEMO_POOL" "default" 4 "last"
run_fusion "E5-09" "iemocap" "$IEMO_POOL" "default" 4 "weighted"

echo ""
echo "========================================="
echo " B4 COMPLETED: $(date)"
echo "========================================="
echo ""
echo "=== AUTO-CHAINING TO B5 ==="
nohup bash scripts/launch_b5.sh > b5_output.log 2>&1 &
echo "B5 launched in background"
