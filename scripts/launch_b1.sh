#!/bin/bash
# B1: In-Domain Pooling Baseline — 9 configs x 3 seeds
# batch_size=32, num_workers=0 (stable, no deadlock)
# Run on AutoDL: nohup bash launch_b1.sh > b1_output.log 2>&1 &
set -uo pipefail

export SER_C_BESD_PATH=/root/autodl-tmp/datasets/BESD/BESD/MY
export SER_IEMOCAP_PATH=/root/autodl-tmp/IEMOCAP/wavs
export SER_FAU_AIBO_PATH=/root/autodl-tmp/IS2009EmotionChallenge/IS2009EmotionChallenge/wav
export PYTHONPATH=/root/autodl-tmp/d-ser
export PYTHONUNBUFFERED=1

cd /root/autodl-tmp/d-ser
PYTHON=/root/miniconda3/bin/python

mkdir -p checkpoints/b1

SEEDS="42 123 456"

run_exp() {
    local exp_id=$1 dataset=$2 pooling=$3 reg=$4 ncls=$5
    echo ""
    echo "================================================================"
    echo "  $exp_id: $dataset (${ncls}cls) | $pooling | reg=$reg"
    echo "================================================================"

    for seed in $SEEDS; do
        echo "--- [$((++run_count))/27] seed=$seed ---"
        mkdir -p "checkpoints/b1/${exp_id}_s${seed}"
        $PYTHON -m src.train \
            --train_data "$dataset" \
            --pooling_type "$pooling" \
            --num_classes "$ncls" \
            --reg_profile "$reg" \
            --seed "$seed" \
            --data_split_seed 42 \
            --exp_name "${exp_id}_s${seed}" \
            --output_dir "checkpoints/b1/${exp_id}_s${seed}" \
            --epochs 100 \
            --batch_size 32 \
            --num_workers 0 \
            --patience 15
        echo "  seed=$seed DONE"
    done
}

echo "========================================="
echo " B1 MATRIX LAUNCHER"
echo " Started: $(date)"
echo " 9 configs x 3 seeds = 27 runs"
echo " GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "========================================="

run_count=0

# C-BESD 6-class (reg=default)
run_exp "E1-01" "c-besd" "mean"           "default" 6
run_exp "E1-02" "c-besd" "self_attention" "default" 6
run_exp "E1-03" "c-besd" "prosody_guided" "default" 6

# FAU Aibo 4-class (reg=fau)
run_exp "E1-04" "fau-aibo" "mean"           "fau" 4
run_exp "E1-05" "fau-aibo" "self_attention" "fau" 4
run_exp "E1-06" "fau-aibo" "prosody_guided" "fau" 4

# IEMOCAP 4-class (reg=default)
run_exp "E1-07" "iemocap" "mean"           "default" 4
run_exp "E1-08" "iemocap" "self_attention" "default" 4
run_exp "E1-09" "iemocap" "prosody_guided" "default" 4

echo ""
echo "========================================="
echo " B1 COMPLETED: $(date)"
echo "========================================="
