#!/bin/bash
# B3: E4 Data Augmentation Sensitivity (36 runs)
# 3 datasets x 4 conditions (C1/C2/C3/C4) x 3 seeds
# C1=clean, C2=data mixing, C3=SafeAWGN, C4=extreme
#
# NOTE: Uses E1's best pooling for each dataset:
#   C-BESD -> self_attention (E1-02 best)
#   FAU    -> self_attention (E1-05 best)
#   IEMOCAP -> TBD from E1 results
# Update these after B1 completes!
set -uo pipefail
export SER_C_BESD_PATH=/root/autodl-tmp/datasets/BESD/BESD/MY
export SER_IEMOCAP_PATH=/root/autodl-tmp/IEMOCAP/wavs
export SER_FAU_AIBO_PATH=/root/autodl-tmp/IS2009EmotionChallenge/IS2009EmotionChallenge/wav
export PYTHONPATH=/root/autodl-tmp/d-ser
cd /root/autodl-tmp/d-ser
PYTHON=/root/miniconda3/bin/python
find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
mkdir -p checkpoints/b3 results/b3

# E1 best pooling per dataset (UPDATE AFTER B1 COMPLETES)
CBESD_POOL="self_attention"
FAU_POOL="self_attention"
IEMO_POOL="prosody_guided"  # E1-09 best (WA=0.6505)

echo "========================================="
echo " B3: AUGMENTATION SENSITIVITY - $(date)"
echo " 3 datasets x 4 conditions x 3 seeds = 36 runs"
echo " GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "========================================="

run_aug() {
    local exp_id=$1 dataset=$2 pooling=$3 reg=$4 ncls=$5 cond=$6
    echo ""
    echo "  $exp_id: $dataset | $pooling | $cond"
    for seed in 42 123 456; do
        echo "--- seed=$seed ---"
        mkdir -p "checkpoints/b3/${exp_id}_s${seed}"
        $PYTHON -m src.train \
            --train_data "$dataset" --pooling_type "$pooling" \
            --num_classes "$ncls" --reg_profile "$reg" \
            --augment_condition "$cond" \
            --seed "$seed" --data_split_seed 42 --epochs 100 --batch_size 16 --patience 15 \
            --exp_name "${exp_id}_s${seed}" \
            --output_dir "checkpoints/b3/${exp_id}_s${seed}"
        echo "  seed=$seed DONE"
    done
}

# ============================================================
# C-BESD (6-class, reg=default)
# ============================================================
run_aug "E4-01" "c-besd" "$CBESD_POOL" "default" 6 "C1"
run_aug "E4-02" "c-besd" "$CBESD_POOL" "default" 6 "C2"  # +IEMOCAP mix
run_aug "E4-03" "c-besd" "$CBESD_POOL" "default" 6 "C3"  # +SafeAWGN
run_aug "E4-04" "c-besd" "$CBESD_POOL" "default" 6 "C4"  # +all extreme

# ============================================================
# FAU Aibo (4-class, reg=fau)
# ============================================================
run_aug "E4-05" "fau-aibo" "$FAU_POOL" "fau" 4 "C1"
run_aug "E4-06" "fau-aibo" "$FAU_POOL" "fau" 4 "C2"
run_aug "E4-07" "fau-aibo" "$FAU_POOL" "fau" 4 "C3"
run_aug "E4-08" "fau-aibo" "$FAU_POOL" "fau" 4 "C4"

# ============================================================
# IEMOCAP (4-class, reg=default)
# C2 uses FAU mixing (adult data + child spontaneous = max shift)
# ============================================================
run_aug "E4-09" "iemocap" "$IEMO_POOL" "default" 4 "C1"
run_aug "E4-10" "iemocap" "$IEMO_POOL" "default" 4 "C2"
run_aug "E4-11" "iemocap" "$IEMO_POOL" "default" 4 "C3"
run_aug "E4-12" "iemocap" "$IEMO_POOL" "default" 4 "C4"

echo ""
echo "========================================="
echo " B3 COMPLETED: $(date)"
echo "========================================="
echo ""
echo "=== AUTO-CHAINING TO B4 ==="
nohup bash scripts/launch_b4.sh > b4_output.log 2>&1 &
echo "B4 launched in background"
