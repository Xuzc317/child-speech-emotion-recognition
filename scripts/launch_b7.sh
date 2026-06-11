#!/bin/bash
# B7: E7 Model Transfer via Fine-tuning (18 runs)
# 6 directions x 3 seeds
# Load E1 best checkpoint → fine-tune on target domain → test on target
#
# REQUIRES: B1 completed with best_model.pt for each dataset.
# Checkpoint paths must be updated after B1 finishes.
set -uo pipefail
export SER_C_BESD_PATH=/root/autodl-tmp/datasets/BESD/BESD/MY
export SER_IEMOCAP_PATH=/root/autodl-tmp/IEMOCAP/wavs
export SER_FAU_AIBO_PATH=/root/autodl-tmp/IS2009EmotionChallenge/IS2009EmotionChallenge/wav
export PYTHONPATH=/root/autodl-tmp/d-ser
cd /root/autodl-tmp/d-ser
PYTHON=/root/miniconda3/bin/python
find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
mkdir -p checkpoints/b7 results/b7

# E1 best checkpoints (UPDATE AFTER B1 COMPLETES — pick best seed per dataset)
# These are examples; replace with actual best seed paths from B1 results.
CBESD_CKPT="checkpoints/b1/E1-02_s42/best_model.pt"      # self_attn seed42
FAU_CKPT="checkpoints/b1/E1-05_s456/best_model.pt"       # self_attn seed456
IEMO_CKPT="checkpoints/b1/E1-08_s42/best_model.pt"       # placeholder

echo "========================================="
echo " B7: MODEL TRANSFER (FINE-TUNE) - $(date)"
echo " 6 directions x 3 seeds = 18 runs"
echo " GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "========================================="

run_transfer() {
    local exp_id=$1 tgt_dataset=$2 pooling=$3 reg=$4 ncls=$5 ckpt=$6
    echo ""
    echo "  $exp_id: checkpoint=$(basename $(dirname $ckpt)) -> $tgt_dataset | $pooling"
    for seed in 42 123 456; do
        echo "--- seed=$seed ---"
        mkdir -p "checkpoints/b7/${exp_id}_s${seed}"
        $PYTHON -m src.train \
            --train_data "$tgt_dataset" --pooling_type "$pooling" \
            --num_classes "$ncls" --reg_profile "$reg" \
            --load_checkpoint "$ckpt" \
            --seed "$seed" --epochs 100 --batch_size 16 --patience 15 \
            --exp_name "${exp_id}_s${seed}" \
            --output_dir "checkpoints/b7/${exp_id}_s${seed}"
        echo "  seed=$seed DONE"
    done
}

# ============================================================
# C-BESD pretrained -> FAU / IEMOCAP
# ============================================================
run_transfer "E7-01" "fau-aibo" "self_attention" "fau" 4 "$CBESD_CKPT"
run_transfer "E7-02" "iemocap"  "self_attention" "default" 4 "$CBESD_CKPT"

# ============================================================
# FAU pretrained -> C-BESD / IEMOCAP
# ============================================================
run_transfer "E7-03" "c-besd"   "self_attention" "default" 6 "$FAU_CKPT"
run_transfer "E7-04" "iemocap"  "self_attention" "default" 4 "$FAU_CKPT"

# ============================================================
# IEMOCAP pretrained -> C-BESD / FAU
# ============================================================
run_transfer "E7-05" "c-besd"   "self_attention" "default" 6 "$IEMO_CKPT"
run_transfer "E7-06" "fau-aibo" "self_attention" "fau" 4 "$IEMO_CKPT"

echo ""
echo "========================================="
echo " B7 COMPLETED: $(date)"
echo "========================================="
