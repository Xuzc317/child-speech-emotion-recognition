#!/bin/bash
# B7-ext: E7-07~12 Unfrozen Model Transfer via Fine-tuning (18 runs)
# 6 directions x 3 seeds — WITH unfrozen WavLM backbone
#
# This is the unfrozen counterpart to launch_b7.sh (E7-01~06, frozen).
# Same source checkpoints, same transfer directions.
# ONLY variable: --unfreeze_ssl + differential LR (backbone 1e-5, head 3e-4)
#
# Differential LR scheme: REUSED from B5 (launch_b5.sh, verified).
# batch_size=8: same as B5 — unfrozen ~95M backbone + optimizer states
#                won't fit in 24GB VRAM with batch_size=16.
set -uo pipefail
export SER_C_BESD_PATH=/root/autodl-tmp/datasets/BESD/BESD/MY
export SER_IEMOCAP_PATH=/root/autodl-tmp/IEMOCAP/wavs
export SER_FAU_AIBO_PATH=/root/autodl-tmp/IS2009EmotionChallenge/IS2009EmotionChallenge/wav
export PYTHONPATH=/root/autodl-tmp/d-ser
cd /root/autodl-tmp/d-ser
PYTHON=/root/miniconda3/bin/python
find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
mkdir -p checkpoints/b7_ext results/b7_ext

# E1 best checkpoints — EXACT SAME as launch_b7.sh (apples-to-apples)
CBESD_CKPT="checkpoints/b1/E1-02_s42/best_model.pt"      # self_attn seed42 (WA=0.9292)
FAU_CKPT="checkpoints/b1/E1-05_s42/best_model.pt"        # self_attn seed42 (WA=0.6781)
IEMO_CKPT="checkpoints/b1/E1-09_s42/best_model.pt"       # prosody_guided seed42 (WA=0.6505)

echo "========================================="
echo " B7-ext: UNFROZEN MODEL TRANSFER - $(date)"
echo " 6 directions x 3 seeds = 18 runs"
echo " batch_size=8, --unfreeze_ssl, backbone_lr=1e-5, head_lr=3e-4"
echo " GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "========================================="

run_transfer_unfrozen() {
    local exp_id=$1 tgt_dataset=$2 pooling=$3 reg=$4 ncls=$5 ckpt=$6
    echo ""
    echo "  $exp_id: checkpoint=$(basename $(dirname $ckpt)) -> $tgt_dataset | $pooling | UNFROZEN"
    for seed in 42 123 456; do
        echo "--- seed=$seed ---"
        mkdir -p "checkpoints/b7_ext/${exp_id}_s${seed}"
        $PYTHON -m src.train \
            --train_data "$tgt_dataset" --pooling_type "$pooling" \
            --num_classes "$ncls" --reg_profile "$reg" \
            --load_checkpoint "$ckpt" \
            --unfreeze_ssl --ssl_lr 1e-5 --lr 3e-4 \
            --seed "$seed" --data_split_seed 42 --epochs 100 --batch_size 8 --patience 15 \
            --exp_name "${exp_id}_s${seed}" \
            --output_dir "checkpoints/b7_ext/${exp_id}_s${seed}"
        echo "  seed=$seed DONE"
    done
}

# ============================================================
# C-BESD pretrained -> FAU / IEMOCAP
# ============================================================
run_transfer_unfrozen "E7-07" "fau-aibo" "self_attention" "fau" 4 "$CBESD_CKPT"
run_transfer_unfrozen "E7-08" "iemocap"  "self_attention" "default" 4 "$CBESD_CKPT"

# ============================================================
# FAU pretrained -> C-BESD / IEMOCAP
# ============================================================
run_transfer_unfrozen "E7-09" "c-besd"   "self_attention" "default" 6 "$FAU_CKPT"
run_transfer_unfrozen "E7-10" "iemocap"  "self_attention" "default" 4 "$FAU_CKPT"

# ============================================================
# IEMOCAP pretrained -> C-BESD / FAU
# ============================================================
run_transfer_unfrozen "E7-11" "c-besd"   "self_attention" "default" 6 "$IEMO_CKPT"
run_transfer_unfrozen "E7-12" "fau-aibo" "self_attention" "fau" 4 "$IEMO_CKPT"

echo ""
echo "========================================="
echo " B7-ext COMPLETED: $(date)"
echo "========================================="
