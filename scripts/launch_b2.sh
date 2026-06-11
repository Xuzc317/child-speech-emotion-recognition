#!/bin/bash
# B2: E3 Zero-shot Transfer Matrix (18 runs)
# Train on source, test on target — NO target-domain fine-tuning.
# C-BESD uses 4-class subset (c-besd-4cl) for label-space alignment.
set -uo pipefail
export SER_C_BESD_PATH=/root/autodl-tmp/datasets/BESD/BESD/MY
export SER_IEMOCAP_PATH=/root/autodl-tmp/IEMOCAP/wavs
export SER_FAU_AIBO_PATH=/root/autodl-tmp/IS2009EmotionChallenge/IS2009EmotionChallenge/wav
export PYTHONPATH=/root/autodl-tmp/d-ser
cd /root/autodl-tmp/d-ser
PYTHON=/root/miniconda3/bin/python
find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
mkdir -p checkpoints/b2 results/b2

echo "========================================="
echo " B2: ZERO-SHOT TRANSFER MATRIX - $(date)"
echo " 6 directions x 3 pooling = 18 runs"
echo " GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "========================================="

# Zero-shot: train on source, test directly on target
run_zs() {
    local exp_id=$1 src=$2 tgt=$3 pooling=$4 ncls=$5
    echo "  $exp_id: $src -> $tgt | $pooling"
    $PYTHON -m src.train \
        --train_data "$src" --test_data "$tgt" \
        --pooling_type "$pooling" --num_classes "$ncls" \
        --seed 42 --epochs 100 --batch_size 16 --patience 15 \
        --exp_name "$exp_id" --output_dir "checkpoints/b2/$exp_id"
    echo "  $exp_id DONE"
}

# --- C-BESD -> FAU (child acted -> child spontaneous) ---
run_zs "E3-01" "c-besd-4cl" "fau-aibo" "mean" 4
run_zs "E3-02" "c-besd-4cl" "fau-aibo" "self_attention" 4
run_zs "E3-03" "c-besd-4cl" "fau-aibo" "prosody_guided" 4

# --- C-BESD -> IEMOCAP (child acted -> adult acted) ---
run_zs "E3-04" "c-besd-4cl" "iemocap" "mean" 4
run_zs "E3-05" "c-besd-4cl" "iemocap" "self_attention" 4
run_zs "E3-06" "c-besd-4cl" "iemocap" "prosody_guided" 4

# --- FAU -> C-BESD (child spontaneous -> child acted) ---
run_zs "E3-07" "fau-aibo" "c-besd-4cl" "mean" 4
run_zs "E3-08" "fau-aibo" "c-besd-4cl" "self_attention" 4
run_zs "E3-09" "fau-aibo" "c-besd-4cl" "prosody_guided" 4

# --- FAU -> IEMOCAP (child spontaneous -> adult acted) ---
run_zs "E3-10" "fau-aibo" "iemocap" "mean" 4
run_zs "E3-11" "fau-aibo" "iemocap" "self_attention" 4
run_zs "E3-12" "fau-aibo" "iemocap" "prosody_guided" 4

# --- IEMOCAP -> C-BESD (adult acted -> child acted) ---
run_zs "E3-13" "iemocap" "c-besd-4cl" "mean" 4
run_zs "E3-14" "iemocap" "c-besd-4cl" "self_attention" 4
run_zs "E3-15" "iemocap" "c-besd-4cl" "prosody_guided" 4

# --- IEMOCAP -> FAU (adult acted -> child spontaneous) ---
run_zs "E3-16" "iemocap" "fau-aibo" "mean" 4
run_zs "E3-17" "iemocap" "fau-aibo" "self_attention" 4
run_zs "E3-18" "iemocap" "fau-aibo" "prosody_guided" 4

echo ""
echo "========================================="
echo " B2 COMPLETED: $(date)"
echo "========================================="
