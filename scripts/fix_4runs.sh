#!/bin/bash
# Fix 4 broken experimental runs on AutoDL
# Usage: nohup bash fix_4runs.sh > fix_4runs_output.log 2>&1 &
set -uo pipefail

export SER_C_BESD_PATH=/root/autodl-tmp/datasets/BESD/BESD/MY
export SER_IEMOCAP_PATH=/root/autodl-tmp/IEMOCAP/wavs
export SER_FAU_AIBO_PATH=/root/autodl-tmp/IS2009EmotionChallenge/IS2009EmotionChallenge/wav
export PYTHONPATH=/root/autodl-tmp/d-ser
export PYTHONUNBUFFERED=1

cd /root/autodl-tmp/d-ser
PYTHON=/root/miniconda3/bin/python

# backup old logs
mkdir -p results/logs/backup_20260622
for f in E1-08_s42.json E4-04_s42.json E4-10_s42.json E4-10_s123.json; do
    if [ -f "results/logs/$f" ]; then
        cp "results/logs/$f" "results/logs/backup_20260622/$f.bak"
        echo "backed up $f"
    fi
done

echo ""
echo "========================================="
echo " FIX RUN: 4 experiments"
echo " Started: $(date)"
echo " GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "========================================="

# ========================================
# 1/4: E1-08 seed=42
# ========================================
echo ""
echo "=== [1/4] E1-08_s42: IEMOCAP self_attn reg=default ==="
mkdir -p checkpoints/b1/E1-08_s42
$PYTHON -m src.train \
    --train_data iemocap \
    --pooling_type self_attention \
    --num_classes 4 \
    --reg_profile default \
    --seed 42 \
    --data_split_seed 42 \
    --exp_name E1-08_s42 \
    --output_dir checkpoints/b1/E1-08_s42 \
    --epochs 100 \
    --batch_size 32 \
    --num_workers 0 \
    --patience 15
echo "[1/4] E1-08_s42 DONE (exit $?)"

# ========================================
# 2/4: E4-04 seed=42
# ========================================
echo ""
echo "=== [2/4] E4-04_s42: C-BESD C4 self_attn reg=default ==="
mkdir -p checkpoints/b3/E4-04_s42
$PYTHON -m src.train \
    --train_data c-besd \
    --pooling_type self_attention \
    --num_classes 6 \
    --reg_profile default \
    --augment_condition C4 \
    --seed 42 \
    --data_split_seed 42 \
    --exp_name E4-04_s42 \
    --output_dir checkpoints/b3/E4-04_s42 \
    --epochs 100 \
    --batch_size 16 \
    --patience 15
echo "[2/4] E4-04_s42 DONE (exit $?)"

# ========================================
# 3/4: E4-10 seed=42
# ========================================
echo ""
echo "=== [3/4] E4-10_s42: IEMOCAP C2 prosody_guided reg=default ==="
mkdir -p checkpoints/b3/E4-10_s42
$PYTHON -m src.train \
    --train_data iemocap \
    --pooling_type prosody_guided \
    --num_classes 4 \
    --reg_profile default \
    --augment_condition C2 \
    --seed 42 \
    --data_split_seed 42 \
    --exp_name E4-10_s42 \
    --output_dir checkpoints/b3/E4-10_s42 \
    --epochs 100 \
    --batch_size 16 \
    --patience 15
echo "[3/4] E4-10_s42 DONE (exit $?)"

# ========================================
# 4/4: E4-10 seed=123
# ========================================
echo ""
echo "=== [4/4] E4-10_s123: IEMOCAP C2 prosody_guided reg=default ==="
mkdir -p checkpoints/b3/E4-10_s123
$PYTHON -m src.train \
    --train_data iemocap \
    --pooling_type prosody_guided \
    --num_classes 4 \
    --reg_profile default \
    --augment_condition C2 \
    --seed 123 \
    --data_split_seed 42 \
    --exp_name E4-10_s123 \
    --output_dir checkpoints/b3/E4-10_s123 \
    --epochs 100 \
    --batch_size 16 \
    --patience 15
echo "[4/4] E4-10_s123 DONE (exit $?)"

echo ""
echo "========================================="
echo " ALL DONE: $(date)"
echo "========================================="
