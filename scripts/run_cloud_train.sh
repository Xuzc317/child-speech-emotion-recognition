#!/bin/bash
# ==============================================================================
# Cloud training launch script for child speech emotion recognition
#
# Usage:
#   bash scripts/run_cloud_train.sh --model DrseCNN --epochs 150
#   bash scripts/run_cloud_train.sh --model OptimizedBiLSTM --lr 1e-3
#
# Options:
#   --gpu GPU_ID         GPU device index (default: 0, or "all")
#   --name NAME          Run name for logs/WandB
#   --resume CKPT        Resume from checkpoint
#   --dry-run            Print config and exit
#
# All other arguments are forwarded to train.py.
# ==============================================================================

set -euo pipefail

# ── defaults ──
GPU_ID="0"
RUN_NAME=""
RESUME=""
DRY_RUN=""

# ── parse custom flags, forward rest ──
FORWARD_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            GPU_ID="$2"; shift 2 ;;
        --name)
            RUN_NAME="$2"; shift 2 ;;
        --resume)
            RESUME="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN="--dry_run"; shift ;;
        *)
            FORWARD_ARGS+=("$1"); shift ;;
    esac
done

# ── project root ──
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# ── environment ──
echo "========================================="
echo "Cloud Training Launcher"
echo "========================================="
echo "Project root: $PROJECT_ROOT"
echo "Python:       $(which python)"
echo "GPU:          $GPU_ID"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
if [ "$GPU_ID" != "all" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

# ── dependencies (uncomment for first run) ──
# pip install -r requirements_lock.txt

# ── check data ──
if [ ! -f "train_data.npy" ]; then
    echo "ERROR: train_data.npy not found."
    echo "Run preprocessing first: python src/data/preprocess.py"
    exit 1
fi

# ── directories ──
mkdir -p experiments/logs checkpoints

# ── timestamp ──
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_NAME=$(echo "${FORWARD_ARGS[@]}" | grep -oP '(?<=--model )\S+' || echo "model")
LOG_FILE="experiments/logs/train_${MODEL_NAME}_${TIMESTAMP}.log"
PID_FILE="experiments/logs/train_${MODEL_NAME}_${TIMESTAMP}.pid"

# ── build command ──
CMD="python -u src/training/train.py"
CMD="$CMD $DRY_RUN"

if [ -n "$RUN_NAME" ]; then
    CMD="$CMD --run_name $RUN_NAME"
fi

if [ -n "$RESUME" ]; then
    CMD="$CMD --resume $RESUME"
fi

CMD="$CMD ${FORWARD_ARGS[@]:-}"

echo "Command: $CMD"
echo "Log:     $LOG_FILE"
echo "========================================="

if [ -n "$DRY_RUN" ]; then
    echo "[Dry run]"
    eval "$CMD"
    exit 0
fi

# ── launch ──
nohup $CMD > "$LOG_FILE" 2>&1 &
PID=$!
echo $PID > "$PID_FILE"

echo "Training launched."
echo "  PID:  $PID"
echo "  Log:  $LOG_FILE"
echo ""
echo "Monitor: tail -f $LOG_FILE"
echo "Watch eval: python evaluate_model.py --ckpt checkpoints/best_${MODEL_NAME}.pth --model $MODEL_NAME --watch"
echo "Stop: kill $PID"
