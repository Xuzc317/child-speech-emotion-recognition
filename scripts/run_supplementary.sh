#!/bin/bash
# Supplementary experiments: CM, XAI, Layer Weights, FD plot
# Run on AutoDL: bash scripts/run_supplementary.sh
set -e
cd /root/autodl-tmp/d-ser

export SER_C_BESD_PATH=/root/autodl-tmp/datasets/BESD/BESD/MY
export SER_IEMOCAP_PATH=/root/autodl-tmp/IEMOCAP/wavs
export SER_FAU_AIBO_PATH=/root/autodl-tmp/IS2009EmotionChallenge/IS2009EmotionChallenge/wav

mkdir -p results/supplementary/figures results/supplementary/data

CKPT_DIR=checkpoints

echo "============================================"
echo "1. CONFUSION MATRICES"
echo "============================================"

declare -A CM_EXPS
CM_EXPS=(
  ["E1-02_s42"]="${CKPT_DIR}/b1/E1-02_s42/best_model.pt|c-besd|6|C-BESD_SelfAttn_B1"
  ["E1-05_s42"]="${CKPT_DIR}/b1/E1-05_s42/best_model.pt|fau-aibo|4|FAU_SelfAttn_B1"
  ["E1-09_s42"]="${CKPT_DIR}/b1/E1-09_s42/best_model.pt|iemocap|4|IEMOCAP_Prosody_B1"
  ["E2-01_s456"]="${CKPT_DIR}/b5/E2-01_s456/best_model.pt|c-besd|6|C-BESD_Unfreeze_B5"
  ["E2-02_s456"]="${CKPT_DIR}/b5/E2-02_s456/best_model.pt|fau-aibo|4|FAU_Unfreeze_B5"
  ["E2-03_s123"]="${CKPT_DIR}/b5/E2-03_s123/best_model.pt|iemocap|4|IEMOCAP_Unfreeze_B5"
  ["E6-04_s42"]="${CKPT_DIR}/b6/E6-04_s42/best_model.pt|c-besd|6|C-BESD_SelfAttenWF_B6"
  ["E7-03_s42"]="${CKPT_DIR}/b7/E7-03_s42/best_model.pt|c-besd|6|C-BESD_Transfer_B7"
)

for exp_id in "${!CM_EXPS[@]}"; do
  IFS='|' read -r ckpt dataset ncls label <<< "${CM_EXPS[$exp_id]}"
  echo ""
  echo "--- $label ($exp_id) ---"
  python scripts/plot_confusion_matrix.py \
    --checkpoint "$ckpt" \
    --dataset "$dataset" \
    --num-classes "$ncls" \
    --output "results/supplementary/figures/cm_${label}" \
    --seed 42
done

echo ""
echo "============================================"
echo "2. XAI VISUALIZATION + APC"
echo "============================================"

for exp_id in "E1-02_s42" "E6-04_s42"; do
  if [ "$exp_id" == "E1-02_s42" ]; then
    ckpt="${CKPT_DIR}/b1/E1-02_s42/best_model.pt"
    dataset="c-besd"
  else
    ckpt="${CKPT_DIR}/b6/E6-04_s42/best_model.pt"
    dataset="c-besd"
  fi
  echo "--- $exp_id ---"
  python src/extract_diagnostics.py \
    --checkpoint "$ckpt" \
    --dataset "$dataset" \
    --output-dir "results/supplementary/data" \
    --prefix "${exp_id}" \
    --mode xai
done

echo ""
echo "============================================"
echo "3. LAYER FUSION WEIGHTS"
echo "============================================"

for exp_id in "E1-02_s42" "E5-01_s42"; do
  if [ "$exp_id" == "E1-02_s42" ]; then
    ckpt="${CKPT_DIR}/b1/E1-02_s42/best_model.pt"
  else
    ckpt="${CKPT_DIR}/b4/E5-01_s42/best_model.pt"
  fi
  echo "--- $exp_id ---"
  python src/extract_diagnostics.py \
    --checkpoint "$ckpt" \
    --dataset "c-besd" \
    --output-dir "results/supplementary/data" \
    --prefix "${exp_id}_layer" \
    --mode layer_weights
done

echo ""
echo "============================================"
echo "4. FD vs ACCURACY PLOT"
echo "============================================"

python scripts/plot_fd_vs_accuracy.py \
  --fd-file results/canonical_fd_pairs.json \
  --output results/supplementary/figures/fig_fd_vs_accuracy

echo ""
echo "============================================"
echo "ALL DONE"
echo "============================================"
