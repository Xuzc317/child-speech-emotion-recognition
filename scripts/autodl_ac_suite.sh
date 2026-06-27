#!/usr/bin/env bash
# AC rigorous suite (AutoDL): per-exp checkpoints, confusion matrices, FAU multi-seed, XAI/APC.
# Protocol: docs/experiment_suite_AC.md
set -euo pipefail

cd "${SER_PROJECT_ROOT:-/root/autodl-tmp/d-ser}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi

pick_existing_dir() {
  for p in "$@"; do
    if [ -d "$p" ]; then echo "$p"; return 0; fi
  done
  return 1
}

export SER_C_BESD_PATH="${SER_C_BESD_PATH:-$(pick_existing_dir \
  /root/autodl-tmp/datasets/BESD/BESD/MY \
  /root/autodl-tmp/BESD/BESD/MY || true)}"
export SER_IEMOCAP_PATH="${SER_IEMOCAP_PATH:-$(pick_existing_dir \
  /root/autodl-tmp/IEMOCAP/wavs \
  /root/autodl-tmp/datasets/IEMOCAP/wavs || true)}"
export SER_FAU_AIBO_PATH="${SER_FAU_AIBO_PATH:-$(pick_existing_dir \
  /root/autodl-tmp/IS2009EmotionChallenge/wav \
  /root/autodl-tmp/IS2009EmotionChallenge/IS2009EmotionChallenge/IS2009EmotionChallenge/wav || true)}"

mkdir -p results/logs results/figures publication_package checkpoints ac_suite_logs
SUITE_LOG="ac_suite_logs/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$SUITE_LOG") 2>&1

echo "=== AC SUITE START $(date -Iseconds) ==="
echo "PYTHON=$PYTHON_BIN"
echo "C-BESD=${SER_C_BESD_PATH:-MISSING}"
echo "IEMOCAP=${SER_IEMOCAP_PATH:-MISSING}"
echo "FAU=${SER_FAU_AIBO_PATH:-MISSING}"

SEED_MAIN=42
SEEDS_FAU="42 123 456"

train_one() {
  local data="$1" pooling="$2" exp="$3" out="$4"
  shift 4
  echo "--- TRAIN $exp (seed=$SEED_MAIN) -> $out ---"
  "$PYTHON_BIN" -m src.train \
    --train_data "$data" \
    --pooling_type "$pooling" \
    --exp_name "$exp" \
    --output_dir "$out" \
    --seed "$SEED_MAIN" \
    "$@"
}

# ── A1: per-experiment checkpoints (Exp2 skip if present) ──
if [ -n "${SER_C_BESD_PATH:-}" ]; then
  train_one c-besd self_attention exp1_self_attention checkpoints/exp1_self_attention --reg_profile default
  if [ ! -f checkpoints/exp2_prosody_guided/best_model.pt ]; then
    train_one c-besd prosody_guided exp2_prosody_guided checkpoints/exp2_prosody_guided --reg_profile default
  else
    echo "--- SKIP exp2 train (checkpoint exists) ---"
  fi
else
  echo "ERROR: C-BESD missing — cannot run A1 Exp1/2"
  exit 1
fi

if [ -n "${SER_IEMOCAP_PATH:-}" ]; then
  train_one iemocap prosody_guided exp3_adult_iemocap checkpoints/exp3_adult_iemocap --reg_profile default
else
  echo "WARN: IEMOCAP missing — skip Exp3"
fi

if [ -n "${SER_C_BESD_PATH:-}" ] && [ -n "${SER_FAU_AIBO_PATH:-}" ]; then
  train_one c-besd prosody_guided exp4_zero_shot_fau checkpoints/exp4_zero_shot_fau \
    --reg_profile default --test_data fau-aibo
else
  echo "WARN: skip Exp4 (need C-BESD + FAU)"
fi

if [ -n "${SER_FAU_AIBO_PATH:-}" ]; then
  train_one fau-aibo prosody_guided exp5_fau_indomain checkpoints/exp5_fau_indomain --reg_profile fau
  train_one fau-aibo self_attention exp5b_self_attention_fau checkpoints/exp5b_self_attention_fau --reg_profile fau
else
  echo "WARN: FAU missing — skip Exp5/5b"
fi

# ── A2/A3: confusion matrices (seed=42, same splits as training) ──
plot_cm() {
  local ckpt="$1" train="$2" pooling="$3" out="$4"
  shift 4
  local extra=("$@")
  echo "--- CM $out ---"
  "$PYTHON_BIN" scripts/plot_confusion_matrix.py \
    --checkpoint "$ckpt" \
    --train_data $train \
    --pooling_type "$pooling" \
    --seed "$SEED_MAIN" \
    --output "results/figures/${out}" \
    "${extra[@]}"
}

plot_cm checkpoints/exp1_self_attention/best_model.pt c-besd self_attention confusion_exp1_self_attention
plot_cm checkpoints/exp2_prosody_guided/best_model.pt c-besd prosody_guided confusion_exp2_prosody_guided
if [ -f checkpoints/exp3_adult_iemocap/best_model.pt ]; then
  plot_cm checkpoints/exp3_adult_iemocap/best_model.pt iemocap prosody_guided confusion_exp3_adult_iemocap
fi
if [ -f checkpoints/exp4_zero_shot_fau/best_model.pt ]; then
  plot_cm checkpoints/exp4_zero_shot_fau/best_model.pt c-besd prosody_guided confusion_exp4_zero_shot_fau \
    --test_data fau-aibo
fi
if [ -f checkpoints/exp5_fau_indomain/best_model.pt ]; then
  plot_cm checkpoints/exp5_fau_indomain/best_model.pt fau-aibo prosody_guided confusion_exp5_fau_indomain
fi
if [ -f checkpoints/exp5b_self_attention_fau/best_model.pt ]; then
  plot_cm checkpoints/exp5b_self_attention_fau/best_model.pt fau-aibo self_attention confusion_exp5b_self_attention_fau
fi

# ── C1: FAU multi-seed (additional seeds; seed=42 already trained above) ──
if [ -n "${SER_FAU_AIBO_PATH:-}" ]; then
  for s in $SEEDS_FAU; do
  if [ "$s" = "42" ]; then continue; fi
  echo "--- MULTISEED exp5 seed=$s ---"
  "$PYTHON_BIN" -m src.train \
    --train_data fau-aibo --pooling_type prosody_guided \
    --exp_name "exp5_fau_indomain_s${s}" \
    --output_dir "checkpoints/exp5_fau_indomain_s${s}" \
    --reg_profile fau --seed "$s"
  echo "--- MULTISEED exp5b seed=$s ---"
  "$PYTHON_BIN" -m src.train \
    --train_data fau-aibo --pooling_type self_attention \
    --exp_name "exp5b_self_attention_fau_s${s}" \
    --output_dir "checkpoints/exp5b_self_attention_fau_s${s}" \
    --reg_profile fau --seed "$s"
  done
  "$PYTHON_BIN" scripts/aggregate_fau_multiseed.py || true
fi

# ── C2: layer weights + APC + XAI from Exp2 Prosody checkpoint ──
if [ -f checkpoints/exp2_prosody_guided/best_model.pt ]; then
  echo "--- C2 diagnostics (Exp2 prosody) ---"
  "$PYTHON_BIN" scripts/run_prosody_diagnostics.py \
    --checkpoint checkpoints/exp2_prosody_guided/best_model.pt \
    --seed "$SEED_MAIN"
  cp -f results/layer_weights.json publication_package/layer_weights.json 2>/dev/null || true
fi

# ── Manifest ──
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
root = Path(".")
manifest = {"protocol": "ac_suite_2026-05", "artifacts": []}
for p in sorted(root.glob("checkpoints/**/best_model.pt")):
    manifest["artifacts"].append({"type": "checkpoint", "path": str(p)})
for p in sorted(root.glob("results/logs/*.json")):
    manifest["artifacts"].append({"type": "log", "path": str(p)})
for p in sorted(root.glob("results/figures/confusion_*")):
    manifest["artifacts"].append({"type": "figure", "path": str(p)})
Path("results/logs/ac_suite_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("manifest:", len(manifest["artifacts"]), "artifacts")
PY

echo "=== AC SUITE DONE $(date -Iseconds) log=$SUITE_LOG ==="
