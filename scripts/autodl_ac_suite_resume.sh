#!/usr/bin/env bash
# Resume AC suite after A1 training (CM + C1 multiseed + C2). Idempotent.
set -euo pipefail
cd "${SER_PROJECT_ROOT:-/root/autodl-tmp/d-ser}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
SEED_MAIN=42
SEEDS_FAU="42 123 456"

pick_existing_dir() {
  for p in "$@"; do
    if [ -d "$p" ]; then echo "$p"; return 0; fi
  done
  return 1
}
export SER_C_BESD_PATH="${SER_C_BESD_PATH:-$(pick_existing_dir \
  /root/autodl-tmp/datasets/BESD/BESD/MY /root/autodl-tmp/BESD/BESD/MY || true)}"
export SER_IEMOCAP_PATH="${SER_IEMOCAP_PATH:-$(pick_existing_dir \
  /root/autodl-tmp/IEMOCAP/wavs /root/autodl-tmp/datasets/IEMOCAP/wavs || true)}"
export SER_FAU_AIBO_PATH="${SER_FAU_AIBO_PATH:-$(pick_existing_dir \
  /root/autodl-tmp/IS2009EmotionChallenge/wav \
  /root/autodl-tmp/IS2009EmotionChallenge/IS2009EmotionChallenge/IS2009EmotionChallenge/wav || true)}"
echo "SER_C_BESD_PATH=${SER_C_BESD_PATH:-MISSING}"
echo "SER_IEMOCAP_PATH=${SER_IEMOCAP_PATH:-MISSING}"
echo "SER_FAU_AIBO_PATH=${SER_FAU_AIBO_PATH:-MISSING}"

plot_cm() {
  local ckpt="$1" train="$2" pooling="$3" out="$4"
  shift 4
  echo "--- CM $out ---"
  "$PYTHON_BIN" scripts/plot_confusion_matrix.py \
    --checkpoint "$ckpt" --train_data $train --pooling_type "$pooling" \
    --seed "$SEED_MAIN" --output "results/figures/${out}" "$@"
}

mkdir -p results/figures publication_package ac_suite_logs
LOG="ac_suite_logs/resume_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "=== AC SUITE RESUME $(date -Iseconds) ==="

plot_cm checkpoints/exp1_self_attention/best_model.pt c-besd self_attention confusion_exp1_self_attention
plot_cm checkpoints/exp2_prosody_guided/best_model.pt c-besd prosody_guided confusion_exp2_prosody_guided
plot_cm checkpoints/exp3_adult_iemocap/best_model.pt iemocap prosody_guided confusion_exp3_adult_iemocap
plot_cm checkpoints/exp4_zero_shot_fau/best_model.pt c-besd prosody_guided confusion_exp4_zero_shot_fau --test_data fau-aibo
plot_cm checkpoints/exp5_fau_indomain/best_model.pt fau-aibo prosody_guided confusion_exp5_fau_indomain
plot_cm checkpoints/exp5b_self_attention_fau/best_model.pt fau-aibo self_attention confusion_exp5b_self_attention_fau

for s in $SEEDS_FAU; do
  if [ "$s" = "42" ]; then continue; fi
  if [ ! -f "results/logs/exp5_fau_indomain_s${s}.json" ]; then
    echo "--- MULTISEED exp5 seed=$s ---"
    "$PYTHON_BIN" -m src.train --train_data fau-aibo --pooling_type prosody_guided \
      --exp_name "exp5_fau_indomain_s${s}" --output_dir "checkpoints/exp5_fau_indomain_s${s}" \
      --reg_profile fau --seed "$s"
  fi
  if [ ! -f "results/logs/exp5b_self_attention_fau_s${s}.json" ]; then
    echo "--- MULTISEED exp5b seed=$s ---"
    "$PYTHON_BIN" -m src.train --train_data fau-aibo --pooling_type self_attention \
      --exp_name "exp5b_self_attention_fau_s${s}" --output_dir "checkpoints/exp5b_self_attention_fau_s${s}" \
      --reg_profile fau --seed "$s"
  fi
done
"$PYTHON_BIN" scripts/aggregate_fau_multiseed.py

echo "--- C2 diagnostics ---"
"$PYTHON_BIN" scripts/run_prosody_diagnostics.py \
  --checkpoint checkpoints/exp2_prosody_guided/best_model.pt --seed "$SEED_MAIN"
cp -f results/layer_weights.json publication_package/layer_weights.json 2>/dev/null || true

"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
root = Path(".")
manifest = {"protocol": "ac_suite_2026-05", "phase": "resume", "artifacts": []}
for p in sorted(root.glob("checkpoints/**/best_model.pt")):
    manifest["artifacts"].append({"type": "checkpoint", "path": str(p)})
for p in sorted(root.glob("results/logs/*.json")):
    manifest["artifacts"].append({"type": "log", "path": str(p)})
for p in sorted(root.glob("results/figures/confusion_*")):
    manifest["artifacts"].append({"type": "figure", "path": str(p)})
Path("results/logs/ac_suite_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("manifest:", len(manifest["artifacts"]), "artifacts")
PY

echo "=== AC SUITE RESUME DONE $(date -Iseconds) ==="
