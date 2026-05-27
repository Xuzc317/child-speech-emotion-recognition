"""Re-evaluate Exp4 zero-shot with test-only split (N=3389) and regenerate CM.

Fixes the split='all' → split='test' issue for cross-corpus zero-shot evaluation.
The original Exp4 JSON test_wa=18.70% was computed on all 18216 FAU samples.
The corrected test_wa uses only the held-out test split (3389 samples, 9 speakers).

Outputs:
  - Updated results/logs/exp4_zero_shot_fau.json (with test_split field)
  - paper_draft/figures/figA2_confusion_exp4_zero_shot.* (PNG/PDF/JSON)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.train import SERModel, evaluate  # noqa: E402
from src.data import get_cross_corpus_dataloaders  # noqa: E402

CLASSES = ["angry", "happy", "neutral", "sad"]
CHECKPOINT_PATH = ROOT / "checkpoints" / "autodl" / "exp4_zero_shot_fau" / "best_model.pt"
JSON_PATH = ROOT / "results" / "logs" / "exp4_zero_shot_fau.json"
FIG_OUT = ROOT / "paper_draft" / "figures" / "figA2_confusion_exp4_zero_shot"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not CHECKPOINT_PATH.exists():
        alt = ROOT / "checkpoints" / "exp4_zero_shot_fau" / "best_model.pt"
        if alt.exists():
            ckpt_path = alt
        else:
            print(f"ERROR: checkpoint not found at {CHECKPOINT_PATH}")
            sys.exit(1)
    else:
        ckpt_path = CHECKPOINT_PATH

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    print(f"  pooling_type={config.get('pooling_type')}, config={list(config.keys())}")

    model = SERModel(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Re-evaluate with test-only split
    print("Building dataloaders with test_split='test' (N=3389 expected)...")
    dls = get_cross_corpus_dataloaders(
        ["c-besd"], ["fau-aibo"],
        batch_size=16, seed=42,
        test_split="test",  # <-- the fix
    )
    test_ds = dls["test"].dataset
    print(f"  Test samples: {len(test_ds)} (expected 3389)")

    print("Evaluating on test split...")
    test_loss, test_wa, test_uar, test_preds, test_labels = evaluate(model, dls["test"])
    print(f"  test_wa={test_wa:.4f} ({test_wa*100:.2f}%)")
    print(f"  test_uar={test_uar:.4f} ({test_uar*100:.2f}%)")

    # Compare with original
    if JSON_PATH.exists():
        orig = json.loads(JSON_PATH.read_text(encoding="utf-8"))
        print(f"\n  ORIGINAL (split='all', 18216 samples): test_wa={orig['test_wa']:.4f} ({orig['test_wa']*100:.2f}%)")
        print(f"  CORRECTED (split='test', {len(test_ds)} samples): test_wa={test_wa:.4f} ({test_wa*100:.2f}%)")
        print(f"  Delta: {(test_wa - orig['test_wa'])*100:+.2f}pp")

    # Save updated JSON (preserve original fields, update metrics)
    if JSON_PATH.exists():
        updated = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    else:
        updated = {}
    updated["test_wa"] = float(test_wa)
    updated["test_uar"] = float(test_uar)
    updated["test_split"] = "test"
    updated["test_n"] = len(test_ds)
    updated["test_split_note"] = "Re-evaluated 2026-05-27 with test_split='test' (was split='all'=18216). Speaker-independent test speakers only."
    updated["original_test_wa_all_split"] = updated.get("test_wa", None)  # backup if needed
    JSON_PATH.write_text(json.dumps(updated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  Updated: {JSON_PATH}")

    # Generate confusion matrix
    cm = confusion_matrix(test_labels, test_preds, labels=list(range(4)))
    cm_sum = cm.sum()
    print(f"\n  Confusion matrix sum: {cm_sum} (expected {len(test_ds)})")

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(CLASSES)
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Exp4 Zero-Shot C-BESD→FAU (test-only, N={cm_sum})")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_OUT.with_suffix(f".{ext}"), bbox_inches="tight", dpi=150)
    plt.close(fig)

    report = classification_report(
        test_labels, test_preds, labels=list(range(4)),
        target_names=CLASSES, digits=3, zero_division=0,
    )
    cm_meta = {
        "checkpoint": str(ckpt_path),
        "train_data": ["c-besd"],
        "test_data": ["fau-aibo"],
        "test_split": "test",
        "test_n": int(cm_sum),
        "pooling_type": config.get("pooling_type"),
        "confusion_matrix": cm.tolist(),
        "note": "Recomputed 2026-05-27 with test_split='test'. Previous version used split='all' (18216 samples).",
    }
    with FIG_OUT.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(cm_meta, f, indent=2)
    print(f"  Saved {FIG_OUT}.png / .pdf / .json")
    print(report)
    print("\nDONE: Exp4 re-evaluation complete.")


if __name__ == "__main__":
    main()
