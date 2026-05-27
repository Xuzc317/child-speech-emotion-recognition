"""Plot confusion matrix from a train.py checkpoint (when available locally).

Example:
  python scripts/plot_confusion_matrix.py \\
    --checkpoint checkpoints/autodl/best_model.pt \\
    --train_data c-besd --pooling_type prosody_guided \\
    --output paper_draft/figures/fig06_confusion_exp2_prosody
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.train import SERModel, evaluate  # noqa: E402
from src.data import get_dataloaders, get_cross_corpus_dataloaders  # noqa: E402

CLASSES = ["angry", "happy", "neutral", "sad"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train_data", nargs="+", default=["c-besd"])
    parser.add_argument("--test_data", nargs="+", default=None)
    parser.add_argument("--pooling_type", default=None, help="Override if not in checkpoint")
    parser.add_argument("--output", default="paper_draft/figures/fig06_confusion")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        print("Sync from AutoDL: python scripts/tmp_paramiko_autodl_runner.py --sync-checkpoints")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    if args.pooling_type:
        config["pooling_type"] = args.pooling_type

    model = SERModel(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if args.test_data:
        dls = get_cross_corpus_dataloaders(
            args.train_data, args.test_data, batch_size=args.batch_size, seed=args.seed,
        )
    else:
        dls = get_dataloaders(args.train_data, batch_size=args.batch_size, seed=args.seed)

    _, _, _, preds, labels = evaluate(model, dls["test"])
    cm = confusion_matrix(labels, preds, labels=list(range(4)))

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(CLASSES)
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix ({ckpt_path.name})")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)

    report = classification_report(
        labels, preds, labels=list(range(4)), target_names=CLASSES, digits=3, zero_division=0
    )
    meta = {
        "checkpoint": str(ckpt_path),
        "train_data": args.train_data,
        "test_data": args.test_data or args.train_data,
        "pooling_type": config.get("pooling_type"),
        "confusion_matrix": cm.tolist(),
    }
    with out.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(report)
    print(f"Saved {out}.png / .pdf / .json")


if __name__ == "__main__":
    main()
