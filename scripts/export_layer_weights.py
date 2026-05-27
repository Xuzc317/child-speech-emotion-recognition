"""Export softmax WavLM layer fusion weights from a train.py checkpoint.

Usage:
  python scripts/export_layer_weights.py --checkpoint checkpoints/exp2_prosody_guided/best_model.pt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt from src.train")
    parser.add_argument(
        "--output",
        default=str(ROOT / "results" / "layer_weights.json"),
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "AutoDL did not sync checkpoints locally. Re-sync from remote or re-train."
        )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    key = "layer_fusion.layer_weights"
    if key not in state:
        raise KeyError(f"{key} not in checkpoint. Keys sample: {list(state.keys())[:8]}")

    raw = state[key].float()
    weights = torch.softmax(raw, dim=0).numpy().tolist()
    argmax = int(torch.argmax(raw).item())  # 0-indexed transformer layer (1..12 → index 0..11)
    # Paper uses 1-indexed "Layer 8" = index 7
    entropy = float(-(torch.softmax(raw, dim=0) * torch.log_softmax(raw, dim=0)).sum().item())

    out = {
        "layer_weights": weights,
        "argmax_layer_index_0based": argmax,
        "argmax_layer_1based": argmax + 1,
        "entropy": entropy,
        "checkpoint": str(ckpt_path),
        "epoch": ckpt.get("epoch"),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Saved {out_path}")
    print(f"  argmax layer (1-based): {out['argmax_layer_1based']}, entropy={entropy:.4f}")


if __name__ == "__main__":
    main()
