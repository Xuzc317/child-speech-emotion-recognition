"""Recompute layer weights, APC, and XAI bundle from a Prosody-Guided checkpoint (C2).

Uses the same test-sample protocol as training (C-BESD, seed=42).
Outputs:
  - results/layer_weights.json
  - results/logs/apc_metrics.json
  - publication_package/xai_raw_data.npz
  - results/xai_final.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.extract_diagnostics import DiagnosticWrapper  # noqa: E402
from src.evaluation import AttentionProsodyExplainer  # noqa: E402
from src.data import get_dataloaders  # noqa: E402


def export_layer_weights(checkpoint_path: Path, output: Path) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    key = "layer_fusion.layer_weights"
    if key not in state:
        raise KeyError(f"{key} missing in {checkpoint_path}")
    raw = state[key].float()
    weights = torch.softmax(raw, dim=0).numpy().tolist()
    argmax_0 = int(torch.argmax(raw).item())
    entropy = float(-(torch.softmax(raw, dim=0) * torch.log_softmax(raw, dim=0)).sum().item())
    out = {
        "layer_weights": weights,
        "argmax_layer_index_0based": argmax_0,
        "argmax_layer_1based": argmax_0 + 1,
        "entropy": entropy,
        "checkpoint": str(checkpoint_path),
        "epoch": ckpt.get("epoch"),
        "protocol": "ac_suite_2026-05",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/exp2_prosody_guided/best_model.pt",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer-out", default=str(ROOT / "results" / "layer_weights.json"))
    parser.add_argument("--apc-out", default=str(ROOT / "results" / "logs" / "apc_metrics.json"))
    parser.add_argument(
        "--xai-npz",
        default=str(ROOT / "publication_package" / "xai_raw_data.npz"),
    )
    parser.add_argument("--xai-png", default=str(ROOT / "results" / "xai_final.png"))
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lw = export_layer_weights(ckpt_path, Path(args.layer_out))
    print(f"layer_weights: argmax_layer_1based={lw['argmax_layer_1based']}, entropy={lw['entropy']:.4f}")

    model = DiagnosticWrapper(str(ckpt_path))
    dls = get_dataloaders(["c-besd"], batch_size=1, seed=args.seed)
    waveform = next(iter(dls["test"]))[0][0]
    _, attn, f0, rms, wav_np = model.forward_with_attention(waveform)

    explainer = AttentionProsodyExplainer(sr=16000, ssl_frame_rate=50)
    apc = explainer.compute_apc(attn, f0, rms)
    apc_rec = {
        "pooling_type": "prosody_guided",
        "checkpoint": str(ckpt_path),
        "seed": args.seed,
        "protocol": "ac_suite_2026-05",
        **apc,
    }
    apc_path = Path(args.apc_out)
    apc_path.parent.mkdir(parents=True, exist_ok=True)
    apc_path.write_text(json.dumps(apc_rec, indent=2) + "\n", encoding="utf-8")
    print(f"APC_wav={apc['apc_wav']:.4f}, APC_delta={apc['apc_delta']:.4f}")

    xai_npz = Path(args.xai_npz)
    xai_npz.parent.mkdir(parents=True, exist_ok=True)
    explainer.plot_saliency(
        wav_np,
        f0,
        rms,
        attn,
        save_path=args.xai_png,
        title="XAI Saliency Map (prosody_guided, C-BESD test sample)",
    )
    # plot_saliency also writes results/xai_raw_data.npz — copy to publication path
    default_npz = ROOT / "results" / "xai_raw_data.npz"
    if default_npz.exists() and default_npz.resolve() != xai_npz.resolve():
        xai_npz.write_bytes(default_npz.read_bytes())
    print(f"saved {xai_npz}")


if __name__ == "__main__":
    main()
