"""Bar plot of WavLM layer fusion weights from layer_weights.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "results" / "layer_weights.json"
OUT = ROOT / "paper_draft" / "figures" / "fig06_layer_fusion_weights"


def main() -> None:
    path = DEFAULT
    if not path.exists():
        path = ROOT / "publication_package" / "layer_weights.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    w = np.array(data["layer_weights"])
    layers = np.arange(1, len(w) + 1)
    argmax = int(data.get("argmax_layer_1based", data.get("argmax_layer", 0) + 1))

    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = ["#E76F51" if i + 1 == argmax else "#457B9D" for i in range(len(w))]
    ax.bar(layers, w, color=colors, edgecolor="white")
    ax.set_xlabel("WavLM transformer layer (1–12)")
    ax.set_ylabel("Softmax weight")
    ax.set_title(f"Learned layer fusion (peak: Layer {argmax}, entropy={data.get('entropy', 0):.2f})")
    ax.set_xticks(layers)
    ax.grid(axis="y", alpha=0.25)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}.png / .pdf")


if __name__ == "__main__":
    main()
