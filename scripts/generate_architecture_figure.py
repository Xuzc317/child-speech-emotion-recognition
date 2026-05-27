"""Generate fig00_system_architecture (block diagram for paper)."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper_draft" / "figures"


def _box(ax, xy, w, h, text, fc, ec="#333"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, wrap=True)


def main():
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis("off")

    # Row y centers
    y_main = 1.55
    h = 0.85
    gap = 0.35

    _box(ax, (0.2, y_main), 1.0, h, "Waveform\n16 kHz", "#E8E8E8")
    _box(ax, (1.5, y_main), 1.3, h, "WavLM Base+\n(frozen)", "#D4E4F7")
    _box(ax, (3.1, y_main), 1.2, h, "Layer Fusion\n12 weights", "#C5E0B4")
    _box(ax, (4.6, y_main), 1.5, h, "Frame feats\n(B,T,768)", "#F5F5F5")

    # Pooling split
    y_p = 2.35
    y_s = 0.75
    _box(ax, (6.4, y_p), 1.55, h, "Prosody Pooling\nF0+RMS → attn", "#F4A582")
    _box(ax, (6.4, y_s), 1.55, h, "Self-Attn Pooling\nSSL only", "#92C5DE")
    ax.text(7.175, 1.55, "111,105\nparams each", ha="center", va="center", fontsize=7, style="italic")

    _box(ax, (8.2, y_main), 1.0, h, "SE-MLP\n~590K", "#E6CCE6")
    _box(ax, (9.4, y_main), 0.5, h, "4-class\nlogits", "#FFF3B0")

    def arrow(p0, p1):
        ax.add_patch(FancyArrowPatch(
            p0, p1, arrowstyle="-|>", mutation_scale=10,
            linewidth=1.0, color="#444",
        ))

    arrow((1.2, y_main + h / 2), (1.5, y_main + h / 2))
    arrow((2.8, y_main + h / 2), (3.1, y_main + h / 2))
    arrow((4.3, y_main + h / 2), (4.6, y_main + h / 2))
    arrow((6.1, y_main + h / 2), (6.4, y_p + h / 2))
    arrow((6.1, y_main + h / 2), (6.4, y_s + h / 2))
    arrow((7.95, y_p + h / 2), (8.2, y_main + h * 0.65))
    arrow((7.95, y_s + h / 2), (8.2, y_main + h * 0.35))
    arrow((9.2, y_main + h / 2), (9.4, y_main + h / 2))

    # Prosody side input
    _box(ax, (4.6, 2.55), 1.2, 0.55, "librosa\nF0, RMS", "#FFE0B2")
    arrow((5.2, 2.55), (6.4, y_p + h))

    ax.set_title("Distribution-driven child SER pipeline", fontsize=11, pad=8)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig00_system_architecture.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved fig00_system_architecture to {OUT}")


if __name__ == "__main__":
    main()
