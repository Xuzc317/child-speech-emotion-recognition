"""Generate publication figures from publication_package/ (A6).

Outputs PDF + PNG under paper_draft/figures/
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "publication_package"
OUT = ROOT / "paper_draft" / "figures"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  saved {name}.png / .pdf")


def fig_main_matrix() -> None:
    """Grouped WA / UAR for Exp1–5b."""
    df = pd.read_csv(PKG / "experiment_results.csv")
    labels = [
        "Exp1\nC-BESD\nSelf-Attn",
        "Exp2\nC-BESD\nProsody",
        "Exp3\nIEMOCAP\nProsody",
        "Exp4\nZS→FAU\nProsody",
        "Exp5\nFAU\nProsody",
        "Exp5b\nFAU\nSelf-Attn",
    ]
    wa = df["test_wa"].values * 100
    uar = df["test_uar"].values * 100
    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 4.5))
    b1 = ax.bar(x - w / 2, wa, w, label="WA (%)", color="#2E86AB", edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x + w / 2, uar, w, label="UAR (%)", color="#A23B72", edgecolor="white", linewidth=0.5)
    ax.axhline(25, color="#888", linestyle="--", linewidth=0.8, label="Chance (4-class)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Main experiment matrix (AutoDL final, autodl_final_2026-05-19)")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(axis="y", alpha=0.25)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7)
    _save(fig, "fig01_main_matrix_wa_uar")


def fig_fau_pooling_compare() -> None:
    """Exp5 vs Exp5b on FAU (fau reg)."""
    df = pd.read_csv(PKG / "experiment_results.csv")
    sub = df[df["experiment"].isin(["exp5_fau_indomain", "exp5b_self_attention_fau"])]
    names = ["Prosody\n(Exp5)", "Self-Attn\n(Exp5b)"]
    wa = sub["test_wa"].values * 100
    uar = sub["test_uar"].values * 100
    x = np.arange(2)
    w = 0.35
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(x - w / 2, wa, w, label="WA", color="#1B998B")
    ax.bar(x + w / 2, uar, w, label="UAR", color="#E84855")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(50, 70)
    ax.set_title("FAU Aibo in-domain (reg_profile=fau)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "fig02_fau_prosody_vs_selfattn")


def fig_fd_accuracy() -> None:
    """FD vs WA monotonic trend (paper Table)."""
    with (PKG / "fd_accuracy_table.json").open(encoding="utf-8") as f:
        data = json.load(f)
    rows = data["rows"]
    labels = [
        "In-domain\nacted (C-BESD)",
        "Age shift\n(child→adult)",
        "Style shift\n(spont., in-domain)",
        "Zero-shot\n(acted→FAU)",
    ]
    fd = [r["fd"] for r in rows]
    wa = [r["best_wa_percent"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    colors = ["#264653", "#2A9D8F", "#E9C46A", "#E76F51"]
    ax1.bar(range(len(labels)), wa, color=colors, edgecolor="white")
    ax1.set_ylabel("Best WA (%)", color="#264653")
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylim(0, 100)
    ax1.set_title("Distribution distance (FD) vs. accuracy")

    ax2 = ax1.twinx()
    ax2.plot(range(len(labels)), fd, "o-", color="#6D597A", linewidth=2, markersize=8, label="FD")
    ax2.set_ylabel("Fréchet distance (FD)", color="#6D597A")
    ax2.tick_params(axis="y", labelcolor="#6D597A")
    lines, labs = ax2.get_legend_handles_labels()
    ax1.legend(lines, labs, loc="center right")
    _save(fig, "fig03_fd_vs_accuracy")


def fig_xai_saliency() -> None:
    """3-panel: waveform, prosody, attention (from xai_raw_data.npz)."""
    npz = np.load(PKG / "xai_raw_data.npz")
    wav = npz["waveform"]
    t_wav = npz["time_wav"]
    t_attn = npz["time_attn"]
    f0 = npz["f0_contour"]
    rms = npz["rms_contour"]
    attn = npz["attention_weights"]

    f0_n = f0 / (np.max(f0) + 1e-8)
    rms_n = rms / (np.max(rms) + 1e-8)
    attn_n = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=False)
    axes[0].plot(t_wav, wav, color="#555", linewidth=0.4)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("(a) Waveform")
    axes[0].set_xlabel("Time (s)")

    axes[1].plot(t_attn, f0_n, label="F0 (norm.)", color="#1f77b4", linewidth=1.2)
    axes[1].plot(t_attn, rms_n, label="RMS (norm.)", color="#ff7f0e", linewidth=1.2)
    axes[1].set_ylabel("Prosody")
    axes[1].set_title("(b) Frame-level prosody")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_xlabel("Time (s)")

    axes[2].fill_between(t_attn, attn_n, color="#d62728", alpha=0.5)
    axes[2].plot(t_attn, attn_n, color="#b2182b", linewidth=1)
    axes[2].set_ylabel("Attention")
    axes[2].set_title("(c) Prosody-guided attention weights")
    axes[2].set_xlabel("Time (s)")

    apc_path = PKG / "logs" / "apc_metrics.json"
    if apc_path.exists():
        with apc_path.open(encoding="utf-8") as f:
            apc = json.load(f)
        fig.suptitle(
            f"XAI saliency (APC_wav = {apc.get('apc_wav', 0):.3f})",
            fontsize=12,
            y=1.02,
        )
    plt.tight_layout()
    _save(fig, "fig04_xai_saliency_triple")


def fig_cbesd_pooling_compare() -> None:
    """Exp1 vs Exp2 on C-BESD."""
    df = pd.read_csv(PKG / "experiment_results.csv")
    sub = df[df["experiment"].isin(["exp1_self_attention", "exp2_prosody_guided"])]
    names = ["Self-Attn\n(Exp1)", "Prosody\n(Exp2)"]
    wa = sub["test_wa"].values * 100
    epochs = sub["best_epoch"].values
    fig, ax = plt.subplots(figsize=(4.5, 4))
    colors = ["#457B9D", "#E63946"]
    bars = ax.bar(names, wa, color=colors, edgecolor="white")
    ax.set_ylabel("Test WA (%)")
    ax.set_ylim(85, 95)
    ax.set_title("Acted child speech (C-BESD, default reg)")
    for bar, ep in zip(bars, epochs):
        ax.annotate(f"epoch {ep}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "fig05_cbesd_selfattn_vs_prosody")


def write_figure_manifest() -> None:
    manifest = {
        "generated_by": "scripts/generate_paper_figures.py",
        "data_sources": [
            "publication_package/experiment_results.csv",
            "publication_package/fd_accuracy_table.json",
            "publication_package/xai_raw_data.npz",
            "publication_package/logs/apc_metrics.json",
        ],
        "figures": [
            {"file": "fig00_system_architecture", "use": "System pipeline (see generate_architecture_figure.py)"},
            {"file": "fig01_main_matrix_wa_uar", "use": "Main results overview"},
            {"file": "fig02_fau_prosody_vs_selfattn", "use": "Pillar 2: naturalistic FAU"},
            {"file": "fig03_fd_vs_accuracy", "use": "Unified distribution-shift framework"},
            {"file": "fig04_xai_saliency_triple", "use": "Pillar 3: explainability"},
            {"file": "fig05_cbesd_selfattn_vs_prosody", "use": "Acted C-BESD trade-off"},
        ],
        "missing": [
            "fig_layer_fusion_weights (needs checkpoint export)",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "FIGURES_MANIFEST.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"  saved FIGURES_MANIFEST.json")


def main() -> None:
    print(f"Output directory: {OUT}")
    fig_main_matrix()
    fig_fau_pooling_compare()
    fig_fd_accuracy()
    fig_xai_saliency()
    fig_cbesd_pooling_compare()
    write_figure_manifest()
    print("Done.")


if __name__ == "__main__":
    main()
