"""生成论文整体框架图 (增强版) —— 包含公式、维度、双分支池化、SEMLP内部结构。

用法: python scripts/fig_architecture_v2.py
输出: paper_draft/figures/fig00_architecture_v2.png (和 .pdf)
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper_draft" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── 配色 ──
C_WAVEFORM = "#E8E8E8"
C_WAVLM = "#D4E4F7"
C_FUSION = "#C5E0B4"
C_SELFATTN = "#92C5DE"
C_PROSODY = "#F4A582"
C_PROSODY_IN = "#FFE0B2"
C_SEMLP = "#E6CCE6"
C_OUTPUT = "#FFF3B0"
C_ARROW = "#555555"
C_BORDER = "#333333"

# ── 绘图工具 ──
def box(ax, xy, w, h, text, fc, ec=C_BORDER, fontsize=7.5, title_fontsize=8.5,
        bold_title=True, sub_texts=None):
    """绘制圆角矩形框，支持多行文本（第一行作为标题加粗）"""
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        linewidth=1.0, edgecolor=ec, facecolor=fc, zorder=2,
    )
    ax.add_patch(patch)

    lines = text.split("\n")
    total_lines = len(lines)
    line_h = h / (total_lines + 1) * 0.85

    for i, line in enumerate(lines):
        if i == 0 and bold_title:
            ax.text(x + w / 2, y + h - (i + 1) * line_h - h * 0.08,
                    line, ha="center", va="center", fontsize=title_fontsize,
                    fontweight="bold", zorder=3)
        else:
            ax.text(x + w / 2, y + h - (i + 1) * line_h - h * 0.08,
                    line, ha="center", va="center", fontsize=fontsize, zorder=3)

    if sub_texts:
        for sx, sy, stext, ssize in sub_texts:
            ax.text(sx, sy, stext, ha="center", va="center", fontsize=ssize,
                    style="italic", color="#666666", zorder=3)


def arrow(ax, p0, p1, color=C_ARROW, style="-|>", lw=0.8):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=10,
        linewidth=lw, color=color, zorder=1,
    ))


def dashed_group(ax, xy, w, h, label):
    """虚线圈出分组"""
    x, y = xy
    rect = plt.Rectangle((x, y), w, h, fill=False, linestyle="--",
                          edgecolor="#999999", linewidth=0.8, zorder=0)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h + 0.03, label, ha="center", va="bottom",
            fontsize=7, color="#666666", style="italic")


def main():
    fig = plt.figure(figsize=(18, 7))
    ax = fig.add_axes([0.02, 0.05, 0.96, 0.90])
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # ── 尺寸常量 ──
    BH = 1.1       # 主模块高度
    BW = [1.2, 1.8, 1.4, 3.2, 2.0, 0.8]  # 各模块宽度
    GAP = 0.5      # 模块间距
    Y_MAIN = 3.8   # 主行 y 坐标

    # 各模块的 x 起始坐标
    xs = [0.3]
    for i in range(5):
        xs.append(xs[-1] + BW[i] + GAP)

    # ── ① Waveform ──
    box(ax, (xs[0], Y_MAIN), BW[0], BH,
        "Waveform\n16kHz mono\n4s · 200 frames\n(B, T_wav=64000)",
        C_WAVEFORM, title_fontsize=8.5)

    # ── ② WavLM Base ──
    box(ax, (xs[1], Y_MAIN), BW[1], BH,
        "WavLM Base\nwavlm-base-sv\n12-layer Transformer\n94.6M params · FROZEN\noutput: 13 hidden states\nH₀..H₁₂, each (T,768)",
        C_WAVLM)

    # ── ③ LayerFusion ──
    lf_text = "WavLM LayerFusion\n12 learnable weights\nwᵢ = softmax(θᵢ)\nF = Σ wᵢ·Hᵢ\n→ (B, T, 768)"
    box(ax, (xs[2], Y_MAIN), BW[2], BH, lf_text, C_FUSION)

    # ── ④ Pooling (双分支) ──
    # 分支高度布局
    BRANCH_H = 1.3
    Y_PROSODY = Y_MAIN + 1.3
    Y_SELFATTN = Y_MAIN - 0.5

    # 虚线分组框
    dashed_group(ax, (xs[3] - 0.05, Y_SELFATTN - 0.2),
                 BW[3] + 0.1, (Y_PROSODY - Y_SELFATTN) + BRANCH_H + 0.2,
                 "111,105 params each (strictly matched)")

    # Self-Attention Pooling
    sa_text = ("Self-Attention Pooling\n"
               "aₜ = MLP(fₜ)\n"
               "   [768→116→100→100→1]\n"
               "αₜ = softmax(aₜ)\n"
               "z = Σ αₜ·fₜ  →  (B, 768)")
    box(ax, (xs[3], Y_SELFATTN), BW[3], BRANCH_H, sa_text, C_SELFATTN, fontsize=6.5, title_fontsize=7.5)

    # Prosody Guided Pooling
    pg_text = ("Prosody Guided Pooling\n"
               "pₜ = MLP_pros([f₀⁽ᵗ⁾; e⁽ᵗ⁾])  [2→64→64]\n"
               "cₜ = [fₜ; pₜ]                 832-dim\n"
               "aₜ = MLP_fusion(cₜ)   [832→128→1]\n"
               "αₜ = softmax(aₜ),  z = Σ αₜ·fₜ  →  (B, 768)")
    box(ax, (xs[3], Y_PROSODY), BW[3], BRANCH_H, pg_text, C_PROSODY, fontsize=6.2, title_fontsize=7.5)

    # F0 + RMS 输入 (Prosody 的侧输入)
    box(ax, (xs[3] - 0.8, Y_PROSODY + BRANCH_H + 0.15), 0.75, 0.55,
        "F0 (YIN)\nRMS (librosa)", C_PROSODY_IN, fontsize=6, title_fontsize=7)

    arrow(ax, (xs[3] - 0.05, Y_PROSODY + BRANCH_H + 0.42),
          (xs[3] + 0.3, Y_PROSODY + BRANCH_H + 0.05), style="-", lw=0.6)

    # ── ⑤ SEMLP ──
    se_text = ("SEMLP Classifier\n"
               "768→512→SE(32)→256→128→4\n"
               "BN+ReLU+Dropout(0.3/0.2)\n"
               "SEBlock: h⊙σ(W₂·ReLU(W₁·h))\n"
               "~593K params")
    box(ax, (xs[4], Y_MAIN), BW[4], BH, se_text, C_SEMLP, fontsize=6.2)

    # ── ⑥ Output ──
    emotions = ["angry", "happy", "neutral", "sad"]
    colors_e = ["#E74C3C", "#F1C40F", "#95A5A6", "#3498DB"]
    for i, (emo, ce) in enumerate(zip(emotions, colors_e)):
        ey = Y_MAIN + BH - 0.15 - i * 0.25
        ax.add_patch(plt.Rectangle((xs[5] + 0.05, ey), 0.6, 0.2,
                                    facecolor=ce, edgecolor=C_BORDER, linewidth=0.5, zorder=3))
        ax.text(xs[5] + 0.35, ey + 0.1, emo, ha="center", va="center", fontsize=6.5, zorder=4)

    box(ax, (xs[5], Y_MAIN), BW[5], BH, "", "#FFFFFF00", ec="none")

    # ── 箭头 ──
    arrow(ax, (xs[0] + BW[0], Y_MAIN + BH / 2), (xs[1], Y_MAIN + BH / 2))
    arrow(ax, (xs[1] + BW[1], Y_MAIN + BH / 2), (xs[2], Y_MAIN + BH / 2))
    arrow(ax, (xs[2] + BW[2], Y_MAIN + BH / 2), (xs[3], Y_MAIN + BH / 2))

    # 分支箭头：从中间分叉到两个 pooling
    mid_x = xs[3] - 0.05
    mid_y = Y_MAIN + BH / 2
    arrow(ax, (mid_x, mid_y), (xs[3] + 0.15, Y_SELFATTN + BRANCH_H / 2))
    arrow(ax, (mid_x, mid_y), (xs[3] + 0.15, Y_PROSODY + BRANCH_H / 2))

    # 分支汇合到 SEMLP
    arrow(ax, (xs[3] + BW[3], Y_SELFATTN + BRANCH_H / 2), (xs[4], Y_MAIN + BH * 0.7))
    arrow(ax, (xs[3] + BW[3], Y_PROSODY + BRANCH_H / 2), (xs[4], Y_MAIN + BH * 0.3))
    arrow(ax, (xs[4] + BW[4], Y_MAIN + BH / 2), (xs[5], Y_MAIN + BH / 2))

    # ── 维度标注 ──
    dim_annotations = [
        (xs[0] + BW[0] / 2, Y_MAIN - 0.25, "(B, 64000)", 6.5),
        (xs[1] + BW[1] / 2, Y_MAIN - 0.25, "13 × (B, T, 768)", 6.5),
        (xs[2] + BW[2] / 2, Y_MAIN - 0.25, "(B, T, 768)", 6.5),
    ]
    for dx, dy, dt, ds in dim_annotations:
        ax.text(dx, dy, dt, ha="center", va="top", fontsize=ds, color="#888888", style="italic")

    # ── 底部参数汇总 ──
    ax.text(9, 0.3,
            "Total Trainable Params ≈ 704K  |  WavLM Backbone: 94.6M (frozen, < 1% trained)  |  "
            "Optimizer: AdamW (lr=3e-4)  |  Loss: Label-Smoothing CE (ε=0.1 or 0.15)",
            ha="center", va="center", fontsize=7.5, style="italic", color="#555555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8F8F8", edgecolor="#CCCCCC"))

    # ── 右上角引用 ──
    ax.text(17.5, 6.7, "WavLM: Chen et al. 2022\nF0/YIN: de Cheveigné & Kawahara 2002\nRMS: librosa (McFee et al. 2015)\nSE-Block: Hu et al. CVPR 2018",
            ha="right", va="top", fontsize=5.5, color="#999999")

    # ── 标题 (English only to avoid CJK font issues) ──
    fig.suptitle("Distribution-Driven Child Speech Emotion Recognition Pipeline",
                 fontsize=12, fontweight="bold", y=0.98)

    # ── 保存 ──
    for ext in ("png", "pdf"):
        fpath = OUT / f"fig00_architecture_v2.{ext}"
        fig.savefig(fpath, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {fpath}")

    plt.close(fig)


if __name__ == "__main__":
    main()
