"""Generate publication-quality figures for the final HTML paper.
Design: clean academic style — no 3D, no heavy grids, colorblind-friendly palette.
"""
import json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'paper_draft' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

# ── Global Style ──────────────────────────────────────
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.unicode_minus': False,
    'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 9.5, 'legend.frameon': False,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.6,
    'grid.linestyle': '--', 'grid.linewidth': 0.4, 'grid.color': '#e0e0e0',
    'grid.alpha': 0.7,
})
# Colorblind-friendly palette (Wong, 2011)
C = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860']
sns.set_palette(sns.color_palette(C))

# ── Data ──────────────────────────────────────────────
EXP_LABELS = ['Exp1\nSelf-Attn\nC-BESD', 'Exp2\nProsody\nC-BESD', 'Exp3\nProsody\nIEMOCAP',
              'Exp4\nZero-Shot\nC-BESD→FAU', 'Exp5\nProsody\nFAU', 'Exp5b\nSelf-Attn\nFAU']
WA  = [92.78, 91.30, 58.67, 19.56, 66.36, 66.18]
UAR = [92.79, 91.35, 59.11, 23.83, 56.35, 58.23]

FAU_WA_M  = [65.15, 66.46]
FAU_WA_S  = [1.12, 0.72]
FAU_UAR_M = [53.95, 55.39]
FAU_UAR_S = [2.22, 2.05]
FAU_LAB   = ['Prosody-Guided\n(Exp5)', 'Self-Attention\n(Exp5b)']

FD_DATA = [
    ('In-domain acted\n(C-BESD)',         0.0,  92.78, 'Exp1'),
    ('Age shift\n(C-BESD vs IEMOCAP)',   7.20,  58.67, 'Exp3'),
    ('Style shift in-domain\n(C-BESD vs FAU)', 8.50, 66.36, 'Exp5'),
    ('Style shift zero-shot\n(C-BESD vs FAU)', 8.50, 19.56, 'Exp4'),
]

SHIFT_LAB = ['C-BESD\nvs C-BESD', 'C-BESD\nvs IEMOCAP', 'C-BESD\nvs FAU Aibo', 'C-BESD\nvs CREMA-D']
SHIFT_FD  = [0.0, 7.20, 8.50, 16.33]
SHIFT_SMMD =[0.0, 0.315, 0.378, 0.412]

LAYER_W = [0.0808, 0.0810, 0.0808, 0.0801, 0.0778, 0.0810,
           0.0876, 0.0888, 0.0912, 0.0854, 0.0824, 0.0832]

# ── Helper ────────────────────────────────────────────
def finish(ax, title=None):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', linewidth=0.4, color='#e0e0e0', alpha=0.7)
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)

# ── Fig 1: Main Results ───────────────────────────────
def fig1():
    fig, ax = plt.subplots(figsize=(9.5, 5))
    x = np.arange(len(EXP_LABELS))
    w = 0.35
    b1 = ax.bar(x - w/2, WA, w, color=C[0], edgecolor='white', linewidth=0.3)
    b2 = ax.bar(x + w/2, UAR, w, color=C[2], edgecolor='white', linewidth=0.3)
    for b in b1:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.8, f'{b.get_height():.1f}',
                ha='center', fontsize=7.5, fontweight='bold', color=C[0])
    for b in b2:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.8, f'{b.get_height():.1f}',
                ha='center', fontsize=7.5, color=C[2])
    ax.set_xticks(x)
    ax.set_xticklabels(EXP_LABELS, fontsize=8)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_ylim(0, 106)
    ax.legend(['WA', 'UAR'], loc='upper right', fontsize=10)
    ax.axhline(y=25, color='#999', linestyle=':', linewidth=1, alpha=0.6)
    ax.text(5.7, 26, 'Chance (25%)', fontsize=8.5, color='#999', ha='right')
    finish(ax, 'Figure 1: Main experimental results across six conditions.')
    fig.tight_layout()
    for fmt in ['png', 'pdf']:
        fig.savefig(OUT / f'fig01_main_results.{fmt}')
    plt.close(fig)

# ── Fig 2: FAU Multiseed ──────────────────────────────
def fig2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.2))
    for ax, v, s, yl, c in [(ax1, FAU_WA_M, FAU_WA_S, 'WA (%)', C[0]),
                              (ax2, FAU_UAR_M, FAU_UAR_S, 'UAR (%)', C[2])]:
        bars = ax.bar(FAU_LAB, v, color=[C[1], C[3]], edgecolor='white', linewidth=0.3, width=0.48)
        ax.errorbar(np.arange(len(FAU_LAB)), v, yerr=s, fmt='none', ecolor='#333',
                    capsize=5, capthick=1.2, linewidth=1.2)
        for b, val in zip(bars, v):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f'{val:.2f}%',
                    ha='center', fontsize=11, fontweight='bold')
        ax.set_xticklabels(FAU_LAB, fontsize=10)
        ax.set_ylabel(yl, fontsize=12)
        ax.set_ylim(min(v)-4, max(v)+6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', linewidth=0.4, color='#e0e0e0', alpha=0.7)
    ax1.set_title('Weighted Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Unweighted Avg. Recall', fontsize=12, fontweight='bold')
    fig.suptitle('Figure 2: FAU Aibo multi-seed comparison (Mean ± Std, n=3).', fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    for fmt in ['png', 'pdf']:
        fig.savefig(OUT / f'fig02_fau_multiseed.{fmt}')
    plt.close(fig)

# ── Fig 3: FD vs Accuracy ─────────────────────────────
def fig3():
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for i, (label, fd, wa, exp) in enumerate(FD_DATA):
        color = [C[0], C[2], C[1], C[3]][i]
        ax.scatter(fd, wa, s=160, c=color, edgecolors='#333', linewidth=0.8, zorder=5)
        ox, oy = (0.35, -6) if i == 2 else (0.35, 3.5)
        ax.annotate(f'{label}\nFD={fd:.1f}, WA={wa:.1f}%',
                    (fd, wa), (fd+ox, wa+oy), fontsize=8.5,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc', alpha=0.85, linewidth=0.5))
    # Trend line: same-corpus through age shift
    tx, ty = [0.0, 7.20, 8.50], [92.78, 58.67, 66.36]
    ax.plot(tx, ty, '--', color='#aaa', linewidth=1.2, alpha=0.5, zorder=1)
    ax.set_xlabel('Frechet Distance (FD)', fontsize=12)
    ax.set_ylabel('Best WA (%)', fontsize=12)
    ax.set_xlim(-0.6, 9.6)
    ax.set_ylim(0, 104)
    ax.axhline(y=25, color='#999', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.text(9.0, 26, 'Chance (25%)', fontsize=8.5, color='#999', ha='right')
    finish(ax, 'Figure 3: Distribution shift (FD) vs. classification accuracy (WA).')
    fig.tight_layout()
    for fmt in ['png', 'pdf']:
        fig.savefig(OUT / f'fig03_fd_accuracy.{fmt}')
    plt.close(fig)

# ── Fig 4: Distribution Shift ─────────────────────────
def fig4():
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    x = np.arange(len(SHIFT_LAB))
    w = 0.35
    b1 = ax.bar(x - w/2, SHIFT_FD, w, color=C[0], edgecolor='white', linewidth=0.3)
    ax2 = ax.twinx()
    b2 = ax2.bar(x + w/2, SHIFT_SMMD, w, color=C[2], edgecolor='white', linewidth=0.3)
    for b, v in zip(b1, SHIFT_FD):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.25, f'{v:.2f}',
                ha='center', fontsize=9.5, fontweight='bold', color=C[0])
    for b, v in zip(b2, SHIFT_SMMD):
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.005, f'{v:.3f}',
                 ha='center', fontsize=8.5, color=C[2])
    ax.set_xticks(x)
    ax.set_xticklabels(SHIFT_LAB, fontsize=9)
    ax.set_ylabel('FD', fontsize=12, color=C[0])
    ax2.set_ylabel('SMMD', fontsize=12, color=C[2])
    ax.tick_params(axis='y', colors=C[0])
    ax2.tick_params(axis='y', colors=C[2])
    ax.spines['top'].set_visible(False)
    ax.grid(True, linestyle='--', linewidth=0.4, color='#e0e0e0', alpha=0.7)
    finish(ax, 'Figure 4: Cross-corpus distribution shift diagnostics (FD and SMMD).')
    fig.tight_layout()
    for fmt in ['png', 'pdf']:
        fig.savefig(OUT / f'fig04_distribution_shift.{fmt}')
    plt.close(fig)

# ── Fig 5: Layer Fusion Weights ───────────────────────
def fig5():
    fig, ax = plt.subplots(figsize=(8, 3.8))
    layers = np.arange(1, 13)
    norm = plt.Normalize(0.077, 0.092)
    colors = plt.cm.Blues(0.3 + 0.6 * norm(LAYER_W))
    bars = ax.bar(layers, [w*100 for w in LAYER_W], color=colors, edgecolor='#555', linewidth=0.3)
    ax.axhline(y=100/12, color='#999', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.text(12.5, 100/12, f'Uniform ({100/12:.1f}%)', fontsize=8, color='#999', va='center')
    for l_idx, w in enumerate(LAYER_W):
        c = '#fff' if w > 0.088 else '#333'
        ax.text(l_idx+1, w*100+0.18, f'{w*100:.2f}%', ha='center', fontsize=8, color=c, fontweight='bold')
    ax.set_xlabel('WavLM Layer (1-based)', fontsize=12)
    ax.set_ylabel('Softmax Weight (%)', fontsize=12)
    ax.set_xticks(layers)
    ax.set_xticklabels([str(l) for l in layers])
    finish(ax, 'Figure 5: Learned softmax weights over WavLM layers 1-12 (peak at Layer 9).')
    fig.tight_layout()
    for fmt in ['png', 'pdf']:
        fig.savefig(OUT / f'fig05_layer_weights.{fmt}')
    plt.close(fig)

# ── Fig 6: C-BESD Comparison ──────────────────────────
def fig6():
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    lab = ['Self-Attention\n(Exp1)', 'Prosody-Guided\n(Exp2)']
    wa_c = [92.78, 91.30]
    uar_c = [92.79, 91.35]
    x = np.arange(len(lab))
    w = 0.32
    b1 = ax.bar(x-w/2, wa_c, w, color=C[0], edgecolor='white', linewidth=0.3)
    b2 = ax.bar(x+w/2, uar_c, w, color=C[2], edgecolor='white', linewidth=0.3)
    for b in b1:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.2, f'{b.get_height():.2f}%',
                ha='center', fontsize=10, fontweight='bold', color=C[0])
    for b in b2:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.2, f'{b.get_height():.2f}%',
                ha='center', fontsize=10, color=C[2])
    ax.set_xticks(x)
    ax.set_xticklabels(lab, fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_ylim(89, 94.5)
    ax.legend(['WA', 'UAR'], loc='lower left', fontsize=10)
    # Annotate gap
    ax.annotate('', xy=(0, 92.78), xytext=(1, 91.30),
                arrowprops=dict(arrowstyle='<->', color=C[3], lw=1.2))
    ax.text(0.5, 92.1, 'Δ = 1.48 pp', ha='center', fontsize=10, color=C[3], fontweight='bold')
    finish(ax, 'Figure 6: C-BESD acted corpus: self-attention vs prosody-guided pooling.')
    fig.tight_layout()
    for fmt in ['png', 'pdf']:
        fig.savefig(OUT / f'fig06_cbesd_comparison.{fmt}')
    plt.close(fig)

# ── Run ───────────────────────────────────────────────
if __name__ == '__main__':
    for f in [fig1, fig2, fig3, fig4, fig5, fig6]:
        f()
        print(f'{f.__name__} done')
    print(f'All figures saved to {OUT}')
