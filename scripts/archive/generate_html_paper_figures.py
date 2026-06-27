"""Generate all figures for the HTML paper draft. Outputs to paper_draft/figures_html/."""
import json, csv, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)
# Use a font stack that supports both Latin and CJK characters
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial'],
    'axes.unicode_minus': False,
    'mathtext.fontset': 'dejavusans',
    'axes.titlesize': 14, 'axes.labelsize': 13,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})
OUT = Path(__file__).resolve().parent.parent / 'paper_draft' / 'figures_html'
OUT.mkdir(parents=True, exist_ok=True)

ROOT = Path(__file__).resolve().parent.parent
PALETTE = sns.color_palette("colorblind", 6)

# ── Data ────────────────────────────────────────────
EXPS = ['Exp1\nSelf-Attn\nC-BESD', 'Exp2\nProsody\nC-BESD', 'Exp3\nProsody\nIEMOCAP',
        'Exp4\nZero-Shot\nC-BESD→FAU', 'Exp5\nProsody\nFAU', 'Exp5b\nSelf-Attn\nFAU']
WA_vals  = [92.78, 91.30, 58.67, 19.56, 66.36, 66.18]
UAR_vals = [92.79, 91.35, 59.11, 23.83, 56.35, 58.23]

# FAU multiseed
fau_labels = ['Exp5\nProsody', 'Exp5b\nSelf-Attn']
fau_wa_mean  = [65.15, 66.46]
fau_wa_std   = [1.12, 0.72]
fau_uar_mean = [53.95, 55.39]
fau_uar_std  = [2.22, 2.05]

# FD-Accuracy
fd_data = [
    ('域内表演\n(C-BESD)', 0.0, 92.78, 'Exp1'),
    ('年龄偏移\n(→IEMOCAP)', 7.20, 58.67, 'Exp3'),
    ('风格偏移域内\n(→FAU)', 8.50, 66.36, 'Exp5'),
    ('风格偏移零样本\n(→FAU)', 8.50, 19.56, 'Exp4'),
]

# Layer weights
layer_weights = [
    0.08081564, 0.08096483, 0.08079173, 0.08010375, 0.07780716,
    0.08099194, 0.08756305, 0.08877978, 0.09116169, 0.08540043,
    0.08240680, 0.08321330,
]

# Distribution shift
shift_labels = ['C-BESD\nvs C-BESD', 'C-BESD\nvs IEMOCAP', 'C-BESD\nvs FAU Aibo', 'C-BESD\nvs CREMA-D']
shift_fd   = [0.0, 7.20, 8.50, 16.33]
shift_smmd = [0.0, 0.315, 0.378, 0.412]

# ── Fig 1: Main Results Bar Chart ──────────────────
def fig1():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(EXPS))
    w = 0.35
    bars1 = ax.bar(x - w/2, WA_vals, w, label='WA (%)', color=PALETTE[0], edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + w/2, UAR_vals, w, label='UAR (%)', color=PALETTE[2], edgecolor='white', linewidth=0.5)
    for b in bars1:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.6, f'{b.get_height():.1f}',
                ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    for b in bars2:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.6, f'{b.get_height():.1f}',
                ha='center', va='bottom', fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(EXPS, fontsize=8)
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', frameon=True, fancybox=False)
    ax.axhline(y=25, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.text(5.6, 26, 'Chance (25%)', fontsize=8, color='gray', ha='right')
    ax.set_title('图1：6组实验主结果矩阵（WA 与 UAR 对比）', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT/'fig01_main_results.png')
    plt.close(fig)

# ── Fig 2: FAU Pooling Comparison with error bars ──
def fig2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
    for ax, vals, stds, ylab in [(ax1, fau_wa_mean, fau_wa_std, 'WA (%)'),
                                   (ax2, fau_uar_mean, fau_uar_std, 'UAR (%)')]:
        bars = ax.bar(fau_labels, vals, yerr=stds, capsize=6, color=[PALETTE[1], PALETTE[3]],
                      edgecolor='white', linewidth=0.8, width=0.5)
        for b, v, s in zip(bars, vals, stds):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+s+0.8, f'{v:.2f}%',
                    ha='center', fontsize=11, fontweight='bold')
        ax.set_ylabel(ylab)
        ax.set_ylim(min(vals)-5, max(vals)+8)
    ax1.set_title('加权准确率 (WA)', fontsize=12)
    ax2.set_title('非加权平均召回率 (UAR)', fontsize=12)
    fig.suptitle('图2：FAU Aibo 自然式语音上两种池化方式对比（3种子，Mean±Std）', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT/'fig02_fau_multiseed.png')
    plt.close(fig)

# ── Fig 3: FD vs Accuracy Scatter ──
def fig3():
    fig, ax = plt.subplots(figsize=(7.5, 5))
    fds = [d[1] for d in fd_data]
    was = [d[2] for d in fd_data]
    names = [d[0] for d in fd_data]
    colors = [PALETTE[0], PALETTE[2], PALETTE[1], PALETTE[3]]
    for i in range(len(fd_data)):
        ax.scatter(fds[i], was[i], s=180, c=[colors[i]], edgecolors='black', linewidth=1.2, zorder=5)
        offset_y = 3.5 if i != 2 else -7
        offset_x = 0.25 if i != 3 else 0.25
        ha = 'left'
        ax.annotate(f'{names[i]}\nFD={fds[i]:.1f}, WA={was[i]:.1f}%',
                    (fds[i], was[i]), (fds[i]+offset_x, was[i]+offset_y),
                    fontsize=8.5, ha=ha,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=0.5))

    # Add connecting line to show monotonic trend (from same-corpus through age shift to style shift)
    trend_x = [0.0, 7.20, 8.50]
    trend_y = [92.78, 58.67, 66.36]
    ax.plot(trend_x, trend_y, '--', color='gray', alpha=0.4, linewidth=1.5, zorder=1)

    ax.set_xlabel('Frechet Distance (FD)')
    ax.set_ylabel('Best WA (%)')
    ax.set_title('图3：分布偏移 (FD) 与分类准确率 (WA) 的关系', fontsize=13, fontweight='bold')
    ax.set_xlim(-0.8, 10)
    ax.set_ylim(0, 105)
    ax.axhline(y=25, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.text(9.3, 26, 'Chance', fontsize=8, color='gray')
    fig.tight_layout()
    fig.savefig(OUT/'fig03_fd_accuracy.png')
    plt.close(fig)

# ── Fig 4: Distribution Shift Bar ──
def fig4():
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(shift_labels))
    w = 0.35
    bars_fd = ax.bar(x - w/2, shift_fd, w, label='FD', color=PALETTE[0], edgecolor='white')
    ax2 = ax.twinx()
    bars_sm = ax2.bar(x + w/2, shift_smmd, w, label='SMMD', color=PALETTE[2], edgecolor='white')
    for b, v in zip(bars_fd, shift_fd):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f'{v:.2f}',
                ha='center', fontsize=9, fontweight='bold')
    for b, v in zip(bars_sm, shift_smmd):
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.005, f'{v:.3f}',
                 ha='center', fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(shift_labels, fontsize=8.5)
    ax.set_ylabel('FD', color=PALETTE[0])
    ax2.set_ylabel('SMMD', color=PALETTE[2])
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, loc='upper left', frameon=True)
    ax.set_title('图4：跨语料库分布偏移诊断 (FD 与 SMMD)', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT/'fig04_distribution_shift.png')
    plt.close(fig)

# ── Fig 5: Layer Fusion Weights ──
def fig5():
    fig, ax = plt.subplots(figsize=(8, 4))
    layers = np.arange(1, 13)
    cmap = plt.cm.Blues
    norm = plt.Normalize(0.076, 0.093)
    colors = cmap(norm(layer_weights))
    bars = ax.bar(layers, [w*100 for w in layer_weights], color=colors, edgecolor='gray', linewidth=0.5)
    ax.axhline(y=100/12, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.text(12.3, 100/12, f'均匀\n{100/12:.1f}%', fontsize=8, color='gray', va='center')
    for i, (l, w) in enumerate(zip(layers, layer_weights)):
        c = 'white' if w > 0.088 else 'black'
        ax.text(l, w*100+0.15, f'{w*100:.2f}%', ha='center', fontsize=8, color=c, fontweight='bold')
    ax.set_xlabel('WavLM 层编号 (1-based)')
    ax.set_ylabel('Softmax 权重 (%)')
    ax.set_xticks(layers)
    ax.set_xticklabels([str(l) for l in layers])
    ax.set_title('图5：WavLM 12层可学习融合权重分布（峰值在第9层）', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT/'fig05_layer_weights.png')
    plt.close(fig)

# ── Fig 6: C-BESD Pooling Comparison ──
def fig6():
    fig, ax = plt.subplots(figsize=(5.5, 4))
    labels = ['Self-Attention\n(Exp1)', 'Prosody-Guided\n(Exp2)']
    wa_vals_c = [92.78, 91.30]
    uar_vals_c = [92.79, 91.35]
    x = np.arange(len(labels))
    w = 0.3
    b1 = ax.bar(x-w/2, wa_vals_c, w, label='WA', color=PALETTE[0], edgecolor='white')
    b2 = ax.bar(x+w/2, uar_vals_c, w, label='UAR', color=PALETTE[2], edgecolor='white')
    for b in b1:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f'{b.get_height():.2f}%',
                ha='center', fontsize=10, fontweight='bold')
    for b in b2:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f'{b.get_height():.2f}%',
                ha='center', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(88, 95)
    ax.legend(loc='lower left')
    ax.text(0.5, 91.5, 'Δ = 1.48 pp', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title('图6：C-BESD 表演式语音上两种池化对比', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT/'fig06_cbesd_comparison.png')
    plt.close(fig)

# ── Run all ──
if __name__ == '__main__':
    fig1(); print('fig1 done')
    fig2(); print('fig2 done')
    fig3(); print('fig3 done')
    fig4(); print('fig4 done')
    fig5(); print('fig5 done')
    fig6(); print('fig6 done')
    print(f'All figures saved to {OUT}')
