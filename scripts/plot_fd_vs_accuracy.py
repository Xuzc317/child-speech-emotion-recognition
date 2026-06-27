"""生成 FD vs Accuracy 负相关曲线（论文图表）。

用法:
  python scripts/plot_fd_vs_accuracy.py
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

# 设置中文字体
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 数据
experiments = ['C1\n(No Aug)', 'C3\n(Child\nConstrained)', 'C2\n(Adult\nParams)', 'C4\n(Extreme\nParams)']
fd_values = [0, 8.71, 9.87, 11.99]
acc_values = [86.42, 66.85, 59.51, 43.25]
colors = ['#2ecc71', '#3498db', '#e74c3c', '#c0392b']

fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar chart for FD
bars = ax1.bar(experiments, fd_values, color=colors, alpha=0.7, width=0.5, label='Frechet Distance')
ax1.set_xlabel('Augmentation Configuration', fontsize=12)
ax1.set_ylabel('Frechet Distance (FD)', fontsize=12, color='#e74c3c')
ax1.tick_params(axis='y', labelcolor='#e74c3c')

# Add FD values on bars
for bar, val in zip(bars, fd_values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f'{val:.2f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold', color='#e74c3c')

# Line chart for Accuracy (secondary y-axis)
ax2 = ax1.twinx()
line = ax2.plot(experiments, acc_values, 'o-', color='#2980b9', linewidth=2.5, markersize=10,
                label='Val Accuracy (%)', markerfacecolor='white', markeredgewidth=2)
ax2.set_ylabel('Validation Accuracy (%)', fontsize=12, color='#2980b9')
ax2.tick_params(axis='y', labelcolor='#2980b9')
ax2.set_ylim(35, 95)

# Add accuracy labels
for i, (x, y) in enumerate(zip(experiments, acc_values)):
    ax2.annotate(f'{y:.2f}%', (x, y), textcoords="offset points", xytext=(0, 10),
                 ha='center', fontsize=11, fontweight='bold', color='#2980b9')

# Title and grid
ax1.set_title('Distribution Shift (FD) vs Emotion Recognition Accuracy\n'
              'BESD MY — WavLM + Adapter + Prosody Pooling',
              fontsize=14, fontweight='bold', pad=20)
ax1.grid(axis='y', alpha=0.3)

# Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

# Annotation
ax1.annotate('FD ↑  →  Accuracy ↓\nStrong negative correlation',
             xy=(2, 7), fontsize=11, color='#8e44ad',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5e6ff', alpha=0.8))

plt.tight_layout()
os.makedirs('assets/figures', exist_ok=True)
plt.savefig('assets/figures/fd_vs_accuracy.png', dpi=200, bbox_inches='tight')
plt.savefig('assets/figures/fd_vs_accuracy.pdf', bbox_inches='tight')
print("Saved to assets/figures/fd_vs_accuracy.png and .pdf")
plt.close()
