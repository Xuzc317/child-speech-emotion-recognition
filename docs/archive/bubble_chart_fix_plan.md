# Bubble Chart Fix Plan

**Author**: Agent 3 (Evaluation & Fix Plan)
**Target**: Agent 2 (Implementation)
**Date**: 2026-06-12
**Status**: Ready for execution

---

## 1. Current State Audit

### 1.1 Existing Output Files

| File | Dimensions | Description |
|------|-----------|-------------|
| `fig_acoustic_bubbles.png` | 2300×2297 | Combined: 3 datasets × 2 features (6 panels, 3×2 grid) |
| `fig_acoustic_bubbles_cbesd.png` | 1928×825 | C-BESD: 2 panels side-by-side (F0 left, RMS right) |
| `fig_acoustic_bubbles_fauaibo.png` | 1928×825 | FAU-AIBO: same layout |
| `fig_acoustic_bubbles_iemocap.png` | 1928×825 | IEMOCAP: same layout |
| `fig_acoustic_bubbles_c-besd_f0.png` | 1380×977 | C-BESD F0 only (single panel) |
| `fig_acoustic_bubbles_c-besd_rms.png` | 1380×977 | C-BESD RMS only |
| `fig_acoustic_bubbles_iemocap_f0.png` | 1379×977 | IEMOCAP F0 only |
| `fig_acoustic_bubbles_iemocap_rms.png` | 1379×977 | IEMOCAP RMS only |

Missing: `fauaibo_f0.png` and `fauaibo_rms.png` are not generated as separate per-feature files.

### 1.2 Generation Script

`scripts/plot_acoustic_bubbles.py` — 411 lines.

Core plotting function (`draw_bubble_panel`, lines 287–308):

```python
def draw_bubble_panel(ax, matrix, title):
    emotions = UNIFIED_4CLASS
    colors = plt.cm.tab10(np.linspace(0, 1, len(emotions)))
    max_prop = max(matrix[b].get(e, 0) for b in range(5) for e in emotions)
    max_prop = max(max_prop, 1e-6)

    for bi in range(5):
        for ei, emo in enumerate(emotions):
            prop = matrix[bi].get(emo, 0)
            if prop > 0:
                size = 80 + 2000 * (prop / max_prop)
                ax.scatter(bi, ei, s=size, c=[colors[ei]], alpha=0.75,
                           edgecolors='black', linewidth=0.5, zorder=3)

    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, len(emotions) - 0.5)
    ax.set_xticks(range(5))
    ax.set_xticklabels(BIN_LABELS, fontsize=7, rotation=30, ha='right')
    ax.set_yticks(range(len(emotions)))
    ax.set_yticklabels([e.capitalize() for e in emotions], fontsize=8)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
```

---

## 2. Gap Analysis: Current vs. Reference (Tsangko Fig.1)

### 2.1 Reference Style (Tsangko et al. 2026, Fig.1)

Based on the extracted description and the project study notes (`docs/实验设计方案_v3_含学习笔记.md`):

| Property | Tsangko Fig.1 |
|----------|---------------|
| Rows | 6 acoustic features (ENERGY, PITCH, BRIGHTNESS, DYNAMICS, FORMANTS, VOICE_QUALITY) |
| Columns | 4 emotions (angry, happy, neutral, sad) |
| Per-cell content | 5 bubbles, one per quantile bin (VERY_LOW — VERY_HIGH) |
| Bubble sizing | Area proportional to frequency proportion |
| Color | tab10 colormap to distinguish emotions |
| Style | Very clean, easy-to-read, minimal clutter |

### 2.2 Our Current Implementation

| Property | Our Code | Problem |
|----------|----------|---------|
| Layout | Per-dataset: 2 panels side-by-side (F0 left, RMS right) | Not a per-dataset figure with features-as-rows |
| Axes within panel | X = bin level (5 ticks), Y = emotion (4 ticks) | Emotions should be columns not Y-axis |
| Bubble sizing | `size = 80 + 2000 * (prop / max_prop)` | Offset `+80` creates spurious bubbles for zero-proportion cells |
| Proportionality | Linear in diameter (not area) | Matplotlib `s` parameter is area; the linear formula maps diameter not area to proportion — the offset makes this worse |
| Colors | tab10 colored by emotion (4 colors) | With emotions as columns, bubbles should be colored by BIN LEVEL |
| Font sizes | xlabels 7pt, ylabels 8pt, title 9pt | Too small for print |
| Grid | Default matplotlib dashed grid | Not polished |
| Number of features | 2 (F0, RMS) vs 6 in reference | Acceptable for our scope but needs clear labeling |

### 2.3 Critical Disconnect with Paper Description

The paper draft (`v3_complete.tex`, lines 706–714) describes a **joint F0×RMS distribution** discretized into 5×5 bins. However, the current code computes **independent marginal distributions** (F0 and RMS separately). This means:

- Paper says: 25 bins per cell (5 F0 levels × 5 RMS levels), bubble shows proportion falling in that 2D bin
- Code does: 5 bins per cell (either F0-only or RMS-only), no joint distribution

The user's instructions resolve this by directing us toward the simpler Tsangko-style marginal approach (one feature per row, independent distributions).

---

## 3. Fix Specifications

### 3.1 Target Layout

**One figure per dataset.** Each figure contains a 2×4 grid:

```
             Angry        Happy        Neutral        Sad
F0 (Hz)     [cell 1]     [cell 2]     [cell 3]      [cell 4]
RMS Energy  [cell 5]     [cell 6]     [cell 7]      [cell 8]
```

**Within each cell**: 5 bubbles arranged vertically, one per quantile bin:
- Top to bottom: VERY_HIGH → VERY_LOW (or vice versa, be consistent)
- Bubble area ∝ proportion of utterances/frames in that bin for that (feature, emotion) pair
- Bubbles colored by bin level (sequential or categorical colormap)
- Empty cells (proportion = 0) show NO bubble

### 3.2 Bubble Sizing (Critical Fix)

Replace:
```python
size = 80 + 2000 * (prop / max_prop)
```

With area-proportional sizing:
```python
# Option A: Global max across all cells in the figure
global_max = max of all proportions across all 2×4 cells
if prop > 0:
    size = MAX_BUBBLE_SIZE * (prop / global_max)  # area ∝ proportion
    # MAX_BUBBLE_SIZE suggestion: 800–1200 (matplotlib s units)
```

Key rules:
1. **No offset** — zero proportion = no bubble drawn
2. **Area proportional to frequency** — `s ∝ prop` (matplotlib `s` is area, so linear mapping is correct)
3. **Consistent scaling across all cells** in one figure — use a single `global_max` for all 8 cells

### 3.3 Color Scheme

Since columns already encode emotion, color bubbles by **bin level**:

```python
# Bin-level colormap (5 bins)
BIN_COLORS = {
    'VERY_LOW':  '#2166ac',  # dark blue
    'LOW':       '#67a9cf',  # medium blue
    'MID':       '#f7f7f7',  # near-white (or light gray with dark edge)
    'HIGH':      '#ef8a62',  # medium red
    'VERY_HIGH': '#b2182b',  # dark red
}
```

Or use a perceptually uniform sequential colormap:
```python
from matplotlib.colors import LinearSegmentedColormap
bin_cmap = plt.cm.RdYlBu_r  # reversed: blue (low) → red (high)
bin_colors = [bin_cmap(i / 4) for i in range(5)]
```

For the "MID" bin (index 2), use a lighter edge color or slightly darker fill to ensure visibility against a white background.

### 3.4 Per-Cell Axis Design

Each cell is a small axes with:
- **X-axis**: Not labeled (hidden ticks). Bubbles are arranged vertically within the cell.
  - ALL bubbles at x=0 (stacked vertically)
  - Or spread slightly: bin 0 at x=-0.2, bin 4 at x=+0.2 to create a gentle horizontal spread (like a miniature beeswarm)
- **Y-axis**: Implicit order (VERY_LOW at bottom, VERY_HIGH at top, or vice versa)
  - Hidden ticks; order is conveyed by color gradient
- **X limits**: [-0.5, 0.5] for centered single-column; or [-0.6, 0.6] for slight spread
- **Y limits**: [0, 4] with bins at integer positions

**Preferred approach** (simpler, cleaner): All 5 bubbles in a vertical stack at x=0 within each cell. Color gradient conveys the bin ordering.

### 3.5 Cell Borders and Grid

Each cell gets a very thin light gray border (spine) to visually separate the grid:
```python
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('#cccccc')
    spine.set_linewidth(0.5)
```

### 3.6 Row and Column Labels

- **Row labels** (left side): "F0 (Hz)" and "RMS Energy" — placed as ylabel on the leftmost axes of each row, or as a shared text annotation
- **Column labels** (top): "Angry", "Happy", "Neutral", "Sad" — placed as title on the topmost axes of each column
- **Figure title** (top): Dataset name + sample count, e.g., "C-BESD (n=XXX test utterances)"
- **Bin legend** (right side or below): 5 colored dots with labels VERy_LOW through VERY_HIGH

### 3.7 Typography

| Element | Current | Target |
|---------|---------|--------|
| Column labels | N/A | 11pt, bold |
| Row labels | N/A | 11pt, bold |
| Bin labels (legend) | 7pt | 9pt |
| Figure title | 9pt | 12pt, bold |
| Tick labels (if any) | 7-8pt | Removed or 8pt |

### 3.8 Figure Dimensions and Output

- **Figure size**: 10" × 4.5" (width × height) for the 2×4 grid
- **DPI**: 200 (keep as-is)
- **Format**: PNG, RGBA
- **File naming**:
  - `fig_acoustic_bubbles_cbesd.png` — C-BESD 2×4 grid
  - `fig_acoustic_bubbles_fauaibo.png` — FAU-AIBO 2×4 grid
  - `fig_acoustic_bubbles_iemocap.png` — IEMOCAP 2×4 grid
- **No combined figure** — each dataset is standalone

### 3.9 Data Structure

Reuse the existing `compute_bubble_data()` function. The returned structure:
```python
{
    'f0':  {bin_idx: {emotion: proportion}},   # 5 bins × 4 emotions
    'rms': {bin_idx: {emotion: proportion}},
    'n_valid': int,
    'n_total': int
}
```
is sufficient. No changes needed to the data pipeline — only the plotting function must change.

---

## 4. Implementation Checklist (for Agent 2)

### Phase 1: Rewrite `draw_bubble_panel` → `draw_bubble_figure`

- [ ] **P1.1** Create new function `draw_dataset_bubble_figure(bubble_data, dataset_key, output_path)` that draws the 2×4 grid for one dataset
- [ ] **P1.2** Use `plt.subplots(2, 4, figsize=(10, 4.5))` to create the grid
- [ ] **P1.3** Row 0 = F0 data, Row 1 = RMS data; Columns 0–3 = angry, happy, neutral, sad
- [ ] **P1.4** Each cell: call new `draw_single_cell(ax, proportions, bin_colors)` that plots 5 bubbles vertically at x=0

### Phase 2: Fix Bubble Sizing

- [ ] **P2.1** Compute `global_max` = max of all proportions across all 8 cells in the figure
- [ ] **P2.2** Set `MAX_BUBBLE_SIZE = 800` (tunable parameter)
- [ ] **P2.3** Use `size = MAX_BUBBLE_SIZE * (prop / global_max)` — no offset
- [ ] **P2.4** Skip drawing when `prop == 0`

### Phase 3: Fix Color Scheme

- [ ] **P3.1** Define `bin_colors` list of 5 RGBA tuples using `plt.cm.RdYlBu_r`
- [ ] **P3.2** Apply bin color (not emotion color) to each bubble
- [ ] **P3.3** Ensure MID bin (index 2) is visible — use `edgecolors='#888888'` if fill is too light

### Phase 4: Cell Styling

- [ ] **P4.1** Remove X and Y tick labels within each cell
- [ ] **P4.2** Set `ax.set_xticks([]); ax.set_yticks([])`
- [ ] **P4.3** Set X limits [-0.5, 0.5], Y limits [-0.5, 4.5] for 5-bin vertical arrangement
- [ ] **P4.4** Configure spines as thin light gray border
- [ ] **P4.5** Remove or hide the matplotlib grid within cells

### Phase 5: Row/Column Labels and Legend

- [ ] **P5.1** Set column titles on Row 0 axes: "Angry", "Happy", "Neutral", "Sad" (11pt, bold)
- [ ] **P5.2** Set row labels on Column 0 axes as ylabels: "F0 (Hz)", "RMS Energy" (11pt, bold)
- [ ] **P5.3** Add figure-level suptitle: dataset name + n (12pt, bold), e.g., "C-BESD (n=XXX)"
- [ ] **P5.4** Add a shared legend for the 5 bin levels — place at bottom-center or right of figure

### Phase 6: Output

- [ ] **P6.1** Generate 3 output files: one per dataset
- [ ] **P6.2** Use `fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')`
- [ ] **P6.3** Remove or deprecate the combined figure (or keep as separate optional output)
- [ ] **P6.4** Remove per-feature single-panel outputs (`*_f0.png`, `*_rms.png`) — the new per-dataset 2×4 figure supersedes them

### Phase 7: Paper Draft Alignment

- [ ] **P7.1** Update `v3_complete.tex` Figure X caption to describe the new 2×4 per-dataset layout
- [ ] **P7.2** Note: The paper described a 5×5 joint F0-RMS distribution. The new marginal approach (F0 row, RMS row) is simpler and closer to Tsangko's style. Update the text accordingly.

---

## 5. Reference Pseudocode

```python
def draw_single_cell(ax, bin_proportions, bin_colors, global_max, MAX_SIZE=800):
    """
    Draw 5 bubbles vertically within one cell.
    
    Args:
        ax: matplotlib Axes
        bin_proportions: dict {bin_idx: proportion} for 5 bins
        bin_colors: list of 5 RGBA color tuples
        global_max: float, max proportion across all cells (for consistent scaling)
        MAX_SIZE: int, matplotlib scatter size for the largest bubble
    """
    for bin_idx in range(5):
        prop = bin_proportions.get(bin_idx, 0.0)
        if prop <= 0:
            continue
        size = MAX_SIZE * (prop / global_max)
        color = bin_colors[bin_idx]
        edge_color = '#888888' if bin_idx == 2 else 'none'  # outline for MID bin
        ax.scatter(
            0, bin_idx,           # x=0 center, y=bin_idx (stacked vertically)
            s=size,
            c=[color],
            alpha=0.85,
            edgecolors=edge_color,
            linewidth=0.5,
            zorder=3
        )
    
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_xticks([])
    ax.set_yticks([])
    # Light gray cell border
    for spine in ax.spines.values():
        spine.set_color('#dddddd')
        spine.set_linewidth(0.5)


def draw_dataset_bubble_figure(bubble_data, dataset_key, output_path):
    """Draw 2×4 grid bubble chart for one dataset."""
    name_map = {'c-besd': 'C-BESD', 'fau-aibo': 'FAU-AIBO', 'iemocap': 'IEMOCAP'}
    ds_name = name_map.get(dataset_key, dataset_key)
    n = bubble_data['n_valid']
    
    emotions = ['angry', 'happy', 'neutral', 'sad']
    emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad']
    features = ['f0', 'rms']
    feature_labels = ['F0 (Hz)', 'RMS Energy']
    
    bin_colors = [plt.cm.RdYlBu_r(i / 4) for i in range(5)]
    
    # Compute global max for consistent bubble sizing
    global_max = 0.0
    for feat in features:
        matrix = bubble_data[feat]
        for bin_idx in range(5):
            for emo in emotions:
                global_max = max(global_max, matrix[bin_idx].get(emo, 0.0))
    global_max = max(global_max, 1e-6)
    
    fig, axes = plt.subplots(2, 4, figsize=(10, 4.5))
    
    for row, feat in enumerate(features):
        matrix = bubble_data[feat]
        for col, emo in enumerate(emotions):
            ax = axes[row, col]
            proportions = {bi: matrix[bi].get(emo, 0.0) for bi in range(5)}
            draw_single_cell(ax, proportions, bin_colors, global_max)
            
            # Column labels (top row only)
            if row == 0:
                ax.set_title(emotion_labels[col], fontsize=11, fontweight='bold', pad=8)
        
        # Row labels
        axes[row, 0].set_ylabel(feature_labels[row], fontsize=11, fontweight='bold')
    
    fig.suptitle(f'{ds_name} (n={n})', fontsize=12, fontweight='bold', y=1.02)
    
    # Add bin legend below the figure
    bin_labels = ['VL', 'L', 'M', 'H', 'VH']  # or full names
    # ... legend construction code ...
    
    fig.tight_layout(pad=1.5)
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
```

---

## 6. Edge Cases and Notes

1. **IEMOCAP gender-aware binning**: The current code does gender-specific F0 discretization for IEMOCAP (lines 211–243). This logic should be preserved in the data computation phase. The plotting phase is unaffected.

2. **Small sample datasets**: If a dataset has very few utterances for a specific emotion class, the proportions may be noisy. This is acceptable — the bubble chart is exploratory, not inferential.

3. **The MID bin visibility**: The middle bin (index 2) may appear nearly white when using `RdYlBu_r` at its center. Always add a thin edge outline for this bin.

4. **Global max edge case**: If all proportions are zero (should not happen with valid data), `global_max` defaults to `1e-6` to avoid division by zero. In this case all bubbles would have size 0 and nothing is drawn — acceptable.

5. **FAU-AIBO label file**: The FAU label file path is hardcoded. No change needed unless the path changes on the target machine.

---

## 7. Summary of Changes

| # | Severity | Issue | Fix |
|---|----------|-------|-----|
| 1 | **Critical** | Bubble sizing formula `80 + 2000*x` creates spurious bubbles and non-proportional sizing | Remove offset; use `s ∝ prop` with global max |
| 2 | **Critical** | Layout: bins on X, emotions on Y — wrong for per-dataset per-emotion comparison | Restructure to 2 rows × 4 columns, features on rows, emotions on columns |
| 3 | **High** | Colors keyed to emotion (redundant with column position) | Color by bin level using sequential colormap |
| 4 | **High** | Missing FAU-AIBO per-feature plots | Superseded by new per-dataset 2×4 layout |
| 5 | **Medium** | Font sizes too small (7-9pt) | Increase to 9-12pt |
| 6 | **Medium** | Grid styling is default matplotlib | Clean spines, no internal grid lines |
| 7 | **Medium** | No per-cell bin arrangement logic | Bubbles stacked vertically at x=0 within each cell |
| 8 | **Low** | Combined figure includes all 3 datasets | Keep as optional; primary output is per-dataset figures |
| 9 | **Low** | Per-feature individual plots partly missing | Remove; 2×4 grid is the canonical output |
