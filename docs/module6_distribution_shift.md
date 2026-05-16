# Module 6: Distribution Shift Analytics (FD & SMMD Probes)

## Overview

Upgraded distribution shift diagnostic tools to include a non-parametric alternative to Frechet Distance. The dual-metric `DistributionShiftProbe` combines FD (parametric, Gaussian assumption) and SMMD (non-parametric, kernel-based) for rigorous out-of-domain quantification.

## Mathematical Definitions

### Frechet Distance (FD)

FD is the squared Wasserstein-2 distance under the multivariate Gaussian assumption:

$$\text{FD}^2(\mathcal{N}_1, \mathcal{N}_2) = \|\mu_1 - \mu_2\|^2 + \text{Tr}\left(\Sigma_1 + \Sigma_2 - 2(\Sigma_1 \Sigma_2)^{1/2}\right)$$

where:
- $\mu_1, \mu_2 \in \mathbb{R}^D$ are the sample means
- $\Sigma_1, \Sigma_2 \in \mathbb{R}^{D \times D}$ are the sample covariance matrices
- $(\Sigma_1 \Sigma_2)^{1/2}$ is the matrix square root

**Limitation:** Assumes WavLM embeddings follow a multivariate normal distribution — an assumption often violated in practice.

### Speech Maximum Mean Discrepancy (SMMD)

Non-parametric kernel two-sample test. No distributional assumptions.

The **unbiased MMD² estimate** with kernel $k(\cdot, \cdot)$:

$$\text{MMD}^2_u(X, Y) = \frac{1}{n(n-1)}\sum_{i \neq j} k(x_i, x_j) + \frac{1}{m(m-1)}\sum_{i \neq j} k(y_i, y_j) - \frac{2}{nm}\sum_{i=1}^{n}\sum_{j=1}^{m} k(x_i, y_j)$$

where:
- $X = \{x_1, ..., x_n\}$ are reference samples ($n \times D$)
- $Y = \{y_1, ..., y_m\}$ are target samples ($m \times D$)

**Gaussian RBF Kernel:**

$$k(x, y) = \exp\left(-\frac{\|x - y\|^2}{2\sigma^2}\right)$$

**Bandwidth selection:** Median heuristic — $\sigma = \text{median}(\{\|x_i - y_j\| : i \in [1, n], j \in [1, m]\})$

**SMMD** is defined as $\sqrt{\max(\text{MMD}^2, 0)}$ for interpretability on a comparable scale to FD.

## Dual-Metric Comparison

| Property | FD | SMMD |
|----------|-----|------|
| Distribution assumption | Multivariate Gaussian | None (non-parametric) |
| Complexity | $O(D^3)$ (matrix sqrt) | $O((n+m)^2 D)$ (kernel) |
| Sensitive to | Mean + covariance shifts | Any distributional shift |
| Zero for identical data | Yes (within numerical precision) | Yes |
| Robust to non-normality | No | Yes |

## Code Location

| File | Component | Description |
|------|-----------|-------------|
| `src/evaluation/distribution_metrics.py:33` | `compute_frechet_distance()` | FD (Wasserstein-2) |
| `src/evaluation/distribution_metrics.py:59` | `_rbf_kernel()` | Gaussian RBF kernel matrix |
| `src/evaluation/distribution_metrics.py:72` | `_median_heuristic()` | Bandwidth selection |
| `src/evaluation/distribution_metrics.py:94` | `compute_smmd()` | Unbiased MMD² with RBF |
| `src/evaluation/distribution_metrics.py:154` | `DistributionShiftProbe` | Unified diagnostic class |
| `src/evaluation/distribution_metrics.py:193` | `extract_features()` | WavLM feature extraction |
| `src/evaluation/distribution_metrics.py:234` | `evaluate()` | Compute both metrics |
| `src/evaluation/distribution_metrics.py:246` | `evaluate_shift()` | High-level API |

## Usage

### Standalone metrics

```python
from src.evaluation import compute_frechet_distance, compute_smmd
import numpy as np

feats_child = np.random.randn(500, 768)      # C-BESD
feats_adult = np.random.randn(500, 768) + 1   # IEMOCAP (shifted)

fd = compute_frechet_distance(feats_child, feats_adult)
smmd = compute_smmd(feats_child, feats_adult)
# smmd with custom bandwidth:
smmd_custom = compute_smmd(feats_child, feats_adult, sigma=0.5)
```

### Full diagnostic probe with WavLM backbone

```python
from src.evaluation import DistributionShiftProbe
from src.models import SSLBackbone
from src.data import get_dataloaders

backbone = SSLBackbone(model_name='wavlm', frozen=True, device='cuda')
probe = DistributionShiftProbe(backbone=backbone, pooling='mean', device='cuda')

dataloaders_child = get_dataloaders(['c-besd'], batch_size=32, seed=42)
dataloaders_adult = get_dataloaders(['iemocap'], batch_size=32, seed=42)

result = probe.evaluate_shift(
    dataloaders_child['train'],
    dataloaders_adult['train'],
    max_samples=500,
)
# → {'fd': 6.87, 'smmd': 0.42}
```

## Dry-Run Results

```
Setup: 100 samples × 768-dim, shift = N(0,1) vs N(0.5,1)

FD   = 1284.0106    (> 0, detects shift)
SMMD = 0.2624       (> 0, detects shift)

Self-test (identical distributions):
FD(self)   = 0.000000  ✓
SMMD(self) = 0.000000  ✓

Monotonicity (shift 0.5 → 2.0):
FD:   1284.01  →  4157.92  (3.2× increase)  ✓
SMMD: 0.2624  →  0.6926   (2.6× increase)   ✓

All NaN/Inf checks: PASS
```

## Three-Dimensional Shift Taxonomy

Using the dual metrics, distribution shifts can be classified along three axes:

| Axis | Reference | Target | Expected FD | Expected SMMD |
|------|-----------|--------|-------------|---------------|
| Age | C-BESD (child) | CREMA-D/IEMOCAP (adult) | ~6.87 | moderate |
| Augmentation | Clean child | +AWGN 10-20dB | 8-12 | low-moderate |
| Language | C-BESD-EN | C-BESD-TE (Telugu) | ~16.48 | high |

FD and SMMD should be monotonically correlated across these conditions, providing convergent validity for out-of-domain detection.
