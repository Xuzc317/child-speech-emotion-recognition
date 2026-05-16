"""Distribution Shift Analytics: FD & SMMD Dual-Metric Probes.

Frechet Distance (FD):
  Parametric — assumes multivariate Gaussian. Fast but potentially
  unreliable when WavLM embeddings fail normality tests.

Speech Maximum Mean Discrepancy (SMMD):
  Non-parametric — uses RBF kernel MMD. No distributional assumptions.
  More robust for high-dimensional speech embeddings.

Combined, they provide rigorous quantification of out-of-domain
distribution shifts (adult vs. child, clean vs. augmented, etc.).
"""

import numpy as np
from scipy import linalg
from typing import Dict, Optional
import torch


# ============================================================
# FD: Frechet Distance (Wasserstein-2 under Gaussian assumption)
# ============================================================

def compute_frechet_distance(
    feats_ref: np.ndarray,
    feats_tgt: np.ndarray,
) -> float:
    """Frechet Distance between two feature distributions.

    FD² = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2(Σ₁Σ₂)^{1/2})

    Under the Gaussian assumption, this equals the squared
    Wasserstein-2 distance.

    Args:
        feats_ref: (N, D) reference distribution features
        feats_tgt: (M, D) target distribution features

    Returns:
        FD scalar (≥ 0). Lower = more similar distributions.
    """
    mu1 = feats_ref.mean(axis=0)
    mu2 = feats_tgt.mean(axis=0)
    sigma1 = np.cov(feats_ref, rowvar=False)
    sigma2 = np.cov(feats_tgt, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fd2 = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(max(fd2, 0.0))


# ============================================================
# SMMD: Speech Maximum Mean Discrepancy (non-parametric)
# ============================================================

def _rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian RBF kernel matrix.

    K_{ij} = exp(-||x_i - y_j||² / (2 * σ²))

    Args:
        X: (N, D) first set
        Y: (M, D) second set
        sigma: kernel bandwidth

    Returns:
        (N, M) kernel matrix
    """
    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)  # (N, 1)
    Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)  # (1, M)
    dist_sq = X_norm + Y_norm - 2 * (X @ Y.T)         # (N, M)
    dist_sq = np.clip(dist_sq, 0, None)               # numerical stability
    return np.exp(-dist_sq / (2 * sigma ** 2))


def _median_heuristic(X: np.ndarray, Y: np.ndarray) -> float:
    """Median heuristic for RBF bandwidth.

    σ = median({||x_i - y_j|| : i=1..N, j=1..M}) for paired distances,
    or median pairwise distance across the combined set.

    Uses a random subset to avoid O(N²) memory on large feature sets.
    """
    n_sample = min(len(X), len(Y), 500)
    idx_x = np.random.choice(len(X), n_sample, replace=False)
    idx_y = np.random.choice(len(Y), n_sample, replace=False)
    X_sub = X[idx_x]
    Y_sub = Y[idx_y]

    Xn = np.sum(X_sub ** 2, axis=1).reshape(-1, 1)
    Yn = np.sum(Y_sub ** 2, axis=1).reshape(1, -1)
    dist_sq = Xn + Yn - 2 * (X_sub @ Y_sub.T)
    dist_sq = np.clip(dist_sq, 0, None)

    sigma = np.sqrt(np.median(dist_sq))
    return max(float(sigma), 1e-8)


def compute_smmd(
    feats_ref: np.ndarray,
    feats_tgt: np.ndarray,
    sigma: Optional[float] = None,
) -> float:
    """Unbiased MMD² estimate with RBF kernel.

    MMD²_u = 1/(n*(n-1)) Σᵢ≠ⱼ k(xᵢ, xⱼ)
           + 1/(m*(m-1)) Σᵢ≠ⱼ k(yᵢ, yⱼ)
           - 2/(n*m) Σᵢ,ⱼ k(xᵢ, yⱼ)

    where k is the Gaussian RBF kernel.

    SMMD = sqrt(max(MMD², 0)) for interpretability on same scale as FD.

    Args:
        feats_ref: (N, D) reference distribution
        feats_tgt: (M, D) target distribution
        sigma: RBF bandwidth. If None, uses median heuristic.

    Returns:
        SMMD scalar (≥ 0). Lower = more similar distributions.
    """
    X = feats_ref.astype(np.float64)
    Y = feats_tgt.astype(np.float64)
    n, m = len(X), len(Y)

    if sigma is None:
        sigma = _median_heuristic(X, Y)

    # Kernel matrices
    K_xx = _rbf_kernel(X, X, sigma)  # (n, n)
    K_yy = _rbf_kernel(Y, Y, sigma)  # (m, m)
    K_xy = _rbf_kernel(X, Y, sigma)  # (n, m)

    # Unbiased MMD²
    # Within-X: sum of off-diagonal / (n*(n-1))
    if n > 1:
        term_xx = (K_xx.sum() - np.trace(K_xx)) / (n * (n - 1))
    else:
        term_xx = 0.0

    # Within-Y: sum of off-diagonal / (m*(m-1))
    if m > 1:
        term_yy = (K_yy.sum() - np.trace(K_yy)) / (m * (m - 1))
    else:
        term_yy = 0.0

    # Cross: mean of all K_xy pairs
    term_xy = K_xy.mean()

    mmd2 = term_xx + term_yy - 2 * term_xy
    smmd = np.sqrt(max(mmd2, 0.0))
    return float(smmd)


# ============================================================
# DistributionShiftProbe: Unified Diagnostic Class
# ============================================================

class DistributionShiftProbe:
    """Dual-metric distribution shift diagnostic tool.

    Combines FD (parametric) and SMMD (non-parametric) for
    rigorous out-of-domain detection.

    Usage:
        probe = DistributionShiftProbe(backbone, device='cuda')
        feats_a = probe.extract_features(dataloader_a)
        feats_b = probe.extract_features(dataloader_b)
        results = probe.evaluate(feats_a, feats_b)
        # → {'fd': 6.87, 'smmd': 0.42}
    """

    def __init__(self, backbone=None, pooling='mean', device='cpu'):
        """
        Args:
            backbone: SSLBackbone instance (optional, for feature extraction)
            pooling: 'mean' (default) — how to pool frame-level features
            device: torch device
        """
        self.backbone = backbone
        self.pooling = pooling
        self.device = device

    @torch.no_grad()
    def extract_features(
        self,
        dataloader,
        max_samples: Optional[int] = None,
    ) -> np.ndarray:
        """Extract sequence-level embeddings from a DataLoader.

        For each batch: waveform → WavLM (frozen) → mean pool → (768,)
        Result: (N, 768) array of utterance-level embeddings.

        Args:
            dataloader: PyTorch DataLoader yielding (waveforms, labels, lengths, ...)
            max_samples: optional cap on total samples

        Returns:
            (N, 768) float64 numpy array
        """
        if self.backbone is None:
            raise RuntimeError(
                "DistributionShiftProbe.backbone is not set. "
                "Set it to an SSLBackbone instance before calling extract_features."
            )

        features = []
        for batch in dataloader:
            waveforms = batch[0].to(self.device)

            # WavLM forward (returns all hidden states if configured)
            result = self.backbone(waveforms)
            if isinstance(result, tuple):
                frame_feats = result[0]  # last_hidden_state
            else:
                frame_feats = result

            # Pool: mean over time dimension
            if self.pooling == 'mean':
                lengths = batch[2] if len(batch) >= 3 else None
                if lengths is not None:
                    mask = torch.arange(
                        frame_feats.shape[1], device=self.device
                    ).unsqueeze(0) < lengths.unsqueeze(1)
                    mask_f = mask.to(dtype=frame_feats.dtype).unsqueeze(-1)
                    pooled = (frame_feats * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
                else:
                    pooled = frame_feats.mean(dim=1)
            else:
                pooled = frame_feats.mean(dim=1)

            features.append(pooled.cpu().numpy())

            if max_samples and sum(len(f) for f in features) >= max_samples:
                break

        feats = np.concatenate(features, axis=0)
        if max_samples and len(feats) > max_samples:
            feats = feats[:max_samples]

        return feats.astype(np.float64)

    def evaluate(
        self,
        feats_ref: np.ndarray,
        feats_tgt: np.ndarray,
        sigma: Optional[float] = None,
    ) -> Dict[str, float]:
        """Compute both FD and SMMD between two feature sets.

        Args:
            feats_ref: (N, D) reference distribution
            feats_tgt: (M, D) target distribution
            sigma: SMMD kernel bandwidth (None → median heuristic)

        Returns:
            {'fd': float, 'smmd': float}
        """
        fd = compute_frechet_distance(feats_ref, feats_tgt)
        smmd = compute_smmd(feats_ref, feats_tgt, sigma=sigma)
        return {'fd': fd, 'smmd': smmd}

    def evaluate_shift(
        self,
        dataloader_ref,
        dataloader_tgt,
        max_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """High-level API: extract features + compute both metrics.

        Args:
            dataloader_ref: DataLoader for reference distribution
            dataloader_tgt: DataLoader for target distribution
            max_samples: cap on features per distribution

        Returns:
            {'fd': float, 'smmd': float}
        """
        feats_ref = self.extract_features(dataloader_ref, max_samples=max_samples)
        feats_tgt = self.extract_features(dataloader_tgt, max_samples=max_samples)
        return self.evaluate(feats_ref, feats_tgt)


# ============================================================
# Dry-Run
# ============================================================

if __name__ == '__main__':
    rng = np.random.RandomState(42)

    # Simulate two distributions with a known shift
    N, D = 100, 768
    feats_ref = rng.randn(N, D).astype(np.float32)          # ~ N(0, 1)
    feats_tgt = rng.randn(N, D).astype(np.float32) + 0.5    # ~ N(0.5, 1)

    # FD
    fd = compute_frechet_distance(feats_ref, feats_tgt)
    assert np.isfinite(fd) and fd >= 0, f'FD invalid: {fd}'

    # SMMD
    smmd = compute_smmd(feats_ref, feats_tgt)
    assert np.isfinite(smmd) and smmd >= 0, f'SMMD invalid: {smmd}'

    # Probe
    probe = DistributionShiftProbe()
    result = probe.evaluate(feats_ref, feats_tgt)
    assert np.isfinite(result['fd']) and np.isfinite(result['smmd']), \
        f'Probe result invalid: {result}'

    print(f'FD   = {result["fd"]:.4f}')
    print(f'SMMD = {result["smmd"]:.4f}')

    # Sensitivity: identical distributions should yield near-zero
    fd_zero = compute_frechet_distance(feats_ref, feats_ref)
    smmd_zero = compute_smmd(feats_ref, feats_ref)
    print(f'FD(self)   = {fd_zero:.6f}  (expect ~0)')
    print(f'SMMD(self) = {smmd_zero:.6f}  (expect ~0)')
    assert fd_zero < 1e-6, f'FD on identical data should be near 0, got {fd_zero}'
    assert smmd_zero < 1e-6, f'SMMD on identical data should be near 0, got {smmd_zero}'

    # Larger shift → larger metrics
    feats_big = rng.randn(N, D).astype(np.float32) + 2.0
    fd_big = compute_frechet_distance(feats_ref, feats_big)
    smmd_big = compute_smmd(feats_ref, feats_big)
    assert fd_big > fd, f'FD should increase with larger shift: {fd_big} <= {fd}'
    assert smmd_big > smmd, f'SMMD should increase with larger shift: {smmd_big} <= {smmd}'
    print(f'FD(large shift)   = {fd_big:.4f}  (>> {fd:.4f})')
    print(f'SMMD(large shift) = {smmd_big:.4f}  (>> {smmd:.4f})')

    print('SUCCESS: Distribution Analytics ready')
