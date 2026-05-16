"""XAI Visualization: Attention-Prosody Correlation & 1D Saliency Maps.

Proves that Prosody-Guided Temporal Importance Pooling successfully
attends to frames with high acoustic density (F0 mutations, RMS peaks).

Metrics:
  - APC_wav: Pearson r between attention weights and RMS energy
  - APC_delta: Pearson r between attention weights and |dF0/dt|
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.stats import pearsonr
from typing import Tuple, Dict


class AttentionProsodyExplainer:
    """Quantify attention-prosody alignment for a single audio sample.

    Usage:
        explainer = AttentionProsodyExplainer(sr=16000, ssl_frame_rate=50)
        metrics = explainer.compute_apc(attn_weights, f0_contour, rms_contour)
        explainer.plot_saliency(waveform, f0_contour, rms_contour, attn_weights,
                                save_path='results/xai_sample.png')
    """

    def __init__(self, sr: int = 16000, ssl_frame_rate: float = 50.0):
        """
        Args:
            sr: audio sampling rate in Hz
            ssl_frame_rate: WavLM output frame rate in Hz (default 50)
        """
        self.sr = sr
        self.ssl_frame_rate = ssl_frame_rate
        self.ssl_hop_samples = int(sr / ssl_frame_rate)  # 320 @ 16kHz/50Hz

    # ── Temporal Alignment ───────────────────────────────────

    def _interpolate_to_length(self, signal: np.ndarray, target_len: int) -> np.ndarray:
        """1D linear interpolation to match target length.

        Args:
            signal: (N,) array at source resolution
            target_len: desired output length
        Returns:
            (target_len,) interpolated array
        """
        if len(signal) == target_len:
            return signal.astype(np.float32)

        x_src = np.linspace(0, 1, len(signal))
        x_tgt = np.linspace(0, 1, target_len)
        f = interpolate.interp1d(x_src, signal, kind='linear',
                                 fill_value='extrapolate')
        return f(x_tgt).astype(np.float32)

    def align_all(
        self,
        f0: np.ndarray,
        rms: np.ndarray,
        attn: np.ndarray,
        target_len: int = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Align F0, RMS, and attention to common temporal length.

        If target_len is None, uses the attention length as reference.
        """
        if target_len is None:
            target_len = len(attn)

        f0_aligned = self._interpolate_to_length(f0, target_len)
        rms_aligned = self._interpolate_to_length(rms, target_len)
        attn_aligned = self._interpolate_to_length(attn, target_len)

        return f0_aligned, rms_aligned, attn_aligned

    # ── APC Metrics ──────────────────────────────────────────

    def compute_apc(
        self,
        attn_weights: np.ndarray,
        f0_contour: np.ndarray,
        rms_contour: np.ndarray,
    ) -> Dict[str, float]:
        """Compute Attention-Prosody Correlation metrics.

        APC_wav (waveform energy):
          Pearson r between attention weights and RMS contour.
          Measures: does attention track vocal energy?

        APC_delta (F0 mutation):
          Pearson r between attention weights and |dF0/dt|.
          Measures: does attention track pitch inflection points?

        Mathematical definitions:
          Let a[t] be attention weights, e[t] be RMS energy,
          f[t] be F0 contour.
          APC_wav  = ρ(a[t], e[t])
          APC_delta = ρ(a[t], |f[t] - f[t-1]|)  for t >= 1

        Args:
            attn_weights: (T_attn,) attention weights from pooling layer
            f0_contour: (T_f0,) F0 values in Hz
            rms_contour: (T_rms,) RMS energy values

        Returns:
            {'apc_wav': float, 'apc_delta': float}
        """
        # Align to attention length
        T = len(attn_weights)
        f0_a, rms_a, attn_a = self.align_all(f0_contour, rms_contour, attn_weights, T)

        # Normalize to [0, 1] for stability
        def safe_norm(x):
            if x.std() < 1e-12:
                return np.zeros_like(x)
            return (x - x.mean()) / x.std()

        attn_n = safe_norm(attn_a)
        rms_n = safe_norm(rms_a)

        # APC_wav: attention vs energy
        apc_wav, _ = pearsonr(attn_n, rms_n)

        # APC_delta: attention vs |dF0/dt|
        delta_f0 = np.abs(np.diff(f0_a, prepend=f0_a[0]))
        delta_n = safe_norm(delta_f0)
        apc_delta, _ = pearsonr(attn_n, delta_n)

        return {'apc_wav': float(apc_wav), 'apc_delta': float(apc_delta)}

    # ── 1D Saliency Map Plotting ─────────────────────────────

    def plot_saliency(
        self,
        waveform: np.ndarray,
        f0_contour: np.ndarray,
        rms_contour: np.ndarray,
        attn_weights: np.ndarray,
        save_path: str = 'results/xai_sample_visualization.png',
        title: str = None,
    ):
        """Generate 3-stack saliency visualization.

        Subplot 1: Raw waveform (grey)
        Subplot 2: Normalized F0 (blue) + RMS (orange)
        Subplot 3: Attention weights (red heatmap)

        All subplots share the same X-axis (Time in seconds).

        Args:
            waveform: (T_wav,) raw audio samples
            f0_contour: (T_f0,) F0 contour in Hz
            rms_contour: (T_rms,) RMS energy
            attn_weights: (T_attn,) attention weights
            save_path: output PNG path
            title: optional figure title
        """
        # Align all prosody to attention temporal resolution
        T = len(attn_weights)
        f0_a, rms_a, attn_a = self.align_all(f0_contour, rms_contour, attn_weights, T)

        # Compute APC for annotation
        apc = self.compute_apc(attn_weights, f0_contour, rms_contour)

        # Time axes
        t_wav = np.arange(len(waveform)) / self.sr
        t_attn = np.arange(T) * self.ssl_hop_samples / self.sr

        # Normalize prosody for display
        f0_norm = f0_a / max(f0_a.max(), 1.0)
        rms_norm = rms_a / max(rms_a.max(), 1e-8)
        attn_norm = attn_a / max(attn_a.max(), 1e-8)

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        # ── Top: Waveform ──
        ax0 = axes[0]
        ax0.plot(t_wav, waveform, color='#8899aa', linewidth=0.5, alpha=0.8)
        ax0.set_ylabel('Amplitude', fontsize=10)
        ax0.set_title(title or 'Attention-Prosody Saliency Map', fontsize=13, fontweight='bold')
        ax0.grid(axis='y', alpha=0.3)
        ax0.set_xlim(0, t_wav[-1])

        # ── Middle: F0 + RMS ──
        ax1 = axes[1]
        ax1.plot(t_attn, f0_norm, color='#2266aa', linewidth=1.2, label='F0 (norm)')
        ax1.plot(t_attn, rms_norm, color='#ee8833', linewidth=1.2, label='RMS (norm)')
        ax1.set_ylabel('Normalized Value', fontsize=10)
        ax1.legend(loc='upper right', fontsize=9, framealpha=0.7)
        ax1.grid(axis='y', alpha=0.3)

        # ── Bottom: Attention Weights ──
        ax2 = axes[2]
        ax2.fill_between(t_attn, 0, attn_norm, color='#cc3333', alpha=0.5, linewidth=0)
        ax2.plot(t_attn, attn_norm, color='#cc3333', linewidth=1.0, alpha=0.9)
        ax2.set_ylabel('Attention Weight', fontsize=10)
        ax2.set_xlabel('Time (s)', fontsize=11)
        ax2.grid(axis='y', alpha=0.3)

        # APC annotation
        ax2.text(0.02, 0.95,
                 f'APC_wav = {apc["apc_wav"]:.3f}  |  APC_delta = {apc["apc_delta"]:.3f}',
                 transform=ax2.transAxes, fontsize=10,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


# ============================================================
# Dry-Run
# ============================================================

if __name__ == '__main__':
    # Generate dummy data simulating a 3-second utterance
    sr = 16000
    duration = 3.0
    ssl_frame_rate = 50.0
    hop_samples = int(sr / ssl_frame_rate)  # 320

    n_samples = int(sr * duration)
    n_frames = int(duration * ssl_frame_rate)  # 150

    rng = np.random.RandomState(42)
    t = np.linspace(0, duration, n_samples)
    t_frame = np.linspace(0, duration, n_frames)

    # Dummy waveform: mixture of sines + noise
    waveform = (
        0.3 * np.sin(2 * np.pi * 220 * t) +
        0.15 * np.sin(2 * np.pi * 440 * t) +
        0.05 * rng.randn(n_samples)
    ).astype(np.float32)

    # Dummy F0: rising then falling (child-like contour ~300-500 Hz)
    f0_mid = 350 + 100 * np.sin(2 * np.pi * 0.5 * t_frame)
    f0_contour = f0_mid + 20 * rng.randn(n_frames).astype(np.float32)

    # Dummy RMS: envelope following waveform amplitude
    frame_samples = n_samples // n_frames
    rms_contour = np.array([
        np.sqrt(np.mean(waveform[i*frame_samples:(i+1)*frame_samples]**2))
        for i in range(n_frames)
    ], dtype=np.float32)

    # Dummy attention: correlated with RMS + some noise
    attn_weights = (
        0.7 * rms_contour / rms_contour.max() +
        0.3 * rng.rand(n_frames).astype(np.float32)
    )
    attn_weights = attn_weights / attn_weights.sum()

    # Run explainer
    explainer = AttentionProsodyExplainer(sr=sr, ssl_frame_rate=ssl_frame_rate)

    apc_metrics = explainer.compute_apc(attn_weights, f0_contour, rms_contour)
    print(f'APC_wav = {apc_metrics["apc_wav"]:.4f}')
    print(f'APC_delta = {apc_metrics["apc_delta"]:.4f}')

    save_path = 'results/xai_sample_visualization.png'
    explainer.plot_saliency(waveform, f0_contour, rms_contour, attn_weights,
                            save_path=save_path,
                            title='XAI Saliency Map (Dummy Data)')
    print(f'SUCCESS: XAI saliency map saved to {save_path}')
