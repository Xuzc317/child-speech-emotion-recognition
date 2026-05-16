"""Safe data augmentation for children's speech emotion recognition.

Only additive noise augmentation is applied — pitch shifting and time stretching
are explicitly avoided as they destroy children's natural emotional prosody.
"""

import numpy as np
import torch


class SafeAWGN:
    """Additive White Gaussian Noise with random SNR.

    Applied dynamically during training only. Does NOT alter pitch or tempo.

    Args:
        snr_db_range: (min_snr, max_snr) in dB. Default (10, 20).
        apply_prob: probability of applying augmentation (default 0.5)
    """

    def __init__(
        self,
        snr_db_range: tuple = (10.0, 20.0),
        apply_prob: float = 0.5,
    ):
        self.snr_min, self.snr_max = snr_db_range
        self.apply_prob = apply_prob

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        """Apply AWGN with probability apply_prob.

        Args:
            waveform: (T,) float32 numpy array

        Returns:
            (T,) float32 numpy array (augmented or original)
        """
        if np.random.random() > self.apply_prob:
            return waveform

        snr_db = np.random.uniform(self.snr_min, self.snr_max)
        signal_power = np.mean(waveform ** 2)
        if signal_power < 1e-12:
            return waveform

        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), size=waveform.shape)
        return (waveform + noise).astype(np.float32)

    def apply_torch(self, waveform: torch.Tensor) -> torch.Tensor:
        """Torch tensor version for in-batch augmentation.

        Args:
            waveform: (T,) or (B, T) float tensor

        Returns:
            augmented tensor of same shape
        """
        if np.random.random() > self.apply_prob:
            return waveform

        snr_db = np.random.uniform(self.snr_min, self.snr_max)
        signal_power = waveform.pow(2).mean(dim=-1, keepdim=True)
        signal_power = signal_power.clamp(min=1e-12)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        return waveform + noise
