"""Standardized audio preprocessing: resample 16kHz, mono, peak normalization."""

import librosa
import numpy as np


class StandardAudioProcessor:
    """Enforce uniform audio format across all datasets.

    Operations (in order):
      1. Load with librosa at native sr
      2. Resample to target_sr (16000 Hz)
      3. Convert to mono
      4. Peak normalize to [-1, 1]
    """

    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr

    def process(self, waveform: np.ndarray, orig_sr: int) -> np.ndarray:
        """Process an already-loaded waveform.

        Args:
            waveform: (n_samples,) or (n_channels, n_samples) numpy array
            orig_sr: original sampling rate

        Returns:
            (T,) float32 numpy array at target_sr, mono, peak-normalized
        """
        # Mono conversion
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)

        # Resample
        if orig_sr != self.target_sr:
            waveform = librosa.resample(
                waveform.astype(np.float64), orig_sr=orig_sr, target_sr=self.target_sr
            )

        # Peak normalization
        peak = np.abs(waveform).max()
        if peak > 0:
            waveform = waveform / peak

        return waveform.astype(np.float32)

    def load_and_process(self, wav_path: str) -> np.ndarray:
        """Load from file path and apply full processing pipeline.

        Args:
            wav_path: path to .wav file

        Returns:
            (T,) float32 numpy array
        """
        waveform, orig_sr = librosa.load(wav_path, sr=None, mono=False)
        return self.process(waveform, orig_sr)
