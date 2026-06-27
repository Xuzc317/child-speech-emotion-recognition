"""Strict speaker-disjoint data splitting with deterministic hashing.

Guarantees: len(train_speakers ∩ test_speakers) == 0 at runtime via assertion.
"""

import hashlib
from typing import List, Tuple, Set, Dict
import numpy as np


def _hash_speaker(speaker_id: str, seed: int = 42) -> float:
    """Deterministic hash of speaker_id → float in [0, 1)."""
    h = hashlib.md5(f"{speaker_id}:{seed}".encode()).hexdigest()
    return int(h, 16) / (16 ** len(h))


def split_speakers(
    speakers: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Split speakers into train/val/test using deterministic hashing.

    Hash each speaker_id → [0, 1), then threshold:
      - hash < train_ratio                    → train
      - train_ratio <= hash < train_ratio + val_ratio → val
      - otherwise                             → test

    Args:
        speakers: list of unique speaker IDs
        train_ratio: fraction for training (default 0.70)
        val_ratio: fraction for validation (default 0.15)
        test_ratio: fraction for test (default 0.15)
        seed: random seed for hashing

    Returns:
        (train_speakers, val_speakers, test_speakers) as sets

    Raises:
        RuntimeError: if any speaker appears in more than one split
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, \
        f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    train_speakers: Set[str] = set()
    val_speakers: Set[str] = set()
    test_speakers: Set[str] = set()

    for spk in speakers:
        h = _hash_speaker(spk, seed)
        if h < train_ratio:
            train_speakers.add(spk)
        elif h < train_ratio + val_ratio:
            val_speakers.add(spk)
        else:
            test_speakers.add(spk)

    # ── Strict zero-leakage assertion ──
    if len(train_speakers & test_speakers) != 0:
        raise RuntimeError(
            f"DATA LEAKAGE: {len(train_speakers & test_speakers)} speakers "
            f"in both train and test splits!"
        )
    if len(train_speakers & val_speakers) != 0:
        raise RuntimeError(
            f"DATA LEAKAGE: {len(train_speakers & val_speakers)} speakers "
            f"in both train and val splits!"
        )
    if len(val_speakers & test_speakers) != 0:
        raise RuntimeError(
            f"DATA LEAKAGE: {len(val_speakers & test_speakers)} speakers "
            f"in both val and test splits!"
        )

    return train_speakers, val_speakers, test_speakers


def get_split_for_speaker(
    speaker_id: str,
    train_speakers: Set[str],
    val_speakers: Set[str],
    test_speakers: Set[str],
) -> str:
    """Return 'train', 'val', or 'test' for a given speaker."""
    if speaker_id in train_speakers:
        return 'train'
    elif speaker_id in val_speakers:
        return 'val'
    elif speaker_id in test_speakers:
        return 'test'
    raise ValueError(f"Speaker {speaker_id} not in any split!")
