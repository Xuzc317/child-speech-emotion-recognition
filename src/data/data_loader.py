"""Modular DataLoader builder for cross-corpus SER.

Provides get_dataloaders() for combined-corpus or single-corpus dataloaders,
supporting cross-corpus zero-shot evaluation.
"""

from typing import List, Dict, Optional, Tuple
import torch
from torch.utils.data import DataLoader, Dataset

from .dataset import UnifiedSERDataset
from .audio_processor import StandardAudioProcessor


def collate_waveforms(batch: List[Tuple[torch.Tensor, int, str, str]]):
    """Collate variable-length waveforms with zero-padding.

    Args:
        batch: list of (waveform, label_idx, speaker_id, dataset_name)

    Returns:
        waveforms: (B, T_max) padded float tensor
        labels: (B,) int64 tensor
        lengths: (B,) int64 tensor of original lengths
        speaker_ids: list of str
        dataset_names: list of str
    """
    waveforms, labels, speaker_ids, dataset_names = zip(*batch)

    lengths = torch.tensor([w.shape[0] for w in waveforms], dtype=torch.int64)
    labels = torch.tensor(labels, dtype=torch.int64)

    max_len = lengths.max().item()
    padded = torch.zeros(len(waveforms), max_len)
    for i, w in enumerate(waveforms):
        padded[i, :w.shape[0]] = w

    return padded, labels, lengths, list(speaker_ids), list(dataset_names)


def get_dataloaders(
    dataset_names: List[str],
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    splits: Optional[List[str]] = None,
) -> Dict[str, DataLoader]:
    """Build DataLoaders for one or more corpora.

    Args:
        dataset_names: e.g. ['c-besd'] or ['c-besd', 'iemocap']
        batch_size: samples per batch
        num_workers: DataLoader worker processes
        seed: random seed for speaker splitting
        splits: which splits to return (default: ['train', 'val', 'test'])

    Returns:
        dict mapping split name → DataLoader, e.g. {'train': dl, 'val': dl, 'test': dl}
    """
    if splits is None:
        splits = ['train', 'val', 'test']

    processor = StandardAudioProcessor(target_sr=16000)
    dataloaders: Dict[str, DataLoader] = {}

    for split in splits:
        dataset = UnifiedSERDataset(
            dataset_names=dataset_names,
            split=split,
            seed=seed,
            processor=processor,
        )
        shuffle = (split == 'train')
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_waveforms,
            pin_memory=True,
        )

    return dataloaders


def get_cross_corpus_dataloaders(
    train_datasets: List[str],
    test_datasets: List[str],
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """Build dataloaders for cross-corpus zero-shot evaluation.

    Train on one set of corpora, evaluate on a disjoint set.

    Args:
        train_datasets: corpora for training, e.g. ['c-besd']
        test_datasets: corpora for testing, e.g. ['iemocap']
        batch_size: samples per batch
        num_workers: DataLoader worker processes
        seed: random seed

    Returns:
        dict with 'train', 'val', 'test' DataLoaders.
        Note: val uses train_datasets, test uses test_datasets (all speakers).
    """
    processor = StandardAudioProcessor(target_sr=16000)

    train_ds = UnifiedSERDataset(train_datasets, split='train', seed=seed, processor=processor)
    val_ds = UnifiedSERDataset(train_datasets, split='val', seed=seed, processor=processor)
    # For zero-shot test, use ALL speakers from target dataset
    test_ds = UnifiedSERDataset(test_datasets, split='all', seed=seed, processor=processor)

    return {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, collate_fn=collate_waveforms, pin_memory=True),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, collate_fn=collate_waveforms, pin_memory=True),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, collate_fn=collate_waveforms, pin_memory=True),
    }
