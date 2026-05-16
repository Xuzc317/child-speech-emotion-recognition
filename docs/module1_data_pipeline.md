# Module 1: Data Pipeline Standardization & Cross-Corpus Infrastructure

## Overview

Rebuilt the data ingestion, preprocessing, and DataLoader pipeline to:
- Eliminate data leakage via strict speaker-disjoint splitting
- Support cross-corpus zero-shot evaluation
- Standardize all inputs for WavLM-based architecture
- Map all datasets into a unified 4-class emotion space

## Architecture

```
src/data/
├── __init__.py              # Public API exports
├── audio_processor.py       # StandardAudioProcessor
├── label_mapper.py          # UniversalLabelMapper + per-dataset mappings
├── speaker_splitter.py      # Deterministic hash-based splitting
├── dataset.py               # UnifiedSERDataset (PyTorch Dataset)
└── data_loader.py           # get_dataloaders() + cross-corpus builder

src/augmentation/
├── __init__.py              # SafeAWGN export
└── safe_augmentation.py     # Additive White Gaussian Noise (replaces old constrained_aug.py)
```

## Dataset Summary

| Dataset   | Samples | Speakers | Classes Present          | Discarded    |
|-----------|---------|----------|--------------------------|--------------|
| C-BESD    | 2,780   | 237      | angry, happy, neutral, sad | disgust, fear     |
| IEMOCAP   | 8,525   | 10       | angry, happy, neutral, sad | fear, disgust, surprise, other |
| CREMA-D   | 4,900   | 91       | angry, happy, neutral, sad | fear, disgust     |
| FAU Aibo  | 18,216  | 51       | angry, happy, neutral    | (no sad class)   |

## Usage

### 1. Single-corpus training

```python
from src.data import get_dataloaders

dataloaders = get_dataloaders(['c-besd'], batch_size=32, seed=42)
train_dl = dataloaders['train']
val_dl = dataloaders['val']
test_dl = dataloaders['test']

for waveforms, labels, lengths, speaker_ids, dataset_names in train_dl:
    # waveforms: (B, T_max) padded float tensor
    # labels: (B,) int64 (0=angry, 1=happy, 2=neutral, 3=sad)
    ...
```

### 2. Multi-corpus combined training

```python
dataloaders = get_dataloaders(['c-besd', 'iemocap'], batch_size=32, seed=42)
```

### 3. Cross-corpus zero-shot evaluation

```python
from src.data import get_cross_corpus_dataloaders

dataloaders = get_cross_corpus_dataloaders(
    train_datasets=['c-besd'],
    test_datasets=['iemocap'],
    batch_size=32, seed=42
)
# train: C-BESD train split
# val:   C-BESD val split
# test:  IEMOCAP all speakers (zero-shot)
```

### 4. Standalone audio processing

```python
from src.data import StandardAudioProcessor

processor = StandardAudioProcessor(target_sr=16000)
waveform = processor.load_and_process('path/to/audio.wav')  # (T,) float32, mono, peak-normed
```

### 5. Label mapping

```python
from src.data import UniversalLabelMapper, UNIFIED_LABEL_TO_IDX

mapper = UniversalLabelMapper('iemocap')
unified = mapper('exc')       # → 'happy'
index = mapper.to_index('ang')  # → 0
```

### 6. Safe augmentation

```python
from src.augmentation import SafeAWGN

aug = SafeAWGN(snr_db_range=(10., 20.), apply_prob=0.5)
augmented = aug(waveform)  # AWGN applied with 50% probability
```

## Label Mapping Logic

### Unified 4-class space: `{angry(0), happy(1), neutral(2), sad(3)}`

### Per-dataset mapping rules:

**C-BESD**: `angry/anger→angry`, `happy→happy`, `neutral→neutral`, `sad→sad`. Disgust/Fear folders not scanned.

**IEMOCAP**: `ang/frustrated→angry`, `hap/happy/exc/excited→happy`, `neu/neutral→neutral`, `sad→sad`. `fea/dis/sur/xxx/oth` → discard.

**CREMA-D**: `ANG→angry`, `HAP→happy`, `NEU→neutral`, `SAD→sad`. `FEA/DIS` → discard.

**FAU Aibo** (5-class IS2009 labels): `A/R→angry`, `E/P→happy`, `N→neutral`. No sad class available.

## Speaker-Disjoint Splitting

- **Method**: Deterministic MD5 hash of `(speaker_id, seed)` → [0, 1), thresholded at 0.70/0.85
- **Split ratios**: Train 70%, Val 15%, Test 15%
- **Runtime assertion**: `assert len(train_spk ∩ test_spk) == 0` in `speaker_splitter.py:split_speakers()`
- **Seed**: Default 42, configurable for reproducibility

### Verified zero-leakage results

| Dataset   | Train Spk | Val Spk | Test Spk | Overlap |
|-----------|-----------|---------|----------|---------|
| C-BESD    | 162       | 36      | 39       | 0       |
| IEMOCAP   | 5         | 2       | 3        | 0       |
| CREMA-D   | 58        | 21      | 12       | 0       |
| FAU Aibo  | 35        | 7       | 9        | 0       |

## Audio Processing Pipeline

1. Load WAV via librosa (preserve native SR)
2. Convert to mono (channel averaging)
3. Resample to 16,000 Hz
4. Peak normalize to [-1, 1]
5. Output as float32 numpy array

## Augmentation

- **REMOVED**: Pitch shifting, time stretching, SpecAugment (destroy children's emotional prosody)
- **ADDED**: SafeAWGN — Additive White Gaussian Noise with random SNR (10-20 dB), applied dynamically with 50% probability during training only

## Changes from Old Pipeline

| Old | New |
|-----|-----|
| `src/data/preprocess.py` (BESD-only) | `src/data/dataset.py` (multi-corpus) |
| 6-class labels | 4-class unified labels |
| `constrained_aug.py` (pitch/stretch/noise) | `safe_augmentation.py` (AWGN only) |
| `dataset_ssl.py` (.npy pre-extracted features) | `dataset.py` (on-the-fly waveform loading) |
| Profile-stratified random split | Deterministic hash-based split |
