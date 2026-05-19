"""Unified PyTorch Dataset for cross-corpus SER.

Yields: (waveform, label_idx, speaker_id, dataset_name)
"""

import os
import re
from typing import List, Tuple, Optional, Dict, Set
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import Dataset

from .audio_processor import StandardAudioProcessor
from .label_mapper import UniversalLabelMapper, UNIFIED_LABEL_TO_IDX
from .speaker_splitter import split_speakers, get_split_for_speaker


# ============================================================
# Dataset path configuration
# ============================================================

# 提交到团队 is in the parent directory of the project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SHARED_ROOT = os.path.dirname(_PROJECT_ROOT)  # D:\...\儿童语音情绪识别\

DATASET_PATHS: Dict[str, str] = {
    'c-besd': os.path.join(_SHARED_ROOT, '提交到团队', '数据集', 'BESD', 'BESD', 'MY'),
    'iemocap': os.path.join(_SHARED_ROOT, '提交到团队', '数据集', 'IEMOCAP', 'wavs'),
    'crema-d': r'D:\大学\crema_temp\AudioWAV',
    'fau-aibo': r'D:\大学\数据集IS2009EmotionChallenge\IS2009EmotionChallenge\IS2009EmotionChallenge\wav',
}

# Optional path overrides for cross-platform training (e.g., Linux cloud).
_ENV_OVERRIDES = {
    'c-besd': os.environ.get('SER_C_BESD_PATH'),
    'iemocap': os.environ.get('SER_IEMOCAP_PATH'),
    'crema-d': os.environ.get('SER_CREMA_D_PATH'),
    'fau-aibo': os.environ.get('SER_FAU_AIBO_PATH'),
}
for _k, _v in _ENV_OVERRIDES.items():
    if _v:
        DATASET_PATHS[_k] = _v

# IEMOCAP emotion → folder name mapping
IEMOCAP_EMOTION_FOLDERS = {
    'ang': 'ANGER',     # includes frustrated
    'hap': 'HAPPY',     # includes happy, excited
    'neu': 'NEUTRAL',
    'sad': 'SAD',
    # DISGUST, FEAR are scanned but discarded by label mapper
}

# C-BESD emotion → folder mapping
C_BESD_EMOTION_FOLDERS = {
    'angry': 'ANGER',
    'happy': 'HAPPY',
    'neutral': 'NEUTRAL',
    'sad': 'SAD',
}

# FAU Aibo label file path
FAU_AIBO_LABEL_FILE = os.path.join(
    os.path.dirname(DATASET_PATHS['fau-aibo']),
    'labels', 'IS2009EmotionChallenge', 'chunk_labels_5cl_corpus.txt'
)


# ============================================================
# Speaker ID extraction per dataset
# ============================================================

def extract_speaker_cbesd(filename: str) -> str:
    """Extract speaker ID from C-BESD filename.

    '1.EF_12 Angry_1.wav' → '1.EF_12'
    '1.TF_12_angry_1.wav'  → '1.TF_12'
    Handles 'anger' variant, 'disguist' typo, missing-dot anomalies.
    """
    # Normalize
    sid = filename.upper().replace('-', '_')
    # Fix missing dot: '3EF_9' → '3.EF_9'
    sid = re.sub(r'^(\d+)([ET][FM])', r'\1.\2', sid)

    basename = os.path.splitext(filename)[0].lower()
    emotion_patterns = ['angry', 'anger', 'disgust', 'disguist',
                        'fear', 'happy', 'neutral', 'sad']
    for emo in emotion_patterns:
        idx = basename.find(emo)
        if idx != -1:
            raw = filename[:idx].rstrip(' _-')
            raw = raw.upper().replace('-', '_')
            raw = re.sub(r'^(\d+)([ET][FM])', r'\1.\2', raw)
            return raw
    return sid


def extract_speaker_iemocap(filename: str) -> str:
    """Extract speaker ID from IEMOCAP filename.

    'Ses01F_impro01_F006_frustrated.wav' → 'Ses01F'
    Session + gender uniquely identifies the speaker.
    """
    return filename[:6]  # 'Ses01F', 'Ses05M', etc.


def extract_speaker_cremad(filename: str) -> str:
    """Extract speaker ID from CREMA-D filename.

    '1001_DFA_ANG_XX.wav' → '1001'
    """
    return filename.split('_')[0]


def extract_speaker_fau_aibo(filename: str) -> str:
    """Extract speaker ID from FAU Aibo filename.

    'Mont_01_000_00.wav' → 'Mont_01'
    """
    parts = filename.split('_')
    return f"{parts[0]}_{parts[1]}"  # 'Mont_01'


# Per-dataset extractors
SPEAKER_EXTRACTORS = {
    'c-besd': extract_speaker_cbesd,
    'iemocap': extract_speaker_iemocap,
    'crema-d': extract_speaker_cremad,
    'fau-aibo': extract_speaker_fau_aibo,
}


# ============================================================
# IEMOCAP raw label extraction from filename
# ============================================================

def extract_iemocap_label(filename: str) -> str:
    """Extract emotion label from IEMOCAP filename.

    'Ses01F_impro01_F006_frustrated.wav' → 'frustrated'
    'Ses01F_impro03_F003_excited.wav' → 'excited'
    'Ses01F_impro01_F000_ang.wav' → 'ang'
    """
    stem = os.path.splitext(filename)[0]
    parts = stem.split('_')
    if len(parts) >= 4:
        return parts[-1]  # last segment is emotion label
    return parts[-1]


def extract_cremad_label(filename: str) -> str:
    """Extract emotion label from CREMA-D filename.

    '1001_DFA_ANG_XX.wav' → 'ANG'
    """
    return filename.split('_')[2]


# ============================================================
# File collection per dataset
# ============================================================

def _collect_cbesd(root: str) -> List[Tuple[str, str, str]]:
    """Collect C-BESD files. Returns [(filepath, raw_label, speaker_id), ...]."""
    entries = []
    for emo_label, folder in C_BESD_EMOTION_FOLDERS.items():
        folder_path = os.path.join(root, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if not fname.endswith('.wav'):
                continue
            if 'copy' in fname.lower():
                continue
            fpath = os.path.join(folder_path, fname)
            sid = extract_speaker_cbesd(fname)
            entries.append((fpath, emo_label, sid))
    return entries


def _collect_iemocap(root: str) -> List[Tuple[str, str, str]]:
    """Collect IEMOCAP files from emotion subfolders."""
    entries = []
    for raw_emo, folder in IEMOCAP_EMOTION_FOLDERS.items():
        folder_path = os.path.join(root, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if not fname.endswith('.wav'):
                continue
            fpath = os.path.join(folder_path, fname)
            # Extract actual label from filename (handles frustrated→ang, excited→hap)
            actual_emo = extract_iemocap_label(fname)
            # Map 'frustrated' back to 'ang' so it goes through the mapper
            if actual_emo == 'frustrated':
                actual_emo = 'ang'
            # Map 'excited' to 'exc'
            elif actual_emo == 'excited':
                actual_emo = 'exc'
            sid = extract_speaker_iemocap(fname)
            entries.append((fpath, actual_emo, sid))
    return entries


def _collect_cremad(root: str) -> List[Tuple[str, str, str]]:
    """Collect CREMA-D files."""
    entries = []
    if not os.path.isdir(root):
        return entries
    for fname in os.listdir(root):
        if not fname.endswith('.wav'):
            continue
        fpath = os.path.join(root, fname)
        raw_emo = extract_cremad_label(fname)
        sid = extract_speaker_cremad(fname)
        entries.append((fpath, raw_emo, sid))
    return entries


def _collect_fau_aibo(root: str) -> List[Tuple[str, str, str]]:
    """Collect FAU Aibo files with labels from chunk_labels_5cl_corpus.txt."""
    entries = []

    # Load label file
    label_file = FAU_AIBO_LABEL_FILE
    if not os.path.exists(label_file):
        print(f"  WARNING: FAU Aibo label file not found: {label_file}")
        return entries

    file_to_label: Dict[str, str] = {}
    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                fname = parts[0] + '.wav'
                label = parts[1]
                file_to_label[fname] = label

    if not os.path.isdir(root):
        return entries

    for fname in os.listdir(root):
        if not fname.endswith('.wav'):
            continue
        fpath = os.path.join(root, fname)
        label = file_to_label.get(fname)
        if label is None:
            continue  # skip files without labels
        sid = extract_speaker_fau_aibo(fname)
        entries.append((fpath, label, sid))

    return entries


DATASET_COLLECTORS = {
    'c-besd': _collect_cbesd,
    'iemocap': _collect_iemocap,
    'crema-d': _collect_cremad,
    'fau-aibo': _collect_fau_aibo,
}


# ============================================================
# Unified Dataset class
# ============================================================

class UnifiedSERDataset(Dataset):
    """Unified speech emotion recognition dataset across multiple corpora.

    Each item: (waveform_tensor, label_idx, speaker_id, dataset_name)

    Waveforms are loaded on-the-fly and processed through StandardAudioProcessor.
    Label mapping and discard filtering happen at construction time.

    Args:
        dataset_names: list of dataset keys, e.g. ['c-besd', 'iemocap']
        split: 'train', 'val', 'test', or 'all'
        seed: random seed for speaker splitting (default 42)
        processor: StandardAudioProcessor instance (created if None)
    """

    def __init__(
        self,
        dataset_names: List[str],
        split: str = 'all',
        seed: int = 42,
        processor: Optional[StandardAudioProcessor] = None,
    ):
        self.dataset_names = [d.lower() for d in dataset_names]
        self.split = split
        self.seed = seed
        self.processor = processor or StandardAudioProcessor(target_sr=16000)

        # Internal storage: list of (wav_path, unified_label_idx, speaker_id, dataset_name)
        self._samples: List[Tuple[str, int, str, str]] = []

        for ds_name in self.dataset_names:
            self._load_dataset(ds_name)

        if len(self._samples) == 0:
            print(f"  WARNING: No samples loaded for datasets={self.dataset_names}, split={split}")

    def _load_dataset(self, ds_name: str):
        """Collect, label-map, split, and filter samples for one dataset."""
        root = DATASET_PATHS.get(ds_name)
        if root is None or not os.path.isdir(root):
            print(f"  WARNING: Dataset path not found for '{ds_name}': {root}")
            return

        collector = DATASET_COLLECTORS.get(ds_name)
        if collector is None:
            print(f"  WARNING: No collector for dataset '{ds_name}'")
            return

        mapper = UniversalLabelMapper(ds_name)
        entries = collector(root)

        if len(entries) == 0:
            print(f"  WARNING: No entries found for '{ds_name}'")
            return

        # Apply label mapping, discard None
        mapped: List[Tuple[str, int, str]] = []
        discarded = 0
        for fpath, raw_label, sid in entries:
            unified_idx = mapper.to_index(raw_label)
            if unified_idx is None:
                discarded += 1
                continue
            mapped.append((fpath, unified_idx, sid))

        # Split speakers
        all_speakers = sorted(set(s[2] for s in mapped))
        train_spk, val_spk, test_spk = split_speakers(
            all_speakers, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=self.seed
        )

        # Filter by split
        split_sets = {'train': train_spk, 'val': val_spk, 'test': test_spk, 'all': set(all_speakers)}
        target_speakers = split_sets.get(self.split, set(all_speakers))

        for fpath, label_idx, sid in mapped:
            if sid in target_speakers:
                self._samples.append((fpath, label_idx, sid, ds_name))

        # Log summary
        split_count = sum(1 for s in mapped if s[2] in target_speakers)
        print(f"  [{ds_name}] {len(mapped)} samples ({len(all_speakers)} speakers), "
              f"discarded={discarded}, split='{self.split}' → {split_count} samples "
              f"(train={len(train_spk)}/val={len(val_spk)}/test={len(test_spk)} speakers)")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, str]:
        wav_path, label_idx, speaker_id, ds_name = self._samples[idx]
        waveform = self.processor.load_and_process(wav_path)
        return (
            torch.from_numpy(waveform),
            label_idx,
            speaker_id,
            ds_name,
        )

    def get_speaker_splits(self) -> Dict[str, Set[str]]:
        """Return the speaker ID sets for each split across all datasets."""
        speakers: Dict[str, Set[str]] = defaultdict(set)
        for _, _, sid, _ in self._samples:
            speakers[self.split].add(sid)
        return dict(speakers)

    @property
    def class_distribution(self) -> Dict[str, int]:
        """Return per-class sample counts."""
        from collections import Counter
        from .label_mapper import IDX_TO_UNIFIED_LABEL
        cnt = Counter()
        for _, label_idx, _, _ in self._samples:
            cnt[IDX_TO_UNIFIED_LABEL[label_idx]] += 1
        return dict(cnt)
