"""Unified label mapping supporting both 6-class (C-BESD) and 4-class (FAU/IEMOCAP).

6-class (C-BESD):    angry, disgust, fear, happy, neutral, sad
4-class (cross-corpus): angry, happy, neutral, sad
  - FAU Aibo: A+E→angry, P→happy, N→neutral, R→sad (2026-06-10 decision)
  - IEMOCAP:  frustrated→angry, excited→happy (standard)
  - C-BESD→4cl:  disgust/fear discarded for cross-corpus comparison
"""

from typing import Optional


# ── Per-dataset raw → unified mappings ──

C_BESD_MAP = {
    # 6-class preserved
    'angry': 'angry',
    'anger': 'angry',
    'disgust': 'disgust',
    'disguist': 'disgust',  # known typo in dataset
    'fear': 'fear',
    'happy': 'happy',
    'neutral': 'neutral',
    'sad': 'sad',
}

C_BESD_MAP_4CL = {
    # Cross-corpus 4-class subset (disgust/fear dropped)
    'angry': 'angry',
    'anger': 'angry',
    'happy': 'happy',
    'neutral': 'neutral',
    'sad': 'sad',
    'disgust': None,
    'disguist': None,
    'fear': None,
}

IEMOCAP_MAP = {
    'ang': 'angry',
    'anger': 'angry',
    'frustrated': 'angry',   # frustration → angry (standard IEMOCAP merge)
    'hap': 'happy',
    'happy': 'happy',
    'exc': 'happy',
    'excited': 'happy',
    'neu': 'neutral',
    'neutral': 'neutral',
    'sad': 'sad',
    'fea': None,   # fear → discard
    'fear': None,
    'dis': None,   # disgust → discard
    'disgust': None,
    'sur': None,   # surprise → discard
    'surprise': None,
    'xxx': None,   # unknown → discard
    'oth': None,   # other → discard
}

CREMAD_MAP = {
    'ang': 'angry',
    'hap': 'happy',
    'neu': 'neutral',
    'sad': 'sad',
    'fea': None,   # fear → discard
    'dis': None,   # disgust → discard
}

FAU_AIBO_MAP = {
    # 2026-06-10 decision: A+E→Angry, P→Happy, N→Neutral, R→Sad
    'a': 'angry',    # Anger (1492)
    'e': 'angry',    # Emphatic (1200) → merged into Angry
    'p': 'happy',    # Positive (889) → Happy
    'n': 'neutral',  # Neutral (1200)
    'r': 'sad',      # Rest (1267) → Sad
}

# ── Unified label → index (6-class for C-BESD domain) ──
UNIFIED_LABEL_TO_IDX_6CL = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
}

# 4-class space (cross-corpus / non-C-BESD)
UNIFIED_LABEL_TO_IDX_4CL = {
    'angry': 0,
    'happy': 1,
    'neutral': 2,
    'sad': 3,
}

# Default (backward compatibility): 4-class
UNIFIED_LABEL_TO_IDX = UNIFIED_LABEL_TO_IDX_4CL
IDX_TO_UNIFIED_LABEL = {v: k for k, v in UNIFIED_LABEL_TO_IDX.items()}

# Dataset-level class count
DATASET_NUM_CLASSES = {
    'c-besd': 6,
    'iemocap': 4,
    'crema-d': 4,
    'fau-aibo': 4,
}

# ── Per-dataset mapping dicts ──
DATASET_MAPS = {
    'c-besd': C_BESD_MAP,
    'c-besd-4cl': C_BESD_MAP_4CL,  # for cross-corpus zero-shot
    'iemocap': IEMOCAP_MAP,
    'crema-d': CREMAD_MAP,
    'fau-aibo': FAU_AIBO_MAP,
}


class UniversalLabelMapper:
    """Map dataset-specific emotion labels into unified label space.

    Uses 6-class space for C-BESD (angry/disgust/fear/happy/neutral/sad)
    and 4-class space for FAU/IEMOCAP/CREMA-D (angry/happy/neutral/sad).

    Usage:
        mapper = UniversalLabelMapper('iemocap')
        mapper.to_index('ang')    # → 0 (4-class)
        mapper = UniversalLabelMapper('c-besd')
        mapper.to_index('fear')   # → 2 (6-class)
    """

    def __init__(self, dataset_name: str):
        dataset_key = dataset_name.lower()
        if dataset_key not in DATASET_MAPS:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Must be one of: {list(DATASET_MAPS.keys())}"
            )
        self.dataset_name = dataset_key
        self._map = DATASET_MAPS[dataset_key]
        self._num_classes = DATASET_NUM_CLASSES.get(dataset_key, 4)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def label_to_idx(self) -> dict:
        """Return the correct index mapping for this dataset."""
        if self._num_classes == 6:
            return UNIFIED_LABEL_TO_IDX_6CL
        return UNIFIED_LABEL_TO_IDX_4CL

    def __call__(self, raw_label: str) -> Optional[str]:
        """Map a raw label to unified label. Returns None if discard."""
        key = raw_label.strip().lower()
        return self._map.get(key)

    def to_index(self, raw_label: str) -> Optional[int]:
        """Map raw label directly to class index, or None if discard."""
        unified = self(raw_label)
        if unified is None:
            return None
        return self.label_to_idx[unified]


def get_label_index(unified_label: str, num_classes: int = 4) -> int:
    """Get class index for a unified label."""
    if num_classes == 6:
        return UNIFIED_LABEL_TO_IDX_6CL[unified_label]
    return UNIFIED_LABEL_TO_IDX_4CL[unified_label]
