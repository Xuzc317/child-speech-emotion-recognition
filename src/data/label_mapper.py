"""Universal 4-class label mapping across all datasets.

Unified space: {'angry', 'happy', 'neutral', 'sad'}
  - "excited" merged into "happy"
  - "fear", "disgust", "surprise", "boredom" discarded (return None)
"""

from typing import Optional


# ── Per-dataset raw → unified mappings ──

C_BESD_MAP = {
    'angry': 'angry',
    'anger': 'angry',
    'happy': 'happy',
    'neutral': 'neutral',
    'sad': 'sad',
    'disgust': None,   # discard
    'fear': None,       # discard
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
    'a': 'angry',    # Angry
    'r': 'angry',    # Reprimanding → angry
    'e': 'happy',    # Empathic → happy
    'p': 'happy',    # Pleasure/Positive → happy
    'n': 'neutral',  # Neutral
}

# ── Unified label → index ──
UNIFIED_LABEL_TO_IDX = {
    'angry': 0,
    'happy': 1,
    'neutral': 2,
    'sad': 3,
}

IDX_TO_UNIFIED_LABEL = {v: k for k, v in UNIFIED_LABEL_TO_IDX.items()}

# ── Per-dataset mapping dicts ──
DATASET_MAPS = {
    'c-besd': C_BESD_MAP,
    'iemocap': IEMOCAP_MAP,
    'crema-d': CREMAD_MAP,
    'fau-aibo': FAU_AIBO_MAP,
}


class UniversalLabelMapper:
    """Map dataset-specific emotion labels into unified 4-class space.

    Usage:
        mapper = UniversalLabelMapper('iemocap')
        unified = mapper('ang')    # → 'angry'
        unified = mapper('fea')    # → None (discard)
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

    def __call__(self, raw_label: str) -> Optional[str]:
        """Map a raw label to unified label.

        Returns None if the label should be discarded.
        """
        key = raw_label.strip().lower()
        return self._map.get(key)

    def to_index(self, raw_label: str) -> Optional[int]:
        """Map raw label directly to class index (0-3), or None if discard."""
        unified = self(raw_label)
        if unified is None:
            return None
        return UNIFIED_LABEL_TO_IDX[unified]


def get_label_index(unified_label: str) -> int:
    """Get class index (0-3) for a unified label."""
    return UNIFIED_LABEL_TO_IDX[unified_label]
