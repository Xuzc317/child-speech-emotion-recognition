#!/usr/bin/env python3
"""plot_bubbles_v5.py — Acoustic bubble charts per dataset (test set only), v5.

v5 refinements over v4:
  1. No blue solid dots at nodes — label text below baseline instead.
  2. Bubble alpha=0.55, edgecolors='none' (no border).
  3. Bubbles vertically cross the baseline — largest centered on y=0,
     smaller ones interlaced nearby.
  4. Only the largest bubble per node gets a label: "emo_abbrev XX%",
     fontsize 8, bold, placed above the bubble. Only >= 10%.
  5. Color legend on right (unchanged from v4).

Outputs to /root/autodl-tmp/acoustic_analysis/
"""

import os, sys, re, hashlib, warnings
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import soundfile as sf
import librosa

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════
FMIN, FMAX = 65, 2093
HOP_LENGTH = 320
SR = 16000
SPLIT_SEED = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15

BIN_LABELS_SHORT = ['VL', 'L', 'M', 'H', 'VH']
BIN_LABELS_FULL = ['VERY_LOW', 'LOW', 'MID', 'HIGH', 'VERY_HIGH']
N_BINS = 5

OUTPUT_DIR = '/root/autodl-tmp/acoustic_analysis'

# Emotion lists per dataset
CBESD_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
FAU_EMOTIONS  = ['angry', 'happy', 'neutral', 'sad']
IEMOCAP_EMOTIONS = ['angry', 'happy', 'neutral', 'sad']

# Fixed emotion colors
EMOTION_COLORS: Dict[str, Tuple[float, float, float]] = {
    'angry':   (255/255, 184/255, 122/255),  # orange
    'happy':   (139/255, 202/255, 139/255),  # green
    'neutral': (232/255, 136/255, 136/255),  # pink/red
    'sad':     (196/255, 171/255, 218/255),  # purple
    'disgust': (180/255, 180/255, 210/255),  # purple-grey
    'fear':    (160/255, 200/255, 180/255),  # grey-green
}

# Emotion abbreviations and display names
EMOTION_ABBREV = {
    'angry': 'ang', 'disgust': 'dis', 'fear': 'fea',
    'happy': 'hap', 'neutral': 'neu', 'sad': 'sad',
}
EMOTION_DISPLAY = {
    'angry': 'Angry', 'disgust': 'Disgust', 'fear': 'Fear',
    'happy': 'Happy', 'neutral': 'Neutral', 'sad': 'Sad',
}

# Dataset paths (from env vars)
C_BESD_PATH = os.environ.get('SER_C_BESD_PATH',
    '/root/autodl-tmp/datasets/BESD/BESD/MY')
IEMOCAP_PATH = os.environ.get('SER_IEMOCAP_PATH',
    '/root/autodl-tmp/IEMOCAP/wavs')
FAU_AIBO_PATH = os.environ.get('SER_FAU_AIBO_PATH',
    '/root/autodl-tmp/IS2009EmotionChallenge/IS2009EmotionChallenge/wav')

# FAU label file candidates
FAU_LABEL_FILE_CANDIDATES = [
    os.path.join(os.path.dirname(FAU_AIBO_PATH), 'labels',
                 'IS2009EmotionChallenge', 'chunk_labels_5cl_corpus.txt'),
    os.path.join(os.path.dirname(os.path.dirname(FAU_AIBO_PATH)),
                 'labels', 'IS2009EmotionChallenge', 'chunk_labels_5cl_corpus.txt'),
]

# Label mappers
IEMOCAP_MAP = {
    'ang': 'angry', 'anger': 'angry', 'angry': 'angry',
    'frustrated': 'angry',
    'hap': 'happy', 'happy': 'happy', 'exc': 'happy', 'excited': 'happy',
    'neu': 'neutral', 'neutral': 'neutral',
    'sad': 'sad',
}
IEMOCAP_DISCARD = {'fea', 'fear', 'dis', 'disgust', 'sur', 'surprise', 'xxx', 'oth'}

FAU_AIBO_MAP = {
    'a': 'angry', 'e': 'angry', 'p': 'happy', 'n': 'neutral', 'r': 'sad',
}

C_BESD_FOLDERS = {
    'angry': 'ANGER', 'disgust': 'DISGUST', 'fear': 'FEAR',
    'happy': 'HAPPY', 'neutral': 'NEUTRAL', 'sad': 'SAD',
}

IEMOCAP_FOLDERS = {'ANGER', 'HAPPY', 'NEUTRAL', 'SAD'}


# ═══════════════════════════════════════════════════════════════════
# Speaker splitting (replicates speaker_splitter.py logic)
# ═══════════════════════════════════════════════════════════════════

def _hash_speaker(speaker_id: str, seed: int = 42) -> float:
    h = hashlib.md5(f"{speaker_id}:{seed}".encode()).hexdigest()
    return int(h, 16) / (16 ** len(h))


def is_test_speaker(speaker_id: str) -> bool:
    h = _hash_speaker(speaker_id, SPLIT_SEED)
    return h >= (TRAIN_RATIO + VAL_RATIO)


# ═══════════════════════════════════════════════════════════════════
# Speaker ID extraction
# ═══════════════════════════════════════════════════════════════════

def extract_speaker_cbesd(filename: str) -> str:
    m = re.match(r'^(\d+)', filename)
    if m:
        return f'C{m.group(1).zfill(2)}'
    basename = os.path.splitext(filename)[0].lower()
    return f'C{basename[:2]}'


def extract_speaker_iemocap(filename: str) -> str:
    return filename[:6]


def extract_speaker_fau(filename: str) -> str:
    parts = filename.split('_')
    return f"{parts[0]}_{parts[1]}"


def iemocap_gender(speaker_id: str) -> str:
    return speaker_id[-1].upper()


# ═══════════════════════════════════════════════════════════════════
# Data collection per dataset (test set only)
# ═══════════════════════════════════════════════════════════════════

def collect_cbesd(root: str) -> List[Tuple[str, str, str]]:
    """Return [(wav_path, unified_label, speaker_id), ...] for test speakers, 6-class."""
    entries = []
    for emo_label, folder in C_BESD_FOLDERS.items():
        folder_path = os.path.join(root, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if not fname.endswith('.wav') or 'copy' in fname.lower():
                continue
            sid = extract_speaker_cbesd(fname)
            if not is_test_speaker(sid):
                continue
            entries.append((os.path.join(folder_path, fname), emo_label, sid))
    return entries


def collect_iemocap(root: str) -> List[Tuple[str, str, str]]:
    """Return [(wav_path, unified_label, speaker_id), ...] for test speakers, 4-class."""
    entries = []
    for folder in IEMOCAP_FOLDERS:
        folder_path = os.path.join(root, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if not fname.endswith('.wav'):
                continue
            stem = os.path.splitext(fname)[0]
            parts = stem.split('_')
            raw_label = parts[-1] if len(parts) >= 4 else parts[-1]
            raw_label = raw_label.strip().lower()

            if raw_label in IEMOCAP_DISCARD:
                continue
            unified = IEMOCAP_MAP.get(raw_label)
            if unified is None:
                continue

            sid = extract_speaker_iemocap(fname)
            if not is_test_speaker(sid):
                continue
            entries.append((os.path.join(folder_path, fname), unified, sid))
    return entries


def collect_fau_aibo(root: str) -> List[Tuple[str, str, str]]:
    """Return [(wav_path, unified_label, speaker_id), ...] for test speakers."""
    label_file = None
    for cand in FAU_LABEL_FILE_CANDIDATES:
        if os.path.isfile(cand):
            label_file = cand
            break
    if label_file is None:
        print(f"  ERROR: FAU label file not found. Tried: {FAU_LABEL_FILE_CANDIDATES}")
        return []

    file_to_label: Dict[str, str] = {}
    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                file_to_label[parts[0] + '.wav'] = parts[1].strip().lower()

    entries = []
    for fname in os.listdir(root):
        if not fname.endswith('.wav'):
            continue
        raw_label = file_to_label.get(fname)
        if raw_label is None:
            continue
        unified = FAU_AIBO_MAP.get(raw_label)
        if unified is None:
            continue
        sid = extract_speaker_fau(fname)
        if not is_test_speaker(sid):
            continue
        entries.append((os.path.join(root, fname), unified, sid))
    return entries


# ═══════════════════════════════════════════════════════════════════
# Acoustic feature extraction
# ═══════════════════════════════════════════════════════════════════

def extract_features(wav_path: str) -> Tuple[Optional[float], Optional[float]]:
    """Extract mean F0 (voiced frames) and mean RMS energy.

    Returns (mean_f0, mean_rms) or (None, None) on failure.
    """
    try:
        y, sr = librosa.load(wav_path, sr=SR, mono=True)
        if len(y) < HOP_LENGTH:
            return None, None

        f0 = librosa.yin(y, fmin=FMIN, fmax=FMAX, sr=SR, hop_length=HOP_LENGTH)
        voiced = f0[f0 > 0]
        mean_f0 = float(np.mean(voiced)) if len(voiced) > 0 else None

        rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
        mean_rms = float(np.mean(rms))

        return mean_f0, mean_rms
    except Exception:
        return None, None


# ═══════════════════════════════════════════════════════════════════
# Discretization
# ═══════════════════════════════════════════════════════════════════

def compute_quantile_bins(values: np.ndarray) -> np.ndarray:
    """Compute 4 boundaries (q0.2, q0.4, q0.6, q0.8) from all values."""
    return np.quantile(values, [0.2, 0.4, 0.6, 0.8])


def assign_bin(value: float, boundaries: np.ndarray) -> int:
    """Map value to bin index 0-4 based on quantile boundaries."""
    for i, b in enumerate(boundaries):
        if value < b:
            return i
    return len(boundaries)


# ═══════════════════════════════════════════════════════════════════
# Frequency matrix building
# ═══════════════════════════════════════════════════════════════════

def build_freq_matrix(
    features: Dict[str, List[Tuple[float, float]]],
    f0_boundaries: np.ndarray,
    rms_boundaries: np.ndarray,
    emotions: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (n_emotions x 5) frequency matrices for F0 and RMS.

    f0_matrix[ei, bi] = proportion of emotion ei's samples that fall in F0 bin bi.
    """
    n = len(emotions)
    f0_matrix = np.zeros((n, N_BINS))
    rms_matrix = np.zeros((n, N_BINS))

    for ei, emo in enumerate(emotions):
        samples = features.get(emo, [])
        total = len(samples)
        if total == 0:
            continue
        for f0_val, rms_val in samples:
            if f0_val is not None:
                b = assign_bin(f0_val, f0_boundaries)
                f0_matrix[ei, b] += 1
            if rms_val is not None:
                b = assign_bin(rms_val, rms_boundaries)
                rms_matrix[ei, b] += 1

        # Normalize: proportion of this emotion's samples in each bin
        if f0_matrix[ei].sum() > 0:
            f0_matrix[ei] /= f0_matrix[ei].sum()
        if rms_matrix[ei].sum() > 0:
            rms_matrix[ei] /= rms_matrix[ei].sum()

    return f0_matrix, rms_matrix


def build_freq_matrix_gendered(
    features: Dict[str, List[Tuple[float, float, str]]],
    f0_boundaries_f: np.ndarray,
    f0_boundaries_m: np.ndarray,
    rms_boundaries: np.ndarray,
    emotions: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """IEMOCAP: gender-specific F0 boundaries, shared RMS boundaries."""
    n = len(emotions)
    f0_matrix = np.zeros((n, N_BINS))
    rms_matrix = np.zeros((n, N_BINS))

    for ei, emo in enumerate(emotions):
        samples = features.get(emo, [])
        total = len(samples)
        if total == 0:
            continue
        for f0_val, rms_val, gender in samples:
            if f0_val is not None:
                boundaries = f0_boundaries_f if gender == 'F' else f0_boundaries_m
                b = assign_bin(f0_val, boundaries)
                f0_matrix[ei, b] += 1
            if rms_val is not None:
                b = assign_bin(rms_val, rms_boundaries)
                rms_matrix[ei, b] += 1

        if f0_matrix[ei].sum() > 0:
            f0_matrix[ei] /= f0_matrix[ei].sum()
        if rms_matrix[ei].sum() > 0:
            rms_matrix[ei] /= rms_matrix[ei].sum()

    return f0_matrix, rms_matrix


# ═══════════════════════════════════════════════════════════════════
# Plotting (v5 layout — Tsangko et al. reference style, refined)
# ═══════════════════════════════════════════════════════════════════

def plot_dataset_figure_v5(
    f0_matrix: np.ndarray,
    rms_matrix: np.ndarray,
    emotions: List[str],
    dataset_label: str,
    n_samples: int,
    output_path: str,
):
    """Plot v5: N rows (acoustic features) x 1 column, refined bubbles.

    v5 changes:
      - No node markers (blue dots).
      - Node labels below baseline, centered.
      - Bubbles alpha=0.55, no edge border.
      - Bubbles vertically cross the baseline (largest at y=0).
      - Only largest bubble per node labelled: "emo_abbrev XX%", size 8, bold.
      - Label threshold >= 10%.
    """
    matrices = [f0_matrix, rms_matrix]
    row_names = ['F0 (Hz)', 'RMS Energy']
    n_rows = 2  # F0, RMS

    is_cbesd = (len(emotions) == 6)
    class_label = '6-class' if is_cbesd else '4-class'

    fig = plt.figure(figsize=(12, 8), facecolor='white')

    # Define layout: 2 rows of axes + legend space on right
    gs = fig.add_gridspec(n_rows, 1, hspace=0.45,
                          left=0.06, right=0.78, top=0.92, bottom=0.08)

    axes = [fig.add_subplot(gs[row, 0]) for row in range(n_rows)]

    # Global max proportion for consistent bubble sizing
    global_max = max(f0_matrix.max(), rms_matrix.max(), 0.01)
    MAX_BUBBLE = 1800  # marker size for proportion=1.0

    # ── Layout of bubbles at each node ──
    for row in range(n_rows):
        matrix = matrices[row]
        ax = axes[row]

        n_emo = len(emotions)

        # Draw horizontal baseline (blue)
        ax.axhline(y=0, color='#4a90d9', linewidth=1.2, zorder=1)

        # 5 node positions
        nodes_x = [0, 1, 2, 3, 4]

        for bi, (node_label, node_x) in enumerate(
            zip(BIN_LABELS_FULL, nodes_x)
        ):
            # v5: NO blue dot — only text label below baseline
            label_y = -0.35
            ax.text(node_x, label_y, node_label.replace('_', '\n'),
                    ha='center', va='top', fontsize=7.5,
                    color='#333333', fontweight='normal', zorder=4)

            # Collect (emotion_idx, proportion) for this bin
            items = []
            for ei in range(n_emo):
                prop = matrix[ei, bi]
                if prop >= 0.005:  # minimum threshold to show
                    items.append((ei, prop))

            # Sort by proportion descending (largest first)
            items.sort(key=lambda x: -x[1])

            # v5 bubble positioning: largest at y=0 (crossing baseline),
            # smaller ones interlaced with small offsets above/below.
            # Offsets: [0, 0.08, -0.08, 0.14, -0.14, 0.20]
            base_offsets = [0, 0.08, -0.08, 0.14, -0.14, 0.20]

            # Per-node deterministic jitter
            rng = np.random.RandomState(42 + row * 10 + bi)
            x_jitters = rng.uniform(-0.05, 0.05, len(items))

            for position, (ei, prop) in enumerate(items):
                emo = emotions[ei]
                color = EMOTION_COLORS[emo]

                # v5: y-offset — largest bubble at 0, others interlaced
                y_off = base_offsets[min(position, len(base_offsets) - 1)]

                # Bubble size proportional to proportion
                bubble_size = MAX_BUBBLE * prop
                bubble_size = max(bubble_size, 30)  # minimum visible

                # v5: alpha=0.55, edgecolors='none' (no border)
                ax.scatter(
                    node_x + x_jitters[position], y_off,
                    s=bubble_size, c=[color], alpha=0.55,
                    edgecolors='none', zorder=6,
                )

                # v5: Only label the largest bubble (position==0) and >= 10%
                if position == 0 and prop >= 0.10:
                    abbrev = EMOTION_ABBREV[emo]
                    # Label placed above the bubble center
                    text_y = y_off + 0.06
                    ax.text(node_x + x_jitters[position], text_y,
                            f'{abbrev} {prop*100:.0f}%',
                            ha='center', va='center',
                            fontsize=8, fontweight='bold',
                            color='#1a1a1a', alpha=0.9, zorder=7)

        # ── Row label on left ──
        ax.text(-0.55, 0.08, row_names[row],
                ha='right', va='center', fontsize=12, fontweight='bold',
                color='#333333', transform=ax.transData)

        # ── Axis formatting ──
        ax.set_xlim(-0.8, 4.2)
        ax.set_ylim(-1.0, 0.8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('white')

        # No spines
        for spine_name in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine_name].set_visible(False)

    # ── Right-side color legend ──
    legend_ax = fig.add_axes([0.80, 0.15, 0.16, 0.70])
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, len(emotions) + 1)
    legend_ax.axis('off')

    legend_ax.text(0.5, len(emotions) + 0.6, 'Emotion',
                   ha='center', va='bottom', fontsize=10, fontweight='bold',
                   transform=legend_ax.transData)

    for ei, emo in enumerate(emotions):
        y_pos = len(emotions) - ei - 0.5
        color = EMOTION_COLORS[emo]

        # Colored circle (v5: alpha=0.55 too, no edge for consistency)
        legend_ax.scatter(0.25, y_pos, s=200, c=[color], alpha=0.82,
                         edgecolors='white', linewidths=0.8,
                         transform=legend_ax.transData, zorder=5)

        # Emotion full name
        legend_ax.text(0.55, y_pos, EMOTION_DISPLAY[emo],
                       ha='left', va='center', fontsize=9.5,
                       transform=legend_ax.transData, zorder=5)

    # ── Title ──
    fig.suptitle(
        f'{dataset_label}  ({class_label},  n={n_samples})',
        fontsize=14, fontweight='bold', x=0.42, y=0.98,
    )

    # ── Save ──
    fig.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def combine_vertically(input_paths: List[str], output_path: str):
    """Stack PNGs vertically into one combined figure."""
    from PIL import Image

    images = [Image.open(p) for p in input_paths if os.path.isfile(p)]
    if not images:
        print("  No images to combine")
        return

    widths = [im.width for im in images]
    max_width = max(widths)
    total_height = sum(im.height for im in images)

    combined = Image.new('RGB', (max_width, total_height), 'white')
    y_offset = 0
    for im in images:
        x_offset = (max_width - im.width) // 2
        combined.paste(im, (x_offset, y_offset))
        y_offset += im.height

    combined.save(output_path)
    print(f"  Saved combined: {output_path}")


# ═══════════════════════════════════════════════════════════════════
# Dataset processing pipeline
# ═══════════════════════════════════════════════════════════════════

def process_dataset(name: str, collect_fn, root: str, emotions: List[str],
                    gendered: bool = False):
    """Collect test data, extract features, compute bins, return matrices."""
    print(f"\n{'='*60}")
    print(f"Processing {name} ({len(emotions)}-class, gendered={gendered})")
    print(f"  Root: {root}")
    print(f"  Collecting test-set files...")

    entries = collect_fn(root)
    print(f"  Collected {len(entries)} test-set files")

    if len(entries) == 0:
        print(f"  SKIP: No test samples for {name}")
        return None, None, 0

    # Count per emotion
    emo_counts = defaultdict(int)
    for _, emo, _ in entries:
        emo_counts[emo] += 1
    print(f"  Per emotion: {dict(emo_counts)}")
    n_samples = sum(emo_counts.values())

    # Extract features
    print(f"  Extracting acoustic features...")
    if gendered:
        features: Dict = {e: [] for e in emotions}
    else:
        features: Dict = {e: [] for e in emotions}

    missing_f0, missing_rms = 0, 0
    for i, (wav_path, emo, sid) in enumerate(entries):
        if (i + 1) % 500 == 0:
            print(f"    ... {i+1}/{len(entries)}")
        f0_val, rms_val = extract_features(wav_path)
        if f0_val is None:
            missing_f0 += 1
        if rms_val is None:
            missing_rms += 1
        if gendered:
            gender = iemocap_gender(sid)
            features[emo].append((f0_val, rms_val, gender))
        else:
            features[emo].append((f0_val, rms_val))

    print(f"  Feature extraction done. Missing F0: {missing_f0}, Missing RMS: {missing_rms}")

    # Compute bin boundaries
    if gendered:
        f0_f_vals = []
        f0_m_vals = []
        rms_vals = []
        for emo in emotions:
            for f0_val, rms_val, gender in features[emo]:
                if f0_val is not None:
                    if gender == 'F':
                        f0_f_vals.append(f0_val)
                    else:
                        f0_m_vals.append(f0_val)
                if rms_val is not None:
                    rms_vals.append(rms_val)

        f0_f_arr = np.array(f0_f_vals)
        f0_m_arr = np.array(f0_m_vals)
        rms_arr = np.array(rms_vals)

        f0_bounds_f = compute_quantile_bins(f0_f_arr) if len(f0_f_arr) > 0 else np.zeros(4)
        f0_bounds_m = compute_quantile_bins(f0_m_arr) if len(f0_m_arr) > 0 else np.zeros(4)
        rms_bounds = compute_quantile_bins(rms_arr) if len(rms_arr) > 0 else np.zeros(4)

        print(f"  F0 boundaries (F): {np.round(f0_bounds_f, 1)}")
        print(f"  F0 boundaries (M): {np.round(f0_bounds_m, 1)}")
        print(f"  RMS boundaries:    {np.round(rms_bounds, 4)}")

        f0_matrix, rms_matrix = build_freq_matrix_gendered(
            features, f0_bounds_f, f0_bounds_m, rms_bounds, emotions,
        )
    else:
        f0_vals = []
        rms_vals = []
        for emo in emotions:
            for f0_val, rms_val in features[emo]:
                if f0_val is not None:
                    f0_vals.append(f0_val)
                if rms_val is not None:
                    rms_vals.append(rms_val)

        f0_arr = np.array(f0_vals)
        rms_arr = np.array(rms_vals)
        f0_bounds = compute_quantile_bins(f0_arr) if len(f0_arr) > 0 else np.zeros(4)
        rms_bounds = compute_quantile_bins(rms_arr) if len(rms_arr) > 0 else np.zeros(4)

        print(f"  F0 boundaries:  {np.round(f0_bounds, 1)}")
        print(f"  RMS boundaries: {np.round(rms_bounds, 4)}")

        f0_matrix, rms_matrix = build_freq_matrix(features, f0_bounds, rms_bounds, emotions)

    return f0_matrix, rms_matrix, n_samples


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {}

    # 1. C-BESD (6-class, non-gendered)
    f0_m, rms_m, n = process_dataset(
        'C-BESD', collect_cbesd, C_BESD_PATH, CBESD_EMOTIONS, gendered=False,
    )
    if f0_m is not None:
        results['cbesd'] = (f0_m, rms_m, n, CBESD_EMOTIONS)

    # 2. FAU Aibo (4-class, non-gendered)
    f0_m, rms_m, n = process_dataset(
        'FAU Aibo', collect_fau_aibo, FAU_AIBO_PATH, FAU_EMOTIONS, gendered=False,
    )
    if f0_m is not None:
        results['fauaibo'] = (f0_m, rms_m, n, FAU_EMOTIONS)

    # 3. IEMOCAP (4-class, gendered F0 bins)
    f0_m, rms_m, n = process_dataset(
        'IEMOCAP', collect_iemocap, IEMOCAP_PATH, IEMOCAP_EMOTIONS, gendered=True,
    )
    if f0_m is not None:
        results['iemocap'] = (f0_m, rms_m, n, IEMOCAP_EMOTIONS)

    # ── Plotting ──
    print(f"\n{'='*60}")
    print("Generating v5 figures (refined bubbles, no dots, translucent)...")

    name_map = {'cbesd': 'C-BESD', 'fauaibo': 'FAU Aibo', 'iemocap': 'IEMOCAP'}
    output_paths = []

    for key in ['cbesd', 'fauaibo', 'iemocap']:
        if key in results:
            f0_m, rms_m, n, emotions = results[key]
            path = os.path.join(OUTPUT_DIR, f'fig_bubbles_v4_{key}.png')
            plot_dataset_figure_v5(
                f0_m, rms_m, emotions, name_map[key], n, path,
            )
            output_paths.append(path)

    # Combined
    if len(output_paths) >= 2:
        combined_path = os.path.join(OUTPUT_DIR, 'fig_bubbles_v4_combined.png')
        combine_vertically(output_paths, combined_path)

    print(f"\nDone! Output directory: {OUTPUT_DIR}")
    for p in sorted(os.listdir(OUTPUT_DIR)):
        if p.endswith('.png') and 'v4' in p:
            fpath = os.path.join(OUTPUT_DIR, p)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  {p} ({size_kb:.1f} KB)")


if __name__ == '__main__':
    main()
