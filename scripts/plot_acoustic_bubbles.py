"""
Acoustic feature bubble charts for C-BESD, FAU-AIBO, IEMOCAP test sets.
Pure acoustic analysis using librosa only -- no trained models needed.
Extracts frame-level F0 (YIN) and RMS, discretizes into 5 bins,
and plots bubble charts (3 datasets x 2 metrics = 6 panels).
Output: /root/autodl-tmp/acoustic_analysis/fig_acoustic_bubbles*.png
"""

import os, sys, hashlib, warnings, re
from collections import defaultdict
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# ---- Paths ----
DATASET_ROOTS = {
    'c-besd': os.environ.get('SER_C_BESD_PATH', '/root/autodl-tmp/datasets/BESD/BESD/MY'),
    'fau-aibo': os.environ.get('SER_FAU_AIBO_PATH', '/root/autodl-tmp/IS2009EmotionChallenge/IS2009EmotionChallenge/wav'),
    'iemocap': os.environ.get('SER_IEMOCAP_PATH', '/root/autodl-tmp/IEMOCAP/wavs'),
}
FAU_LABEL_FILE = '/root/autodl-tmp/IS2009EmotionChallenge/labels/IS2009EmotionChallenge/chunk_labels_5cl_corpus.txt'
OUTPUT_DIR = '/root/autodl-tmp/acoustic_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Speaker split (replicating speaker_splitter.py) ----
def _hash_speaker(speaker_id, seed=42):
    h = hashlib.md5(f"{speaker_id}:{seed}".encode()).hexdigest()
    return int(h, 16) / (16 ** len(h))

def is_test_speaker(speaker_id, seed=42):
    return _hash_speaker(speaker_id, seed) >= 0.85

# ---- Speaker extraction (replicating dataset.py) ----
def extract_speaker_cbesd(filename):
    m = re.match(r'^(\d+)', filename)
    if m:
        return 'C' + m.group(1).zfill(2)
    return 'C' + os.path.splitext(filename)[0][:2]

def extract_speaker_iemocap(filename):
    return filename[:6]

def extract_speaker_fau(filename):
    parts = filename.split('_')
    return parts[0] + '_' + parts[1]

# ---- Label mapping (replicating label_mapper.py, 4-class) ----
CBESD_DIR_MAP_4CL = {
    'ANGER': 'angry', 'HAPPY': 'happy', 'NEUTRAL': 'neutral', 'SAD': 'sad',
}

IEMOCAP_FOLDERS = ['ANGER', 'HAPPY', 'NEUTRAL', 'SAD']

IEMOCAP_LABEL_MAP = {
    'ang': 'angry', 'anger': 'angry', 'angry': 'angry', 'frustrated': 'angry',
    'hap': 'happy', 'happy': 'happy', 'exc': 'happy', 'excited': 'happy',
    'neu': 'neutral', 'neutral': 'neutral',
    'sad': 'sad',
}

FAU_LABEL_MAP = {'A': 'angry', 'E': 'angry', 'P': 'happy', 'N': 'neutral', 'R': 'sad'}

UNIFIED_4CLASS = ['angry', 'happy', 'neutral', 'sad']

def get_gender_iemocap(filename):
    """Determine gender from IEMOCAP session prefix.
    Sessions: Ses01F/Ses02F/Ses03F/Ses04F/Ses05F → female
              Ses01M/Ses02M/Ses03M/Ses04M/Ses05M → male
    """
    session = filename[:6]
    if len(session) >= 6 and session[5] == 'F':
        return 'female'
    elif len(session) >= 6 and session[5] == 'M':
        return 'male'
    return 'unknown'


# ---- Data collection ----
def collect_cbesd(root):
    results = []
    folder_map = {'anger': 'angry', 'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad'}
    for dirname in sorted(os.listdir(root)):
        dirpath = os.path.join(root, dirname)
        if not os.path.isdir(dirpath):
            continue
        unified = CBESD_DIR_MAP_4CL.get(dirname.upper())
        if unified is None:
            continue
        for fname in os.listdir(dirpath):
            if not fname.endswith('.wav') or 'copy' in fname.lower():
                continue
            sid = extract_speaker_cbesd(fname)
            if is_test_speaker(sid):
                results.append((os.path.join(dirpath, fname), unified, 'c-besd'))
    return results

def collect_iemocap(root):
    results = []
    for folder_name in IEMOCAP_FOLDERS:
        folder = os.path.join(root, folder_name)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if not fname.endswith('.wav'):
                continue
            stem = os.path.splitext(fname)[0]
            parts = stem.split('_')
            raw_label = parts[-1] if len(parts) >= 4 else parts[-1]
            if raw_label == 'frustrated':
                raw_label = 'ang'
            elif raw_label == 'excited':
                raw_label = 'exc'
            unified = IEMOCAP_LABEL_MAP.get(raw_label)
            if unified is None:
                continue
            sid = extract_speaker_iemocap(fname)
            if is_test_speaker(sid):
                results.append((os.path.join(folder, fname), unified, 'iemocap'))
    return results

def collect_fau(root, label_file):
    file_to_label = {}
    if not os.path.exists(label_file):
        print("  WARNING: FAU label file not found: " + label_file)
        return []
    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                file_to_label[parts[0] + '.wav'] = parts[1]

    results = []
    for fname in os.listdir(root):
        if not fname.endswith('.wav'):
            continue
        raw_label = file_to_label.get(fname)
        if raw_label is None:
            continue
        unified = FAU_LABEL_MAP.get(raw_label)
        if unified is None:
            continue
        sid = extract_speaker_fau(fname)
        if is_test_speaker(sid):
            results.append((os.path.join(root, fname), unified, 'fau-aibo'))
    return results


# ---- Acoustic feature extraction ----
def extract_acoustic_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        if len(y) < sr * 0.1:
            return None, None
        f0 = librosa.yin(y, fmin=65, fmax=2093, sr=sr, hop_length=320)
        voiced = f0[f0 > 0]
        mean_f0 = float(np.mean(voiced)) if len(voiced) > 0 else 0.0
        rms = librosa.feature.rms(y=y, hop_length=320)[0]
        mean_rms = float(np.mean(rms))
        return mean_f0, mean_rms
    except Exception:
        return None, None


# ---- Discretization ----
BIN_LABELS = ['VERY_LOW', 'LOW', 'MID', 'HIGH', 'VERY_HIGH']

def discretize(values, quantiles=None):
    if quantiles is None:
        arr = np.array([v for v in values if v is not None and v > 0])
        if len(arr) < 5:
            return np.zeros(len(values), dtype=int), None
        quantiles = np.percentile(arr, [20, 40, 60, 80])
    bins = np.zeros(len(values), dtype=int)
    for i, v in enumerate(values):
        if v is None or v <= 0:
            bins[i] = -1
        elif v <= quantiles[0]:
            bins[i] = 0
        elif v <= quantiles[1]:
            bins[i] = 1
        elif v <= quantiles[2]:
            bins[i] = 2
        elif v <= quantiles[3]:
            bins[i] = 3
        else:
            bins[i] = 4
    return bins, quantiles


def compute_bubble_data(f0_values, rms_values, labels, genders, dataset_name):
    # Filter valid entries
    valid_idx = [i for i, (f0, r) in enumerate(zip(f0_values, rms_values))
                 if f0 is not None and r is not None and f0 > 0 and r > 0]
    f0_valid = [f0_values[i] for i in valid_idx]
    rms_valid = [rms_values[i] for i in valid_idx]
    labels_valid = [labels[i] for i in valid_idx]
    genders_valid = [genders[i] for i in valid_idx] if genders else []
    n = len(labels_valid)
    print("  Valid utterances: {} (filtered {})".format(n, len(f0_values) - n))

    if n == 0:
        return None

    # Discretize F0
    if dataset_name == 'iemocap' and genders_valid:
        female_idx = [i for i, g in enumerate(genders_valid) if g == 'female']
        male_idx = [i for i, g in enumerate(genders_valid) if g == 'male']
        print("  Gender split: female={}, male={}, unknown={}".format(
            len(female_idx), len(male_idx),
            n - len(female_idx) - len(male_idx)))
        f0_bins = np.full(n, -1, dtype=int)
        success = False
        for idx_list in [female_idx, male_idx]:
            if len(idx_list) < 5:
                continue
            vals = [f0_valid[i] for i in idx_list]
            _, fq = discretize(vals)
            if fq is not None:
                bins_sub, _ = discretize(vals, fq)
                for j, ii in enumerate(idx_list):
                    f0_bins[ii] = bins_sub[j]
                success = True
        if success:
            keep = f0_bins >= 0
            f0_bins = f0_bins[keep]
            rms_f = [rms_valid[i] for i in range(n) if keep[i]]
            labels_f = [labels_valid[i] for i in range(n) if keep[i]]
        else:
            # Fallback: global discretization
            print("  Gender-split failed, falling back to global F0 discretization")
            f0_bins, fq = discretize(f0_valid)
            if fq is None:
                return None
            keep = f0_bins >= 0
            f0_bins = f0_bins[keep]
            rms_f = [rms_valid[i] for i in range(n) if keep[i]]
            labels_f = [labels_valid[i] for i in range(n) if keep[i]]
    else:
        f0_bins, fq = discretize(f0_valid)
        if fq is None:
            return None
        keep = f0_bins >= 0
        f0_bins = f0_bins[keep]
        rms_f = [rms_valid[i] for i in range(n) if keep[i]]
        labels_f = [labels_valid[i] for i in range(n) if keep[i]]

    # Discretize RMS
    rms_bins, rq = discretize(rms_f)
    if rq is None:
        return None
    keep_rms = rms_bins >= 0
    f0_bins = f0_bins[keep_rms]
    rms_bins = rms_bins[keep_rms]
    labels_f = [labels_f[i] for i in range(len(labels_f)) if keep_rms[i]]
    labels_rms = list(labels_f)

    n_final = len(labels_f)

    def build_matrix(bins_arr, labels_arr):
        data = {}
        for b in range(5):
            data[b] = {e: 0.0 for e in UNIFIED_4CLASS}
        total = len(labels_arr)
        if total == 0:
            return data
        for b, lbl in zip(bins_arr, labels_arr):
            if 0 <= b < 5:
                data[b][lbl] += 1.0
        for b in range(5):
            for e in UNIFIED_4CLASS:
                data[b][e] /= total
        return data

    f0_matrix = build_matrix(f0_bins, labels_f)
    rms_matrix = build_matrix(rms_bins, labels_rms)

    return {'f0': f0_matrix, 'rms': rms_matrix, 'n_valid': n_final, 'n_total': len(f0_values)}


# ---- Plotting ----
def draw_bubble_panel(ax, matrix, title):
    emotions = UNIFIED_4CLASS
    colors = plt.cm.tab10(np.linspace(0, 1, len(emotions)))
    max_prop = max(matrix[b].get(e, 0) for b in range(5) for e in emotions)
    max_prop = max(max_prop, 1e-6)

    for bi in range(5):
        for ei, emo in enumerate(emotions):
            prop = matrix[bi].get(emo, 0)
            if prop > 0:
                size = 80 + 2000 * (prop / max_prop)
                ax.scatter(bi, ei, s=size, c=[colors[ei]], alpha=0.75,
                           edgecolors='black', linewidth=0.5, zorder=3)

    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, len(emotions) - 0.5)
    ax.set_xticks(range(5))
    ax.set_xticklabels(BIN_LABELS, fontsize=7, rotation=30, ha='right')
    ax.set_yticks(range(len(emotions)))
    ax.set_yticklabels([e.capitalize() for e in emotions], fontsize=8)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')


def plot_dataset(dataset_key, bubble_data, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    name_map = {'c-besd': 'C-BESD', 'fau-aibo': 'FAU-AIBO', 'iemocap': 'IEMOCAP'}
    ds_name = name_map.get(dataset_key, dataset_key)
    n = bubble_data['n_valid']
    draw_bubble_panel(axes[0], bubble_data['f0'], '{} - Mean F0 (n={})'.format(ds_name, n))
    draw_bubble_panel(axes[1], bubble_data['rms'], '{} - Mean RMS (n={})'.format(ds_name, n))
    fig.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: " + output_path)


def plot_combined(all_bubble_data, output_path):
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    name_map = {'c-besd': 'C-BESD', 'fau-aibo': 'FAU-AIBO', 'iemocap': 'IEMOCAP'}
    for row, ds_key in enumerate(['c-besd', 'fau-aibo', 'iemocap']):
        data = all_bubble_data.get(ds_key)
        if data is None:
            continue
        n = data['n_valid']
        ds_name = name_map.get(ds_key, ds_key)
        draw_bubble_panel(axes[row, 0], data['f0'], '{} - Mean F0 (n={})'.format(ds_name, n))
        draw_bubble_panel(axes[row, 1], data['rms'], '{} - Mean RMS (n={})'.format(ds_name, n))
    fig.tight_layout(pad=2.5)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: " + output_path)


# ---- Main ----
def main():
    print("=" * 60)
    print("Acoustic Bubble Chart Generation")
    print("=" * 60)

    collectors = {
        'c-besd': lambda: collect_cbesd(DATASET_ROOTS['c-besd']),
        'fau-aibo': lambda: collect_fau(DATASET_ROOTS['fau-aibo'], FAU_LABEL_FILE),
        'iemocap': lambda: collect_iemocap(DATASET_ROOTS['iemocap']),
    }

    all_data = {}

    for ds_key, collector_fn in collectors.items():
        print("\n" + "-" * 40)
        print("Processing " + ds_key + "...")

        entries = collector_fn()
        print("  Test set entries: " + str(len(entries)))

        if not entries:
            print("  WARNING: No test entries, skipping.")
            continue

        # Extract acoustic features
        print("  Extracting acoustic features...")
        f0_list, rms_list, label_list, gender_list = [], [], [], []

        for i, (fpath, label, source) in enumerate(entries):
            if (i + 1) % 500 == 0:
                print("    ... {}/{}".format(i + 1, len(entries)))
            f0, rms = extract_acoustic_features(fpath)
            f0_list.append(f0)
            rms_list.append(rms)
            label_list.append(label)
            if ds_key == 'iemocap':
                gender_list.append(get_gender_iemocap(os.path.basename(fpath)))
            else:
                gender_list.append('unknown')

        # Compute bubble data
        print("  Computing bubble data...")
        bubble_data = compute_bubble_data(f0_list, rms_list, label_list, gender_list, ds_key)
        if bubble_data is None:
            print("  WARNING: Could not compute bubble data, skipping.")
            continue
        all_data[ds_key] = bubble_data

        # Individual plot
        safe_name = ds_key.replace('-', '')
        single_path = os.path.join(OUTPUT_DIR, 'fig_acoustic_bubbles_' + safe_name + '.png')
        plot_dataset(ds_key, bubble_data, single_path)

    # Combined plot
    if all_data:
        combined_path = os.path.join(OUTPUT_DIR, 'fig_acoustic_bubbles.png')
        plot_combined(all_data, combined_path)

    print("\n" + "=" * 60)
    print("Done! Output files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print("  {}  ({:.1f} KB)".format(fpath, size_kb))
    print("=" * 60)


if __name__ == '__main__':
    main()
