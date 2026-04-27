"""Speaker-independent preprocessing for BESD MY dataset.

Splits unique speakers 70/30 to prevent data leakage.
Both train and test: 3x processing per WAV (original + noise + stretch+pitch).
"""
import os
import re
import sys
import numpy as np

# ---- Config ----
WAV_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "数据集", "BESD", "BESD", "MY")
OUT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_RATIO = 0.7
SEED = 42
CLASS_NAMES = ['ANGER', 'DISGUST', 'FEAR', 'HAPPY', 'NEUTRAL', 'SAD']

TRAIN_DATA = os.path.join(OUT_DIR, "train_data.npy")
TRAIN_LABEL = os.path.join(OUT_DIR, "train_labels.npy")
TEST_DATA = os.path.join(OUT_DIR, "test_data.npy")
TEST_LABEL = os.path.join(OUT_DIR, "test_labels.npy")

EMOTION_PATTERNS = ['angry', 'anger', 'disgust', 'disguist', 'fear', 'happy', 'neutral', 'sad']


def normalize_speaker_id(raw_id):
    """Normalize speaker ID: fix hyphens, missing dots, case."""
    sid = raw_id.upper()
    sid = sid.replace('-', '_')  # '1.EF-12' -> '1.EF_12'
    # Fix missing dot: '3EF_9' -> '3.EF_9'
    sid = re.sub(r'^(\d+)([ET][FM])', r'\1.\2', sid)
    return sid


def extract_speaker_id(filename):
    """Extract speaker ID from WAV filename.

    ENGLISH: '1.EF_12 Angry_1.wav' -> '1.EF_12'
    TELUGU:  '1.TF_12_angry_1.wav' -> '1.TF_12'
    Handles 'anger' variant, 'disguist' typo, missing-dot anomalies.
    """
    basename = os.path.splitext(filename)[0].lower()
    for emotion in EMOTION_PATTERNS:
        idx = basename.find(emotion)
        if idx != -1:
            raw = filename[:idx].rstrip(' _-')
            return normalize_speaker_id(raw)
    # Fallback: take everything before the last dot segment
    raw = filename[:filename.rfind('.')].rstrip(' _-')
    return normalize_speaker_id(raw)


def collect_wav_files(wav_dir=None):
    """Collect all WAV files from MY/ grouped by class.

    Returns list of (filepath, label_idx, speaker_id).
    Excludes .pk files, Copy duplicates, and _not_for_github.
    """
    if wav_dir is None:
        wav_dir = WAV_DIR
    entries = []
    for cls_name in CLASS_NAMES:
        cls_dir = os.path.join(wav_dir, cls_name)
        if not os.path.isdir(cls_dir):
            print(f"  WARNING: directory not found: {cls_dir}")
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if not fname.endswith('.wav'):
                continue
            if 'copy' in fname.lower():
                print(f"  SKIP (Copy): {fname}")
                continue
            fpath = os.path.join(cls_dir, fname)
            sid = extract_speaker_id(fname)
            entries.append((fpath, CLASS_NAMES.index(cls_name), sid))
    return entries


def split_speakers(entries):
    """Split unique speaker IDs 70/30 stratified by emotion-class profile.

    Speakers are grouped by which combination of emotion classes they have
    recordings for (their "profile"). Each profile group is split 70/30.
    This preserves per-class balance better than a global speaker shuffle
    while guaranteeing zero speaker overlap between train and test.
    """
    speaker_map = {}
    speaker_profile = {}
    for path, label, sid in entries:
        speaker_map.setdefault(sid, []).append((path, label))
        speaker_profile.setdefault(sid, set()).add(label)

    # Group speakers by their emotion profile (frozenset of class indices)
    profile_groups = {}
    for sid, profile in speaker_profile.items():
        key = frozenset(profile)
        profile_groups.setdefault(key, []).append(sid)

    rng = np.random.RandomState(SEED)
    train_sids = set()
    test_sids = set()

    for profile, sids in profile_groups.items():
        sorted_sids = sorted(sids)
        perm = rng.permutation(len(sorted_sids))
        n_train = max(1, int(len(sorted_sids) * TRAIN_RATIO))
        for i in perm[:n_train]:
            train_sids.add(sorted_sids[i])
        for i in perm[n_train:]:
            test_sids.add(sorted_sids[i])

    assert train_sids.isdisjoint(test_sids), "FATAL: speaker overlap detected!"

    train_entries = []
    test_entries = []
    for sid in train_sids:
        for path, label in speaker_map[sid]:
            train_entries.append((path, label, sid))
    for sid in test_sids:
        for path, label in speaker_map[sid]:
            test_entries.append((path, label, sid))

    return train_entries, test_entries, train_sids, test_sids


def extract_features_from_entries(entries, split_name):
    """Extract 3x processing per WAV (original + noise + stretch+pitch)."""
    from src.data.dataset import get_features
    features = []
    labels = []
    skipped = 0
    for path, label, _ in entries:
        try:
            feat = get_features(path)  # shape: (3, 162)
            for k in range(3):
                features.append(feat[k])
                labels.append(label)
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"  SKIP {split_name}: {os.path.basename(path)} — {e}")
    if skipped:
        print(f"  Total skipped {split_name} files: {skipped}")
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64)


def main():
    print("=" * 60)
    print("Speaker-Independent Preprocessing: MY Dataset")
    print(f"WAV dir: {WAV_DIR}")
    print(f"Seed: {SEED}, Train ratio: {TRAIN_RATIO}")
    print(f"Output: {OUT_DIR}")
    print("=" * 60)

    # Step 1: Collect WAV files
    print("\n[1/5] Collecting WAV files from MY/ ...")
    entries = collect_wav_files()
    print(f"  Total: {len(entries)} WAV files")
    unique = len(set(sid for _, _, sid in entries))
    print(f"  Unique speakers: {unique}")

    # Step 2: Split by speaker
    print("\n[2/5] Splitting speakers 70/30 (seed=42)...")
    train_entries, test_entries, train_sids, test_sids = split_speakers(entries)
    print(f"  Train speakers: {len(train_sids)}")
    print(f"  Test speakers:  {len(test_sids)}")
    print(f"  Train WAVs:     {len(train_entries)}")
    print(f"  Test WAVs:      {len(test_entries)}")
    assert train_sids.isdisjoint(test_sids), "FATAL: speaker overlap!"

    # Show a few speaker IDs from each split
    print(f"  Train speakers (sample): {sorted(train_sids)[:5]}...")
    print(f"  Test speakers (sample):  {sorted(test_sids)[:5]}...")

    # Step 3: Extract training features
    print("\n[3/5] Extracting training features (3x processing per WAV)...")
    X_train, y_train = extract_features_from_entries(train_entries, "train")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  Label distribution: {np.bincount(y_train)}")

    # Step 4: Extract test features
    print("\n[4/5] Extracting test features (3x processing per WAV)...")
    X_test, y_test = extract_features_from_entries(test_entries, "test")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"  Label distribution: {np.bincount(y_test)}")

    # Step 5: Save
    print("\n[5/5] Saving files...")
    np.save(TRAIN_DATA, X_train)
    np.save(TRAIN_LABEL, y_train)
    np.save(TEST_DATA, X_test)
    np.save(TEST_LABEL, y_test)
    print(f"  {TRAIN_DATA} ({os.path.getsize(TRAIN_DATA) / 1024 / 1024:.1f} MB)")
    print(f"  {TRAIN_LABEL} ({os.path.getsize(TRAIN_LABEL) / 1024:.1f} KB)")
    print(f"  {TEST_DATA} ({os.path.getsize(TEST_DATA) / 1024 / 1024:.1f} MB)")
    print(f"  {TEST_LABEL} ({os.path.getsize(TEST_LABEL) / 1024:.1f} KB)")

    print("\n" + "=" * 60)
    print("Preprocessing complete — speaker-independent, no data leakage.")
    print(f"Train: {len(X_train)} samples from {len(train_sids)} speakers (3x processing)")
    print(f"Test:  {len(X_test)} samples from {len(test_sids)} speakers (3x processing)")
    print("=" * 60)


if __name__ == "__main__":
    main()
