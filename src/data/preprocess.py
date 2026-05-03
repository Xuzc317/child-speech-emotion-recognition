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
    """[DEPRECATED v5] 旧 70/30 划分，仅 train/test 两路。

    已被 split_speakers_7_3_with_inner_val() 替代。
    Phase 1 基线实验 (emotion2vec, WavLM linear probe) 来自此函数。

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


def split_speakers_3way(entries, train_ratio=0.6, val_ratio=0.2):
    """[DEPRECATED v5] 60/20/20 三路 speaker 划分。

    已被 split_speakers_7_3_with_inner_val() 替代。
    历史 Phase 3 实验结果来自此函数。
    仅保留用于复现历史实验。

    - train: 用于模型训练
    - val:   用于 early stopping 和超参选择
    - test:  仅用于最终评估（严格 hold-out）

    test_ratio = 1 - train_ratio - val_ratio
    """
    speaker_map = {}
    speaker_profile = {}
    for path, label, sid in entries:
        speaker_map.setdefault(sid, []).append((path, label))
        speaker_profile.setdefault(sid, set()).add(label)

    profile_groups = {}
    for sid, profile in speaker_profile.items():
        key = frozenset(profile)
        profile_groups.setdefault(key, []).append(sid)

    rng = np.random.RandomState(SEED)
    train_sids, val_sids, test_sids = set(), set(), set()

    for profile, sids in profile_groups.items():
        sorted_sids = sorted(sids)
        perm = rng.permutation(len(sorted_sids))
        n_train = max(1, int(len(sorted_sids) * train_ratio))
        n_val = max(1, int(len(sorted_sids) * val_ratio))
        for i in perm[:n_train]:
            train_sids.add(sorted_sids[i])
        for i in perm[n_train:n_train + n_val]:
            val_sids.add(sorted_sids[i])
        for i in perm[n_train + n_val:]:
            test_sids.add(sorted_sids[i])

    assert train_sids.isdisjoint(val_sids), "FATAL: train/val overlap!"
    assert train_sids.isdisjoint(test_sids), "FATAL: train/test overlap!"
    assert val_sids.isdisjoint(test_sids), "FATAL: val/test overlap!"

    def gather(sids):
        result = []
        for sid in sids:
            for path, label in speaker_map[sid]:
                result.append((path, label, sid))
        return result

    return gather(train_sids), gather(val_sids), gather(test_sids), train_sids, val_sids, test_sids


def split_speakers_7_3_with_inner_val(entries, outer_train_ratio=0.8, inner_val_ratio=0.15, seed=42):
    """外部 8:2 + 内部 val 的说话人独立划分（最终协议 v5）。

    协议：
      1. 外部 speaker-independent 8:2 划分：
         - 80% speaker → trainval
         - 20% speaker → final test (严格 hold-out)
      2. 内部从 trainval 中划分 validation：
         - 从 80% trainval speaker 中再划出 inner_val_ratio 作为 val
         - 默认 inner_val_ratio=0.15 → train 约 68%, val 约 12%
      3. adapter_init 只用 train speaker，不用 val/test
      4. val 只用于 early stopping 和模型选择
      5. test 严格 hold-out，训练中完全不可见，只在最终评估时使用一次

    保证：
      - train/val/test speaker 三者互斥
      - profile-stratified 每层都保持情绪分布均衡
      - 对过小的 profile 分组做安全处理（避免 max(1) 导致比例严重失真）

    Returns:
      train_entries, val_entries, test_entries,
      train_sids, val_sids, test_sids,
      stats: dict with speaker counts, sample counts, class distribution per split
    """
    rng = np.random.RandomState(seed)

    # ── 建立 speaker → entries 映射和 profile 分组 ──
    speaker_map = {}
    speaker_profile = {}
    for path, label, sid in entries:
        speaker_map.setdefault(sid, []).append((path, label))
        speaker_profile.setdefault(sid, set()).add(label)

    profile_groups = {}
    for sid, profile in speaker_profile.items():
        key = frozenset(profile)
        profile_groups.setdefault(key, []).append(sid)

    total_speakers = len(speaker_map)
    trainval_sids = set()
    test_sids = set()

    # ── 第 1 步：外部 8:2 划分 (trainval vs test) ──
    for profile, sids in profile_groups.items():
        sorted_sids = sorted(sids)
        perm = rng.permutation(len(sorted_sids))
        n_trainval = max(1, int(len(sorted_sids) * outer_train_ratio))
        # 对小 profile 做安全处理：至少留 1 个给 test（如果 >=2 人）
        if len(sorted_sids) >= 2:
            n_trainval = max(1, min(n_trainval, len(sorted_sids) - 1))
        for i in perm[:n_trainval]:
            trainval_sids.add(sorted_sids[i])
        for i in perm[n_trainval:]:
            test_sids.add(sorted_sids[i])

    assert trainval_sids.isdisjoint(test_sids), \
        "FATAL: trainval/test speaker overlap in outer split!"

    # ── 第 2 步：内部从 trainval 划分 val ──
    # 对 trainval 内部再做 profile-stratified split
    train_sids = set()
    val_sids = set()

    for profile, sids in profile_groups.items():
        # 只考虑 trainval 中的 speaker
        trainval_in_profile = [sid for sid in sorted(sids) if sid in trainval_sids]
        if len(trainval_in_profile) == 0:
            continue
        perm = rng.permutation(len(trainval_in_profile))
        n_val = max(1, int(len(trainval_in_profile) * inner_val_ratio))
        # 安全处理：至少保留 1 个给 train
        if len(trainval_in_profile) >= 2:
            n_val = max(1, min(n_val, len(trainval_in_profile) - 1))
        for i in perm[:n_val]:
            val_sids.add(trainval_in_profile[i])
        for i in perm[n_val:]:
            train_sids.add(trainval_in_profile[i])

    # ── 三层互斥断言 ──
    assert train_sids.isdisjoint(val_sids), \
        "FATAL: train/val speaker overlap in inner split!"
    assert train_sids.isdisjoint(test_sids), \
        "FATAL: train/test speaker overlap!"
    assert val_sids.isdisjoint(test_sids), \
        "FATAL: val/test speaker overlap!"

    # ── 组装 entries ──
    def gather(sids):
        result = []
        for sid in sids:
            for path, label in speaker_map[sid]:
                result.append((path, label, sid))
        return result

    train_entries = gather(train_sids)
    val_entries = gather(val_sids)
    test_entries = gather(test_sids)

    # ── 统计信息 ──
    def class_dist(entries_list):
        from collections import Counter
        cnt = Counter()
        for _, label, _ in entries_list:
            cls_name = CLASS_NAMES[label]
            cnt[cls_name] += 1
        return dict(cnt)

    stats = {
        'outer_train_ratio': outer_train_ratio,
        'inner_val_ratio': inner_val_ratio,
        'seed': seed,
        'total_speakers': total_speakers,
        'total_samples': len(entries),
        'train_speakers': len(train_sids),
        'val_speakers': len(val_sids),
        'test_speakers': len(test_sids),
        'train_samples': len(train_entries),
        'val_samples': len(val_entries),
        'test_samples': len(test_entries),
        'train_ratio': len(train_entries) / len(entries),
        'val_ratio': len(val_entries) / len(entries),
        'test_ratio': len(test_entries) / len(entries),
        'train_class_dist': class_dist(train_entries),
        'val_class_dist': class_dist(val_entries),
        'test_class_dist': class_dist(test_entries),
        'total_class_dist': class_dist(entries),
    }

    # ── 打印统计 ──
    print(f"\n{'='*60}")
    print(f"Speaker split: outer 8:2 + inner val (seed={seed})")
    print(f"{'='*60}")
    print(f"Total: {total_speakers} speakers, {len(entries)} samples")
    print(f"  Train: {stats['train_speakers']} speakers, {stats['train_samples']} samples "
          f"({stats['train_ratio']:.1%})")
    print(f"  Val:   {stats['val_speakers']} speakers, {stats['val_samples']} samples "
          f"({stats['val_ratio']:.1%})")
    print(f"  Test:  {stats['test_speakers']} speakers, {stats['test_samples']} samples "
          f"({stats['test_ratio']:.1%})")
    print(f"Class distribution (total): {stats['total_class_dist']}")
    print(f"  Train: {stats['train_class_dist']}")
    print(f"  Val:   {stats['val_class_dist']}")
    print(f"  Test:  {stats['test_class_dist']}")

    # 验证 speaker 数守恒
    assert stats['train_speakers'] + stats['val_speakers'] + stats['test_speakers'] == total_speakers, \
        "FATAL: speaker count mismatch!"

    return train_entries, val_entries, test_entries, train_sids, val_sids, test_sids, stats


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
    """DEPRECATED: 此入口仅为旧 162-dim 特征提取保留。

    新项目使用 SSL 帧级特征，通过 scripts/extract_ssl_features.py
    直接调用 collect_wav_files() + split_speakers_7_3_with_inner_val()
    完成外部 8:2 + 内部 val 的严格划分。
    请勿使用本函数生成数据。
    """
    print("=" * 60)
    print("WARNING: This entry point is DEPRECATED for the new project.")
    print("Use scripts/extract_ssl_features.py for SSL feature extraction.")
    print("It uses split_speakers_7_3_with_inner_val() (outer 8:2 + inner val).")
    print("=" * 60)

    # Step 1: Collect WAV files
    print("\n[1/3] Collecting WAV files from MY/ ...")
    entries = collect_wav_files()
    print(f"  Total: {len(entries)} WAV files")
    unique = len(set(sid for _, _, sid in entries))
    print(f"  Unique speakers: {unique}")

    # Step 2: 3-way split (updated from old 70/30)
    print("\n[2/3] Splitting speakers 60/20/20 (seed=42)...")
    train_entries, val_entries, test_entries, train_sids, val_sids, test_sids = split_speakers_3way(entries)
    print(f"  Train: {len(train_entries)} WAVs ({len(train_sids)} speakers)")
    print(f"  Val:   {len(val_entries)} WAVs ({len(val_sids)} speakers)")
    print(f"  Test:  {len(test_entries)} WAVs ({len(test_sids)} speakers)")

    # Step 3: Extract features (old 162-dim method, for backward compat only)
    print("\n[3/3] Extracting features (3x processing per WAV, old method)...")
    X_train, y_train = extract_features_from_entries(train_entries, "train")
    X_test, y_test = extract_features_from_entries(test_entries, "test")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")

    np.save(TRAIN_DATA, X_train)
    np.save(TRAIN_LABEL, y_train)
    np.save(TEST_DATA, X_test)
    np.save(TEST_LABEL, y_test)

    print("\n" + "=" * 60)
    print("Preprocessing complete (3-way split).")
    print(f"Train: {len(X_train)} | Val: {len(val_entries)} | Test: {len(X_test)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
