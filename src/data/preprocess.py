"""Speaker-independent preprocessing for BESD MY dataset.

Uses outer 8:2 + inner val split (split_speakers_7_3_with_inner_val)
to ensure strict speaker-independent train/val/test separation.
"""
import os
import re
import numpy as np

# ---- Config ----
WAV_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "数据集", "BESD", "BESD", "MY")
SEED = 42
CLASS_NAMES = ['ANGER', 'DISGUST', 'FEAR', 'HAPPY', 'NEUTRAL', 'SAD']
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


def split_speakers_7_3_with_inner_val(entries, outer_train_ratio=0.8, inner_val_ratio=0.25, seed=42):
    """外部 8:2 + 内部 val 的说话人独立划分（最终协议 v5, 目标 ~6:2:2）。

    协议：
      1. 外部 speaker-independent 8:2 划分：
         - 80% speaker → trainval
         - 20% speaker → final test (严格 hold-out)
      2. 内部从 trainval 中划分 validation：
         - 从 80% trainval speaker 中再划出 inner_val_ratio 作为 val
         - 默认 outer=0.8, inner=0.25 → train 约 60%, val 约 20%, test 约 20%
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
    print(f"Speaker split: outer 8:2 + inner val — target 6:2:2 (seed={seed})")
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
