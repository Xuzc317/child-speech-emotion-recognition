#!/usr/bin/env python3
"""Independent speaker-split integrity verification.
Reuses the project's exact speaker_splitter and dataset collector logic.
Does NOT modify any files, run experiments, or read .pt / binaries.
"""

import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collections import defaultdict, Counter
from src.data.speaker_splitter import _hash_speaker, split_speakers
from src.data.dataset import (
    _collect_cbesd, _collect_iemocap, _collect_fau_aibo,
    DATASET_PATHS, C_BESD_EMOTION_FOLDERS, IEMOCAP_EMOTION_FOLDERS,
    extract_speaker_cbesd, extract_speaker_iemocap, extract_speaker_fau_aibo,
)
from src.data.label_mapper import UniversalLabelMapper

SEED = 42


def verify_dataset(name: str, collector, root: str, mapper_key: str = None):
    print(f"\n{'='*70}")
    print(f"  DATASET: {name}")
    print(f"{'='*70}")

    if not os.path.isdir(root):
        print(f"  !! PATH NOT FOUND: {root}")
        return

    entries = collector(root)
    print(f"  Raw files found: {len(entries)}")

    if mapper_key:
        mapper = UniversalLabelMapper(mapper_key)
    else:
        mapper = UniversalLabelMapper(name)

    mapped = []
    discarded = 0
    for fpath, raw_label, sid in entries:
        unified_idx = mapper.to_index(raw_label)
        if unified_idx is None:
            discarded += 1
            continue
        mapped.append((fpath, unified_idx, sid))

    print(f"  After label mapping: {len(mapped)} samples (discarded {discarded})")

    all_speakers = sorted(set(s[2] for s in mapped))
    print(f"  Unique speakers: {len(all_speakers)}")
    print(f"  Speaker IDs: {all_speakers}")

    spk_to_files = defaultdict(list)
    for fpath, _, sid in mapped:
        spk_to_files[sid].append(os.path.basename(fpath))
    print(f"\n  -- Speaker -> utterance count --")
    for sid in all_speakers:
        print(f"    {sid}: {len(spk_to_files[sid])} utterances")

    train_spk, val_spk, test_spk = split_speakers(
        all_speakers, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=SEED
    )

    print(f"\n  -- Split sizes (speakers) --")
    print(f"    Train: {len(train_spk)} speakers  ({len(train_spk)/len(all_speakers)*100:.1f}%)")
    print(f"    Val:   {len(val_spk)} speakers  ({len(val_spk)/len(all_speakers)*100:.1f}%)")
    print(f"    Test:  {len(test_spk)} speakers  ({len(test_spk)/len(all_speakers)*100:.1f}%)")

    train_utt = sum(1 for _, _, sid in mapped if sid in train_spk)
    val_utt   = sum(1 for _, _, sid in mapped if sid in val_spk)
    test_utt  = sum(1 for _, _, sid in mapped if sid in test_spk)
    print(f"\n  -- Split sizes (utterances) --")
    print(f"    Train: {train_utt} utterances  ({train_utt/len(mapped)*100:.1f}%)")
    print(f"    Val:   {val_utt} utterances  ({val_utt/len(mapped)*100:.1f}%)")
    print(f"    Test:  {test_utt} utterances  ({test_utt/len(mapped)*100:.1f}%)")

    inter_tv = train_spk & val_spk
    inter_tt = train_spk & test_spk
    inter_vt = val_spk & test_spk
    total_covered = len(train_spk) + len(val_spk) + len(test_spk)

    print(f"\n  -- ZERO LEAKAGE CHECK --")
    print(f"    Train n Val:  {len(inter_tv)} speakers {'!! LEAK!' if inter_tv else 'OK ZERO'}")
    print(f"    Train n Test: {len(inter_tt)} speakers {'!! LEAK!' if inter_tt else 'OK ZERO'}")
    print(f"    Val n Test:   {len(inter_vt)} speakers {'!! LEAK!' if inter_vt else 'OK ZERO'}")
    print(f"    Union coverage: {total_covered} / {len(all_speakers)} unique speakers "
          f"({'OK ALL COVERED' if total_covered == len(all_speakers) else 'WARNING MISSING SOME'})")

    if inter_tv: print(f"      Leaked: {sorted(inter_tv)}")
    if inter_tt: print(f"      Leaked: {sorted(inter_tt)}")
    if inter_vt: print(f"      Leaked: {sorted(inter_vt)}")

    print(f"\n  -- Speaker hash distribution --")
    for sid in all_speakers:
        h = _hash_speaker(sid, SEED)
        bucket = "train" if h < 0.70 else ("val" if h < 0.85 else "test")
        print(f"    {sid}: hash={h:.4f} -> {bucket}")


def main():
    print("=" * 70)
    print("  SPEAKER SPLIT INTEGRITY VERIFICATION (seed=42)")
    print("  Protocol: ac_suite_2026-06-validated")
    print("=" * 70)

    # ---- C-BESD (6-class, in-domain) ----
    cbesd_root = DATASET_PATHS['c-besd']
    verify_dataset('c-besd', _collect_cbesd, cbesd_root, mapper_key='c-besd')

    # ---- C-BESD-4cl (4-class, cross-corpus subset) ----
    print(f"\n{'~'*70}")
    print(f"  C-BESD-4cl (cross-corpus, 4-class, discards disgust/fear)")
    cbesd4_entries = _collect_cbesd(cbesd_root)
    mapper4 = UniversalLabelMapper('c-besd-4cl')
    mapped4 = []
    discarded4 = 0
    for fpath, raw_label, sid in cbesd4_entries:
        idx = mapper4.to_index(raw_label)
        if idx is None:
            discarded4 += 1
            continue
        mapped4.append((fpath, idx, sid))
    speakers4 = sorted(set(s[2] for s in mapped4))
    print(f"  Samples: {len(mapped4)} (discarded {discarded4}), speakers: {len(speakers4)}")
    train4, val4, test4 = split_speakers(speakers4, seed=SEED)
    i_tv4 = train4 & val4; i_tt4 = train4 & test4; i_vt4 = val4 & test4
    print(f"  Train n Val:  {len(i_tv4)} {'!!' if i_tv4 else 'OK'}")
    print(f"  Train n Test: {len(i_tt4)} {'!!' if i_tt4 else 'OK'}")
    print(f"  Val n Test:   {len(i_vt4)} {'!!' if i_vt4 else 'OK'}")

    # ---- IEMOCAP ----
    iemocap_root = DATASET_PATHS['iemocap']
    verify_dataset('iemocap', _collect_iemocap, iemocap_root)

    # ---- FAU Aibo ----
    fau_root = DATASET_PATHS['fau-aibo']
    verify_dataset('fau-aibo', _collect_fau_aibo, fau_root)

    # ---- Hidden trap #1: C-BESD speaker ID extraction deep-dive ----
    print(f"\n{'='*70}")
    print(f"  HIDDEN TRAP ANALYSIS")
    print(f"{'='*70}")

    print(f"\n  -- [C-BESD] Speaker ID extraction from filenames --")
    print(f"  Rule: leading digits -> 'C{{NN}}'")

    all_cbesd_fnames = []
    for folder in C_BESD_EMOTION_FOLDERS.values():
        fpath = os.path.join(cbesd_root, folder)
        if os.path.isdir(fpath):
            for fname in os.listdir(fpath):
                if fname.endswith('.wav') and 'copy' not in fname.lower():
                    all_cbesd_fnames.append((folder, fname))

    suspicious = []
    for folder, fname in all_cbesd_fnames:
        m = re.match(r'^(\d+)', fname)
        if m:
            expected_sid = f'C{m.group(1).zfill(2)}'
            actual_sid = extract_speaker_cbesd(fname)
            if expected_sid != actual_sid:
                suspicious.append((fname, expected_sid, actual_sid))
        else:
            suspicious.append((fname, 'NO-LEADING-DIGITS', extract_speaker_cbesd(fname)))

    if suspicious:
        print(f"  !! Found {len(suspicious)} suspicious filename -> speaker_id mappings:")
        for fname, exp, act in suspicious[:20]:
            print(f"      {fname} -> expected '{exp}', got '{act}'")
    else:
        print(f"  OK All {len(all_cbesd_fnames)} filenames produce consistent speaker IDs")

    all_leading_nums = set()
    for folder, fname in all_cbesd_fnames:
        m = re.match(r'^(\d+)', fname)
        if m:
            all_leading_nums.add(int(m.group(1)))
    print(f"  C-BESD unique child numbers detected: {sorted(all_leading_nums)}")
    print(f"  Count: {len(all_leading_nums)} unique child IDs")

    # ---- Hidden trap #2: IEMOCAP session naming ----
    print(f"\n  -- [IEMOCAP] Speaker ID extraction --")
    print(f"  Rule: first 6 chars = 'SesXXG' (session + gender = unique speaker)")
    iemo_fnames = []
    for folder in IEMOCAP_EMOTION_FOLDERS.values():
        fpath = os.path.join(iemocap_root, folder)
        if os.path.isdir(fpath):
            for fname in os.listdir(fpath):
                if fname.endswith('.wav'):
                    iemo_fnames.append(fname)

    iemo_speakers = sorted(set(extract_speaker_iemocap(f) for f in iemo_fnames))
    print(f"  Unique speaker IDs: {iemo_speakers}")
    print(f"  Count: {len(iemo_speakers)} unique speakers")
    for sid in iemo_speakers:
        assert len(sid) == 6, f"Unexpected speaker ID length: {sid}"
        assert sid.startswith('Ses'), f"Unexpected speaker ID prefix: {sid}"
    print(f"  OK All {len(iemo_speakers)} IEMOCAP speaker IDs follow 'SesXXG' convention")

    # ---- Hidden trap #3: FAU Aibo naming ----
    print(f"\n  -- [FAU Aibo] Speaker ID extraction --")
    print(f"  Rule: first two '_' parts = '{{prefix}}_{{number}}'")
    fau_entries = _collect_fau_aibo(fau_root)
    fau_speakers = sorted(set(s[2] for s in fau_entries))
    print(f"  Unique speaker IDs: {fau_speakers}")
    print(f"  Count: {len(fau_speakers)} unique speakers")

    fau_name_prefixes = set(s.rsplit('_', 1)[0] for s in fau_speakers)
    print(f"  Unique name prefixes (excl. digit): {sorted(fau_name_prefixes)}")
    if len(fau_name_prefixes) < len(fau_speakers):
        print(f"  !! Some names have multiple number variants -- check if same child")
        prefix_counts = Counter(s.rsplit('_', 1)[0] for s in fau_speakers)
        multi = {k: v for k, v in prefix_counts.items() if v > 1}
        for name, count in multi.items():
            variants = [s for s in fau_speakers if s.rsplit('_', 1)[0] == name]
            print(f"      {name}: {count} variants -> {variants}")
    else:
        print(f"  OK No name-sharing between speakers (each prefix matches one ID)")

    print(f"\n{'='*70}")
    print(f"  VERIFICATION COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
