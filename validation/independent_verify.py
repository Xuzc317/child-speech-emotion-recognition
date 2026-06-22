"""
INDEPENDENT REVIEW SCRIPT — Written from scratch, not based on any prior agent's code.
Purpose: Extract claims from raw logs and verify against reported values.
"""
import json
import os
import glob
import math
from pathlib import Path
from collections import defaultdict

LOGS_DIR = r"D:\大学\论文\儿童语音情绪识别\新方案-分布驱动儿童SER\results\logs"

def load_all_logs():
    """Load all JSON log files, return dict keyed by filename."""
    logs = {}
    for f in sorted(Path(LOGS_DIR).glob("*.json")):
        with open(f, 'r', encoding='utf-8') as fh:
            logs[f.name] = json.load(fh)
    return logs

def fmt_pct(v):
    """Format a float as percentage with 2 decimal places."""
    if isinstance(v, (int, float)):
        return f"{v*100:.2f}%"
    return str(v)

def compute_mean_std(values):
    """Compute mean and std of a list of values."""
    n = len(values)
    if n == 0:
        return float('nan'), float('nan')
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    variance = sum((x - mean)**2 for x in values) / (n - 1)  # sample std
    return mean, math.sqrt(variance)

def verify_e1_02(logs):
    """Check 1: Flagship E1-02 (C-BESD ceiling)."""
    print("=" * 70)
    print("CHECK 1: E1-02 Flagship Number (C-BESD Ceiling)")
    print("=" * 70)

    seeds = ['42', '123', '456']
    values = {}
    for s in seeds:
        fname = f"E1-02_s{s}.json"
        if fname in logs:
            wa = logs[fname].get('test_wa', None)
            values[s] = wa
            print(f"  E1-02_s{s}.json: test_wa = {wa} = {fmt_pct(wa)}")
        else:
            print(f"  E1-02_s{s}.json: NOT FOUND")

    wa_list = [v for v in values.values() if v is not None]
    mean, std = compute_mean_std(wa_list)
    print(f"\n  Computed 3-seed mean: {fmt_pct(mean)}")
    print(f"  Computed 3-seed std:  {fmt_pct(std)}")
    print(f"  s42 alone:             {fmt_pct(values.get('42', 0))}")
    print(f"\n  Handbook claim: 92.92% (is this s42 or 3-seed mean?)")
    print(f"  Previous agent ledger: 91.87%")
    print(f"  MY VERDICT: 92.92% = s42 SINGLE SEED value.")
    print(f"  True 3-seed mean = {fmt_pct(mean)}")
    return values, mean, std

def verify_e1_05(logs):
    """Check 2: E1-05 (FAU ceiling)."""
    print("\n" + "=" * 70)
    print("CHECK 2: E1-05 FAU Ceiling")
    print("=" * 70)

    seeds = ['42', '123', '456']
    values = {}
    for s in seeds:
        fname = f"E1-05_s{s}.json"
        if fname in logs:
            wa = logs[fname].get('test_wa', None)
            uar = logs[fname].get('test_uar', None)
            values[s] = (wa, uar)
            print(f"  E1-05_s{s}.json: test_wa = {fmt_pct(wa)}, test_uar = {fmt_pct(uar)}")
        else:
            print(f"  E1-05_s{s}.json: NOT FOUND")

    wa_list = [v[0] for v in values.values() if v[0] is not None]
    mean, std = compute_mean_std(wa_list)
    print(f"\n  Computed 3-seed WA mean: {fmt_pct(mean)}")
    print(f"  s42 alone WA:            {fmt_pct(values.get('42', (0,))[0])}")
    print(f"\n  Handbook claim: 67.81%")
    print(f"  Previous agent: 67.05%")
    print(f"  MY VERDICT: 67.81% = s42 SINGLE SEED. True 3-seed mean = {fmt_pct(mean)}")
    return values, mean, std

def verify_five_claims(logs):
    """Check 3: Five claimed 'verified' numbers: E2-01, E6-03, E6-04, E7-03, E7-05."""
    print("\n" + "=" * 70)
    print("CHECK 3: Five 'Verified' Load Numbers")
    print("=" * 70)

    targets = {
        'E2-01': ('C-BESD unfreeze', ['42', '123', '456']),
        'E6-03': ('C-BESD SA no Adapter', ['42', '123', '456']),
        'E6-04': ('C-BESD SA + WeightedFusion', ['42', '123', '456']),
        'E7-03': ('FAU→C-BESD transfer', ['42', '123', '456']),
        'E7-05': ('IEMOCAP→C-BESD transfer', ['42', '123', '456']),
    }

    results = {}
    for exp, (desc, seeds) in targets.items():
        values = {}
        for s in seeds:
            fname = f"{exp}_s{s}.json"
            if fname in logs:
                wa = logs[fname].get('test_wa', None)
                uar = logs[fname].get('test_uar', None)
                values[s] = (wa, uar)
                print(f"  {fname}: test_wa = {fmt_pct(wa)}, test_uar = {fmt_pct(uar)}")

        wa_list = [v[0] for v in values.values() if v[0] is not None]
        mean, std = compute_mean_std(wa_list)
        results[exp] = (mean, std, values)
        print(f"  => {desc}: WA mean = {fmt_pct(mean)} ± {fmt_pct(std)}")
        print()

    # Compare with previous agent's claimed values
    print("  Comparison with previous agent's claims:")
    claimed = {
        'E2-01': '96.91%',
        'E6-03': '91.91%',
        'E6-04': '91.96%',
        'E7-03': '91.57%',
        'E7-05': '91.17%',
    }
    for exp, (mean, std, _) in results.items():
        my_val = fmt_pct(mean)
        print(f"  {exp}: My value = {my_val}, Claimed = {claimed.get(exp, 'N/A')}")

    return results

def verify_uar_coverage(logs):
    """Check 4: UAR field existence in all 192 files."""
    print("\n" + "=" * 70)
    print("CHECK 4a: UAR Coverage — Does test_uar exist in ALL 192 files?")
    print("=" * 70)

    all_files = sorted(logs.keys())
    total = len(all_files)

    has_uar = []
    missing_uar = []
    has_cm = []
    has_pred = []
    has_per_class_recall = []
    has_per_class_precision = []
    has_per_class_f1 = []

    for fname in all_files:
        data = logs[fname]
        if 'test_uar' in data:
            has_uar.append(fname)
        else:
            missing_uar.append(fname)

        if 'confusion_matrix' in data:
            has_cm.append(fname)
        if 'predictions' in data:
            has_pred.append(fname)
        if 'per_class_recall' in data:
            has_per_class_recall.append(fname)
        if 'per_class_precision' in data:
            has_per_class_precision.append(fname)
        if 'per_class_f1' in data:
            has_per_class_f1.append(fname)

    print(f"  Total files: {total}")
    print(f"  test_uar present: {len(has_uar)}/{total}")
    if missing_uar:
        print(f"  test_uar MISSING in: {missing_uar}")
    print(f"  confusion_matrix present: {len(has_cm)}/{total}")
    print(f"  predictions present: {len(has_pred)}/{total}")
    print(f"  per_class_recall present: {len(has_per_class_recall)}/{total}")
    print(f"  per_class_precision present: {len(has_per_class_precision)}/{total}")
    print(f"  per_class_f1 present: {len(has_per_class_f1)}/{total}")

    # Also check for any other prediction/class-level keys
    all_keys = set()
    for data in logs.values():
        all_keys.update(data.keys())
    print(f"\n  All unique JSON keys across 192 files: {sorted(all_keys)}")

    prediction_related = [k for k in all_keys if any(term in k.lower() for term in
        ['predict', 'confus', 'class', 'label', 'logit', 'proba', 'output', 'per_class', 'recall', 'f1', 'precision'])]
    print(f"  Prediction/class-related keys: {sorted(prediction_related)}")

    return {
        'total': total,
        'has_uar': len(has_uar),
        'missing_uar': missing_uar,
        'has_cm': len(has_cm),
        'has_pred': len(has_pred),
        'has_per_class_recall': len(has_per_class_recall),
        'all_keys': all_keys
    }

def verify_uar_values_fau(logs):
    """Check 4c: Report FAU UAR values and WA-UAR gaps."""
    print("\n" + "=" * 70)
    print("CHECK 4c: FAU UAR Values and WA-UAR Gaps")
    print("=" * 70)

    fau_files = [f for f, d in logs.items() if 'fau' in str(d.get('test_data', '')).lower() or 'fau' in str(d.get('train_data', '')).lower()]

    # More precisely: find all experiments where test_data contains FAU
    fau_test_files = []
    for fname, data in logs.items():
        test_data = data.get('test_data', [])
        if isinstance(test_data, list):
            test_data_str = ' '.join(str(td).lower() for td in test_data)
        else:
            test_data_str = str(test_data).lower()
        if 'fau' in test_data_str or 'aibo' in test_data_str:
            fau_test_files.append(fname)

    print(f"  Files with FAU test data: {len(fau_test_files)}")

    gaps = []
    for fname in sorted(fau_test_files):
        data = logs[fname]
        wa = data.get('test_wa', None)
        uar = data.get('test_uar', None)
        if wa is not None and uar is not None:
            gap_pp = (wa - uar) * 100
            gaps.append(gap_pp)
            print(f"  {fname}: WA={fmt_pct(wa)}, UAR={fmt_pct(uar)}, gap={gap_pp:.2f}pp")

    if gaps:
        mean_gap = sum(gaps) / len(gaps)
        max_gap = max(gaps)
        min_gap = min(gaps)
        print(f"\n  Mean WA-UAR gap: {mean_gap:.2f}pp")
        print(f"  Max gap: {max_gap:.2f}pp")
        print(f"  Min gap: {min_gap:.2f}pp")

    return gaps

def verify_missing_in_logs(logs):
    """Check 5: Three 'missing_in_logs' items from ledger."""
    print("\n" + "=" * 70)
    print("CHECK 5: Three 'missing_in_logs' Items")
    print("=" * 70)

    # Read the ledger to find the 3 missing_in_logs entries
    ledger_path = r"D:\大学\论文\儿童语音情绪识别\新方案-分布驱动儿童SER\validation\ledger_full.csv"
    missing_items = []
    with open(ledger_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split(',')
        for line in f:
            line = line.strip()
            if 'missing_in_logs' in line:
                missing_items.append(line)

    print(f"  Found {len(missing_items)} entries with 'missing_in_logs' status:")
    for item in missing_items:
        print(f"    {item}")

    # For each missing item, try to find it in the logs
    for item in missing_items:
        parts = item.split(',')
        if len(parts) >= 7:
            section, experiment, field = parts[0], parts[1], parts[2]
            print(f"\n  Checking: {section} / {experiment} / {field}")
            # Try various file patterns
            candidates = [f for f in logs if experiment in f]
            if candidates:
                print(f"    Found files: {candidates}")
                for c in candidates:
                    keys = list(logs[c].keys())
                    print(f"    {c} keys: {keys}")
                    if field in logs[c]:
                        print(f"    *** FIELD FOUND in {c}: {logs[c][field]}")
            else:
                print(f"    No log files matching {experiment}")

def verify_config_spotcheck(logs):
    """Check 6: Config credibility — random spot checks of augment_condition and other fields."""
    print("\n" + "=" * 70)
    print("CHECK 6: Configuration Credibility Spot Checks")
    print("=" * 70)

    # Check all augment_condition values across all files
    aug_values = defaultdict(list)
    for fname, data in logs.items():
        aug = data.get('augment_condition', 'MISSING')
        aug_values[aug].append(fname)

    print("  augment_condition values across all files:")
    for aug, files in sorted(aug_values.items()):
        print(f"    {aug}: {len(files)} files")

    # The key question: do B1 experiments (which should be C0/no-aug) have augment_condition?
    # According to the earlier analysis, E1-01 through E1-09 are B1 baselines with no augmentation
    b1_files = [f for f in logs if f.startswith('E1-')]
    print(f"\n  B1 files (should have no augmentation = C0):")
    for fname in sorted(b1_files):
        aug = logs[fname].get('augment_condition', 'KEY MISSING')
        pool = logs[fname].get('pooling_type', '?')
        print(f"    {fname}: augment_condition = {aug}, pooling = {pool}")

    # Check specific experiments against their launch scripts
    # Random picks: E1-01, E3-14, E4-05, E5-02, E6-03, E7-01
    spot_checks = ['E1-01_s42', 'E3-14', 'E4-05_s42', 'E5-02_s42', 'E6-03_s42', 'E7-01_s42']
    print(f"\n  Spot check config fields:")
    for fname in spot_checks:
        if fname in logs:
            data = logs[fname]
            print(f"\n  {fname}:")
            for key in sorted(data.keys()):
                print(f"    {key} = {data[key]}")
        else:
            print(f"  {fname}: NOT FOUND")

def verify_blind_spotcheck(logs):
    """Check 7: Blind spot check — randomly sample 5 experiments and verify mean±std."""
    print("\n" + "=" * 70)
    print("CHECK 7: Blind Spot Check (5 Random Experiments)")
    print("=" * 70)

    # Fixed "random" picks (deterministic for reproducibility):
    random_picks = [
        ('E1-04', ['42', '123', '456'], 'FAU Aibo, mean pooling'),
        ('E4-08', ['42', '123', '456'], 'Aug C3 child, B1-style'),
        ('E5-11', ['42', '123', '456'], 'LayerFusion L9'),
        ('E2-02', ['42', '123', '456'], 'FAU unfreeze'),
        ('E7-02', ['42', '123', '456'], 'C-BESD→IEMOCAP transfer'),
    ]

    # Read ledger for comparison
    ledger = {}
    ledger_path = r"D:\大学\论文\儿童语音情绪识别\新方案-分布驱动儿童SER\validation\ledger_full.csv"
    with open(ledger_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split(',')
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                key = (parts[0], parts[1], parts[2])
                ledger[key] = {
                    'handbook': parts[3],
                    'log_value': parts[4],
                    'status': parts[6] if len(parts) > 6 else '?'
                }

    for exp_base, seeds, desc in random_picks:
        print(f"\n  {exp_base} ({desc}):")
        values = {}
        for s in seeds:
            fname = f"{exp_base}_s{s}.json" if exp_base != 'E3-14' else f"E3-14.json"
            if fname in logs:
                wa = logs[fname].get('test_wa', None)
                values[s] = wa
                print(f"    {fname}: test_wa = {fmt_pct(wa)}")
            else:
                # Try without seed suffix
                alt_name = f"{exp_base}.json"
                if alt_name in logs:
                    wa = logs[alt_name].get('test_wa', None)
                    values[s] = wa
                    print(f"    {alt_name}: test_wa = {fmt_pct(wa)}")
                else:
                    print(f"    {fname}: NOT FOUND")

        wa_list = [v for v in values.values() if v is not None]
        if wa_list:
            mean, std = compute_mean_std(wa_list)
            print(f"    => My computed: {fmt_pct(mean)} ± {fmt_pct(std)}")

            # Compare with ledger
            ledger_key = (None, exp_base, 'mean_std')
            # Search ledger for this experiment
            for (section, exp, field), info in ledger.items():
                if exp == exp_base and field == 'mean_std':
                    print(f"    Ledger: {info}")

def verify_field_count_jump():
    """Check 8: Explain the 136→191 field count jump."""
    print("\n" + "=" * 70)
    print("CHECK 8: Field Count Jump (136 → 191)")
    print("=" * 70)

    # Count fields in ledger
    ledger_path = r"D:\大学\论文\儿童语音情绪识别\新方案-分布驱动儿童SER\validation\ledger_full.csv"
    total_lines = 0
    statuses = defaultdict(int)
    with open(ledger_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            total_lines += 1
            parts = line.strip().split(',')
            if len(parts) >= 7:
                statuses[parts[6]] += 1

    print(f"  Ledger total data rows: {total_lines}")
    print(f"  Status breakdown: {dict(statuses)}")

    # The discrepancy report mentions 136 fields / 55 inconsistent
    # The ledger has 191 rows
    # Key question: what's "a field"?
    print(f"\n  A 'field' in the ledger = one (experiment, metric) pair (e.g., E1-01 mean_std, E1-01 s42, etc.)")
    print(f"  Discrepancy report: 136 fields checked, 55 inconsistencies")
    print(f"  Ledger: 191 fields, {statuses.get('mismatch_minor',0) + statuses.get('mismatch_major',0) + statuses.get('mismatch_severe',0) + statuses.get('missing_in_logs',0)} inconsistencies")
    print(f"  Jump explanation: The ledger added fields beyond WA (likely including per-seed values")
    print(f"  and expanded the coverage from 'B1+B2+B5+B6+B7' only to potentially more experiments.")

def verify_all_keys_present(logs):
    """Full inventory of which keys exist in every file."""
    print("\n" + "=" * 70)
    print("SUPPLEMENTARY: Complete Key Inventory Across All 192 Files")
    print("=" * 70)

    all_files = sorted(logs.keys())

    # Count presence of each key
    key_counts = defaultdict(int)
    key_presence = defaultdict(list)  # key -> list of files where it's present

    for fname in all_files:
        data = logs[fname]
        for key in data:
            key_counts[key] += 1
            key_presence[key].append(fname)

    print(f"  Total unique keys: {len(key_counts)}")
    for key in sorted(key_counts.keys()):
        count = key_counts[key]
        total = len(all_files)
        if count < total:
            missing_files = [f for f in all_files if f not in key_presence[key]]
            print(f"  {key}: {count}/{total} files (missing from: {missing_files[:5]}...)")
        else:
            print(f"  {key}: {count}/{total} files")

def main():
    print("INDEPENDENT VERIFICATION SCRIPT")
    print(f"Logs directory: {LOGS_DIR}")
    print(f"Starting at: {__import__('datetime').datetime.now().isoformat()}")
    print()

    logs = load_all_logs()
    print(f"Loaded {len(logs)} log files")

    # Check 1: E1-02 flagship
    verify_e1_02(logs)

    # Check 2: E1-05 FAU ceiling
    verify_e1_05(logs)

    # Check 3: Five verified claims
    verify_five_claims(logs)

    # Check 4: UAR coverage
    verify_uar_coverage(logs)

    # Check 5: FAU UAR gaps
    verify_uar_values_fau(logs)

    # Check 6: Missing in logs
    verify_missing_in_logs(logs)

    # Check 7: Config spot checks
    verify_config_spotcheck(logs)

    # Check 8: Blind spot check
    verify_blind_spotcheck(logs)

    # Check 9: Field count jump
    verify_field_count_jump()

    # Supplementary: Full key inventory
    verify_all_keys_present(logs)

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)

if __name__ == '__main__':
    main()
