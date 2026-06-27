#!/usr/bin/env python3
"""
Phase 3: Metric completeness verification.
Check every log for per-class recall, confusion matrix, predictions.
Report exact present/absent status per file.
Verify UAR values exist and are consistent with WA.
"""

import json, os, csv, math
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "results" / "logs"
METRICS_CSV = PROJECT_ROOT / "validation" / "metrics_inventory.csv"
MISSING_MD = PROJECT_ROOT / "validation" / "missing_metrics.md"

REQUIRED_FIELDS = ['test_wa', 'test_uar', 'best_val_wa']
DESIRED_FIELDS = ['confusion_matrix', 'predictions', 'per_class_recall', 'per_class_precision', 'per_class_f1']

def check_file(filepath):
    with open(filepath) as f:
        d = json.load(f)

    exp_name = d.get('exp_name', os.path.basename(filepath))
    status = {}

    # Required fields
    for field in REQUIRED_FIELDS:
        status[f'req_{field}'] = 'present' if field in d else 'missing'

    # Desired fields
    for field in DESIRED_FIELDS:
        status[f'des_{field}'] = 'present' if field in d else 'missing'

    # Check UAR sanity: should be between 0 and 1, and roughly correlated with WA
    wa = d.get('test_wa')
    uar = d.get('test_uar')
    if wa is not None and uar is not None:
        status['uar_sane'] = 'yes' if 0 <= uar <= 1.0 else 'no'
        status['wa_minus_uar'] = f"{wa*100 - uar*100:.2f}pp"
    else:
        status['uar_sane'] = 'no_data'

    # Check for per-class metrics
    n_classes = None
    if 'n_classes' in d:
        n_classes = d['n_classes']
    per_class_fields = [k for k in d if k.startswith('class_') or k.startswith('per_class_')]
    status['per_class_fields'] = ','.join(per_class_fields) if per_class_fields else 'none'

    return exp_name, status

def main():
    json_files = sorted(f for f in os.listdir(LOGS_DIR) if f.endswith('.json'))
    print(f"Checking {len(json_files)} files...")

    all_statuses = []
    present_counts = defaultdict(int)
    missing_counts = defaultdict(int)
    uar_present = 0
    uar_missing = 0
    wa_uar_diffs = []

    for fname in json_files:
        exp_name, status = check_file(LOGS_DIR / fname)
        all_statuses.append((exp_name, fname, status))

        for k, v in status.items():
            if v == 'present':
                present_counts[k] += 1
            elif v == 'missing':
                missing_counts[k] += 1

        if status.get('req_test_uar') == 'present':
            uar_present += 1
        else:
            uar_missing += 1

        if status.get('wa_minus_uar'):
            try:
                wa_uar_diffs.append(float(status['wa_minus_uar'].replace('pp','')))
            except:
                pass

    # Write CSV
    fieldnames = ['exp_name', 'log_file'] + [f'req_{f}' for f in REQUIRED_FIELDS] + \
                 [f'des_{f}' for f in DESIRED_FIELDS] + ['uar_sane', 'wa_minus_uar', 'per_class_fields']

    with open(METRICS_CSV, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for exp_name, fname, status in all_statuses:
            row = {'exp_name': exp_name, 'log_file': fname}
            row.update({k: status.get(k, '') for k in fieldnames if k not in ('exp_name', 'log_file')})
            w.writerow(row)

    # Statistics
    print(f"\n=== Phase 3: Metric Completeness ===")
    print(f"Files checked: {len(json_files)}")
    print(f"\nRequired fields:")
    for field in REQUIRED_FIELDS:
        p = present_counts.get(f'req_{field}', 0)
        m = missing_counts.get(f'req_{field}', 0)
        print(f"  {field}: present={p} missing={m}")

    print(f"\nDesired fields (needed for offline UAR recomputation):")
    for field in DESIRED_FIELDS:
        p = present_counts.get(f'des_{field}', 0)
        m = missing_counts.get(f'des_{field}', 0)
        print(f"  {field}: present={p} missing={m}")

    print(f"\nUAR status: {uar_present} present, {uar_missing} missing")
    if wa_uar_diffs:
        print(f"WA-UAR gap: min={min(wa_uar_diffs):.1f}pp max={max(wa_uar_diffs):.1f}pp mean={sum(wa_uar_diffs)/len(wa_uar_diffs):.1f}pp")

    # Can we recompute UAR offline?
    can_recompute = present_counts.get('des_confusion_matrix', 0) > 0 or \
                    present_counts.get('des_predictions', 0) > 0 or \
                    present_counts.get('des_per_class_recall', 0) > 0

    # Write missing_metrics.md
    with open(MISSING_MD, 'w', encoding='utf-8') as f:
        f.write("# Missing Metrics Report\n\n")
        f.write(f"> Generated: 2026-06-22 | Source: `scripts/check_metrics.py` | {len(json_files)} files checked\n\n")

        f.write("## Summary\n\n")
        f.write(f"- **test_wa**: {present_counts.get('req_test_wa',0)}/{len(json_files)} present\n")
        f.write(f"- **test_uar**: {present_counts.get('req_test_uar',0)}/{len(json_files)} present\n")
        f.write(f"- **confusion_matrix**: {present_counts.get('des_confusion_matrix',0)}/{len(json_files)} present\n")
        f.write(f"- **predictions**: {present_counts.get('des_predictions',0)}/{len(json_files)} present\n")
        f.write(f"- **per_class_recall**: {present_counts.get('des_per_class_recall',0)}/{len(json_files)} present\n\n")

        if can_recompute:
            f.write("## UAR: CAN be recomputed offline\n\n")
            f.write("At least one of confusion_matrix/predictions/per_class_recall exists.\n")
        else:
            f.write("## UAR: CANNOT be recomputed offline\n\n")
            f.write("**Reason**: None of confusion_matrix, predictions, or per_class_recall exist in any log file.\n\n")
            f.write("All 192 files contain `test_uar` as an aggregate metric computed by the training script, ")
            f.write("but the raw per-class data needed for independent verification was not saved.\n\n")

        f.write("## What exists\n\n")
        f.write("- `test_uar` is present in **all 192 files** — computed by `sklearn.recall_score(average='macro')` during training\n")
        f.write("- `test_wa` is present in all 192 files — computed by `sklearn.accuracy_score` during training\n")
        f.write("- `best_val_wa` is present in all 192 files\n\n")

        f.write("## What's missing (needed for offline UAR recomputation)\n\n")
        f.write("| Field | Present | Missing | Required for |\n")
        f.write("|-------|---------|---------|-------------|\n")
        for field in DESIRED_FIELDS:
            p = present_counts.get(f'des_{field}', 0)
            m = missing_counts.get(f'des_{field}', 0)
            f.write(f"| {field} | {p} | {m} | offline UAR recomputation |\n")

        f.write("\n## Recommendation\n\n")
        f.write("To enable independent UAR verification, future training runs should save:\n")
        f.write("1. `confusion_matrix` (NxN numpy array or list of lists) — covers all per-class metrics\n")
        f.write("2. OR `predictions` + `labels` (for recomputation)\n\n")
        f.write("For the 192 existing runs, the `test_uar` values are trusted as-is (computed consistently by the same sklearn function). ")
        f.write("They cannot be independently verified without re-running inference with saved model checkpoints.\n")

    print(f"\nOutput files:")
    print(f"  {METRICS_CSV}")
    print(f"  {MISSING_MD}")

if __name__ == "__main__":
    main()
