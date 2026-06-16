#!/usr/bin/env python3
"""Verify all 192 experiments across B1-B7 phases.
Usage: python scripts/verify_all_192.py [--dir results_remote/results/logs]
"""
import sys, os, json

LOG_DIR = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == '--dir' else 'results_remote/results/logs'
SEEDS = [42, 123, 456]

# ── Expected experiment matrices ──
B1_EXPECTED = {}  # E1: 9 datasets × 3 pools × 3 seeds = 27
for i, ds in enumerate(['c-besd', 'fau-aibo', 'iemocap']):
    for j, pool in enumerate(['mean', 'self_attention', 'prosody_guided']):
        eid = f'E1-{i*3 + j + 1:02d}'
        B1_EXPECTED[eid] = {'dataset': ds, 'pooling': pool, 'seeds': 3}

B2_EXPECTED = {}  # E3: 18 zero-shot pairs, 1 seed each = 18
b2_pairs = [
    ('c-besd', 'fau-aibo'), ('c-besd', 'iemocap'),
    ('fau-aibo', 'c-besd'), ('fau-aibo', 'iemocap'),
    ('iemocap', 'c-besd'), ('iemocap', 'fau-aibo'),
]
pool_names = ['mean', 'self_attention', 'prosody_guided']
for pi, (src, tgt) in enumerate(b2_pairs):
    for pj, pool in enumerate(pool_names):
        eid = f'E3-{pi*3 + pj + 1:02d}'
        B2_EXPECTED[eid] = {'source': src, 'target': tgt, 'pooling': pool, 'seeds': 1}

B3_EXPECTED = {}  # E4: 3 datasets × 4 conditions × 3 seeds = 36
for di, ds in enumerate(['c-besd', 'fau-aibo', 'iemocap']):
    for ci, cond in enumerate(['C1', 'C2', 'C3', 'C4']):
        eid = f'E4-{di*4 + ci + 1:02d}'
        B3_EXPECTED[eid] = {'dataset': ds, 'condition': cond, 'seeds': 3}

B4_EXPECTED = {}  # E5: 54 experiments
# E5-01: last layer, 3 seeds
B4_EXPECTED['E5-01'] = {'dataset': 'c-besd', 'fusion': 'last', 'seeds': 3}
# E5-02: single layers L1-L12, 1 seed each
for l in range(1, 13):
    B4_EXPECTED[f'E5-02_L{l}'] = {'dataset': 'c-besd', 'fusion': f'L{l}', 'seeds': 1}
# E5-03: weighted, 3 seeds
B4_EXPECTED['E5-03'] = {'dataset': 'c-besd', 'fusion': 'weighted', 'seeds': 3}

# E5-04: last, fau-aibo, 3 seeds
B4_EXPECTED['E5-04'] = {'dataset': 'fau-aibo', 'fusion': 'last', 'seeds': 3}
# E5-05: single layers L1-L12, fau-aibo, 1 seed each
for l in range(1, 13):
    B4_EXPECTED[f'E5-05_L{l}'] = {'dataset': 'fau-aibo', 'fusion': f'L{l}', 'seeds': 1}
# E5-06: weighted, fau-aibo, 3 seeds
B4_EXPECTED['E5-06'] = {'dataset': 'fau-aibo', 'fusion': 'weighted', 'seeds': 3}

# E5-07: last, iemocap, 3 seeds
B4_EXPECTED['E5-07'] = {'dataset': 'iemocap', 'fusion': 'last', 'seeds': 3}
# E5-08: single layers L1-L12, iemocap, 1 seed each
for l in range(1, 13):
    B4_EXPECTED[f'E5-08_L{l}'] = {'dataset': 'iemocap', 'fusion': f'L{l}', 'seeds': 1}
# E5-09: weighted, iemocap, 3 seeds
B4_EXPECTED['E5-09'] = {'dataset': 'iemocap', 'fusion': 'weighted', 'seeds': 3}

B5_EXPECTED = {}  # E2: 3 datasets × 3 seeds = 9
for i, ds in enumerate(['c-besd', 'fau-aibo', 'iemocap']):
    eid = f'E2-{i+1:02d}'
    B5_EXPECTED[eid] = {'dataset': ds, 'unfreeze': True, 'seeds': 3}

B6_EXPECTED = {}  # E6: 10 configs × 3 seeds = 30
for i in range(1, 11):
    eid = f'E6-{i:02d}'
    B6_EXPECTED[eid] = {'config': f'E6-{i:02d}', 'seeds': 3}

B7_EXPECTED = {}  # E7: 6 transfer configs × 3 seeds = 18
for i in range(1, 7):
    eid = f'E7-{i:02d}'
    B7_EXPECTED[eid] = {'config': f'E7-{i:02d}', 'seeds': 3}


def check_phase(name, expected, log_dir):
    """Returns (ok_count, total, missing, empty, corrupt, results_table)"""
    ok_list = []
    missing = []
    empty = []
    corrupt = []

    for exp_id, info in sorted(expected.items()):
        n_seeds = info.get('seeds', 3)
        # Detect naming convention: check if single-seed files use _s42 suffix or not
        if n_seeds == 1:
            # Try without seed suffix first (E3 naming convention)
            json_path_no_suffix = os.path.join(log_dir, f'{exp_id}.json')
            if os.path.exists(json_path_no_suffix):
                seeds_to_check = [None]  # None means no _s suffix
                use_suffix = False
            else:
                seeds_to_check = [42]
                use_suffix = True
        else:
            seeds_to_check = SEEDS
            use_suffix = True

        for seed in seeds_to_check:
            if use_suffix and seed is not None:
                eid = f'{exp_id}_s{seed}'
            else:
                eid = exp_id
            json_path = os.path.join(log_dir, f'{eid}.json')

            if not os.path.exists(json_path):
                missing.append(eid)
                continue

            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                if not content:
                    empty.append(eid)
                    continue
                d = json.loads(content)
                wa = d.get('test_wa', 0)
                if isinstance(wa, (int, float)):
                    wa = wa * 100 if wa <= 1 else wa
                uar = d.get('test_uar', 0)
                if isinstance(uar, (int, float)):
                    uar = uar * 100 if uar <= 1 else uar
                ep = d.get('best_epoch', '?')
                ok_list.append((eid, wa, uar, ep))
            except (json.JSONDecodeError, Exception) as e:
                corrupt.append((eid, str(e)[:50]))

    total = sum(info.get('seeds', 3) for info in expected.values())
    n_ok = len(ok_list)
    has_gaps = len(missing) + len(empty) + len(corrupt) > 0
    status = 'COMPLETE' if not has_gaps else f'GAPS ({n_ok}/{total})'

    print(f'\n{"="*70}')
    print(f'  {name}  |  {status}')
    print(f'{"="*70}')

    if ok_list:
        wa_vals = [x[1] for x in ok_list if isinstance(x[1], (int, float))]
        uar_vals = [x[2] for x in ok_list if isinstance(x[2], (int, float))]
        if wa_vals:
            print(f'  [OK] {n_ok} results  |  WA: mean={sum(wa_vals)/len(wa_vals):.2f}%  min={min(wa_vals):.2f}%  max={max(wa_vals):.2f}%')
        else:
            print(f'  [OK] {n_ok} results')
        # Show top 5
        sorted_ok = sorted(ok_list, key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True)
        for eid, wa, uar, ep in sorted_ok[:5]:
            print(f'    TOP {eid}: WA={wa:.2f}% UAR={uar:.2f}% ep={ep}')

    if missing:
        print(f'  [MISSING] ({len(missing)}):')
        for m in missing:
            print(f'       {m}')
    if empty:
        print(f'  [EMPTY] ({len(empty)}): {empty}')
    if corrupt:
        print(f'  [CORRUPT] ({len(corrupt)}):')
        for cid, err in corrupt:
            print(f'       {cid}: {err}')

    return n_ok, total, missing, empty, corrupt


def main():
    log_dir = LOG_DIR
    if not os.path.isdir(log_dir):
        print(f'ERROR: Log directory not found: {log_dir}')
        return 1

    phases = [
        ('B1 | E1 Pooling × Dataset Baseline (frozen)', B1_EXPECTED),
        ('B2 | E3 Zero-shot Cross-corpus Transfer', B2_EXPECTED),
        ('B3 | E4 Augmentation Sensitivity (C1-C4)', B3_EXPECTED),
        ('B4 | E5 LayerFusion Ablation', B4_EXPECTED),
        ('B5 | E2 WavLM Unfreeze Comparison', B5_EXPECTED),
        ('B6 | E6 Module Ablation (Adapter/Pooling/Fusion)', B6_EXPECTED),
        ('B7 | E7 Model Transfer Fine-tune', B7_EXPECTED),
    ]

    total_ok = 0
    total_expected = 0
    all_missing = []
    all_empty = []
    all_corrupt = []

    for name, expected in phases:
        n_ok, total, missing, empty, corrupt = check_phase(name, expected, log_dir)
        total_ok += n_ok
        total_expected += total
        all_missing.extend(missing)
        all_empty.extend(empty)
        all_corrupt.extend(corrupt)

    print(f'\n{"="*70}')
    print(f'  GRAND TOTAL: {total_ok}/{total_expected} experiments verified')
    print(f'{"="*70}')

    if all_missing:
        print(f'\n  [MISSING FILES] ({len(all_missing)}):')
        for m in all_missing:
            print(f'       {m}')

    if all_empty:
        print(f'\n  [EMPTY FILES] ({len(all_empty)}):')
        for e in all_empty:
            print(f'       {e}')

    if all_corrupt:
        print(f'\n  [CORRUPT FILES] ({len(all_corrupt)}):')
        for c, err in all_corrupt:
            print(f'       {c}: {err}')

    if total_ok == total_expected:
        print(f'\n  *** ALL {total_expected} EXPERIMENTS VERIFIED -- 100% COMPLETE! ***')
    else:
        print(f'\n  *** WARNING: {total_expected - total_ok} gaps remaining out of {total_expected} ***')

    # Also check extra JSON files in the directory
    all_json = sorted(f for f in os.listdir(log_dir) if f.endswith('.json'))
    expected_filenames = set()
    for _, expected in phases:
        for exp_id, info in expected.items():
            n_seeds = info.get('seeds', 3)
            if n_seeds == 1:
                # Check which naming convention is used
                if os.path.exists(os.path.join(log_dir, f'{exp_id}.json')):
                    expected_filenames.add(f'{exp_id}.json')
                else:
                    expected_filenames.add(f'{exp_id}_s42.json')
            else:
                for seed in SEEDS:
                    expected_filenames.add(f'{exp_id}_s{seed}.json')

    extras = [f for f in all_json if f not in expected_filenames]
    if extras:
        print(f'\n  [EXTRA] Non-matrix JSON files ({len(extras)}):')
        for e in extras:
            print(f'       {e}')

    # Write gaps report
    gaps = {'missing': all_missing, 'empty': all_empty, 'corrupt': [c[0] for c in all_corrupt]}
    if any(gaps.values()):
        gaps_path = os.path.join(log_dir, 'VERIFY_192_GAPS.json')
        with open(gaps_path, 'w') as f:
            json.dump(gaps, f, indent=2)
        print(f'\n  Gaps written to {gaps_path}')

    return 0 if total_ok == total_expected else 1


if __name__ == '__main__':
    sys.exit(main())
