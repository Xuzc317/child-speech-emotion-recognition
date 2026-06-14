#!/usr/bin/env python3
"""Comprehensive experiment completeness verifier.
Usage: python scripts/verify_experiments.py [--phase B1] [--phase B3]
"""
import sys, os, json, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('SER_C_BESD_PATH', '/root/autodl-tmp/datasets/BESD/BESD/MY')
os.environ.setdefault('SER_IEMOCAP_PATH', '/root/autodl-tmp/IEMOCAP/wavs')
os.environ.setdefault('SER_FAU_AIBO_PATH', '/root/autodl-tmp/IS2009EmotionChallenge/IS2009EmotionChallenge/wav')

LOG_DIR = os.environ.get('SER_LOG_DIR', 'results_remote/results/logs')
SEEDS = [42, 123, 456]

# Expected experiment matrices per phase
B1_EXPECTED = {
    'E1-01': ['c-besd', 'mean', 3], 'E1-02': ['c-besd', 'self_attention', 3],
    'E1-03': ['c-besd', 'prosody_guided', 3], 'E1-04': ['fau-aibo', 'mean', 3],
    'E1-05': ['fau-aibo', 'self_attention', 3], 'E1-06': ['fau-aibo', 'prosody_guided', 3],
    'E1-07': ['iemocap', 'mean', 3], 'E1-08': ['iemocap', 'self_attention', 3],
    'E1-09': ['iemocap', 'prosody_guided', 3],
}

B3_EXPECTED = {}
for di, ds in enumerate(['c-besd', 'fau-aibo', 'iemocap']):
    for ci, cond in enumerate(['C1', 'C2', 'C3', 'C4']):
        eid = f'E4-{di*4 + ci + 1:02d}'
        B3_EXPECTED[eid] = [ds, cond, 3]

CORE_EXPERIMENTS = [
    'exp1_self_attention', 'exp2_prosody_guided', 'exp3_adult_iemocap',
    'exp4_zero_shot_fau', 'exp5_fau_indomain', 'exp5b_self_attention_fau',
]

def check_phase(name, expected):
    """Check completeness of one experiment phase. Returns True if all OK."""
    print()
    print('=' * 60)
    print(f'  {name}')
    print('=' * 60)

    ok_list = []
    missing_json = []
    empty_json = []

    for exp_id, (dataset, desc, n_seeds) in sorted(expected.items()):
        for seed in SEEDS:
            if n_seeds == 1 and seed != 42:
                continue
            eid = f'{exp_id}_s{seed}'
            json_path = os.path.join(LOG_DIR, f'{eid}.json')

            if not os.path.exists(json_path):
                missing_json.append(eid)
                continue

            try:
                with open(json_path) as f:
                    content = f.read().strip()
                if not content:
                    empty_json.append(eid)
                    continue
                d = json.loads(content)
                wa = d.get('test_wa', 0) * 100
                uar = d.get('test_uar', 0) * 100
                ep = d.get('best_epoch', '?')
                ok_list.append((eid, wa, uar, ep))
            except Exception:
                empty_json.append(eid)

    # Print results
    if ok_list:
        print(f'  OK: {len(ok_list)} results')
        for eid, wa, uar, ep in ok_list:
            print(f'    {eid}: WA={wa:.1f}% UAR={uar:.1f}% ep={ep}')

    if missing_json:
        print(f'  MISSING JSON ({len(missing_json)}): {missing_json}')
    if empty_json:
        print(f'  EMPTY JSON ({len(empty_json)}): {empty_json}')

    n_total = len(expected) * 3
    n_ok = len(ok_list)
    result = len(missing_json) + len(empty_json) == 0
    status = 'COMPLETE' if result else f'GAPS ({n_ok}/{n_total})'
    print(f'  => {status}')
    return result, missing_json, empty_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', default=None, help='Specific phase (B1, B3, core, all)')
    args = p.parse_args()

    all_ok = True
    all_missing = {}

    if not args.phase or args.phase in ('B1', 'all'):
        ok, missing, empty = check_phase('B1 (E1 Pooling Ablation)', B1_EXPECTED)
        if not ok:
            all_ok = False
            all_missing['B1'] = missing + empty

    if not args.phase or args.phase in ('B3', 'all'):
        ok, missing, empty = check_phase('B3 (E4 Augmentation Sensitivity)', B3_EXPECTED)
        if not ok:
            all_ok = False
            all_missing['B3'] = missing + empty

    if not args.phase or args.phase in ('core', 'all'):
        print()
        print('=' * 60)
        print('  Core Experiments')
        print('=' * 60)
        for exp in CORE_EXPERIMENTS:
            path = os.path.join(LOG_DIR, f'{exp}.json')
            if os.path.exists(path):
                with open(path) as f:
                    d = json.load(f)
                print(f'  {exp}: WA={d["test_wa"]*100:.1f}% UAR={d["test_uar"]*100:.1f}%')
            else:
                print(f'  {exp}: MISSING')
                all_ok = False

    print()
    print('=' * 60)
    if all_ok:
        print('  ALL CHECKS PASSED')
    else:
        print('  GAPS DETECTED - see above')
    print('=' * 60)

    # Write gaps file for automated fixup
    if all_missing:
        gaps_path = os.path.join(LOG_DIR, 'VERIFY_GAPS.json')
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(gaps_path, 'w') as f:
            json.dump(all_missing, f, indent=2)
        print(f'  Gaps written to {gaps_path}')

    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
