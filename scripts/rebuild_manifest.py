#!/usr/bin/env python3
"""
Rebuild provenance_manifest.csv with:
- sample std (ddof=1)
- E1-08, E4-04, E4-10 marked as INVALID_AGGREGATION
- aug_trusted column
"""

import json, os, math, csv, re
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "results" / "logs"
OUTPUT = PROJECT_ROOT / "validation" / "provenance_manifest.csv"

E_TO_PHASE = {
    "E1": "B1", "E2": "B5", "E3": "B2", "E4": "B3",
    "E5": "B4", "E6": "B6", "E7": "B7",
}
PHASE_SCRIPTS = {
    "B1": ["launch_b1.sh"], "B2": ["launch_b2.sh"], "B3": ["launch_b3.sh"],
    "B4": ["launch_b4.sh", "launch_b4_resume.sh"], "B5": ["launch_b5.sh"],
    "B6": ["launch_b6.sh", "launch_b6_fill.sh"], "B7": ["launch_b7.sh"],
}

INVALID_AGGREGATION = {"E1-08", "E4-04", "E4-10"}

def corpus_name(dl):
    if not dl: return "unknown"
    n = dl[0].lower()
    if "besd" in n: return "C-BESD"
    if "fau" in n or "aibo" in n: return "FAU_Aibo"
    if "iemocap" in n: return "IEMOCAP"
    return n

def sample_mean_std(vals):
    if not vals: return None, None
    n = len(vals)
    if n == 1: return vals[0], None
    m = sum(vals)/n
    var = sum((v-m)**2 for v in vals)/(n-1)
    return m, math.sqrt(var)

def check_ckpt(exp_name, phase):
    d = PROJECT_ROOT / "checkpoints" / "autodl" / phase.lower() / exp_name
    if not d.is_dir(): return False
    return any(f.endswith(('.pt','.pth')) for f in os.listdir(d))

def csv_q(s):
    s = str(s)
    return f'"{s}"' if (',' in s or '"' in s) else s

# Load all
logs = {}
for fname in sorted(os.listdir(LOGS_DIR)):
    if not fname.endswith('.json'): continue
    with open(LOGS_DIR / fname) as f:
        d = json.load(f)
    logs[d['exp_name']] = d

# Build records
records = []
for fname in sorted(os.listdir(LOGS_DIR)):
    if not fname.endswith('.json'): continue
    with open(LOGS_DIR / fname) as f:
        d = json.load(f)
    exp_name = d['exp_name']
    e_series = exp_name.split('-')[0]
    phase = E_TO_PHASE.get(e_series, '???')
    m = re.match(r'^(.+)_s(\d+)$', exp_name)
    base = m.group(1) if m else exp_name
    seed = int(m.group(2)) if m else d.get('seed')
    records.append({
        'exp_name': exp_name, 'base': base, 'phase': phase, 'e_series': e_series,
        'train': corpus_name(d.get('train_data', [])),
        'test': corpus_name(d.get('test_data', [])),
        'pooling': d.get('pooling_type', '?'),
        'fusion': d.get('fusion_mode', 'last'),
        'adapter': d.get('use_adapter', False),
        'unfreeze': d.get('unfreeze_ssl', False),
        'aug': d.get('augment_condition', 'C0_none'),
        'seed': seed,
        'test_wa': d.get('test_wa'), 'test_uar': d.get('test_uar'),
        'best_val_wa': d.get('best_val_wa'), 'best_epoch': d.get('best_epoch'),
        'output_dir': d.get('output_dir', ''), 'protocol': d.get('protocol', '?'),
        'log_file': fname,
    })

# Group by base
groups = defaultdict(list)
for r in records:
    groups[r['base']].append(r)

# Build CSV
header = ['experiment_id','phase','e_series','corpus','pooling','fusion','adapter',
          'unfreeze','aug','aug_trusted','seeds','test_wa_per_seed','test_wa_mean+-std',
          'test_uar_per_seed','test_uar_mean+-std','launch_script','log_files',
          'ckpt_exists_all_local','protocol','aggregation_valid']

rows_out = [header]
for base in sorted(groups):
    group = sorted(groups[base], key=lambda r: r['seed'] or 0)
    r0 = group[0]
    phase = r0['phase']
    corpus = f"{r0['train']}->{r0['test']}" if r0['train'] != r0['test'] else r0['train']

    seeds = [r['seed'] for r in group if r['seed'] is not None]
    wa_vals = [r['test_wa'] for r in group if r['test_wa'] is not None]
    uar_vals = [r['test_uar'] for r in group if r['test_uar'] is not None]

    wa_m, wa_s = sample_mean_std([v*100 for v in wa_vals])
    uar_m, uar_s = sample_mean_std([v*100 for v in uar_vals])

    seeds_str = '/'.join(str(s) for s in seeds)
    wa_per = '/'.join(f"{v*100:.2f}%" for v in wa_vals)
    uar_per = '/'.join(f"{v*100:.2f}%" for v in uar_vals)
    wa_ms = f"{wa_m:.2f}+-{wa_s:.2f}%" if wa_s is not None else f"{wa_m:.2f}%"
    uar_ms = f"{uar_m:.2f}+-{uar_s:.2f}%" if uar_s is not None else f"{uar_m:.2f}%"

    # aug_trusted
    if phase in ('B3',) and r0['aug'] != 'C0_none':
        aug_trusted = 'TRUE'
    elif phase in ('B1','B2','B4','B5','B6','B7'):
        aug_trusted = 'FALSE (code default)'
    else:
        aug_trusted = 'CHECK'

    ckpt_all = all(check_ckpt(r['exp_name'], phase) for r in group)
    scripts = '+'.join(PHASE_SCRIPTS.get(phase, ['?']))
    log_files = '/'.join(r['log_file'] for r in group)

    agg_valid = 'FALSE' if base in INVALID_AGGREGATION else 'TRUE'

    row = [base, phase, r0['e_series'], corpus, r0['pooling'], r0['fusion'],
           str(r0['adapter']), str(r0['unfreeze']), r0['aug'], aug_trusted,
           seeds_str, wa_per, wa_ms, uar_per, uar_ms,
           scripts, log_files, str(ckpt_all), r0['protocol'], agg_valid]
    rows_out.append(row)

with open(OUTPUT, 'w', encoding='utf-8', newline='') as f:
    for row in rows_out:
        f.write(','.join(csv_q(c) for c in row) + '\n')

print(f"Manifest rebuilt: {OUTPUT}")
print(f"Rows: {len(rows_out)-1} (+ header)")
# Count invalid
invalid_count = sum(1 for r in rows_out[1:] if r[-1] == 'FALSE')
print(f"Invalid aggregation: {invalid_count} ({', '.join(INVALID_AGGREGATION)})")
