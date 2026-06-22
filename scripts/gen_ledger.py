#!/usr/bin/env python3
"""
Phase 2补全: 生成完整逐字段账本 ledger.csv
对全部 handbook 声称的字段，每个给: handbook值 / log值 / 状态
状态: verified | mismatch_<type> | missing_in_handbook | missing_in_logs
"""

import json, math, os, re, csv
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "results" / "logs"
OUTPUT = PROJECT_ROOT / "validation" / "ledger_full.csv"

# ?? helpers ??????????????????????????????????????????????
def load_all_logs():
    logs = {}
    for fname in sorted(os.listdir(LOGS_DIR)):
        if not fname.endswith('.json'): continue
        with open(LOGS_DIR / fname) as f:
            d = json.load(f)
        logs[d['exp_name']] = d
    return logs

def get_seed_runs(logs, base):
    if base in logs:
        return [logs[base]]
    runs = []
    for s in [42, 123, 456]:
        k = f"{base}_s{s}"
        if k in logs: runs.append(logs[k])
    return runs

def mean_std_pct(vals):
    if not vals: return None, None
    if len(vals) == 1: return vals[0]*100, 0.0
    m = sum(vals)/len(vals)
    var = sum((v-m)**2 for v in vals)/len(vals)
    return m*100, math.sqrt(var)*100

def wa(run): return run.get('test_wa')
def uar(run): return run.get('test_uar')

# ?? handbook parser ??????????????????????????????????????
def parse_md_table_rows(text, marker):
    idx = text.find(marker)
    if idx < 0: return []
    text = text[idx:]
    rows, in_table = [], False
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith('|') and line.endswith('|'):
            if re.match(r'^[\|\s\-:]+$', line):
                in_table = True; continue
            if in_table or not rows:
                cells = [c.strip() for c in line.split('|')[1:-1]]
                rows.append(cells); in_table = True
        elif in_table and rows: break
    return rows[1:] if len(rows)>1 else rows

def extract_pct(s):
    s = re.sub(r'\*\*|?|?|?|?','',s).strip()
    m = re.search(r'(\d+\.\d+)%', s)
    return float(m.group(1)) if m else None

def extract_mean_std_str(s):
    s = re.sub(r'\*\*|?|?|?|?','',s).strip()
    m = re.search(r'([\d.]+?[\d.]+)%', s)
    return m.group(1)+'%' if m else s

def parse_all_handbook(path):
    with open(path, encoding='utf-8') as f: text = f.read()
    claims = {}  # (section, eid, field) -> value_str

    def add_table(rows, seed_cols, ms_col):
        for row in rows:
            if len(row) <= max(seed_cols[-1] if seed_cols else 0, ms_col):
                continue
            eid = row[0].strip()
            if not re.match(r'E\d', eid): continue
            for si, ci in enumerate(seed_cols):
                v = extract_pct(row[ci])
                if v is not None:
                    claims[(section, eid, f's{["42","123","456"][si]}')] = v
            claims[(section, eid, 'mean_std')] = extract_mean_std_str(row[ms_col])

    # B1
    section = 'B1'
    rows = parse_md_table_rows(text, "## 1. B1")
    add_table(rows, [3,4,5], 6)

    # B5
    section = 'B5'
    rows = parse_md_table_rows(text, "## 2. B5")
    add_table(rows, [3,4,5], 6)

    # B2
    section = 'B2'
    rows = parse_md_table_rows(text, "## 3. B2")
    for row in rows:
        if len(row) < 5: continue
        eid = row[0].strip()
        if not eid.startswith('E3'): continue
        v = extract_pct(row[-1])
        if v is not None:
            claims[(section, eid, 'WA')] = v

    # B3
    section = 'B3'
    rows = parse_md_table_rows(text, "## 4. B3")
    add_table(rows, [2,3,4], 5)

    # B6
    section = 'B6'
    rows = parse_md_table_rows(text, "## 6. B6")
    add_table(rows, [3,4,5], 6)

    # B7
    section = 'B7'
    rows = parse_md_table_rows(text, "## 7. B7")
    add_table(rows, [4,5,6], 7)

    # Leaderboard
    section = 'LB'
    rows = parse_md_table_rows(text, "## 8. 全局最高分榜单")
    for row in rows:
        if len(row) < 5: continue
        eid = row[1].strip()
        if not eid.startswith('E'): continue
        v = extract_pct(row[4])
        if v is not None:
            claims[(section, eid, 'WA')] = v

    # B2 text claims
    m = re.search(r'最佳零样本方向\s+\S+\s+\((\d+\.\d+)%', text)
    if m: claims[('TEXT', 'B2_best', 'WA')] = float(m.group(1))
    m = re.search(r'Zero-shot 跨语料仅\s+(\d+\.\d+)%-(\d+\.\d+)%', text)
    if m:
        claims[('TEXT', 'B2_min', 'WA')] = float(m.group(1))
        claims[('TEXT', 'B2_max', 'WA')] = float(m.group(2))

    return claims

# ?? log values ???????????????????????????????????????????
def get_log_val(logs, section, eid, field):
    """Return log value as string or None."""
    runs = get_seed_runs(logs, eid)
    if not runs: return None, 'no_runs'

    if field == 'WA':
        v = wa(runs[0])
        return f"{v*100:.2f}%" if v is not None else None, 'single'
    if field == 'mean_std':
        wv = [wa(r) for r in runs if wa(r) is not None]
        if not wv: return None, 'no_wa'
        m, s = mean_std_pct(wv)
        return f"{m:.2f}?{s:.2f}%", 'computed'
    # seed fields
    seed_map = {'s42': 42, 's123': 123, 's456': 456}
    if field in seed_map:
        target_seed = seed_map[field]
        for r in runs:
            if r.get('seed') == target_seed:
                v = wa(r)
                return f"{v*100:.2f}%" if v is not None else None, 'log'
        return None, 'seed_missing'
    return None, 'unknown_field'

def status_label(hb_val, log_val_str, diff_threshold=0.07):
    """Classify field status."""
    if log_val_str is None:
        return 'missing_in_logs'
    if hb_val is None:
        return 'missing_in_handbook'

    try:
        hb_num = float(str(hb_val).replace('%','').replace('?',' ').split()[0])
        log_num = float(str(log_val_str).replace('%','').replace('?',' ').split()[0])
        diff = abs(hb_num - log_num)
        if diff <= diff_threshold:
            return 'verified'
        if diff <= 2.0:
            return 'mismatch_minor'
        if diff <= 5.0:
            return 'mismatch_major'
        return 'mismatch_severe'
    except:
        return 'parse_error'

# ?? main ?????????????????????????????????????????????????
def main():
    logs = load_all_logs()
    print(f"Logs loaded: {len(logs)}")

    hb_path = PROJECT_ROOT / "docs" / "current" / "权威数据手册.md"
    hb = parse_all_handbook(hb_path)
    print(f"Handbook claims: {len(hb)}")

    # Build ledger rows
    rows = []
    verified = 0
    mismatch = 0
    missing_log = 0

    for (section, eid, field), hb_val in sorted(hb.items()):
        log_val, source = get_log_val(logs, section, eid, field)
        status = status_label(hb_val, log_val)

        if status == 'verified':
            verified += 1
        elif status.startswith('mismatch'):
            mismatch += 1
        elif status == 'missing_in_logs':
            missing_log += 1

        rows.append({
            'section': section,
            'experiment': eid,
            'field': field,
            'handbook_value': str(hb_val) if hb_val is not None else '',
            'log_value': log_val if log_val else 'N/A',
            'log_source': source,
            'status': status,
        })

    # Write CSV
    with open(OUTPUT, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['section','experiment','field','handbook_value','log_value','log_source','status'])
        w.writeheader()
        w.writerows(rows)

    # Summary
    print(f"\nLedger: {len(rows)} fields")
    print(f"  verified: {verified}")
    print(f"  mismatch (total): {mismatch}")
    print(f"  missing_in_logs: {missing_log}")

    # Status counts
    from collections import Counter
    sc = Counter(r['status'] for r in rows)
    for s, c in sorted(sc.items()):
        print(f"    {s}: {c}")

    print(f"\nOutput: {OUTPUT}")

    # ?? B: Explicit verification of load-bearing numbers ??
    print("\n" + "="*70)
    print("LOAD-BEARING NUMBER VERIFICATION")
    print("="*70)

    checks = [
        ("E1-02", "C-BESD frozen SA (天花板)"),
        ("E1-05", "FAU frozen SA"),
        ("E2-01", "C-BESD unfreeze 96.91%"),
        ("E6-03", "C-BESD SA + last (B6)"),
        ("E6-04", "C-BESD SA + WF (B6)"),
        ("E7-03", "FAU?C-BESD transfer"),
        ("E7-05", "IEMOCAP?C-BESD transfer"),
    ]

    for eid, desc in checks:
        runs = get_seed_runs(logs, eid)
        if not runs:
            print(f"\n{eid} ({desc}): NO RUNS FOUND")
            continue
        wa_vals = [wa(r) for r in runs if wa(r) is not None]
        uar_vals = [uar(r) for r in runs if uar(r) is not None]
        mw, sw = mean_std_pct(wa_vals)
        mu, su = mean_std_pct(uar_vals)
        seeds_str = " / ".join(f"{r.get('seed','?')}" for r in runs)
        wa_str = " / ".join(f"{v*100:.2f}%" for v in wa_vals)
        uar_str = " / ".join(f"{v*100:.2f}%" for v in uar_vals)
        pooling = runs[0].get('pooling_type','?')
        unfreeze = runs[0].get('unfreeze_ssl', False)
        train = runs[0].get('train_data', ['?'])[0]
        test = runs[0].get('test_data', ['?'])[0]
        print(f"\n{eid} ({desc})")
        print(f"  train={train} test={test} pooling={pooling} unfreeze={unfreeze}")
        print(f"  seeds: {seeds_str}")
        print(f"  WA per seed: {wa_str}")
        print(f"  WA mean?std: {mw:.2f}?{sw:.2f}%")
        print(f"  UAR per seed: {uar_str}")
        print(f"  UAR mean?std: {mu:.2f}?{su:.2f}%")

        # Check against CLAUDE.md claims
        if eid == "E2-01":
            if abs(mw - 96.91) < 0.1:
                print(f"  [OK] 96.91% CONFIRMED (matches CLAUDE.md)")
            else:
                print(f"  [MISMATCH] CLAUDE.md says 96.91% but log mean is {mw:.2f}%")

    # Also verify B2 best
    all_b2 = []
    for eid in [f"E3-{i:02d}" for i in range(1,19)]:
        runs = get_seed_runs(logs, eid)
        if runs:
            w = wa(runs[0])
            if w is not None:
                all_b2.append((eid, w*100, runs[0]['train_data'][0], runs[0]['test_data'][0], runs[0]['pooling_type']))
    all_b2.sort(key=lambda x: -x[1])
    print(f"\nB2 True Best (from logs):")
    for rank, (eid, w, src, tgt, pool) in enumerate(all_b2[:5], 1):
        print(f"  #{rank} {eid}: {w:.2f}% {src}?{tgt} ({pool})")

if __name__ == "__main__":
    main()
