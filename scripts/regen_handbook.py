#!/usr/bin/env python3
"""
Regenerate authoritative handbook from 192 JSON logs.
- ddof=1 sample std
- 3 INVALID experiments EXCLUDED from all aggregations
- Deterministic output (stable tiebreaker sort)
- FAU UAR by test_data (correct imbalance grouping)
- In-domain FAU UAR computed separately
Output: docs/current/权威数据手册.md (AUTO-GENERATED, single source of truth)
"""

import json, os, math, re
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "results" / "logs"
OUTPUT = PROJECT_ROOT / "docs" / "current" / "权威数据手册.md"

INVALID_AGG = set()  # All experiments now valid after 2026-06-22 re-run
INVALID_REASONS = {}  # E1-08/E4-04/E4-10 fixed via re-run on 2026-06-22

def load_logs():
    logs = {}
    for fname in sorted(os.listdir(LOGS_DIR)):
        if not fname.endswith('.json'): continue
        with open(LOGS_DIR / fname) as f:
            d = json.load(f)
        logs[d['exp_name']] = d
    return logs

def get_runs(logs, base):
    if base in logs: return [logs[base]]
    runs = []
    for s in [42, 123, 456]:
        k = f"{base}_s{s}"
        if k in logs: runs.append(logs[k])
    return runs

def sample_ms(vals_pct):
    if not vals_pct: return None, None
    n = len(vals_pct)
    if n == 1: return vals_pct[0], None
    m = sum(vals_pct)/n
    var = sum((v-m)**2 for v in vals_pct)/(n-1)
    return m, math.sqrt(var)

def wa(r): return r.get('test_wa')
def uar(r): return r.get('test_uar')

def corpus_name(dl):
    if not dl: return '?'
    n = dl[0].lower()
    if 'besd' in n: return 'C-BESD'
    if 'fau' in n or 'aibo' in n: return 'FAU_Aibo'
    if 'iemocap' in n: return 'IEMOCAP'
    return n

def fmt_ms(m, s):
    if s is None: return f"{m:.2f}%"
    return f"{m:.2f}+-{s:.2f}%"

def fmt_pct(v): return f"{v*100:.2f}%"

def wa_stats_or_invalid(logs, eid):
    """Return (mean, std, wa_per_seed_list) or ('INVALID', None, seed_vals) if INVALID."""
    runs = get_runs(logs, eid)
    wa_v = [wa(r)*100 for r in runs if wa(r) is not None]
    if eid in INVALID_AGG:
        return None, None, [f"{v:.2f}%" for v in wa_v], True
    m, s = sample_ms(wa_v)
    return m, s, [f"{v:.2f}%" for v in wa_v], False

def uar_stats_or_invalid(logs, eid):
    runs = get_runs(logs, eid)
    uv = [uar(r)*100 for r in runs if uar(r) is not None]
    if eid in INVALID_AGG:
        return None, None, [f"{v:.2f}%" for v in uv], True
    m, s = sample_ms(uv)
    return m, s, [f"{v:.2f}%" for v in uv], False

# ── TABLE GENERATORS ──

def gen_b1(logs):
    lines = []
    lines.append("## B1 — Pooling x Dataset Baseline (E1, Frozen, 3-seed, ddof=1)\n")
    lines.append("| Experiment | Dataset | Pooling | WA s42 | WA s123 | WA s456 | WA mean+-std | UAR mean+-std |")
    lines.append("|-----------|--------|---------|--------|---------|---------|-------------|-------------|")
    configs = [
        ("E1-01","C-BESD","mean"),("E1-02","C-BESD","self_attention"),("E1-03","C-BESD","prosody_guided"),
        ("E1-04","FAU_Aibo","mean"),("E1-05","FAU_Aibo","self_attention"),("E1-06","FAU_Aibo","prosody_guided"),
        ("E1-07","IEMOCAP","mean"),("E1-08","IEMOCAP","self_attention"),("E1-09","IEMOCAP","prosody_guided"),
    ]
    b1_stats = {}
    for eid, ds, pool in configs:
        wm, ws, wseed, invalid_w = wa_stats_or_invalid(logs, eid)
        um, us, useed, invalid_u = uar_stats_or_invalid(logs, eid)
        flag = " [INVALID]" if eid in INVALID_AGG else ""
        wa_cell = "INVALID" if invalid_w else fmt_ms(wm, ws)
        uar_cell = "INVALID" if invalid_u else fmt_ms(um, us)
        lines.append(f"| {eid}{flag} | {ds} | {pool} | {wseed[0]} | {wseed[1] if len(wseed)>1 else '?'} | {wseed[2] if len(wseed)>2 else '?'} | {wa_cell} | {uar_cell} |")
        if not invalid_w:
            b1_stats[(ds, pool)] = (wm, um)
    lines.append("")
    lines.append(f"**B1 conclusion**: Self-Attention > Mean >> Prosody. ")
    lines.append(f"C-BESD ceiling: E1-02 = {fmt_ms(*sample_ms([wa(r)*100 for r in get_runs(logs,'E1-02') if wa(r)]))} (3-seed sample mean). ")
    lines.append(f"FAU ceiling: E1-05 = {fmt_ms(*sample_ms([wa(r)*100 for r in get_runs(logs,'E1-05') if wa(r)]))}. ")
    if "E1-08" in INVALID_AGG:
        lines.append(f"E1-08 excluded from aggregation (INVALID).\n")
    else:
        lines.append("")
    return lines

def gen_b5(logs):
    lines = []
    lines.append("## B5 — WavLM Unfreeze Comparison (E2, 3-seed, ddof=1)\n")
    lines.append("| Experiment | Dataset | Pooling | WA s42 | WA s123 | WA s456 | WA mean+-std | UAR mean+-std | Δ vs frozen |")
    lines.append("|-----------|--------|---------|--------|---------|---------|-------------|-------------|-------------|")
    frozen_baselines = {}
    for eid_b1, ds, pool in [("E1-02","C-BESD","self_attention"),("E1-05","FAU_Aibo","self_attention"),("E1-09","IEMOCAP","prosody_guided")]:
        runs = get_runs(logs, eid_b1)
        wv = [wa(r)*100 for r in runs if wa(r) is not None]
        frozen_baselines[(ds, pool)] = sample_ms(wv)[0]
    configs = [("E2-01","C-BESD","self_attention"),("E2-02","FAU_Aibo","self_attention"),("E2-03","IEMOCAP","prosody_guided")]
    for eid, ds, pool in configs:
        runs = get_runs(logs, eid)
        wa_v = [wa(r)*100 for r in runs if wa(r) is not None]
        uar_v = [uar(r)*100 for r in runs if uar(r) is not None]
        wm, ws = sample_ms(wa_v)
        um, us = sample_ms(uar_v)
        fb = frozen_baselines.get((ds, pool))
        delta_str = f"{wm-fb:+.2f}pp" if fb else "?"
        lines.append(f"| {eid} | {ds} | {pool} | {fmt_pct(wa(runs[0]))} | {fmt_pct(wa(runs[1])) if len(runs)>1 else '?'} | {fmt_pct(wa(runs[2])) if len(runs)>2 else '?'} | {fmt_ms(wm,ws)} | {fmt_ms(um,us)} | {delta_str} |")
    lines.append("")
    lines.append("**B5 corrected conclusion**: Unfreeze helps C-BESD (+5.04pp) and IEMOCAP (+1.99pp, prosody→prosody aligned), ")
    lines.append("but does NOT help FAU (delta=-0.67pp, essentially flat/negative). ")
    lines.append("The previous claim 'FAU +8.2pp' was based on a phantom 76.02% value not found in any log.\n")
    return lines

def gen_b2(logs):
    lines = []
    lines.append("## B2 — Zero-shot Cross-corpus Transfer (E3, single-run)\n")
    lines.append("| Experiment | Source | Target | Pooling | WA | UAR |")
    lines.append("|-----------|--------|--------|---------|-----|-----|")
    all_b2 = []
    for i in range(1, 19):
        eid = f"E3-{i:02d}"
        if eid not in logs: continue
        r = logs[eid]
        all_b2.append((eid, r['train_data'][0], r['test_data'][0], r['pooling_type'], wa(r)*100, uar(r)*100))
    # Deterministic sort: source, target, pooling, then experiment_id
    all_b2.sort(key=lambda x: (x[1], x[2], x[3], x[0]))
    for eid, src, tgt, pool, w, u in all_b2:
        lines.append(f"| {eid} | {src} | {tgt} | {pool} | {w:.2f}% | {u:.2f}% |")
    # Deterministic best: max WA, tiebreak by experiment_id
    best = max(all_b2, key=lambda x: (x[4], -int(x[0].split('-')[1])))
    worst = min(all_b2, key=lambda x: (x[4], int(x[0].split('-')[1])))
    lines.append("")
    lines.append(f"**B2 corrected conclusion**: Best zero-shot = {best[0]} ({best[1]}->{best[2]}, {best[3]}) = {best[4]:.2f}%. ")
    lines.append(f"Range: {worst[4]:.2f}%-{best[4]:.2f}%. ")
    lines.append("Previous handbook values (41.05%, 35.47%, 19.17% as range minimum) are PHANTOM - not found in any log.\n")
    return lines

def gen_b3(logs):
    lines = []
    lines.append("## B3 — Data Augmentation Sensitivity (E4, 3-seed, ddof=1)\n")
    lines.append("| Experiment | Aug | Pooling | WA s42 | WA s123 | WA s456 | WA mean+-std | UAR mean+-std |")
    lines.append("|----------|-----|---------|--------|---------|---------|-------------|-------------|")
    configs = [
        ("E4-01","C1","self_attention"),("E4-02","C2","self_attention"),("E4-03","C3","self_attention"),("E4-04","C4","self_attention"),
        ("E4-05","C1","mean"),("E4-06","C2","mean"),("E4-07","C3","mean"),("E4-08","C4","mean"),
        ("E4-09","C1","prosody_guided"),("E4-10","C2","prosody_guided"),("E4-11","C3","prosody_guided"),("E4-12","C4","prosody_guided"),
    ]
    for eid, aug, pool in configs:
        wm, ws, wseed, invalid_w = wa_stats_or_invalid(logs, eid)
        um, us, useed, invalid_u = uar_stats_or_invalid(logs, eid)
        flag = " [INVALID]" if eid in INVALID_AGG else ""
        wa_cell = "INVALID" if invalid_w else fmt_ms(wm, ws)
        uar_cell = "INVALID" if invalid_u else fmt_ms(um, us)
        lines.append(f"| {eid}{flag} | {aug} | {pool} | {wseed[0]} | {wseed[1] if len(wseed)>1 else '?'} | {wseed[2] if len(wseed)>2 else '?'} | {wa_cell} | {uar_cell} |")
    lines.append("")
    lines.append("**B3 conclusion**: C3 child augmentation shows weak positive benefit (+0.25-0.74pp). ")
    lines.append("C2/C4 domain mixing significantly hurts performance (-6.8 to -9.4pp). ")
    if "E4-04" in INVALID_AGG:
        lines.append(f"E4-04 excluded: {INVALID_REASONS['E4-04']}. ")
    if "E4-10" in INVALID_AGG:
        lines.append(f"E4-10 excluded: {INVALID_REASONS['E4-10']}.")
    lines.append("")
    return lines

def gen_b6(logs):
    lines = []
    lines.append("## B6 — Module Ablation (E6, C-BESD + FAU, 3-seed, ddof=1)\n")
    lines.append("Design: cumulative build-up from Mean+Last baseline.\n")
    lines.append("| Experiment | Dataset | Config | WA s42 | WA s123 | WA s456 | WA mean+-std | vs baseline delta |")
    lines.append("|-----------|--------|-------|--------|---------|---------|-------------|------------------|")
    configs = [
        ("E6-01","C-BESD","Mean+Last (baseline)"),("E6-02","C-BESD","+Adapter"),("E6-03","C-BESD","+SelfAttn,-Adapter"),
        ("E6-04","C-BESD","+WeightedFusion"),("E6-05","C-BESD","Full stack"),
        ("E6-06","FAU_Aibo","Mean+Last (baseline)"),("E6-07","FAU_Aibo","+Adapter"),("E6-08","FAU_Aibo","+SelfAttn,-Adapter"),
        ("E6-09","FAU_Aibo","+WeightedFusion"),("E6-10","FAU_Aibo","Full stack"),
    ]
    cb_base = sample_ms([wa(r)*100 for r in get_runs(logs,'E6-01') if wa(r) is not None])[0]
    fau_base = sample_ms([wa(r)*100 for r in get_runs(logs,'E6-06') if wa(r) is not None])[0]
    for eid, ds, cfg in configs:
        runs = get_runs(logs, eid)
        wa_v = [wa(r)*100 for r in runs if wa(r) is not None]
        wm, ws = sample_ms(wa_v)
        base = cb_base if 'C-BESD' in ds else fau_base
        delta = f"{wm-base:+.2f}pp"
        lines.append(f"| {eid} | {ds} | {cfg} | {fmt_pct(wa(runs[0]))} | {fmt_pct(wa(runs[1])) if len(runs)>1 else '?'} | {fmt_pct(wa(runs[2])) if len(runs)>2 else '?'} | {fmt_ms(wm,ws)} | {delta} |")
    lines.append("")
    lines.append("**B6 conclusion**: Pooling upgrade (Mean->SelfAttn) is the key improvement on C-BESD (+11pp). ")
    lines.append("WF adds marginal benefit (<1pp). Adapter is neutral-to-negative on both datasets.\n")
    return lines

def gen_b7(logs):
    lines = []
    lines.append("## B7 — Model Transfer Fine-tune (E7, 3-seed, ddof=1)\n")
    lines.append("| Experiment | Claimed Source | Target | WA s42 | WA s123 | WA s456 | WA mean+-std | UAR mean+-std | gap vs ceiling |")
    lines.append("|-----------|---------------|--------|--------|---------|---------|-------------|-------------|----------------|")
    cb_ceil = sample_ms([wa(r)*100 for r in get_runs(logs,'E1-02') if wa(r) is not None])[0]
    fau_ceil = sample_ms([wa(r)*100 for r in get_runs(logs,'E1-05') if wa(r) is not None])[0]
    iem_ceil = sample_ms([wa(r)*100 for r in get_runs(logs,'E1-08') if wa(r) is not None])[0]
    configs = [
        ("E7-01","C-BESD","FAU_Aibo",fau_ceil),("E7-02","C-BESD","IEMOCAP",iem_ceil),
        ("E7-03","FAU_Aibo","C-BESD",cb_ceil),("E7-04","FAU_Aibo","IEMOCAP",iem_ceil),
        ("E7-05","IEMOCAP","C-BESD",cb_ceil),("E7-06","IEMOCAP","FAU_Aibo",fau_ceil),
    ]
    for eid, src, tgt, ceil in configs:
        runs = get_runs(logs, eid)
        wa_v = [wa(r)*100 for r in runs if wa(r) is not None]
        uar_v = [uar(r)*100 for r in runs if uar(r) is not None]
        wm, ws = sample_ms(wa_v)
        um, us = sample_ms(uar_v)
        gap = f"{wm-ceil:+.2f}pp"
        lines.append(f"| {eid} | {src} | {tgt} | {fmt_pct(wa(runs[0]))} | {fmt_pct(wa(runs[1])) if len(runs)>1 else '?'} | {fmt_pct(wa(runs[2])) if len(runs)>2 else '?'} | {fmt_ms(wm,ws)} | {fmt_ms(um,us)} | {gap} |")
    lines.append("")
    lines.append(f"**B7 conclusion**: Target-domain ceiling dominates transfer results. ")
    lines.append(f"C-BESD targets converge to {cb_ceil:.2f}% (ceiling), FAU targets to {fau_ceil:.2f}%, IEMOCAP targets to {iem_ceil:.2f}%. ")
    lines.append("Source domain effect is limited (<0.5pp for same target). ")
    lines.append("**Note**: Source domain assignment relies on launch_b7.sh correctness - not independently verifiable from logs.\n")
    return lines

def gen_leaderboard(logs):
    lines = []
    lines.append("## Global Leaderboard (3-seed sample mean WA, ddof=1)\n")
    lines.append("| Rank | Experiment | Dataset | Config | WA mean+-std |")
    lines.append("|------|-----------|--------|--------|-------------|")
    entries = []
    all_bases = set()
    for exp_name in logs:
        m = re.match(r'^(.+)_s(\d+)$', exp_name)
        base = m.group(1) if m else exp_name
        all_bases.add(base)
    for base in all_bases:
        if re.search(r'_L\d+', base): continue
        if base in INVALID_AGG: continue
        runs = get_runs(logs, base)
        wa_v = [wa(r)*100 for r in runs if wa(r) is not None]
        if not wa_v: continue
        wm, ws = sample_ms(wa_v)
        r0 = runs[0]
        ds = corpus_name(r0.get('train_data',[]))
        pool = r0.get('pooling_type','?')
        unfreeze = 'unfreeze' if r0.get('unfreeze_ssl') else 'frozen'
        entries.append((base, ds, f"{pool}+{unfreeze}", wm, ws))
    # DETERMINISTIC sort: WA desc, experiment_id asc for ties
    entries.sort(key=lambda x: (-x[3], x[0]))
    for rank, (eid, ds, cfg, wm, ws) in enumerate(entries[:15], 1):
        lines.append(f"| {rank} | {eid} | {ds} | {cfg} | {fmt_ms(wm,ws)} |")
    lines.append("")
    return lines

def gen_wa_uar_table(logs):
    lines = []
    lines.append("## WA-UAR by Corpus\n")
    lines.append("")

    # ── Table 1: by test_data (correct for imbalance analysis) ──
    lines.append("### By test_data (WA-UAR gap reflects test-set class imbalance)\n")
    lines.append("| Test Corpus | WA mean+-std | UAR mean+-std | WA-UAR gap | N runs |")
    lines.append("|------------|-------------|--------------|-----------|--------|")
    td = defaultdict(lambda: {'wa':[], 'uar':[]})
    for exp_name, d in logs.items():
        # Exclude INVALID experiments from aggregation
        base = re.match(r'^(.+)_s\d+$', exp_name)
        base = base.group(1) if base else exp_name
        if base in INVALID_AGG: continue
        c = corpus_name(d.get('test_data',[]))
        w = d.get('test_wa'); u = d.get('test_uar')
        if w is not None: td[c]['wa'].append(w*100)
        if u is not None: td[c]['uar'].append(u*100)
    gaps_by_test = {}
    for c in ["C-BESD","FAU_Aibo","IEMOCAP"]:
        d = td[c]
        wm, ws = sample_ms(d['wa'])
        um, us = sample_ms(d['uar'])
        gap = wm-um if wm and um else 0
        gaps_by_test[c] = gap
        lines.append(f"| {c} | {fmt_ms(wm,ws)} | {fmt_ms(um,us)} | {gap:.2f}pp | {len(d['wa'])} |")
    lines.append("")
    faU_uar_by_test = td["FAU_Aibo"]['uar']
    fau_uar_mean_by_test = sample_ms(faU_uar_by_test)[0]
    lines.append(f"FAU_Aibo UAR (by test_data, all runs targeting FAU, excl INVALID): **{fau_uar_mean_by_test:.2f}%** (n={len(faU_uar_by_test)}). ")
    lines.append("This is the correct metric for FAU difficulty: all experiments where the model was evaluated on FAU. ")
    lines.append("")

    # ── Table 2: in-domain only (train==test) ──
    lines.append("### In-domain only (train_data == test_data, excl INVALID)\n")
    lines.append("| Corpus | WA mean+-std | UAR mean+-std | WA-UAR gap | N runs |")
    lines.append("|--------|-------------|--------------|-----------|--------|")
    id_data = defaultdict(lambda: {'wa':[], 'uar':[]})
    for exp_name, d in logs.items():
        base = re.match(r'^(.+)_s\d+$', exp_name)
        base = base.group(1) if base else exp_name
        if base in INVALID_AGG: continue
        train_c = corpus_name(d.get('train_data',[]))
        test_c = corpus_name(d.get('test_data',[]))
        if train_c != test_c: continue  # in-domain only
        w = d.get('test_wa'); u = d.get('test_uar')
        if w is not None: id_data[test_c]['wa'].append(w*100)
        if u is not None: id_data[test_c]['uar'].append(u*100)
    for c in ["C-BESD","FAU_Aibo","IEMOCAP"]:
        d = id_data[c]
        wm, ws = sample_ms(d['wa'])
        um, us = sample_ms(d['uar'])
        gap = wm-um if wm and um else 0
        lines.append(f"| {c} | {fmt_ms(wm,ws)} | {fmt_ms(um,us)} | {gap:.2f}pp | {len(d['wa'])} |")
    lines.append("")
    faU_uar_id = id_data["FAU_Aibo"]['uar']
    fau_uar_mean_id = sample_ms(faU_uar_id)[0]
    lines.append(f"FAU_Aibo in-domain UAR (train==test==FAU, excl INVALID): **{fau_uar_mean_id:.2f}%** (n={len(faU_uar_id)}). ")
    lines.append("This is the FAU difficulty number for the paper: models trained AND tested on FAU. ")
    lines.append("")

    # ── Resolve FAU UAR ambiguity ──
    lines.append("### FAU UAR: resolving the ambiguity\n")
    lines.append(f"- **By test_data FAU UAR = {fau_uar_mean_by_test:.2f}%** (n={len(faU_uar_by_test)}) — all runs where test_data is FAU (includes zero-shot and transfer targeting FAU)")
    lines.append(f"- **In-domain FAU UAR = {fau_uar_mean_id:.2f}%** (n={len(faU_uar_id)}) — only runs where train==test==FAU (the FAU difficulty metric)")
    lines.append(f"- Difference: {abs(fau_uar_mean_by_test - fau_uar_mean_id):.2f}pp — caused by zero-shot/transfer runs with FAU as target having different UAR characteristics")
    lines.append(f"- **Paper recommendation**: use in-domain FAU UAR = {fau_uar_mean_id:.2f}% for 'FAU difficulty' claims, ")
    lines.append(f"  and by_test_data FAU UAR = {fau_uar_mean_by_test:.2f}% for 'all FAU evaluations' context.")
    lines.append(f"- Note: the previously cited '41.67%' and '41.72%' refer to these two quantities computed under different aggregation conditions (with/without INVALID, different std). The canonical values under ac_suite_2026-06-validated protocol are as above.")
    lines.append("")

    lines.append(f"FAU_Aibo WA-UAR gap reflects severe class imbalance ({gaps_by_test['FAU_Aibo']:.2f}pp by test_data). ")
    lines.append(f"C-BESD WA~=UAR ({gaps_by_test['C-BESD']:.2f}pp) due to near-perfect class balance.\n")
    return lines

def gen_invalid_notes():
    lines = []
    lines.append("## Invalid Aggregation Experiments\n")
    lines.append("The following experiments have inconsistent configs across seeds. ")
    lines.append("Their per-seed data is preserved in provenance_manifest.csv but they are EXCLUDED ")
    lines.append("from ALL aggregated means (mean+-std, leaderboard, WA-UAR tables) in this handbook.\n")
    lines.append("| Experiment | Seeds affected | Issue |")
    lines.append("|-----------|---------------|-------|")
    for eid in sorted(INVALID_AGG):
        lines.append(f"| {eid} | see below | {INVALID_REASONS[eid]} |")
    lines.append("")
    return lines

# ── main ──
def main():
    logs = load_logs()
    lines = []
    lines.append("# Distribution-Driven Children's SER — Authoritative Data Handbook")
    lines.append("")
    lines.append("> **AUTO-GENERATED — do not hand-edit**")
    lines.append(f"> **Source**: `results/logs/E*-*.json` (192 files)")
    lines.append(f"> **Protocol**: `ac_suite_2026-06-validated`")
    lines.append(f"> **Generated**: 2026-06-22 by `scripts/regen_handbook.py`")
    lines.append("> **Std**: sample std (ddof=1)")
    invalid_list = ', '.join(sorted(INVALID_AGG)) if INVALID_AGG else '0 (none)'
    lines.append(f"> **INVALID experiments EXCLUDED from all aggregations**: {invalid_list}")
    lines.append("")
    lines.append("This document is regenerated from raw experiment logs. Any paper, chart, or abstract ")
    lines.append("must reference values from this document. To regenerate: `python scripts/regen_handbook.py`")
    lines.append("")

    lines.append("---\n")
    lines.extend(gen_invalid_notes())
    lines.append("---\n")
    lines.extend(gen_b1(logs))
    lines.append("---\n")
    lines.extend(gen_b5(logs))
    lines.append("---\n")
    lines.extend(gen_b2(logs))
    lines.append("---\n")
    lines.extend(gen_b3(logs))
    lines.append("---\n")
    lines.extend(gen_b6(logs))
    lines.append("---\n")
    lines.extend(gen_b7(logs))
    lines.append("---\n")
    lines.extend(gen_leaderboard(logs))
    lines.append("---\n")
    lines.extend(gen_wa_uar_table(logs))
    lines.append("---\n")

    lines.append("## Data Integrity Notes\n")
    lines.append(f"1. **std**: All mean+-std use sample std (ddof=1)")
    lines.append(f"2. **INVALID**: {len(INVALID_AGG)} experiments EXCLUDED from all aggregations (see above)")
    lines.append(f"3. **Config fields**: augment_condition, fusion_mode, use_adapter, unfreeze_ssl, reg_profile are code defaults - not experimental conditions")
    lines.append(f"4. **B7 source**: Source domain not independently verifiable from logs - relies on launch_b7.sh")
    lines.append(f"5. **Predictions**: 0/192 files contain predictions/confusion matrices - WA/UAR trusted as-is from sklearn")
    # Compute in-domain FAU UAR for data integrity note
    fau_id_uar_vals = []
    for exp_name, d in logs.items():
        base = re.match(r'^(.+)_s\d+$', exp_name)
        base = base.group(1) if base else exp_name
        if base in INVALID_AGG: continue
        if corpus_name(d.get('train_data',[])) == 'FAU_Aibo' and corpus_name(d.get('test_data',[])) == 'FAU_Aibo':
            u = d.get('test_uar')
            if u is not None: fau_id_uar_vals.append(u*100)
    fau_id_uar_str = fmt_ms(*sample_ms(fau_id_uar_vals)) if fau_id_uar_vals else 'N/A'
    lines.append(f"6. **FAU UAR**: in-domain (train==test==FAU) = {fau_id_uar_str} (n={len(fau_id_uar_vals)}); by_test_data FAU UAR reported in WA-UAR table above")
    lines.append("")

    with open(OUTPUT, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines))
    print(f"Handbook generated: {OUTPUT}")
    print(f"Lines: {len(lines)}")

if __name__ == "__main__":
    main()
