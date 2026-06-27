"""Statistical tests and acoustic statistics for paper v7 revision.

Computes from actual data:
1. Bootstrap CI for A1 vs A3 delta (C-BESD, IEMOCAP, CREMA-D)
2. Three-dataset F0/energy/duration statistics
3. Attention entropy Pearson r + p-value
"""
import os, sys, json
import numpy as np
from scipy import stats

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

# ── 1. Bootstrap CI for key deltas ──
# Load seed-level results from existing experiment JSONs
def bootstrap_ci(values, n_bootstrap=10000, alpha=0.05):
    """Compute 95% bootstrap CI for the mean."""
    values = np.array(values)
    means = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(sample.mean())
    means = np.sort(means)
    lo = means[int(alpha/2 * n_bootstrap)]
    hi = means[int((1-alpha/2) * n_bootstrap)]
    return float(np.mean(values)), float(lo), float(hi)

print("=" * 60)
print("1. BOOTSTRAP CI FOR KEY DELTAS")
print("=" * 60)

# C-BESD A1 vs A3 (from v5_results.json)
v5 = json.load(open('experiments/v5_622/v5_save_results.json'))
cb_a1 = [v['test'] for v in v5['experiments']['A1_baseline'].values()]
cb_a3 = [v['test'] for v in v5['experiments']['A3_prosody_only'].values()]
cb_delta = [a3 - a1 for a1, a3 in zip(cb_a1, cb_a3)]
cb_mean, cb_lo, cb_hi = bootstrap_ci(cb_delta)
print(f"C-BESD A1→A3: mean={cb_mean:.4f}, 95% CI=[{cb_lo:.4f}, {cb_hi:.4f}]")
print(f"  Raw deltas: {[f'{d:.4f}' for d in cb_delta]}")

# IEMOCAP A1 vs A3
ie = json.load(open('experiments/v5_622/iemocap_contrast_result.json'))
# The IEMOCAP result only has aggregate, compute from the two model runs
iem_a1 = ie['A1']['test_wa']
iem_a3 = ie['A3']['test_wa']
iem_delta = iem_a3 - iem_a1
print(f"IEMOCAP A1→A3: WA delta = {iem_delta:+.4f} (single run, no seed-level CI)")
# We need seed-level data for IEMOCAP — not available. Note as limitation.
print("  NOTE: IEMOCAP has single-run data only; seed-level bootstrap not possible.")

# CREMA-D A1 vs A3
cr = json.load(open('experiments/v5_622/cremad_contrast_result.json'))
cr_a1 = cr['A1_mean']['test_wa']
cr_a3 = cr['A3_prosody']['test_wa']
cr_delta = cr_a3 - cr_a1
print(f"CREMA-D A1→A3: WA delta = {cr_delta:+.4f} (single run, no seed-level CI)")
print("  NOTE: CREMA-D has single-run data only.")

# ── 2. Three-dataset acoustic statistics ──
print("\n" + "=" * 60)
print("2. THREE-DATASET ACOUSTIC STATISTICS")
print("=" * 60)

datasets = {
    'C-BESD': {
        'feats': 'data/train_wavlm_prosody.npy',
        'split_name': 'C-BESD (children, portrayed)',
    },
}

# For CREMA-D and IEMOCAP, we need the pre-extracted prosody features
# Check what's available
def compute_prosody_stats(feats_path, name):
    if not os.path.exists(feats_path):
        print(f"  {name}: features not found at {feats_path}")
        return None
    data = np.load(feats_path)  # (N, 200, 2): [F0, energy]
    f0 = data[:, :, 0]  # (N, 200)
    energy = data[:, :, 1]  # (N, 200)

    # Per-utterance statistics (only on non-zero frames)
    f0_means, f0_stds, f0_maxs = [], [], []
    energy_means, energy_stds = [], []
    for i in range(len(data)):
        f0_valid = f0[i][f0[i] > 0]  # only voiced frames
        en_valid = energy[i]
        if len(f0_valid) > 0:
            f0_means.append(f0_valid.mean())
            f0_stds.append(f0_valid.std())
            f0_maxs.append(f0_valid.max())
        energy_means.append(en_valid.mean())
        energy_stds.append(en_valid.std())

    # Use median (not mean) for F0 — mean is inflated by YIN spurious detections at fmax
    all_f0_valid = f0[f0 > 10]
    stats = {
        'n': len(data),
        'f0_mean': float(np.mean(f0_means)),  # inflated by artifacts
        'f0_median': float(np.median(f0)),  # more robust
        'f0_median_of_voiced': float(np.median(all_f0_valid)) if len(all_f0_valid) > 0 else 0,
        'f0_std_of_means': float(np.std(f0_means)),
        'f0_range': f"[{float(np.min(f0_means)):.0f}, {float(np.max(f0_maxs)):.0f}]",
        'energy_mean': float(np.mean(energy_means)),
        'energy_std': float(np.std(energy_means)),
        'note': 'F0 mean is inflated by YIN algorithm spurious detections at upper bound (2093 Hz). Use median for central tendency.',
    }
    return stats

# C-BESD prosody (from pre-extracted)
print("\nC-BESD (train split):")
cb_stats = compute_prosody_stats('data/train_wavlm_prosody.npy', 'C-BESD')
if cb_stats:
    for k, v in cb_stats.items():
        print(f"  {k}: {v}")

# IEMOCAP & CREMA-D — need to compute from their pre-extracted features
# These aren't saved locally. Compute from the experiment scripts or use cloud.
print("\nIEMOCAP & CREMA-D: prosody features need extraction from cloud/local runs")
print("  (Use scripts/run_iemocap_contrast.py and run_cremad_contrast.py output)")

# ── 3. Attention entropy Pearson r ──
print("\n" + "=" * 60)
print("3. ATTENTION ENTROPY vs ACCURACY CORRELATION")
print("=" * 60)

attn_stats = json.load(open('experiments/v5_622/attention_analysis/attention_stats.json'))
entropies = []
accuracies = []
for cls_name, s in attn_stats.items():
    entropies.append(s['entropy'])
    accuracies.append(s['accuracy'])
    print(f"  {cls_name}: entropy={s['entropy']:.4f}, acc={s['accuracy']:.4f}")

r, p = stats.pearsonr(entropies, accuracies)
rho, p_s = stats.spearmanr(entropies, accuracies)
print(f"\n  Pearson r = {r:.4f}, p = {p:.4f}")
print(f"  Spearman ρ = {rho:.4f}, p = {p_s:.4f}")
print(f"  (N=6 emotions; p<0.05 indicates significant correlation)")

max_entropy = np.log(200)  # theoretical max entropy for 200 frames
print(f"  Theoretical max entropy: {max_entropy:.4f}")

# ── Summary ──
summary = {
    'bootstrap': {
        'cb_a1_a3': {'mean': cb_mean, 'ci95': [cb_lo, cb_hi], 'raw_deltas': cb_delta},
        'iemocap': {'delta': iem_delta, 'note': 'single run, no CI available'},
        'cremad': {'delta': cr_delta, 'note': 'single run, no CI available'},
    },
    'acoustic': {
        'cbesd': cb_stats,
        'iemocap': 'needs extraction from cloud run',
        'cremad': 'needs extraction from local run',
    },
    'attention_correlation': {
        'pearson_r': float(r), 'pearson_p': float(p),
        'spearman_rho': float(rho), 'spearman_p': float(p_s),
        'n': 6,
    }
}

os.makedirs('experiments/v5_622', exist_ok=True)
with open('experiments/v5_622/statistical_tests.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print(f"\nSaved: experiments/v5_622/statistical_tests.json")
