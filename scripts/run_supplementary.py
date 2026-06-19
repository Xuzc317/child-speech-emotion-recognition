#!/usr/bin/env python3
"""Supplementary experiments: CM, XAI+APC, Layer Weights, FD plot.
Upload to AutoDL and run: python scripts/run_supplementary.py
"""
import os, sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.train import SERModel, evaluate
from src.data import get_dataloaders, get_cross_corpus_dataloaders
from src.models.pooling import extract_prosody
from src.evaluation import AttentionProsodyExplainer
from src.extract_diagnostics import DiagnosticWrapper, extract_layer_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_FIG = 'results/supplementary/figures'
OUT_DATA = 'results/supplementary/data'
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_DATA, exist_ok=True)

CKPT = 'checkpoints'

# ── Experiment list ────────────────────────────────────────
CM_EXPS = [
    ('B1', 'E1-02_s42', f'{CKPT}/b1/E1-02_s42/best_model.pt', ['c-besd'], 6,
     'C-BESD_SelfAttn_frozen'),
    ('B1', 'E1-05_s42', f'{CKPT}/b1/E1-05_s42/best_model.pt', ['fau-aibo'], 4,
     'FAU_SelfAttn_frozen'),
    ('B1', 'E1-09_s42', f'{CKPT}/b1/E1-09_s42/best_model.pt', ['iemocap'], 4,
     'IEMOCAP_Prosody_frozen'),
    ('B5', 'E2-01_s456', f'{CKPT}/b5/E2-01_s456/best_model.pt', ['c-besd'], 6,
     'C-BESD_SelfAttn_unfrozen'),
    ('B5', 'E2-02_s456', f'{CKPT}/b5/E2-02_s456/best_model.pt', ['fau-aibo'], 4,
     'FAU_SelfAttn_unfrozen'),
    ('B5', 'E2-03_s123', f'{CKPT}/b5/E2-03_s123/best_model.pt', ['iemocap'], 4,
     'IEMOCAP_Prosody_unfrozen'),
    ('B6', 'E6-04_s42', f'{CKPT}/b6/E6-04_s42/best_model.pt', ['c-besd'], 6,
     'C-BESD_MeanPool'),
    ('B7', 'E7-03_s42', f'{CKPT}/b7/E7-03_s42/best_model.pt', ['c-besd'], 6,
     'C-BESD_Transfer'),
]

LABELS_4 = ['angry', 'happy', 'neutral', 'sad']
LABELS_6 = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']


def run_cm(phase, exp_id, ckpt_path, train_data, n_classes, label):
    """Generate confusion matrix for one experiment."""
    print(f'\n  [{phase}] {label} ({exp_id})')
    if not os.path.exists(ckpt_path):
        print(f'    SKIP: checkpoint not found: {ckpt_path}')
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    config.setdefault('pooling_type', 'self_attention')

    model = SERModel(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    dls = get_dataloaders(train_data, batch_size=16, seed=42)
    _, _, _, preds, trues = evaluate(model, dls['test'])

    classes = LABELS_6 if n_classes == 6 else LABELS_4
    cm = confusion_matrix(trues, preds, labels=list(range(n_classes)))

    # Per-class accuracy
    per_class = {}
    for i in range(n_classes):
        row_sum = cm[i].sum()
        per_class[classes[i]] = float(cm[i, i] / row_sum) if row_sum > 0 else 0.0

    # Plot
    fig, ax = plt.subplots(figsize=(max(5, n_classes), max(4.5, n_classes - 0.5)))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(classes, rotation=45 if n_classes == 6 else 0, ha='right')
    ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'{label}\n{exp_id}')
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)

    safe = label.replace(' ', '_')
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT_FIG, f'cm_{safe}.{ext}'), bbox_inches='tight')
    plt.close(fig)

    # Save data
    result = {
        'experiment': exp_id, 'phase': phase, 'label': label,
        'n_classes': n_classes, 'classes': classes,
        'confusion_matrix': cm.tolist(),
        'per_class_accuracy': per_class,
        'overall_wa': float(np.sum(np.diag(cm)) / cm.sum()),
    }
    json_path = os.path.join(OUT_DATA, f'cm_{safe}.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'    WA={result["overall_wa"]:.4f}  saved: cm_{safe}.png')
    return result


def run_xai(ckpt_path, dataset_name, prefix):
    """Extract XAI saliency + APC for one checkpoint."""
    print(f'\n  XAI: {prefix}')
    if not os.path.exists(ckpt_path):
        print(f'    SKIP: checkpoint not found')
        return

    model = DiagnosticWrapper(ckpt_path)
    dls = get_dataloaders([dataset_name], batch_size=1, seed=42)
    batch = next(iter(dls['test']))
    waveform = batch[0][0]

    fused, attn, f0, rms, wav_np = model.forward_with_attention(waveform)

    explainer = AttentionProsodyExplainer(sr=16000, ssl_frame_rate=50)
    apc = explainer.compute_apc(attn, f0, rms)
    explainer.plot_saliency(wav_np, f0, rms, attn,
                            save_path=os.path.join(OUT_FIG, f'xai_{prefix}.png'),
                            title=f'XAI: {prefix} ({model.pooling_type})')

    # Accumulate APC over 50 samples
    apc_vals = {'apc_wav': [], 'apc_delta': []}
    for i, batch in enumerate(dls['test']):
        if i >= 50:
            break
        wf = batch[0][0]
        _, attn_i, f0_i, rms_i, _ = model.forward_with_attention(wf)
        apc_i = explainer.compute_apc(attn_i, f0_i, rms_i)
        apc_vals['apc_wav'].append(apc_i['apc_wav'])
        apc_vals['apc_delta'].append(apc_i['apc_delta'])

    result = {
        'prefix': prefix, 'pooling_type': model.pooling_type,
        'n_samples': len(apc_vals['apc_wav']),
        'apc_wav_mean': float(np.mean(apc_vals['apc_wav'])),
        'apc_wav_std': float(np.std(apc_vals['apc_wav'])),
        'apc_delta_mean': float(np.mean(apc_vals['apc_delta'])),
        'apc_delta_std': float(np.std(apc_vals['apc_delta'])),
    }
    with open(os.path.join(OUT_DATA, f'apc_{prefix}.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(f'    APC_wav={result["apc_wav_mean"]:.4f}  APC_delta={result["apc_delta_mean"]:.4f}')


def run_layer_weights(ckpt_path, prefix):
    """Extract layer fusion weights."""
    print(f'\n  Layer Weights: {prefix}')
    if not os.path.exists(ckpt_path):
        print(f'    SKIP: checkpoint not found')
        return

    # Direct extraction (avoid torch.load weights_only issue)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt['model_state_dict']
    key = 'layer_fusion.layer_weights'
    if key in state:
        weights = state[key].numpy()
        weights = weights / weights.sum()
        result = {
            'layer_weights': weights.tolist(),
            'argmax_layer': int(np.argmax(weights)),
            'entropy': float(-np.sum(weights * np.log(weights + 1e-12))),
        }
    else:
        print(f'    ERROR: layer_weights not found')
        return

    # Plot
    weights = np.array(result['layer_weights'])
    fig, ax = plt.subplots(figsize=(8, 4))
    layers = [f'L{i+1}' for i in range(12)]
    colors = ['#2ecc71' if i == result['argmax_layer'] else '#3498db' for i in range(12)]
    ax.bar(layers, weights, color=colors)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Weight')
    ax.set_title(f'Layer Fusion Weights: {prefix}\nargmax=L{result["argmax_layer"]+1}, entropy={result["entropy"]:.3f}')
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT_FIG, f'layer_weights_{prefix}.{ext}'), bbox_inches='tight')
    plt.close(fig)

    with open(os.path.join(OUT_DATA, f'layer_weights_{prefix}.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(f'    argmax=L{result["argmax_layer"]+1}  entropy={result["entropy"]:.3f}')


# ── Main ──────────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 60)
    print('1. CONFUSION MATRICES')
    print('=' * 60)
    cm_results = []
    for phase, exp_id, ckpt, train_data, ncls, label in CM_EXPS:
        r = run_cm(phase, exp_id, ckpt, train_data, ncls, label)
        if r:
            cm_results.append(r)

    print(f'\n{"=" * 60}')
    print('2. XAI + APC')
    print('=' * 60)
    run_xai(f'{CKPT}/b1/E1-02_s42/best_model.pt', 'c-besd', 'E1-02_C-BESD')
    run_xai(f'{CKPT}/b6/E6-04_s42/best_model.pt', 'c-besd', 'E6-04_C-BESD_MeanPool')

    print(f'\n{"=" * 60}')
    print('3. LAYER FUSION WEIGHTS')
    print('=' * 60)
    run_layer_weights(f'{CKPT}/b1/E1-02_s42/best_model.pt', 'E1-02_C-BESD')
    run_layer_weights(f'{CKPT}/b4/E5-01_s42/best_model.pt', 'E5-01_FAU')

    print(f'\n{"=" * 60}')
    print('4. CM PER-CLASS SUMMARY')
    print('=' * 60)
    for r in cm_results:
        print(f'  {r["label"]}: WA={r["overall_wa"]:.4f}')
        for cls_name, acc in r['per_class_accuracy'].items():
            print(f'    {cls_name}: {acc:.4f}')

    print(f'\nAll done. Output: {OUT_FIG}/ + {OUT_DATA}/')
