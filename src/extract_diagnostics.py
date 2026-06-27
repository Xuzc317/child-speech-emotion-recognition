"""Diagnostics extraction from trained SER model.

Actions:
  A. Layer Weights → results/layer_weights.json
  B. XAI Saliency → results/xai_final.png + APC metrics
  C. Distribution Shift → results/distribution_shift.json
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import SSLBackbone, WavLMLayerFusion
from src.models.pooling import create_pooling, extract_prosody
from src.data import get_dataloaders
from src.evaluation import AttentionProsodyExplainer, DistributionShiftProbe

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs('results/logs', exist_ok=True)


def _interpolate_1d(x, target_len):
    """(B, T, 1) → (B, target_len, 1), same as train.py."""
    x = x.permute(0, 2, 1)
    x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)
    return x.permute(0, 2, 1)


# ── Reconstruction wrapper for diagnostics ─────────────────

class DiagnosticWrapper(torch.nn.Module):
    """Wraps SER model to expose intermediate outputs for XAI."""

    def __init__(self, checkpoint_path):
        super().__init__()
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = ckpt['config']

        self.pooling_type = config['pooling_type']
        self.backbone = SSLBackbone(
            model_name=config.get('ssl_model', 'wavlm'),
            frozen=True, device=device,
        )
        self.layer_fusion = WavLMLayerFusion(num_layers=12)
        self.pooler = create_pooling(pooling_type=self.pooling_type, ssl_dim=768)

        # Load weights
        state = ckpt['model_state_dict']
        self.layer_fusion.load_state_dict(
            {k.replace('layer_fusion.', ''): v for k, v in state.items()
             if k.startswith('layer_fusion.')}
        )
        self.pooler.load_state_dict(
            {k.replace('pooler.', ''): v for k, v in state.items()
             if k.startswith('pooler.')}
        )
        self.layer_fusion.to(device)
        self.pooler.to(device)
        self.eval()

    def forward_with_attention(self, waveform):
        """Return fused features + attention weights + prosody features."""
        wav = waveform.unsqueeze(0).to(device)  # (1, T_wav)
        _, all_hidden = self.backbone(wav, return_all_layers=True)
        fused = self.layer_fusion(all_hidden)  # (1, T, 768)

        if self.pooling_type == 'prosody_guided':
            wav_np = waveform.cpu().numpy()
            f0_np, rms_np = extract_prosody(wav_np, sr=16000, hop_length=320)
            f0 = torch.from_numpy(f0_np).float().unsqueeze(0).unsqueeze(-1).to(device)
            energy = torch.from_numpy(rms_np).float().unsqueeze(0).unsqueeze(-1).to(device)
            T = fused.shape[1]
            if f0.shape[1] != T:
                f0 = _interpolate_1d(f0, T)
                energy = _interpolate_1d(energy, T)
            f0_n = f0 / 2093.0
            energy_n = energy / energy.max(dim=1, keepdim=True).values.clamp(min=1e-8)
            prosody = torch.cat([f0_n, energy_n], dim=-1)
            prosody_emb = self.pooler.prosody_proj(prosody)
            combined = torch.cat([fused, prosody_emb], dim=-1)
            attn_raw = self.pooler.attn(combined).squeeze(-1)  # (1, T)
            attn = torch.softmax(attn_raw, dim=-1)  # (1, T)
            return (
                fused.squeeze(0),
                attn.squeeze(0).detach().cpu().numpy(),
                f0_np,
                rms_np,
                wav_np,
            )
        else:
            attn_raw = self.pooler.attn(fused).squeeze(-1)
            attn = torch.softmax(attn_raw, dim=-1)
            # For self-attention pooler, extract dummy prosody for display
            wav_np = waveform.cpu().numpy()
            f0_np, rms_np = extract_prosody(wav_np, sr=16000, hop_length=320)
            return (
                fused.squeeze(0),
                attn.squeeze(0).detach().cpu().numpy(),
                f0_np,
                rms_np,
                wav_np,
            )


# ── Action A: Layer Weights ────────────────────────────────

def extract_layer_weights(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt['model_state_dict']
    layer_weights_key = 'layer_fusion.layer_weights'
    if layer_weights_key in state:
        weights = state[layer_weights_key].numpy()
        weights = weights / weights.sum()  # softmax
        result = {
            'layer_weights': weights.tolist(),
            'argmax_layer': int(np.argmax(weights)),
            'entropy': float(-np.sum(weights * np.log(weights + 1e-12))),
        }
    else:
        result = {'error': 'layer_weights not found in checkpoint'}

    path = 'results/layer_weights.json'
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    return result


# ── Action B: XAI Saliency ─────────────────────────────────

def extract_xai(checkpoint_path):
    model = DiagnosticWrapper(checkpoint_path)

    # Get one sample from test set
    dls = get_dataloaders(['c-besd'], batch_size=1, seed=42)
    batch = next(iter(dls['test']))
    waveform = batch[0][0]  # (T_wav,)

    fused, attn, f0, rms, wav_np = model.forward_with_attention(waveform)

    explainer = AttentionProsodyExplainer(sr=16000, ssl_frame_rate=50)
    apc = explainer.compute_apc(attn, f0, rms)
    explainer.plot_saliency(wav_np, f0, rms, attn,
                            save_path='results/xai_final.png',
                            title=f'XAI Saliency Map ({model.pooling_type})')

    path = 'results/logs/apc_metrics.json'
    with open(path, 'w') as f:
        json.dump({'pooling_type': model.pooling_type, **apc}, f, indent=2)
    return apc


# ── Action C: Distribution Shift ───────────────────────────

def extract_distribution_shift(checkpoint_path):
    backbone = SSLBackbone(model_name='wavlm', frozen=True, device=device)
    probe = DistributionShiftProbe(backbone=backbone, pooling='mean', device=device)

    dls_child = get_dataloaders(['c-besd'], batch_size=16, seed=42)
    dls_adult = get_dataloaders(['crema-d'], batch_size=16, seed=42)

    feats_child = probe.extract_features(dls_child['train'], max_samples=300)
    feats_adult = probe.extract_features(dls_adult['train'], max_samples=300)

    result = probe.evaluate(feats_child, feats_adult)

    path = 'results/distribution_shift.json'
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    return result


# ── Main ───────────────────────────────────────────────────

def main():
    checkpoint_path = 'checkpoints/best_model.pt'
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Run training first: python src/train.py --exp_name exp1 ...")
        sys.exit(1)

    print("Action A: Extracting layer weights...")
    lw = extract_layer_weights(checkpoint_path)
    print(f"  Layer weights saved, argmax_layer={lw.get('argmax_layer')}, entropy={lw.get('entropy', 'N/A'):.4f}")

    print("Action B: Generating XAI saliency...")
    apc = extract_xai(checkpoint_path)
    print(f"  APC_wav={apc['apc_wav']:.4f}, APC_delta={apc['apc_delta']:.4f}")

    print("Action C: Computing distribution shift...")
    shift = extract_distribution_shift(checkpoint_path)
    print(f"  FD={shift['fd']:.4f}, SMMD={shift['smmd']:.4f}")

    print("SUCCESS: All diagnostics extracted")


if __name__ == '__main__':
    main()
