"""Main training loop for SER ablation studies and cross-corpus evaluation.

Integrates: DataLoader (M1) → WavLM + LayerFusion (M2) → Pooling (M3) → SEMLP
"""

import argparse
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import get_dataloaders, get_cross_corpus_dataloaders
from src.models import SSLBackbone, WavLMLayerFusion, SEMLP, AcousticCalibrationAdapter
from src.models.pooling import create_pooling, extract_prosody

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Augmentation condition helpers ─────────────────────────

_AUG_DATA_MIX_CONDITIONS = {'C2', 'C4'}  # conditions that mix IEMOCAP/FAU into training


def _resolve_train_datasets(train_data, augment_condition, num_classes):
    """Resolve training dataset list based on augmentation condition.

    C2/C4: add cross-domain data for distribution shift.
      - C-BESD → add iemocap (adult, acted)
      - FAU → add iemocap (adult, acted)
      - IEMOCAP → add fau-aibo (child, spontaneous)
    """
    if augment_condition not in _AUG_DATA_MIX_CONDITIONS:
        return train_data, num_classes

    augmented = list(train_data)
    for ds in train_data:
        ds_lower = ds.lower()
        if ds_lower in ('c-besd', 'c-besd-4cl'):
            if 'iemocap' not in augmented:
                augmented.append('iemocap')
        elif ds_lower == 'fau-aibo':
            if 'iemocap' not in augmented:
                augmented.append('iemocap')
        elif ds_lower == 'iemocap':
            # Adult data: mix child spontaneous speech instead
            if 'fau-aibo' not in augmented:
                augmented.append('fau-aibo')

    # C-BESD 6-class → switch to 4-class for cross-corpus mixing
    has_cbesd_6cl = any(d.lower() == 'c-besd' for d in train_data)
    has_cross = any(d.lower() in ('iemocap', 'fau-aibo') for d in augmented)

    if has_cbesd_6cl and has_cross:
        # Replace c-besd with c-besd-4cl (drop disgust/fear)
        augmented = ['c-besd-4cl' if d.lower() == 'c-besd' else d for d in augmented]
        num_classes = 4

    return augmented, num_classes


# ── Full Model ─────────────────────────────────────────────

class SERModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pooling_type = config['pooling_type']
        self.num_classes = config.get('num_classes', 4)
        self.fusion_mode = config.get('fusion_mode', 'weighted')
        self.fusion_best_layer = config.get('fusion_best_layer', 8)
        self.use_adapter = config.get('use_adapter', False)

        # Module 2: WavLM backbone
        unfreeze = config.get('unfreeze_ssl', False)
        self.backbone = SSLBackbone(
            model_name=config.get('ssl_model', 'wavlm'),
            frozen=not unfreeze,
            device=device,
        )

        # Module 2b: Layer fusion (mode-dependent)
        if self.fusion_mode == 'weighted':
            self.layer_fusion = WavLMLayerFusion(num_layers=12)
        else:
            self.layer_fusion = None  # last/best_single — select layer directly

        # Module 1 (optional): Adapter for E6 ablation
        if self.use_adapter:
            self.adapter = AcousticCalibrationAdapter(dim=768)
        else:
            self.adapter = None

        # Module 3: Pooling
        self.pooler = create_pooling(
            pooling_type=self.pooling_type,
            ssl_dim=768,
            dropout=config.get('pooling_dropout', 0.0),
        )

        # Classifier
        self.classifier = SEMLP(input_dim=768, num_classes=self.num_classes)

    def _get_layer_features(self, all_hidden):
        """Extract features based on fusion_mode."""
        if self.fusion_mode == 'weighted' and self.layer_fusion is not None:
            return self.layer_fusion(all_hidden)  # (B, T, 768)
        elif self.fusion_mode == 'last':
            return all_hidden[-1]  # last transformer layer
        elif self.fusion_mode == 'best_single':
            layer_idx = self.fusion_best_layer  # 1-indexed (1..12)
            if layer_idx < 1 or layer_idx > 12:
                raise ValueError(f"fusion_best_layer must be 1..12, got {layer_idx}")
            return all_hidden[layer_idx]  # hidden_states[1..12] → all_hidden[layer_idx]
        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

    def forward(self, waveforms, lengths=None, return_features=False):
        # M2: Extract all hidden layers
        _, all_hidden = self.backbone(waveforms, return_all_layers=True)
        fused = self._get_layer_features(all_hidden)  # (B, T, 768)

        # M1 (optional): Adapter
        if self.adapter is not None:
            fused = self.adapter(fused)

        # Build mask from lengths
        B, T = fused.shape[:2]
        mask = None
        if lengths is not None:
            mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)

        # M3: Pool with or without prosody
        if self.pooling_type == 'mean':
            if mask is not None:
                fused_m = fused * mask.unsqueeze(-1).float()
                pooled = fused_m.sum(dim=1) / mask.sum(dim=1, keepdim=True).float().clamp(min=1)
            else:
                pooled = fused.mean(dim=1)
        elif self.pooling_type == 'prosody_guided':
            f0, energy = _extract_prosody_batch(waveforms)
            f0 = f0.to(device)
            energy = energy.to(device)
            if f0.shape[1] != T:
                f0 = _interpolate_1d(f0, T)
                energy = _interpolate_1d(energy, T)
            pooled = self.pooler(fused, f0, energy, mask=mask)
        else:
            pooled = self.pooler(fused, mask=mask)

        logits = self.classifier(pooled)
        if return_features:
            return logits, pooled
        return logits


def _extract_prosody_batch(waveforms):
    """Extract F0 and RMS for a batch on CPU (librosa)."""
    f0_list, energy_list = [], []
    for wav in waveforms.cpu().numpy():
        f0, energy = extract_prosody(wav, sr=16000, hop_length=320)
        f0_list.append(torch.from_numpy(f0).float().unsqueeze(-1))
        energy_list.append(torch.from_numpy(energy).float().unsqueeze(-1))
    # Pad to max length in batch
    max_t = max(f.shape[0] for f in f0_list)
    f0_batch = torch.zeros(len(f0_list), max_t, 1)
    energy_batch = torch.zeros(len(energy_list), max_t, 1)
    for i, (f, e) in enumerate(zip(f0_list, energy_list)):
        n = f.shape[0]
        f0_batch[i, :n, 0] = f.squeeze(-1)
        energy_batch[i, :n, 0] = e.squeeze(-1)
    return f0_batch, energy_batch


def _interpolate_1d(x, target_len):
    """Linear interpolation to target length. (B, T, 1) → (B, target_len, 1)."""
    x = x.permute(0, 2, 1)  # (B, 1, T)
    x = nn.functional.interpolate(x, size=target_len, mode='linear', align_corners=False)
    return x.permute(0, 2, 1)  # (B, target_len, 1)


# ── Training Utilities ─────────────────────────────────────

def compute_metrics(logits, labels):
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    labels_np = labels.cpu().numpy()
    wa = accuracy_score(labels_np, preds)
    uar = recall_score(labels_np, preds, average='macro', zero_division=0)
    return wa, uar


def train_epoch(model, dataloader, optimizer, criterion, grad_clip=None):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_preds, all_labels = [], []

    for batch in dataloader:
        waveforms, labels, lengths, _, _ = batch
        waveforms = waveforms.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        logits = model(waveforms, lengths=lengths)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item() * waveforms.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    if total_samples == 0:
        return 0.0, 0.0, 0.0
    wa = accuracy_score(all_labels, all_preds)
    uar = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    return total_loss / total_samples, wa, uar


@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_preds, all_labels = [], []
    criterion = nn.CrossEntropyLoss()

    for batch in dataloader:
        waveforms, labels, lengths, _, _ = batch
        waveforms = waveforms.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        logits = model(waveforms, lengths=lengths)
        loss = criterion(logits, labels)

        total_loss += loss.item() * waveforms.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    if total_samples == 0:
        return 0.0, 0.0, 0.0, [], []
    wa = accuracy_score(all_labels, all_preds)
    uar = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    return total_loss / total_samples, wa, uar, all_preds, all_labels


# ── Main ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    # ── Data ──
    parser.add_argument('--train_data', nargs='+', default=['c-besd'])
    parser.add_argument('--test_data', nargs='+', default=None)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--data_split_seed', type=int, default=42,
                        help='Fixed seed for speaker-level data split (independent of --seed)')

    # ── Model architecture ──
    parser.add_argument('--pooling_type', default='prosody_guided',
                        choices=['mean', 'prosody_guided', 'self_attention'])
    parser.add_argument('--ssl_model', default='wavlm')
    parser.add_argument('--fusion_mode', default='weighted',
                        choices=['last', 'best_single', 'weighted'],
                        help='E5: how to combine WavLM 12 layers')
    parser.add_argument('--fusion_best_layer', type=int, default=8,
                        help='E5: which layer to use when fusion_mode=best_single (1..12)')
    parser.add_argument('--use_adapter', action='store_true',
                        help='E6: enable AcousticCalibrationAdapter')
    parser.add_argument('--unfreeze_ssl', action='store_true',
                        help='E2: unfreeze WavLM backbone for full fine-tuning')

    # ── Training hyperparams ──
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--ssl_lr', type=float, default=1e-5,
                        help='E2: learning rate for unfrozen SSL backbone')
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--label_smoothing', type=float, default=None)
    parser.add_argument('--pooling_dropout', type=float, default=None)
    parser.add_argument('--grad_clip', type=float, default=None)
    parser.add_argument('--reg_profile', choices=['default', 'fau'], default='default')
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)

    # ── Augmentation (E4) ──
    parser.add_argument('--augment_condition', default='C1',
                        choices=['C1', 'C2', 'C3', 'C4'],
                        help='E4: C1=clean, C2=data mix, C3=SafeAWGN, C4=extreme')

    # ── Transfer learning (E7) ──
    parser.add_argument('--load_checkpoint', default=None,
                        help='E7: path to pre-trained checkpoint for fine-tuning')

    # ── Output ──
    parser.add_argument('--output_dir', default='checkpoints')
    parser.add_argument('--exp_name', default='exp')
    args = parser.parse_args()

    # ── Resolve augmentation data mixing ──
    train_data, num_classes = _resolve_train_datasets(
        args.train_data, args.augment_condition, args.num_classes
    )

    reg_profiles = {
        'default': {
            'weight_decay': 1e-3,
            'label_smoothing': 0.1,
            'pooling_dropout': 0.0,
            'grad_clip': None,
        },
        'fau': {
            'weight_decay': 5e-3,
            'label_smoothing': 0.15,
            'pooling_dropout': 0.3,
            'grad_clip': 1.0,
        },
    }
    reg_cfg = reg_profiles[args.reg_profile]
    weight_decay = reg_cfg['weight_decay'] if args.weight_decay is None else args.weight_decay
    label_smoothing = reg_cfg['label_smoothing'] if args.label_smoothing is None else args.label_smoothing
    pooling_dropout = reg_cfg['pooling_dropout'] if args.pooling_dropout is None else args.pooling_dropout
    grad_clip = reg_cfg['grad_clip'] if args.grad_clip is None else args.grad_clip

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Data ──
    if args.test_data:
        dls = get_cross_corpus_dataloaders(
            train_data, args.test_data,
            batch_size=args.batch_size, seed=args.data_split_seed,
            num_workers=args.num_workers,
            augment_condition=args.augment_condition,
        )
    else:
        dls = get_dataloaders(
            train_data, batch_size=args.batch_size, seed=args.data_split_seed,
            num_workers=args.num_workers,
            augment_condition=args.augment_condition,
        )

    # ── Model ──
    config = {
        'pooling_type': args.pooling_type,
        'num_classes': num_classes,
        'ssl_model': args.ssl_model,
        'pooling_dropout': pooling_dropout,
        'fusion_mode': args.fusion_mode,
        'fusion_best_layer': args.fusion_best_layer,
        'use_adapter': args.use_adapter,
        'unfreeze_ssl': args.unfreeze_ssl,
    }
    model = SERModel(config).to(device)

    # Load pre-trained checkpoint for E7 fine-tuning
    if args.load_checkpoint:
        print(f"Loading checkpoint: {args.load_checkpoint}")
        ckpt = torch.load(args.load_checkpoint, map_location=device)
        # Load only compatible keys (skip classifier if num_classes differs)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in ckpt['model_state_dict'].items()
                           if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        skipped = len(ckpt['model_state_dict']) - len(pretrained_dict)
        print(f"  Loaded {len(pretrained_dict)}/{len(ckpt['model_state_dict'])} params"
              + (f" (skipped {skipped} incompatible)" if skipped else ""))

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_total:,} total, {n_trainable:,} trainable")

    # ── Optimizer with differential LR for E2 unfreeze ──
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    if args.unfreeze_ssl:
        backbone_params = list(model.backbone.parameters())
        head_params = [p for n, p in model.named_parameters()
                       if not n.startswith('backbone.') and p.requires_grad]
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': args.ssl_lr},
            {'params': head_params, 'lr': args.lr},
        ], weight_decay=weight_decay)
        print(f"Differential LR: backbone={args.ssl_lr}, head={args.lr}")
    else:
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=weight_decay,
        )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    print(
        f"Config: reg={args.reg_profile}, wd={weight_decay}, ls={label_smoothing}, "
        f"pdrop={pooling_dropout}, gclip={grad_clip}, "
        f"fusion={args.fusion_mode}, adapter={args.use_adapter}, "
        f"unfreeze_ssl={args.unfreeze_ssl}, augment={args.augment_condition}"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, 'best_model.pt')

    best_val_wa = 0.0
    patience_counter = 0
    history = defaultdict(list)

    for epoch in range(args.epochs):
        train_loss, train_wa, train_uar = train_epoch(
            model, dls['train'], optimizer, criterion, grad_clip=grad_clip
        )
        val_loss, val_wa, val_uar, _, _ = evaluate(model, dls['val'])

        history['train_wa'].append(train_wa)
        history['val_wa'].append(val_wa)
        history['val_uar'].append(val_uar)

        scheduler.step()
        if (epoch + 1) % 1 == 0:
            print(f'Epoch {epoch+1}: train_wa={train_wa:.4f}, val_wa={val_wa:.4f}, best={best_val_wa:.4f}', flush=True)

        if val_wa > best_val_wa:
            best_val_wa = val_wa
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config,
                'val_wa': val_wa,
                'val_uar': val_uar,
                'args': vars(args),
            }, save_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    # Final test evaluation
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_wa, test_uar, test_preds, test_labels = evaluate(model, dls['test'])

    # Print final results
    results = {
        'exp_name': args.exp_name,
        'pooling_type': args.pooling_type,
        'train_data': train_data,
        'test_data': args.test_data or train_data,
        'seed': int(args.seed),
        'best_val_wa': float(best_val_wa),
        'test_wa': float(test_wa),
        'test_uar': float(test_uar),
        'best_epoch': int(checkpoint['epoch']),
        'reg_profile': args.reg_profile,
        'weight_decay': float(weight_decay),
        'label_smoothing': float(label_smoothing),
        'pooling_dropout': float(pooling_dropout),
        'grad_clip': None if grad_clip is None else float(grad_clip),
        'fusion_mode': args.fusion_mode,
        'fusion_best_layer': args.fusion_best_layer if args.fusion_mode == 'best_single' else None,
        'use_adapter': args.use_adapter,
        'unfreeze_ssl': args.unfreeze_ssl,
        'augment_condition': args.augment_condition,
        'output_dir': args.output_dir,
        'protocol': 'ac_suite_2026-06',
    }
    print(f"RESULT: {json.dumps(results)}")

    # Save results
    os.makedirs('results/logs', exist_ok=True)
    log_path = f"results/logs/{args.exp_name}.json"
    with open(log_path, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
