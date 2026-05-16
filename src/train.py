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
from src.models import SSLBackbone, WavLMLayerFusion, SEMLP
from src.models.pooling import create_pooling, extract_prosody

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Full Model ─────────────────────────────────────────────

class SERModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pooling_type = config['pooling_type']
        self.num_classes = config.get('num_classes', 4)

        # Module 2: WavLM backbone + layer fusion
        self.backbone = SSLBackbone(
            model_name=config.get('ssl_model', 'wavlm'),
            frozen=True,
            device=device,
        )
        self.layer_fusion = WavLMLayerFusion(num_layers=12)

        # Module 3: Pooling
        self.pooler = create_pooling(
            pooling_type=self.pooling_type,
            ssl_dim=768,
        )

        # Classifier
        self.classifier = SEMLP(input_dim=768, num_classes=self.num_classes)

    def forward(self, waveforms, lengths=None):
        # M2: Extract all hidden layers and fuse
        _, all_hidden = self.backbone(waveforms, return_all_layers=True)
        fused = self.layer_fusion(all_hidden)  # (B, T, 768)

        # Build mask from lengths
        B, T = fused.shape[:2]
        mask = None
        if lengths is not None:
            mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)

        # M3: Pool with or without prosody
        if self.pooling_type == 'prosody_guided':
            f0, energy = _extract_prosody_batch(waveforms)
            f0 = f0.to(device)
            energy = energy.to(device)
            # Align prosody length to fused features
            if f0.shape[1] != T:
                f0 = _interpolate_1d(f0, T)
                energy = _interpolate_1d(energy, T)
            pooled = self.pooler(fused, f0, energy, mask=mask)
        else:
            pooled = self.pooler(fused, mask=mask)

        logits = self.classifier(pooled)
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


def train_epoch(model, dataloader, optimizer, criterion):
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
        optimizer.step()

        total_loss += loss.item() * waveforms.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

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

    wa = accuracy_score(all_labels, all_preds)
    uar = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    return total_loss / total_samples, wa, uar, all_preds, all_labels


# ── Main ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', nargs='+', default=['c-besd'])
    parser.add_argument('--test_data', nargs='+', default=None)
    parser.add_argument('--pooling_type', default='prosody_guided',
                        choices=['prosody_guided', 'self_attention'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ssl_model', default='wavlm')
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--output_dir', default='checkpoints')
    parser.add_argument('--exp_name', default='exp')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Data
    if args.test_data:
        dls = get_cross_corpus_dataloaders(
            args.train_data, args.test_data,
            batch_size=args.batch_size, seed=args.seed,
        )
    else:
        dls = get_dataloaders(
            args.train_data, batch_size=args.batch_size, seed=args.seed,
        )

    # Model
    config = {
        'pooling_type': args.pooling_type,
        'num_classes': args.num_classes,
        'ssl_model': args.ssl_model,
    }
    model = SERModel(config).to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_total:,} total, {n_trainable:,} trainable")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, 'best_model.pt')

    best_val_wa = 0.0
    patience_counter = 0
    history = defaultdict(list)

    for epoch in range(args.epochs):
        train_loss, train_wa, train_uar = train_epoch(model, dls['train'], optimizer, criterion)
        val_loss, val_wa, val_uar, _, _ = evaluate(model, dls['val'])

        history['train_wa'].append(train_wa)
        history['val_wa'].append(val_wa)
        history['val_uar'].append(val_uar)

        scheduler.step()

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
        'train_data': args.train_data,
        'test_data': args.test_data or args.train_data,
        'best_val_wa': float(best_val_wa),
        'test_wa': float(test_wa),
        'test_uar': float(test_uar),
        'best_epoch': int(checkpoint['epoch']),
    }
    print(f"RESULT: {json.dumps(results)}")

    # Save results
    os.makedirs('results/logs', exist_ok=True)
    log_path = f"results/logs/{args.exp_name}.json"
    with open(log_path, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
