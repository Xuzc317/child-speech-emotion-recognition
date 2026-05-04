"""新训练脚本 —— 基于 SSL 特征的完整流水线。

与旧训练脚本的主要区别：
  - 数据集基于 SSL 帧级特征（T×768），而非 162-dim 均值向量
  - 支持 Model A（预提取特征）和 Model B（在线 WAV 加载）两种模式
  - 训练时不使用 .npy 数据增强（增强在波形层完成）
  - 新增 adapter + pooling 前向传播

用法：
  python src/training/train_ssl.py \
      --mode precomputed \
      --train_feats data/train_ssl_feats.npy \
      --train_labels data/train_labels.npy \
      --test_feats data/test_ssl_feats.npy \
      --test_labels data/test_labels.npy

  python src/training/train_ssl.py \
      --mode online \
      --wav_dir /path/to/BESD/MY
"""

import argparse
import os
import sys
import math
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.dataset_ssl import SSLFeatureDataset, collate_fn_ssl_features
from src.models.ssl_backbone import SSLBackbone
from src.models.adapter import AcousticCalibrationAdapter
from src.models.pooling import TemporalImportancePooling
from src.models.semlp import SEMLP
from src.utils.experiment_logger import ExperimentLogger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 完整框架 ──────────────────────────────────────

class DistributionCalibratedSER(nn.Module):
    """分布校准的儿童语音情绪识别完整框架。

    模块组成：
      1. SSL Backbone (frozen) —— 帧级特征提取
      2. Distribution Calibration Adapter —— 分布偏移校准
      3. Temporal Importance Pooling —— 韵律引导的时序融合
      4. SEMLP / Simple Classifier —— 情感分类

    前向：
      waveforms / ssl_feats → Backbone → Adapter → Pooling → Classifier
    """

    def __init__(self, config, adapter_init=None):
        super().__init__()
        dim = config.get('feature_dim', 768)
        self.use_backbone = config.get('use_backbone', True)
        self.use_adapter = config.get('use_adapter', True)
        self.use_prosody = config.get('use_prosody', True)

        # 1. SSL Backbone
        if self.use_backbone:
            self.backbone = SSLBackbone(
                model_name=config.get('ssl_model', 'emotion2vec'),
                frozen=config.get('frozen_backbone', True),
                device=device,
            )

        # 2. Distribution Calibration Adapter
        if self.use_adapter:
            init_scale = adapter_init['scale'] if adapter_init else None
            init_bias = adapter_init['bias'] if adapter_init else None
            self.adapter = AcousticCalibrationAdapter(dim=dim, init_scale=init_scale, init_bias=init_bias)

        # 3. Temporal Importance Pooling
        if self.use_prosody:
            self.pooling = TemporalImportancePooling(ssl_dim=dim)

        # 4. Classifier
        self.classifier = SEMLP(input_dim=dim, num_classes=config.get('num_classes', 6))

    def forward(self, waveforms, f0=None, energy=None, mask=None):
        # 1. SSL 帧级特征 (B, T, 768)
        if self.use_backbone:
            ssl_feats = self.backbone(waveforms)
        else:
            ssl_feats = waveforms  # 直接输入特征（预提取模式）

        # 2. 分布校准
        if self.use_adapter:
            ssl_feats = self.adapter(ssl_feats)

        # 3. 时序池化
        if self.use_prosody and f0 is not None and energy is not None:
            pooled = self.pooling(ssl_feats, f0, energy, mask=mask)
        else:
            # Mask-aware mean pooling
            if mask is not None:
                mask_f = mask.float().unsqueeze(-1)  # (B, T, 1)
                pooled = (ssl_feats * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
            else:
                pooled = ssl_feats.mean(dim=1)

        # 4. 分类
        return self.classifier(pooled)


# ── 训练流程 ──────────────────────────────────────

def train():
    parser = argparse.ArgumentParser()
    # Mode
    parser.add_argument('--mode', default='precomputed', choices=['precomputed', 'online'])
    # Precomputed mode
    parser.add_argument('--train_feats', default='data/train_ssl_feats.npy')
    parser.add_argument('--train_labels', default='data/train_labels.npy')
    parser.add_argument('--val_feats', default='data/val_ssl_feats.npy')
    parser.add_argument('--val_labels', default='data/val_labels.npy')
    parser.add_argument('--test_feats', default='data/test_ssl_feats.npy')
    parser.add_argument('--test_labels', default='data/test_labels.npy')
    # Online mode
    parser.add_argument('--wav_dir', default=None)
    # Model components
    parser.add_argument('--ssl_model', default='emotion2vec')
    parser.add_argument('--finetune', action='store_true', default=False,
                        help='Unfreeze backbone for fine-tuning (default: frozen)')
    parser.add_argument('--use_adapter', action='store_true', default=False)
    parser.add_argument('--random_init_adapter', action='store_true', default=False,
                        help='Use random init for adapter (skip statistical prior)')
    parser.add_argument('--use_prosody', action='store_true', default=False)
    parser.add_argument('--feature_dim', type=int, default=768,
                        help='SSL feature dimension (768 for WavLM/emotion2vec, 1024 for emotion2vec_plus_large)')
    # Training
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers (0 for Windows, 4+ for Linux cloud)')
    parser.add_argument('--output_dir', default='checkpoints',
                        help='Directory for model checkpoints')
    parser.add_argument('--allow_test_as_val', action='store_true', default=False,
                        help='DANGEROUS: allow using test set as validation (reintroduces data leakage). '
                             'Only for debugging or backward compatibility with old experiments.')

    args = parser.parse_args()

    # ── 数据加载 ──
    if args.mode == 'precomputed':
        prosody_train = prosody_val = prosody_test = None
        if args.use_prosody:
            base = os.path.dirname(args.train_feats)
            for pfx in ['wavlm', 'e2v']:
                if pfx in args.train_feats:
                    prosody_train = os.path.join(base, f'train_{pfx}_prosody.npy')
                    prosody_val = os.path.join(base, f'val_{pfx}_prosody.npy')
                    prosody_test = os.path.join(base, f'test_{pfx}_prosody.npy')
                    break
            if not prosody_train or not os.path.exists(prosody_train or ''):
                print(f"WARNING: Prosody features not found, falling back to mean pooling")
                args.use_prosody = False
        train_dataset = SSLFeatureDataset(args.train_feats, args.train_labels, prosody_train)
        val_exists = os.path.exists(args.val_feats)
        if val_exists:
            val_dataset = SSLFeatureDataset(args.val_feats, args.val_labels, prosody_val)
            print(f"Val set: {len(val_dataset)} samples")
        elif args.allow_test_as_val:
            print("=" * 60)
            print("DANGER: --allow_test_as_val set — using test set as validation!")
            print("This reintroduces DATA LEAKAGE (test used for early stopping).")
            print("Results from this run MUST NOT be reported as final results.")
            print("=" * 60)
            val_dataset = SSLFeatureDataset(args.test_feats, args.test_labels, prosody_test)
        else:
            print("=" * 60)
            print("ERROR: Validation set NOT FOUND!")
            print(f"  Expected: {args.val_feats}")
            print("  Generate val features with: python scripts/extract_ssl_features.py")
            print("  Or use --allow_test_as_val to force test-as-val (NOT for final results).")
            print("=" * 60)
            sys.exit(1)
        test_dataset = SSLFeatureDataset(args.test_feats, args.test_labels, prosody_test)
    else:
        raise NotImplementedError("Online mode — implement in Phase 2")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn_ssl_features,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn_ssl_features,
        num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn_ssl_features,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── 模型 ──
    config = {
        'use_backbone': False,  # precomputed 模式下 backbone 不在训练图内
        'ssl_model': args.ssl_model,
        'frozen_backbone': not args.finetune,
        'use_adapter': args.use_adapter,
        'use_prosody': args.use_prosody,
        'feature_dim': args.feature_dim,
    }

    # Load adapter statistical prior if available
    adapter_init = None
    if args.use_adapter and not args.random_init_adapter:
        init_path = os.path.join(os.path.dirname(args.train_feats), 'adapter_init.npz')
        if os.path.exists(init_path):
            adapter_init = dict(np.load(init_path))
            print(f"Loaded adapter init from {init_path}")
            print(f"  scale mean={adapter_init['scale'].mean():.4f}, bias mean={adapter_init['bias'].mean():.4f}")
        else:
            print(f"WARNING: --use_adapter set but {init_path} not found, using random init")
    elif args.use_adapter and args.random_init_adapter:
        print(f"Using random Adapter init (--random_init_adapter)")

    model = DistributionCalibratedSER(config, adapter_init=adapter_init).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable:  {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── 实验日志 ──
    run_name = f"{args.ssl_model}_adp{int(args.use_adapter)}_pros{int(args.use_prosody)}_seed{args.seed}"
    exp_log = ExperimentLogger(name=run_name, config={
        "model": args.ssl_model,
        "mode": args.mode,
        "use_adapter": args.use_adapter,
        "use_prosody": args.use_prosody,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "patience": args.patience,
        "epochs": args.epochs,
    })

    # ── 训练循环 ──
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            inputs, labels = batch[0].to(device), batch[1].to(device)
            mask = batch[2].to(device) if len(batch) >= 3 else None
            prosody_batch = batch[3].to(device) if len(batch) >= 4 else None

            optimizer.zero_grad()
            if prosody_batch is not None:
                f0 = prosody_batch[:, :, 0:1]
                energy = prosody_batch[:, :, 1:2]
                outputs = model(inputs, f0=f0, energy=energy, mask=mask)
            else:
                outputs = model(inputs, mask=mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            train_loss += loss.item()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        # Eval on VAL (for early stopping)
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                mask = batch[2].to(device) if len(batch) >= 3 else None
                prosody_batch = batch[3].to(device) if len(batch) >= 4 else None
                if prosody_batch is not None:
                    f0 = prosody_batch[:, :, 0:1]
                    energy = prosody_batch[:, :, 1:2]
                    outputs = model(inputs, f0=f0, energy=energy, mask=mask)
                else:
                    outputs = model(inputs, mask=mask)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        train_acc = train_correct / train_total
        scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs}  "
              f"Train Acc: {train_acc:.2%}  Val Acc: {val_acc:.2%}")
        exp_log.log_epoch(epoch + 1, train_acc=train_acc, val_acc=val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_ser_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}, best val acc: {best_val_acc:.2%}")
                break

    # Final evaluation on held-out TEST set
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_ser_model.pth')))
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            mask = batch[2].to(device) if len(batch) >= 3 else None
            prosody_batch = batch[3].to(device) if len(batch) >= 4 else None
            if prosody_batch is not None:
                f0 = prosody_batch[:, :, 0:1]
                energy = prosody_batch[:, :, 1:2]
                outputs = model(inputs, f0=f0, energy=energy, mask=mask)
            else:
                outputs = model(inputs, mask=mask)
            _, preds = torch.max(outputs, 1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

    test_acc = test_correct / test_total
    print(f"Best val accuracy: {best_val_acc:.2%}")
    print(f"Test accuracy (held-out): {test_acc:.2%}")
    print(f"RESULT: val={best_val_acc:.4f} test={test_acc:.4f}")
    exp_log.finish(best_val=best_val_acc, test_acc=test_acc)


if __name__ == "__main__":
    train()
