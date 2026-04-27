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

from src.data.dataset_ssl import SSLFeatureDataset, SSLOnlineDataset, collate_fn_ssl_features
from src.models.ssl_backbone import SSLBackbone
from src.models.adapter import AcousticCalibrationAdapter, DomainAdversarialAdapter
from src.models.pooling import TemporalImportancePooling
from src.models.drse_cnn import DrseCNN
from src.utils.tracker import ExperimentTracker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 完整框架 ──────────────────────────────────────

class DistributionCalibratedSER(nn.Module):
    """分布校准的儿童语音情绪识别完整框架。

    模块组成：
      1. SSL Backbone (frozen) —— 帧级特征提取
      2. Distribution Calibration Adapter —— 分布偏移校准
      3. Temporal Importance Pooling —— 韵律引导的时序融合
      4. DrseCNN / Simple Classifier —— 情感分类

    前向：
      waveforms / ssl_feats → Backbone → Adapter → Pooling → Classifier
    """

    def __init__(self, config):
        super().__init__()
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
            self.adapter = AcousticCalibrationAdapter(dim=768)

        # 3. Temporal Importance Pooling
        if self.use_prosody:
            self.pooling = TemporalImportancePooling(ssl_dim=768)

        # 4. Classifier
        self.classifier = DrseCNN(input_dim=768, num_classes=config.get('num_classes', 6))

    def forward(self, waveforms, f0=None, energy=None):
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
            pooled = self.pooling(ssl_feats, f0, energy)
        else:
            pooled = ssl_feats.mean(dim=1)  # fallback: mean pooling

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
    parser.add_argument('--test_feats', default='data/test_ssl_feats.npy')
    parser.add_argument('--test_labels', default='data/test_labels.npy')
    # Online mode
    parser.add_argument('--wav_dir', default=None)
    # Model components
    parser.add_argument('--ssl_model', default='emotion2vec')
    parser.add_argument('--finetune', action='store_true', default=False,
                        help='Unfreeze backbone for fine-tuning (default: frozen)')
    parser.add_argument('--use_adapter', action='store_true', default=False)
    parser.add_argument('--use_prosody', action='store_true', default=False)
    # Training
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # ── 数据加载 ──
    if args.mode == 'precomputed':
        train_dataset = SSLFeatureDataset(args.train_feats, args.train_labels)
        test_dataset = SSLFeatureDataset(args.test_feats, args.test_labels)
    else:
        # TODO Phase 2: 在线模式
        raise NotImplementedError("Online mode — implement in Phase 2")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn_ssl_features,
        num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn_ssl_features,
        num_workers=4, pin_memory=True,
    )

    # ── 模型 ──
    config = {
        'use_backbone': False,  # precomputed 模式下 backbone 不在训练图内
        'ssl_model': args.ssl_model,
        'frozen_backbone': not args.finetune,
        'use_adapter': args.use_adapter,
        'use_prosody': args.use_prosody,
    }
    model = DistributionCalibratedSER(config).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable:  {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── 训练循环 ──
    best_acc = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)  # precomputed: 特征直接进 adapter+pooling+classifier
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            train_loss += loss.item()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        # Eval
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels, masks in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        train_acc = train_correct / train_total
        scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs}  "
              f"Train Acc: {train_acc:.2%}  Val Acc: {val_acc:.2%}")

        # Early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), f"checkpoints/best_ser_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}, best val acc: {best_acc:.2%}")
                break

    print(f"Best validation accuracy: {best_acc:.2%}")


if __name__ == "__main__":
    train()
