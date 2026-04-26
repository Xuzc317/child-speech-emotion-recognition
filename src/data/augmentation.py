"""Feature-level data augmentation for small-dataset training.

Mixup is the single most effective regularizer for small tabular/feature datasets.
On a 12.5K-sample 6-class task, it prevents overfitting by creating convex combinations
of training samples during each batch.
"""

import numpy as np
import torch


def mixup_features(features, labels, alpha=0.2, num_classes=6):
    """Apply mixup augmentation to a batch of feature vectors.

    Args:
        features: (B, 1, D) or (B, D) tensor
        labels: (B,) long tensor
        alpha: Beta distribution parameter (smaller = less mixing, 0 = disabled)
        num_classes: number of classes (unused, kept for API compatibility)

    Returns:
        mixed_features, labels_a, labels_b, lam
        labels_b is None if mixup was skipped
    """
    if alpha <= 0:
        return features, labels, None, 1.0

    lam = float(np.random.beta(alpha, alpha))
    if lam < 0.5:
        lam = 1.0 - lam  # ensure lam >= 0.5, keeps majority label meaningful

    batch_size = features.size(0)
    index = torch.randperm(batch_size, device=features.device)

    mixed_features = lam * features + (1.0 - lam) * features[index]
    return mixed_features, labels, labels[index], lam


def feature_noise(features, std=0.01):
    """Add small Gaussian noise to features."""
    if std <= 0:
        return features
    noise = torch.randn_like(features) * std
    return features + noise


def collate_fn_with_augment(collate_fn, mixup_alpha=0.2, noise_std=0.01,
                             mixup_prob=0.5, training=True):
    """Wrap the base collate_fn with online augmentation.

    Only applies augmentation when training=True. Mixup is applied with
    probability mixup_prob per batch; noise is always applied if std > 0.

    Returns a function suitable for DataLoader(collate_fn=...).
    """
    def _collate(batch):
        inputs, labels = collate_fn(batch)

        if training:
            # Feature noise
            if noise_std > 0:
                inputs = feature_noise(inputs, noise_std)

            # Mixup (probabilistic per batch)
            if mixup_alpha > 0 and np.random.random() < mixup_prob:
                inputs, labels_a, labels_b, lam = mixup_features(
                    inputs, labels, mixup_alpha
                )
                return inputs, labels, (labels_a, labels_b, lam)

        return inputs, labels, None

    return _collate
