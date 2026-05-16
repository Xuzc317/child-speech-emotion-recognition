"""Learnable Weighted Sum over WavLM hidden layers.

Enables the model to dynamically learn which transformer layers are most
informative for children's vs. adult speech emotion recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WavLMLayerFusion(nn.Module):
    """Fuse 12 WavLM hidden layers via learnable weighted sum.

    Initialized with equal weights (1/12 each). Softmax ensures weights
    sum to 1. Only the 12 weight parameters are trainable.

    Input:  tuple of 12 tensors, each (B, T, 768)
    Output: (B, T, 768) fused representation
    """

    def __init__(self, num_layers: int = 12):
        super().__init__()
        init_weight = torch.ones(num_layers) / num_layers
        self.layer_weights = nn.Parameter(init_weight)

    def forward(self, hidden_states):
        """Weighted sum over layers.

        Args:
            hidden_states: tuple of Tensors from WavLM output_hidden_states.
                           Each shape (B, T, 768). First is embedding layer,
                           remaining 12 are transformer layers.
                           We use the last 12 (layers 1-12), ignoring layer 0
                           (input embeddings).

        Returns:
            fused: (B, T, 768) float tensor
        """
        # hidden_states[0] is input embeddings, [1..12] are transformer layers
        layers = torch.stack(hidden_states[1:], dim=0)  # (12, B, T, 768)
        weights = F.softmax(self.layer_weights, dim=0)  # (12,)
        weights = weights.view(12, 1, 1, 1)             # (12, 1, 1, 1)
        fused = (layers * weights).sum(dim=0)            # (B, T, 768)
        return fused

    def get_layer_weights(self):
        """Return current layer weights (softmaxed) for analysis."""
        with torch.no_grad():
            return F.softmax(self.layer_weights, dim=0).detach().cpu().numpy()
