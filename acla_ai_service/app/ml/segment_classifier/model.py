"""Pure model code for the segment classifier.

Two PyTorch ``nn.Module`` classes:
  - ``FocalLoss``: binary focal loss with optional positive-class weighting,
    used during training to handle class imbalance across segment labels.
  - ``CNN1DModel``: 1D-CNN feature extractor + linear head that scores each
    timestep against the full label set.

This module imports NOTHING from the rest of the app — it's a pure leaf,
testable in isolation. The training and inference orchestrators in
``app.ml.segment_classifier.service`` import these classes back.

Extracted from app/ml/segment_classifier/service.py in
refactor/hexagonal-v4 (Page 5 of acla-ai-service-architecture.drawio).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='none', pos_weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none', pos_weight=self.pos_weight
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CNN1DModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(CNN1DModel, self).__init__()

        layers = []
        in_channels = input_dim

        # Using padding='same' to keep sequence length equal
        for _ in range(num_layers):
            layers.append(nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding='same'))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            in_channels = hidden_dim

        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        # x: (batch, seq_len, input_dim)
        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)

        out = self.features(x)

        # out: (batch, channels, seq_len) -> (batch, seq_len, channels)
        out = out.transpose(1, 2)
        out = self.fc(out)

        return out, None


__all__ = ["FocalLoss", "CNN1DModel"]
