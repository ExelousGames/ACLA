"""Pretrained ResNet-18 spatial features, a small ConvLSTM, and local XYZ regression."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from .position_model import PositionModel


class ConvLSTMCell(nn.Module):
    """One spatial recurrent step; hidden and cell state belong to the caller's window."""

    def __init__(self, input_channels, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(input_channels + hidden_channels, 4 * hidden_channels, 3, padding=1)

    def forward(self, features, hidden, cell):
        incoming, forget, candidate, output = self.gates(torch.cat((features, hidden), dim=1)).chunk(4, dim=1)
        cell = forget.sigmoid() * cell + incoming.sigmoid() * candidate.tanh()
        hidden = output.sigmoid() * cell.tanh()
        return hidden, cell


class PositionModelV2(PositionModel):
    """Keep the original position inputs/output, retaining space through temporal fusion.

    Training keeps the ImageNet ResNet-18 weights and BatchNorm state frozen.
    Checkpoint restoration uses
    pretrained=False because every encoder parameter is already in the checkpoint.
    """

    model_name = "position_model_v2"
    temporal_type = "convlstm"
    checkpoint_format = "acla-position-resnet18-convlstm-v1"

    def __init__(self, imgsz=768, *, categories, sequence_length=1, pretrained=True):
        self.pretrained = pretrained
        super().__init__(imgsz, categories=categories, sequence_length=sequence_length)
        self.train()

    def _build_network(self):
        weights = ResNet18_Weights.IMAGENET1K_V1 if self.pretrained else None
        backbone = resnet18(weights=weights)
        # Remove global pooling and classification: [N, 512, H/32, W/32] remains spatial.
        self.image_encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.image_encoder.requires_grad_(False)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))
        self.position_encoder = nn.Sequential(nn.Conv2d(512, 64, 1), nn.ReLU())
        auxiliary_size = 3 + self.context_size + 6
        self.temporal_encoder = ConvLSTMCell(64 + auxiliary_size, 32)
        # Pool only after recurrence; keep coarse spatial layout for offset regression.
        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.head = nn.Sequential(nn.Linear(32 * 4 * 4 + auxiliary_size, 128), nn.ReLU(), nn.Linear(128, 3))

    def train(self, mode=True):
        super().train(mode)
        # Keep pretrained BatchNorm state fixed even when the remaining layers train.
        self.image_encoder.eval()
        return self

    def visual_features(self, images):
        # Input is still letterboxed RGB in [0, 1]; normalization is inside the exported graph.
        return self.image_encoder((images - self.image_mean) / self.image_std)

    def forward(self, images, references, context, orientation):
        batch, images, positions, context, orientation = self._prepare_inputs(images, references, context, orientation)
        visual = self.position_encoder(self.visual_features(images))
        auxiliary = torch.cat((positions, context, orientation), dim=1).reshape(batch, self.sequence_length, -1)
        visual = visual.reshape(batch, self.sequence_length, *visual.shape[1:])
        hidden = visual.new_zeros(batch, self.temporal_encoder.hidden_channels, *visual.shape[-2:])
        cell = torch.zeros_like(hidden)
        # Explicit unrolling exports ordinary Conv/Sigmoid/Tanh ops, with no external state.
        for step in range(self.sequence_length):
            conditioning = auxiliary[:, step, :, None, None].expand(-1, -1, *visual.shape[-2:])
            hidden, cell = self.temporal_encoder(torch.cat((visual[:, step], conditioning), dim=1), hidden, cell)
        features = torch.cat((self.spatial_pool(hidden).flatten(1), auxiliary[:, -1]), dim=1)
        return self.head(features).reshape(-1, 1, 3)
