"""Independent length-preserving TCN used by the session cropper."""

from __future__ import annotations

import torch.nn as nn


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.activation = nn.ReLU()

    def forward(self, inputs):
        return self.activation(inputs + self.layers(inputs))


class BoundaryTCN(nn.Module):
    """Emit aligned start, end, and inside logits for every input row."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(*(
            TemporalResidualBlock(hidden_dim, dilation, dropout)
            for dilation in dilations
        ))
        self.start_head = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.end_head = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.inside_head = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def forward(self, inputs):
        hidden = self.input_projection(inputs.transpose(1, 2))
        hidden = self.blocks(hidden)
        return (
            self.start_head(hidden).squeeze(1),
            self.end_head(hidden).squeeze(1),
            self.inside_head(hidden).squeeze(1),
        )


__all__ = ["BoundaryTCN", "TemporalResidualBlock"]
