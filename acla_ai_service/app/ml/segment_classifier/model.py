"""Length-preserving temporal model for behavior segment detection."""

from __future__ import annotations

import torch.nn as nn


class TemporalResidualBlock(nn.Module):
    """Non-causal dilated convolutions with a residual connection."""

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


class TemporalDetectionModel(nn.Module):
    """Emit one logit per label for every input timestep."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
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
        self.output_projection = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)

    def forward(self, inputs):
        # inputs: (batch, sequence, features)
        hidden = self.input_projection(inputs.transpose(1, 2))
        hidden = self.blocks(hidden)
        return self.output_projection(hidden).transpose(1, 2)


__all__ = ["TemporalDetectionModel", "TemporalResidualBlock"]
