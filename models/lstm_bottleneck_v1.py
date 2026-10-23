from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .spatiotemporal_utils import ConvNormAct, ResidualBlock, infer_input_layout, split_spatiotemporal_input


class LSTMBottleneckV1(nn.Module):
    """CNN bottleneck + 2-layer LSTM baseline for 7-day ocean history."""

    def __init__(self, cin: int, cout: int, width: int):
        super().__init__()
        self.layout = infer_input_layout(cin, cout)
        latent = width * 4
        self.day_encoder = nn.Sequential(
            ConvNormAct(cout, width),
            ResidualBlock(width),
            ConvNormAct(width, width * 2, stride=2),
            ResidualBlock(width * 2),
            ConvNormAct(width * 2, latent, stride=2),
            ResidualBlock(latent),
            nn.AdaptiveAvgPool2d(1),
        )
        self.lstm = nn.LSTM(latent, latent, num_layers=2, batch_first=True, dropout=0.1)
        self.context = nn.Sequential(
            ConvNormAct(self.layout.forcing_channels + self.layout.static_channels, width),
            ResidualBlock(width),
            ConvNormAct(width, width * 2),
            ResidualBlock(width * 2),
        )
        self.last_state = nn.Sequential(ConvNormAct(cout, width * 2), ResidualBlock(width * 2))
        self.decoder = nn.Sequential(
            ConvNormAct(latent + width * 2 + width * 2, width * 4),
            ResidualBlock(width * 4),
            ConvNormAct(width * 4, width * 2),
            ResidualBlock(width * 2),
            ConvNormAct(width * 2, width),
            ResidualBlock(width),
            nn.Conv2d(width, cout, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        history, forcing, static = split_spatiotemporal_input(x, self.layout)
        b, l, c, h, w = history.shape
        encoded = self.day_encoder(history.view(b * l, c, h, w)).view(b, l, -1)
        _, (hidden, _) = self.lstm(encoded)
        temporal = hidden[-1].view(b, -1, 1, 1).expand(-1, -1, h, w)
        context = self.context(torch.cat([forcing, static], dim=1))
        last = self.last_state(history[:, -1])
        return self.decoder(torch.cat([temporal, context, last], dim=1))
