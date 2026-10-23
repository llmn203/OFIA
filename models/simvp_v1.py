from __future__ import annotations

import torch
from torch import nn

from .spatiotemporal_utils import ConvNormAct, FrameEncoder, ResidualBlock, UNetDecoder, infer_input_layout, split_spatiotemporal_input


class TemporalMixingBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvNormAct(channels, channels * 2, kernel_size=1),
            nn.Conv2d(channels * 2, channels * 2, 5, padding=2, groups=channels * 2),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class SimVPV1(nn.Module):
    """Pure CNN spatiotemporal baseline inspired by SimVP-style mixing."""

    def __init__(self, cin: int, cout: int, width: int):
        super().__init__()
        self.layout = infer_input_layout(cin, cout)
        hidden = width * 4
        self.frame_encoder = FrameEncoder(cout, width)
        self.context_encoder = FrameEncoder(self.layout.forcing_channels + self.layout.static_channels, width)
        mixed_channels = hidden * self.layout.history_days
        self.temporal_mixer = nn.Sequential(
            ConvNormAct(mixed_channels, hidden),
            TemporalMixingBlock(hidden),
            TemporalMixingBlock(hidden),
            TemporalMixingBlock(hidden),
            ResidualBlock(hidden),
        )
        self.fuse = nn.Sequential(ConvNormAct(hidden * 2, hidden), ResidualBlock(hidden))
        self.decoder = UNetDecoder(hidden, width, cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        history, forcing, static = split_spatiotemporal_input(x, self.layout)
        bottlenecks = []
        for step in range(self.layout.history_days):
            _, _, frame = self.frame_encoder(history[:, step])
            bottlenecks.append(frame)
        last_skip1, last_skip2, _ = self.frame_encoder(history[:, -1])
        _, skip2, force_bottleneck = self.context_encoder(torch.cat([forcing, static], dim=1))
        mixed = self.temporal_mixer(torch.cat(bottlenecks, dim=1))
        bottleneck = self.fuse(torch.cat([mixed, force_bottleneck], dim=1))
        return self.decoder(bottleneck, last_skip2 + skip2, last_skip1)
