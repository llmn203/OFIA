from __future__ import annotations

import torch
from torch import nn

from .spatiotemporal_utils import ConvNormAct, FrameEncoder, ResidualBlock, UNetDecoder, infer_input_layout, split_spatiotemporal_input


class ConvGRUCell(nn.Module):
    def __init__(self, cin: int, hidden: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden = hidden
        self.zr = nn.Conv2d(cin + hidden, hidden * 2, kernel_size, padding=padding)
        self.candidate = nn.Conv2d(cin + hidden, hidden, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        z, r = torch.chunk(self.zr(torch.cat([x, h], dim=1)), 2, dim=1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)
        h_tilde = torch.tanh(self.candidate(torch.cat([x, r * h], dim=1)))
        return (1.0 - z) * h + z * h_tilde


class ConvGRUUNetV1(nn.Module):
    """Two-layer ConvGRU bottleneck with U-Net style decoding."""

    def __init__(self, cin: int, cout: int, width: int):
        super().__init__()
        self.layout = infer_input_layout(cin, cout)
        hidden = width * 4
        self.frame_encoder = FrameEncoder(cout, width)
        self.gru1 = ConvGRUCell(hidden, hidden)
        self.gru2 = ConvGRUCell(hidden, hidden)
        self.context_encoder = FrameEncoder(self.layout.forcing_channels + self.layout.static_channels, width)
        self.fuse = nn.Sequential(ConvNormAct(hidden * 2, hidden), ResidualBlock(hidden))
        self.decoder = UNetDecoder(hidden, width, cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        history, forcing, static = split_spatiotemporal_input(x, self.layout)
        _, skip2, force_bottleneck = self.context_encoder(torch.cat([forcing, static], dim=1))
        last_skip1, last_skip2, _ = self.frame_encoder(history[:, -1])
        h1 = h2 = None
        for step in range(self.layout.history_days):
            _, _, frame = self.frame_encoder(history[:, step])
            if h1 is None:
                h1 = frame.new_zeros(frame.shape)
                h2 = frame.new_zeros(frame.shape)
            h1 = self.gru1(frame, h1)
            h2 = self.gru2(h1, h2)
        bottleneck = self.fuse(torch.cat([h2, force_bottleneck], dim=1))
        return self.decoder(bottleneck, last_skip2 + skip2, last_skip1)
