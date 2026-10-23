from __future__ import annotations

import torch
from torch import nn

from .spatiotemporal_utils import ConvNormAct, FrameEncoder, ResidualBlock, UNetDecoder, groups_for, infer_input_layout, split_spatiotemporal_input


class ConvLSTMCell(nn.Module):
    def __init__(self, cin: int, hidden: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden = hidden
        self.gates = nn.Conv2d(cin + hidden, hidden * 4, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor, state):
        h, c = state
        gates = self.gates(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTMUNetV1(nn.Module):
    """Two-layer ConvLSTM bottleneck with U-Net style decoding."""

    def __init__(self, cin: int, cout: int, width: int):
        super().__init__()
        self.layout = infer_input_layout(cin, cout)
        hidden = width * 4
        self.frame_encoder = FrameEncoder(cout, width)
        self.cell1 = ConvLSTMCell(hidden, hidden)
        self.cell2 = ConvLSTMCell(hidden, hidden)
        self.context_encoder = FrameEncoder(self.layout.forcing_channels + self.layout.static_channels, width)
        self.fuse = nn.Sequential(
            ConvNormAct(hidden * 2, hidden),
            ResidualBlock(hidden),
            nn.GroupNorm(groups_for(hidden), hidden),
        )
        self.decoder = UNetDecoder(hidden, width, cout)

    def _zero_state(self, x: torch.Tensor):
        b, _, h, w = x.shape
        zeros = x.new_zeros((b, self.cell1.hidden, h, w))
        return (zeros, zeros.clone()), (zeros.clone(), zeros.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        history, forcing, static = split_spatiotemporal_input(x, self.layout)
        _, skip2, force_bottleneck = self.context_encoder(torch.cat([forcing, static], dim=1))
        last_skip1, last_skip2, _ = self.frame_encoder(history[:, -1])
        state1 = state2 = None
        for step in range(self.layout.history_days):
            _, _, frame = self.frame_encoder(history[:, step])
            if state1 is None:
                state1, state2 = self._zero_state(frame)
            h1, c1 = self.cell1(frame, state1)
            h2, c2 = self.cell2(h1, state2)
            state1, state2 = (h1, c1), (h2, c2)
        bottleneck = self.fuse(torch.cat([state2[0], force_bottleneck], dim=1))
        return self.decoder(bottleneck, last_skip2 + skip2, last_skip1)
