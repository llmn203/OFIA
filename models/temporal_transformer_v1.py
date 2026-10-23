from __future__ import annotations

import torch
from torch import nn

from .spatiotemporal_utils import ConvNormAct, FrameEncoder, ResidualBlock, UNetDecoder, infer_input_layout, split_spatiotemporal_input


class TemporalTransformerV1(nn.Module):
    """Temporal self-attention over history at bottleneck spatial resolution."""

    def __init__(self, cin: int, cout: int, width: int):
        super().__init__()
        self.layout = infer_input_layout(cin, cout)
        hidden = width * 4
        heads = 8 if hidden % 8 == 0 else 4
        self.frame_encoder = FrameEncoder(cout, width)
        self.context_encoder = FrameEncoder(self.layout.forcing_channels + self.layout.static_channels, width)
        self.position = nn.Parameter(torch.zeros(1, self.layout.history_days, hidden))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=2)
        self.fuse = nn.Sequential(ConvNormAct(hidden * 2, hidden), ResidualBlock(hidden))
        self.decoder = UNetDecoder(hidden, width, cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        history, forcing, static = split_spatiotemporal_input(x, self.layout)
        encoded = []
        for step in range(self.layout.history_days):
            _, _, frame = self.frame_encoder(history[:, step])
            encoded.append(frame)
        sequence = torch.stack(encoded, dim=1)
        b, l, c, h, w = sequence.shape
        tokens = sequence.permute(0, 3, 4, 1, 2).contiguous().view(b * h * w, l, c)
        tokens = self.temporal(tokens + self.position[:, :l])
        attended = tokens[:, -1].view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        last_skip1, last_skip2, _ = self.frame_encoder(history[:, -1])
        _, skip2, force_bottleneck = self.context_encoder(torch.cat([forcing, static], dim=1))
        bottleneck = self.fuse(torch.cat([attended, force_bottleneck], dim=1))
        return self.decoder(bottleneck, last_skip2 + skip2, last_skip1)
