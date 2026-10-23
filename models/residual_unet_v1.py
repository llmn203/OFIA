from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    return 4 if channels % 4 == 0 else 1


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.a = nn.Conv2d(channels, channels, 3, padding=1)
        self.b = nn.Conv2d(channels, channels, 3, padding=1)
        self.n1 = nn.GroupNorm(_groups(channels), channels)
        self.n2 = nn.GroupNorm(_groups(channels), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + F.gelu(self.n2(self.b(F.gelu(self.n1(self.a(x))))))


class ResidualUNet(nn.Module):
    """Original operational residual U-Net kept state_dict-compatible."""

    def __init__(self, cin: int, cout: int, width: int):
        super().__init__()
        self.inp = nn.Conv2d(cin, width, 3, padding=1)
        self.r1 = ResBlock(width)
        self.down = nn.Conv2d(width, width * 2, 3, 2, 1)
        self.r2 = ResBlock(width * 2)
        self.up = nn.ConvTranspose2d(width * 2, width, 2, 2)
        self.r3 = ResBlock(width)
        self.out = nn.Conv2d(width, cout, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.r1(F.gelu(self.inp(x)))
        b = self.r2(F.gelu(self.down(a)))
        u = self.up(b)
        u = u[..., : a.shape[-2], : a.shape[-1]] + a
        return self.out(self.r3(F.gelu(u)))

