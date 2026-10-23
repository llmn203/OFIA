from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = _groups(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.gelu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.gelu(x + y)


class DownBlock(nn.Module):
    def __init__(self, cin: int, cout: int, blocks_per_stage: int = 1):
        super().__init__()
        self.proj = nn.Conv2d(cin, cout, 3, stride=2, padding=1)
        self.norm = nn.GroupNorm(_groups(cout), cout)
        self.blocks = nn.Sequential(*[ResidualConvBlock(cout) for _ in range(blocks_per_stage)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(F.gelu(self.norm(self.proj(x))))


class UpBlock(nn.Module):
    def __init__(self, cin: int, skip_channels: int, cout: int, blocks_per_stage: int = 1):
        super().__init__()
        self.reduce = nn.Conv2d(cin + skip_channels, cout, 1)
        self.norm = nn.GroupNorm(_groups(cout), cout)
        self.blocks = nn.Sequential(*[ResidualConvBlock(cout) for _ in range(blocks_per_stage)])

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.blocks(F.gelu(self.norm(self.reduce(x))))


class MultiScaleResUNetV2(nn.Module):
    """Small-grid multiscale 2D residual U-Net variant."""

    def __init__(self, cin: int, cout: int, width: int, blocks_per_stage: int = 1, mid_blocks: int = 1):
        super().__init__()
        blocks_per_stage = max(int(blocks_per_stage), 1)
        mid_blocks = max(int(mid_blocks), 1)
        w1, w2, w3 = width, width * 2, width * 3
        self.stem = nn.Conv2d(cin, w1, 3, padding=1)
        self.stem_norm = nn.GroupNorm(_groups(w1), w1)
        self.enc1 = nn.Sequential(*[ResidualConvBlock(w1) for _ in range(blocks_per_stage)])
        self.down1 = DownBlock(w1, w2, blocks_per_stage)
        self.down2 = DownBlock(w2, w3, blocks_per_stage)
        self.mid = nn.Sequential(*[ResidualConvBlock(w3) for _ in range(mid_blocks)])
        self.up2 = UpBlock(w3, w2, w2, blocks_per_stage)
        self.up1 = UpBlock(w2, w1, w1, blocks_per_stage)
        self.head = nn.Conv2d(w1, cout, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(F.gelu(self.stem_norm(self.stem(x))))
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.mid(x3)
        x5 = self.up2(x4, x2)
        x6 = self.up1(x5, x1)
        return self.head(x6)
