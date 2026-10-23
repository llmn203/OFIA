from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class InputLayout:
    history_days: int
    forcing_channels: int
    state_channels: int
    static_channels: int


def groups_for(channels: int) -> int:
    for groups in (16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def infer_input_layout(cin: int, state_channels: int, static_channels: int = 4) -> InputLayout:
    dynamic = int(cin) - int(static_channels)
    if dynamic <= int(state_channels):
        raise ValueError(f'cin={cin} is too small for state_channels={state_channels} and static_channels={static_channels}.')
    history_days = dynamic // int(state_channels)
    forcing_channels = dynamic - history_days * int(state_channels)
    if history_days < 1 or forcing_channels < 1:
        raise ValueError(f'Cannot infer input layout from cin={cin}, state_channels={state_channels}, static_channels={static_channels}.')
    return InputLayout(history_days, forcing_channels, int(state_channels), int(static_channels))


def split_spatiotemporal_input(x: torch.Tensor, layout: InputLayout):
    b, c, h, w = x.shape
    expected = layout.history_days * layout.state_channels + layout.forcing_channels + layout.static_channels
    if c != expected:
        raise ValueError(f'Expected {expected} input channels from layout={layout}, got {c}.')
    history_end = layout.history_days * layout.state_channels
    forcing_end = history_end + layout.forcing_channels
    history = x[:, :history_end].contiguous().view(b, layout.history_days, layout.state_channels, h, w)
    forcing = x[:, history_end:forcing_end]
    static = x[:, forcing_end:]
    return history, forcing, static


class ConvNormAct(nn.Module):
    def __init__(self, cin: int, cout: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(cin, cout, kernel_size, stride=stride, padding=padding)
        self.norm = nn.GroupNorm(groups_for(cout), cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvNormAct(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups_for(channels), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.norm2(self.conv2(self.conv1(x))))


class UpFuseBlock(nn.Module):
    def __init__(self, cin: int, skip_channels: int, cout: int):
        super().__init__()
        self.reduce = ConvNormAct(cin + skip_channels, cout, kernel_size=1)
        self.block = ResidualBlock(cout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.block(self.reduce(torch.cat([x, skip], dim=1)))


class FrameEncoder(nn.Module):
    def __init__(self, cin: int, width: int):
        super().__init__()
        self.stem = ConvNormAct(cin, width)
        self.block1 = ResidualBlock(width)
        self.down1 = ConvNormAct(width, width * 2, stride=2)
        self.block2 = ResidualBlock(width * 2)
        self.down2 = ConvNormAct(width * 2, width * 4, stride=2)
        self.block3 = ResidualBlock(width * 4)

    def forward(self, x: torch.Tensor):
        x1 = self.block1(self.stem(x))
        x2 = self.block2(self.down1(x1))
        x3 = self.block3(self.down2(x2))
        return x1, x2, x3


class UNetDecoder(nn.Module):
    def __init__(self, bottleneck_channels: int, width: int, cout: int):
        super().__init__()
        self.up2 = UpFuseBlock(bottleneck_channels, width * 2, width * 2)
        self.up1 = UpFuseBlock(width * 2, width, width)
        self.head = nn.Conv2d(width, cout, 1)

    def forward(self, bottleneck: torch.Tensor, skip2: torch.Tensor, skip1: torch.Tensor) -> torch.Tensor:
        x = self.up2(bottleneck, skip2)
        x = self.up1(x, skip1)
        return self.head(x)
