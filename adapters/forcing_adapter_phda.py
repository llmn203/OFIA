"""Horizon-conditioned physical-forcing Adapter for PHDA."""

from __future__ import annotations

import torch
from torch import nn

from .provenance_horizon_gate import HorizonForcingGate


class ForcingAdapterPHDA(nn.Module):
    forcing_channels = 8
    adapted_channel_names = ("u10", "v10", "t2m", "d2m", "msl", "precipitation")
    passthrough_channel_names = ("sin_doy", "cos_doy")
    adapted_channels = 6
    version = "phda_v1_provenance_horizon"

    def __init__(self, channels: int = 8, rank: int = 4) -> None:
        super().__init__()
        if channels != self.forcing_channels:
            raise ValueError(f"ForcingAdapterPHDA requires 8 channels, got {channels}.")
        self.residual = nn.Sequential(
            nn.Conv2d(self.adapted_channels, rank, 1),
            nn.GELU(),
            nn.Conv2d(rank, self.adapted_channels, 1),
        )
        self.alpha = nn.Parameter(torch.zeros(1, self.adapted_channels, 1, 1))
        self.gate = HorizonForcingGate(self.adapted_channels)

    def reset_identity(self) -> None:
        nn.init.zeros_(self.alpha)

    def effective_alpha(self) -> torch.Tensor:
        return 2.0 * torch.tanh(self.alpha / 2.0)

    def forward(self, forcing: torch.Tensor, lead_day: int | torch.Tensor) -> torch.Tensor:
        if forcing.ndim != 4 or forcing.shape[1] != self.forcing_channels:
            raise ValueError(f"ForcingAdapterPHDA expects [B,8,H,W], got {tuple(forcing.shape)}.")
        gates = self.gate(lead_day)
        if gates.shape[0] not in {1, forcing.shape[0]}:
            raise ValueError("lead_day tensor batch does not match forcing batch.")
        physical = forcing[:, : self.adapted_channels]
        adapted = physical + self.effective_alpha() * gates * self.residual(torch.nan_to_num(physical))
        return torch.cat([adapted, forcing[:, self.adapted_channels :]], dim=1)
