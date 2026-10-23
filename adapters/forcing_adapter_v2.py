"""Lead-conditioned forcing Adapter V2 with calendar passthrough."""

from __future__ import annotations

import torch
from torch import nn

from .lead_conditioned_gate import MonotoneLeadGate


class ForcingAdapterV2(nn.Module):
    forcing_channels = 8
    adapted_channel_names = ("u10", "v10", "t2m", "d2m", "msl", "precipitation")
    passthrough_channel_names = ("sin_doy", "cos_doy")
    channel_names = adapted_channel_names + passthrough_channel_names
    adapted_channels = len(adapted_channel_names)
    version = "v2_lead_conditioned_monotone"

    def __init__(self, channels: int = 8, rank: int = 4) -> None:
        super().__init__()
        if channels != self.forcing_channels:
            raise ValueError(f"ForcingAdapterV2 requires 8 channels, got {channels}.")
        self.residual = nn.Sequential(
            nn.Conv2d(self.adapted_channels, rank, 1),
            nn.GELU(),
            nn.Conv2d(rank, self.adapted_channels, 1),
        )
        self.alpha = nn.Parameter(torch.zeros(1, self.adapted_channels, 1, 1))
        self.lead_gate = MonotoneLeadGate(self.adapted_channels)

    def reset_identity(self) -> None:
        nn.init.zeros_(self.alpha)

    def forward(self, forcing: torch.Tensor, lead_day: int | torch.Tensor) -> torch.Tensor:
        if forcing.ndim != 4 or forcing.shape[1] != self.forcing_channels:
            raise ValueError(f"ForcingAdapterV2 expects [B,8,H,W], got {tuple(forcing.shape)}.")
        gates = self.lead_gate(lead_day)
        if gates.shape[0] not in {1, forcing.shape[0]}:
            raise ValueError("lead_day tensor batch does not match forcing batch.")
        adapted_input = forcing[:, : self.adapted_channels]
        adapted = adapted_input + self.alpha * gates * self.residual(torch.nan_to_num(adapted_input))
        return torch.cat([adapted, forcing[:, self.adapted_channels :]], dim=1)
