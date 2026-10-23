"""Identity-initialized adapter for the frozen eight-channel forcing interface."""

from __future__ import annotations

import torch
from torch import nn


class ForcingAdapter(nn.Module):
    forcing_channels = 8
    adapted_channel_names = ("u10", "v10", "t2m", "d2m", "msl", "precipitation")
    passthrough_channel_names = ("sin_doy", "cos_doy")
    channel_names = adapted_channel_names + passthrough_channel_names
    adapted_channels = len(adapted_channel_names)

    def __init__(self, channels: int = 8, rank: int = 4):
        super().__init__()
        if channels != self.forcing_channels:
            raise ValueError(
                f"Stage5 v1 forcing interface is frozen at {self.forcing_channels} channels, got {channels}."
            )
        if rank < 1:
            raise ValueError("rank must be positive.")
        self.residual = nn.Sequential(
            nn.Conv2d(self.adapted_channels, rank, 1),
            nn.GELU(),
            nn.Conv2d(rank, self.adapted_channels, 1),
        )
        self.alpha = nn.Parameter(torch.zeros(1, self.adapted_channels, 1, 1))

    def reset_identity(self) -> None:
        nn.init.zeros_(self.alpha)

    def forward(
        self,
        forcing: torch.Tensor,
        lead_day: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if forcing.ndim != 4 or forcing.shape[1] != self.forcing_channels:
            raise ValueError(
                f"ForcingAdapter expects [B,{self.forcing_channels},H,W], got {tuple(forcing.shape)}."
            )
        if isinstance(lead_day, int) and lead_day not in range(1, 11):
            raise ValueError(f"lead_day must be 1..10 when provided, got {lead_day}.")
        # lead_day is intentionally reserved for a later, gated enhancement.
        # Stage5 Infrastructure Closure does not implement a lead-conditioned gate.
        adapted_input = forcing[:, : self.adapted_channels]
        finite_forcing = torch.nan_to_num(adapted_input)
        adapted = adapted_input + self.alpha * self.residual(finite_forcing)
        # Calendar channels are a fixed deterministic encoding and are never
        # part of the learned product-domain correction.
        return torch.cat([adapted, forcing[:, self.adapted_channels :]], dim=1)
