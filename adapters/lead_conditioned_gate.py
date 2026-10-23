"""Small monotone lead-conditioned gates for Stage5 Adapter V2."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_GATE_SCHEDULE = (0.0, 0.12, 0.30, 0.50, 0.68, 0.81, 0.90, 0.95, 0.98, 1.0)


class MonotoneLeadGate(nn.Module):
    """Per-channel monotone gates with exact Day-1 bypass and Day-10 scale one.

    Positive increments are learned and cumulatively normalized.  This keeps
    every gate in [0, 1], prevents an oscillatory ten-parameter lookup table,
    and guarantees g(1)=0 and g(10)=1 throughout training.
    """

    def __init__(
        self,
        channels: int,
        *,
        max_lead: int = 10,
        initial_schedule: tuple[float, ...] = DEFAULT_GATE_SCHEDULE,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive.")
        if max_lead < 2:
            raise ValueError("max_lead must be at least two.")
        if len(initial_schedule) != max_lead:
            raise ValueError("initial_schedule length must equal max_lead.")
        schedule = torch.as_tensor(initial_schedule, dtype=torch.float32)
        if schedule[0].item() != 0.0 or schedule[-1].item() != 1.0:
            raise ValueError("initial_schedule must start at zero and end at one.")
        increments = torch.diff(schedule)
        if torch.any(increments <= 0):
            raise ValueError("initial_schedule must be strictly increasing.")
        # Inverse softplus makes the normalized initial gate reproduce the
        # requested schedule up to ordinary float precision.
        raw = torch.log(torch.expm1(increments)).repeat(channels, 1)
        self.raw_increments = nn.Parameter(raw)
        self.channels = int(channels)
        self.max_lead = int(max_lead)

    def schedule(self) -> torch.Tensor:
        increments = F.softplus(self.raw_increments)
        cumulative = torch.cat(
            [
                torch.zeros(
                    (self.channels, 1),
                    dtype=increments.dtype,
                    device=increments.device,
                ),
                torch.cumsum(increments, dim=1),
            ],
            dim=1,
        )
        return cumulative / cumulative[:, -1:].clamp_min(torch.finfo(cumulative.dtype).eps)

    def forward(self, lead_day: int | torch.Tensor) -> torch.Tensor:
        schedule = self.schedule()
        if isinstance(lead_day, int):
            if not 1 <= lead_day <= self.max_lead:
                raise ValueError(f"lead_day must be 1..{self.max_lead}, got {lead_day}.")
            return schedule[:, lead_day - 1].reshape(1, self.channels, 1, 1)
        if not torch.is_tensor(lead_day):
            raise TypeError("lead_day must be an int or tensor.")
        indexes = lead_day.to(device=schedule.device, dtype=torch.long).reshape(-1) - 1
        if indexes.numel() == 0 or torch.any(indexes < 0) or torch.any(indexes >= self.max_lead):
            raise ValueError(f"lead_day tensor values must be 1..{self.max_lead}.")
        return schedule.transpose(0, 1)[indexes].reshape(-1, self.channels, 1, 1)
