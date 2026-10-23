"""Mechanism-constrained gates for the Stage5 PHDA paper-method ablation."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("softplus target must be positive.")
    return math.log(math.expm1(value))


def _normalized_lead(
    lead_day: int | torch.Tensor,
    *,
    max_lead: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(lead_day, int):
        if not 1 <= lead_day <= max_lead:
            raise ValueError(f"lead_day must be 1..{max_lead}, got {lead_day}.")
        return torch.tensor([[(lead_day - 1) / (max_lead - 1)]], device=device, dtype=dtype)
    if not torch.is_tensor(lead_day):
        raise TypeError("lead_day must be an int or tensor.")
    values = lead_day.to(device=device, dtype=dtype).reshape(-1, 1)
    if values.numel() == 0 or torch.any(values < 1) or torch.any(values > max_lead):
        raise ValueError(f"lead_day tensor values must be 1..{max_lead}.")
    return (values - 1) / (max_lead - 1)


class ProvenanceHorizonOceanGate(nn.Module):
    """Bounded ocean gate increasing with lead and observed-frame evidence.

    logit(g_o) = b + softplus(k_h) * normalized_lead
                     + softplus(k_p) * observed_fraction

    The two positive slopes encode the mechanism-level prior without imposing
    a particular lead-wise error curve.  Each ocean variable group has only
    three scalar parameters.
    """

    def __init__(
        self,
        groups: int = 5,
        *,
        max_lead: int = 10,
        initial_intercept: float = -2.0,
        initial_lead_slope: float = 1.0,
        initial_provenance_slope: float = 1.0,
    ) -> None:
        super().__init__()
        if groups < 1 or max_lead < 2:
            raise ValueError("groups must be positive and max_lead at least two.")
        self.intercept = nn.Parameter(torch.full((groups,), float(initial_intercept)))
        self.raw_lead_slope = nn.Parameter(
            torch.full((groups,), _inverse_softplus(initial_lead_slope))
        )
        self.raw_provenance_slope = nn.Parameter(
            torch.full((groups,), _inverse_softplus(initial_provenance_slope))
        )
        self.groups = int(groups)
        self.max_lead = int(max_lead)

    def forward(
        self,
        lead_day: int | torch.Tensor,
        observed_fraction: torch.Tensor,
    ) -> torch.Tensor:
        lead = _normalized_lead(
            lead_day,
            max_lead=self.max_lead,
            device=self.intercept.device,
            dtype=self.intercept.dtype,
        )
        provenance = observed_fraction.to(
            device=self.intercept.device,
            dtype=self.intercept.dtype,
        ).reshape(-1, 1)
        if torch.any(provenance < 0) or torch.any(provenance > 1):
            raise ValueError("observed_fraction must be in [0,1].")
        batch = max(lead.shape[0], provenance.shape[0])
        if lead.shape[0] not in {1, batch} or provenance.shape[0] not in {1, batch}:
            raise ValueError("lead_day and observed_fraction batch dimensions are incompatible.")
        logits = (
            self.intercept.reshape(1, -1)
            + lead * F.softplus(self.raw_lead_slope).reshape(1, -1)
            + provenance * F.softplus(self.raw_provenance_slope).reshape(1, -1)
        )
        return torch.sigmoid(logits).reshape(batch, self.groups, 1, 1)


class HorizonForcingGate(nn.Module):
    """Bounded smooth forcing gate with a positive horizon response."""

    def __init__(
        self,
        channels: int = 6,
        *,
        max_lead: int = 10,
        initial_intercept: float = -2.0,
        initial_lead_slope: float = 2.0,
    ) -> None:
        super().__init__()
        if channels < 1 or max_lead < 2:
            raise ValueError("channels must be positive and max_lead at least two.")
        self.intercept = nn.Parameter(torch.full((channels,), float(initial_intercept)))
        self.raw_lead_slope = nn.Parameter(
            torch.full((channels,), _inverse_softplus(initial_lead_slope))
        )
        self.channels = int(channels)
        self.max_lead = int(max_lead)

    def forward(self, lead_day: int | torch.Tensor) -> torch.Tensor:
        lead = _normalized_lead(
            lead_day,
            max_lead=self.max_lead,
            device=self.intercept.device,
            dtype=self.intercept.dtype,
        )
        logits = self.intercept.reshape(1, -1) + lead * F.softplus(
            self.raw_lead_slope
        ).reshape(1, -1)
        return torch.sigmoid(logits).reshape(-1, self.channels, 1, 1)


class FrameAgeHorizonOceanGate(nn.Module):
    """Per-history-frame gate with positive horizon and recency responses.

    logit(g_o) = b + softplus(k_h) * normalized_lead
                   - softplus(k_a) * normalized_frame_age

    Frame age is zero for the most recent frame and one for the oldest frame.
    Consequently, recent product observations receive no less gate weight than
    older product observations at the same forecast lead.  The provenance mask
    remains the hard authority boundary: predicted frames are never adapted.
    """

    def __init__(
        self,
        groups: int = 5,
        *,
        max_lead: int = 10,
        initial_intercept: float = -2.0,
        initial_lead_slope: float = 1.0,
        initial_age_decay: float = 1.0,
    ) -> None:
        super().__init__()
        if groups < 1 or max_lead < 2:
            raise ValueError("groups must be positive and max_lead at least two.")
        self.intercept = nn.Parameter(torch.full((groups,), float(initial_intercept)))
        self.raw_lead_slope = nn.Parameter(
            torch.full((groups,), _inverse_softplus(initial_lead_slope))
        )
        self.raw_age_decay = nn.Parameter(
            torch.full((groups,), _inverse_softplus(initial_age_decay))
        )
        self.groups = int(groups)
        self.max_lead = int(max_lead)

    def forward(
        self,
        lead_day: int | torch.Tensor,
        frame_age: torch.Tensor,
    ) -> torch.Tensor:
        lead = _normalized_lead(
            lead_day,
            max_lead=self.max_lead,
            device=self.intercept.device,
            dtype=self.intercept.dtype,
        )
        ages = frame_age.to(device=self.intercept.device, dtype=self.intercept.dtype)
        if ages.ndim == 1:
            ages = ages.reshape(1, -1)
        elif ages.ndim != 2:
            raise ValueError("frame_age must be [L] or [B,L].")
        if ages.numel() == 0 or torch.any(ages < 0) or torch.any(ages > 1):
            raise ValueError("frame_age values must be in [0,1].")
        batch = max(lead.shape[0], ages.shape[0])
        if lead.shape[0] not in {1, batch} or ages.shape[0] not in {1, batch}:
            raise ValueError("lead_day and frame_age batch dimensions are incompatible.")
        lead = lead.reshape(lead.shape[0], 1, 1)
        ages = ages.reshape(ages.shape[0], ages.shape[1], 1)
        logits = (
            self.intercept.reshape(1, 1, -1)
            + lead * F.softplus(self.raw_lead_slope).reshape(1, 1, -1)
            - ages * F.softplus(self.raw_age_decay).reshape(1, 1, -1)
        )
        return torch.sigmoid(logits).reshape(batch, ages.shape[1], self.groups, 1, 1)
