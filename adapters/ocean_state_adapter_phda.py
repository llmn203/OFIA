"""Provenance- and horizon-conditioned ocean Adapter for PHDA."""

from __future__ import annotations

import torch
from torch import nn

from .ocean_state_adapter import OCEAN_CHANNEL_GROUPS
from .provenance_horizon_gate import ProvenanceHorizonOceanGate


class _BoundedResidualVariableAdapter(nn.Module):
    def __init__(self, channels: int, rank: int, use_depthwise: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(channels, rank, 1), nn.GELU()]
        if use_depthwise:
            layers.extend([nn.Conv2d(rank, rank, 3, padding=1, groups=rank), nn.GELU()])
        layers.append(nn.Conv2d(rank, channels, 1))
        self.residual = nn.Sequential(*layers)
        self.alpha = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def reset_identity(self) -> None:
        nn.init.zeros_(self.alpha)

    def effective_alpha(self) -> torch.Tensor:
        # Smoothly bounds residual magnitude while preserving alpha=0 identity.
        return 2.0 * torch.tanh(self.alpha / 2.0)

    def forward(self, values: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return values + self.effective_alpha() * gate * self.residual(torch.nan_to_num(values))


class OceanStateAdapterPHDA(nn.Module):
    state_channels = 133
    channel_groups = OCEAN_CHANNEL_GROUPS
    version = "phda_v1_provenance_horizon"

    def __init__(self, rank_3d: int = 8, rank_zos: int = 1, use_depthwise: bool = True) -> None:
        super().__init__()
        self.branches = nn.ModuleDict(
            {
                name: _BoundedResidualVariableAdapter(
                    channel_slice.stop - channel_slice.start,
                    rank_zos if name == "zos" else rank_3d,
                    use_depthwise,
                )
                for name, channel_slice in self.channel_groups.items()
            }
        )
        self.gate = ProvenanceHorizonOceanGate(len(self.channel_groups))

    def reset_identity(self) -> None:
        for branch in self.branches.values():
            branch.reset_identity()

    def forward(
        self,
        history: torch.Tensor,
        observed_history_mask: torch.Tensor | None = None,
        *,
        lead_day: int | torch.Tensor,
    ) -> torch.Tensor:
        if history.ndim not in {4, 5} or history.shape[-3] != self.state_channels:
            raise ValueError(
                "OceanStateAdapterPHDA expects [B,133,H,W] or [B,L,133,H,W], "
                f"got {tuple(history.shape)}."
            )
        batch = history.shape[0]
        if observed_history_mask is None:
            mask = torch.ones(history.shape[:2] if history.ndim == 5 else (batch, 1), device=history.device)
        else:
            mask = observed_history_mask.to(device=history.device)
            expected = (batch, history.shape[1]) if history.ndim == 5 else (batch,)
            if tuple(mask.shape) not in {expected, (batch, 1)}:
                raise ValueError(f"observed_history_mask shape is incompatible; expected {expected}.")
            mask = mask.reshape(batch, -1)
        observed_fraction = mask.to(torch.float32).mean(dim=1)
        gates = self.gate(lead_day, observed_fraction)
        if gates.shape[0] not in {1, batch}:
            raise ValueError("lead_day batch does not match ocean history batch.")
        if gates.shape[0] == 1 and batch > 1:
            gates = gates.expand(batch, -1, -1, -1)

        original_shape = history.shape
        if history.ndim == 5:
            history_days = history.shape[1]
            values = history.reshape(-1, *history.shape[-3:])
            gates = gates.repeat_interleave(history_days, dim=0)
        else:
            values = history
        adapted = torch.cat(
            [
                self.branches[name](values[:, channel_slice], gates[:, index : index + 1])
                for index, (name, channel_slice) in enumerate(self.channel_groups.items())
            ],
            dim=1,
        )
        adapted = adapted if history.ndim == 4 else adapted.reshape(original_shape)
        boolean_mask = mask.to(dtype=torch.bool)
        if history.ndim == 4:
            boolean_mask = boolean_mask.reshape(batch, 1, 1, 1)
        else:
            boolean_mask = boolean_mask[:, :, None, None, None]
        return torch.where(boolean_mask, adapted, history)
