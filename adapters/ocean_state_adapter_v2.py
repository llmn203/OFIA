"""Lead-conditioned, identity-initialized ocean-state Adapter V2."""

from __future__ import annotations

import torch
from torch import nn

from .lead_conditioned_gate import MonotoneLeadGate
from .ocean_state_adapter import OCEAN_CHANNEL_GROUPS


class _GatedResidualVariableAdapter(nn.Module):
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

    def forward(self, values: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return values + self.alpha * gate * self.residual(torch.nan_to_num(values))


class OceanStateAdapterV2(nn.Module):
    """Five variable-group residual branches controlled by monotone lead gates."""

    state_channels = 133
    channel_groups = OCEAN_CHANNEL_GROUPS
    version = "v2_lead_conditioned_monotone"

    def __init__(self, rank_3d: int = 8, rank_zos: int = 1, use_depthwise: bool = True) -> None:
        super().__init__()
        self.branches = nn.ModuleDict(
            {
                name: _GatedResidualVariableAdapter(
                    channel_slice.stop - channel_slice.start,
                    rank_zos if name == "zos" else rank_3d,
                    use_depthwise,
                )
                for name, channel_slice in self.channel_groups.items()
            }
        )
        self.lead_gate = MonotoneLeadGate(len(self.channel_groups))

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
                "OceanStateAdapterV2 expects [B,133,H,W] or [B,L,133,H,W], "
                f"got {tuple(history.shape)}."
            )
        batch = history.shape[0]
        gates = self.lead_gate(lead_day)
        if gates.shape[0] not in {1, batch}:
            raise ValueError("lead_day tensor batch does not match ocean history batch.")
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
        if observed_history_mask is None:
            return adapted
        if history.ndim == 4:
            if tuple(observed_history_mask.shape) not in {(batch,), (batch, 1)}:
                raise ValueError("Single-frame observed_history_mask must be [B] or [B,1].")
            mask = observed_history_mask.reshape(batch, 1, 1, 1)
        else:
            expected = (batch, history.shape[1])
            if tuple(observed_history_mask.shape) != expected:
                raise ValueError(f"History observed_history_mask must be {expected}.")
            mask = observed_history_mask[:, :, None, None, None]
        return torch.where(mask.to(device=history.device, dtype=torch.bool), adapted, history)
