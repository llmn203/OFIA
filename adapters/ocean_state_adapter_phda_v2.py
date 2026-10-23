"""Frame-age and horizon-conditioned ocean Adapter for PHDA V2."""

from __future__ import annotations

import torch
from torch import nn

from .ocean_state_adapter import OCEAN_CHANNEL_GROUPS
from .ocean_state_adapter_phda import _BoundedResidualVariableAdapter
from .provenance_horizon_gate import FrameAgeHorizonOceanGate


class OceanStateAdapterPHDAV2(nn.Module):
    state_channels = 133
    channel_groups = OCEAN_CHANNEL_GROUPS
    version = "phda_v2_frame_age_horizon"

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
        self.gate = FrameAgeHorizonOceanGate(len(self.channel_groups))

    def reset_identity(self) -> None:
        for branch in self.branches.values():
            branch.reset_identity()

    @staticmethod
    def normalized_frame_age(history_days: int, *, device: torch.device) -> torch.Tensor:
        if history_days < 1:
            raise ValueError("history_days must be positive.")
        if history_days == 1:
            return torch.zeros(1, device=device)
        # History is chronological: index 0 is oldest, index L-1 is newest.
        return torch.linspace(1.0, 0.0, history_days, device=device)

    def forward(
        self,
        history: torch.Tensor,
        observed_history_mask: torch.Tensor | None = None,
        *,
        lead_day: int | torch.Tensor,
    ) -> torch.Tensor:
        if history.ndim not in {4, 5} or history.shape[-3] != self.state_channels:
            raise ValueError(
                "OceanStateAdapterPHDAV2 expects [B,133,H,W] or [B,L,133,H,W], "
                f"got {tuple(history.shape)}."
            )
        batch = history.shape[0]
        history_days = history.shape[1] if history.ndim == 5 else 1
        if observed_history_mask is None:
            mask = torch.ones((batch, history_days), device=history.device, dtype=torch.bool)
        else:
            mask = observed_history_mask.to(device=history.device, dtype=torch.bool)
            if tuple(mask.shape) not in {(batch, history_days), (batch,), (batch, 1)}:
                raise ValueError(
                    f"observed_history_mask must match {(batch, history_days)}, got {tuple(mask.shape)}."
                )
            mask = mask.reshape(batch, history_days)

        frame_age = self.normalized_frame_age(history_days, device=history.device)
        gates = self.gate(lead_day, frame_age)
        if gates.shape[0] not in {1, batch}:
            raise ValueError("lead_day batch does not match ocean history batch.")
        if gates.shape[0] == 1 and batch > 1:
            gates = gates.expand(batch, -1, -1, -1, -1)

        original_shape = history.shape
        values = history if history.ndim == 4 else history.reshape(-1, *history.shape[-3:])
        flat_gates = gates.reshape(batch * history_days, len(self.channel_groups), 1, 1)
        adapted = torch.cat(
            [
                self.branches[name](values[:, channel_slice], flat_gates[:, index : index + 1])
                for index, (name, channel_slice) in enumerate(self.channel_groups.items())
            ],
            dim=1,
        )
        adapted = adapted if history.ndim == 4 else adapted.reshape(original_shape)
        boolean_mask = mask.reshape(batch, 1, 1, 1) if history.ndim == 4 else mask[:, :, None, None, None]
        return torch.where(boolean_mask, adapted, history)
