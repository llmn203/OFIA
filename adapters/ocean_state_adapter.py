"""Identity-initialized adapter for the 133-channel ocean-state interface."""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn


OCEAN_CHANNEL_GROUPS = OrderedDict(
    (
        ("thetao", slice(0, 33)),
        ("so", slice(33, 66)),
        ("uo", slice(66, 99)),
        ("vo", slice(99, 132)),
        ("zos", slice(132, 133)),
    )
)


class _ResidualVariableAdapter(nn.Module):
    def __init__(self, channels: int, rank: int, use_depthwise: bool = True):
        super().__init__()
        if channels < 1 or rank < 1:
            raise ValueError("channels and rank must be positive.")
        layers: list[nn.Module] = [nn.Conv2d(channels, rank, 1), nn.GELU()]
        if use_depthwise:
            layers.extend(
                [
                    nn.Conv2d(rank, rank, 3, padding=1, groups=rank),
                    nn.GELU(),
                ]
            )
        layers.append(nn.Conv2d(rank, channels, 1))
        self.residual = nn.Sequential(*layers)
        self.alpha = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def reset_identity(self) -> None:
        nn.init.zeros_(self.alpha)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4:
            raise ValueError(f"Variable adapter expects [B,C,H,W], got {tuple(values.shape)}.")
        # Invalid ocean cells are normally normalized to zero upstream.  This
        # guard also prevents a synthetic NaN from spreading through a 3x3
        # branch and changing the backbone's mask behavior at identity init.
        finite_values = torch.nan_to_num(values)
        return values + self.alpha * self.residual(finite_values)


class OceanStateAdapter(nn.Module):
    """Five independent, history-shared residual adapters.

    Input may be either ``[B,133,H,W]`` or ``[B,L,133,H,W]``.  A 5-D input is
    flattened over ``B*L`` so all history days share exactly the same weights.
    """

    state_channels = 133
    channel_groups = OCEAN_CHANNEL_GROUPS

    def __init__(
        self,
        rank_3d: int = 8,
        rank_zos: int = 1,
        use_depthwise: bool = True,
    ):
        super().__init__()
        self.branches = nn.ModuleDict(
            {
                name: _ResidualVariableAdapter(
                    channels=channel_slice.stop - channel_slice.start,
                    rank=rank_zos if name == "zos" else rank_3d,
                    use_depthwise=use_depthwise,
                )
                for name, channel_slice in self.channel_groups.items()
            }
        )

    def reset_identity(self) -> None:
        for branch in self.branches.values():
            branch.reset_identity()

    def forward(
        self,
        history: torch.Tensor,
        observed_history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if history.ndim not in {4, 5}:
            raise ValueError(
                f"OceanStateAdapter expects [B,133,H,W] or [B,L,133,H,W], got {tuple(history.shape)}."
            )
        original_shape = history.shape
        if history.shape[-3] != self.state_channels:
            raise ValueError(
                f"OceanStateAdapter expects {self.state_channels} state channels, got {history.shape[-3]}."
            )
        values = history if history.ndim == 4 else history.reshape(-1, *history.shape[-3:])
        adapted = torch.cat(
            [
                self.branches[name](values[:, channel_slice])
                for name, channel_slice in self.channel_groups.items()
            ],
            dim=1,
        )
        adapted = adapted if history.ndim == 4 else adapted.reshape(original_shape)
        if observed_history_mask is None:
            return adapted
        if history.ndim == 4:
            expected = (history.shape[0],)
            if tuple(observed_history_mask.shape) not in {expected, (history.shape[0], 1)}:
                raise ValueError(
                    f"Observed mask for a single frame must be [B] or [B,1], got {tuple(observed_history_mask.shape)}."
                )
            mask = observed_history_mask.reshape(history.shape[0], 1, 1, 1).to(
                device=history.device, dtype=torch.bool
            )
        else:
            expected = (history.shape[0], history.shape[1])
            if tuple(observed_history_mask.shape) != expected:
                raise ValueError(
                    f"Observed mask for history must be {expected}, got {tuple(observed_history_mask.shape)}."
                )
            mask = observed_history_mask[:, :, None, None, None].to(
                device=history.device, dtype=torch.bool
            )
        # torch.where blocks adapter gradients for predicted frames and keeps
        # those frames bitwise unchanged.
        return torch.where(mask, adapted, history)
