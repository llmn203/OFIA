"""Backbone-agnostic dual-interface spatiotemporal Adapter (DISTA).

DISTA operates only on the physical interfaces shared by every operational
backbone: seven ocean-history frames, the ten-day forcing trajectory, and the
predicted daily ocean-state tendency.  The frozen backbone implementation is
not modified.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from models.spatiotemporal_utils import InputLayout, groups_for, split_spatiotemporal_input


def _zero_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        nn.init.zeros_(parameter)


def _identity_depthwise(module: nn.Conv2d) -> None:
    nn.init.zeros_(module.weight)
    center_y = module.weight.shape[-2] // 2
    center_x = module.weight.shape[-1] // 2
    with torch.no_grad():
        module.weight[:, 0, center_y, center_x] = 1.0
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class _TemporalDepthwiseMixer(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            channels,
            channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=channels,
            bias=False,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        # [B,T,C,H,W] -> temporal depthwise convolution.
        mixed = self.conv(values.permute(0, 2, 1, 3, 4))
        return values + mixed.permute(0, 2, 1, 3, 4)


class _ConditionedSpatialBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        context_channels: int,
        *,
        expansion: int = 3,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        hidden = channels * expansion
        self.norm = nn.GroupNorm(groups_for(channels), channels)
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
        )
        self.expand = nn.Conv2d(channels, hidden, 1)
        self.project = nn.Conv2d(hidden, channels, 1)
        self.film = nn.Linear(context_channels, channels * 2)

    def forward(self, values: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        residual = self.depthwise(self.norm(values))
        gamma, beta = self.film(context).chunk(2, dim=1)
        residual = residual * (1.0 + 0.1 * torch.tanh(gamma)[:, :, None, None])
        residual = residual + 0.1 * beta[:, :, None, None]
        residual = self.project(F.gelu(self.expand(residual)))
        return values + residual


class _TrajectoryAdapter(nn.Module):
    """Low-resolution temporal-spatial residual Adapter for a channel trajectory."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        context_channels: int,
        *,
        blocks: int = 2,
        max_correction: float = 2.0,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)
        self.max_correction = float(max_correction)
        self.input_projection = nn.Conv2d(channels, hidden_channels, 1)
        self.temporal = _TemporalDepthwiseMixer(hidden_channels)
        self.blocks = nn.ModuleList(
            [
                _ConditionedSpatialBlock(
                    hidden_channels,
                    context_channels,
                    dilation=1 if index % 2 == 0 else 2,
                )
                for index in range(blocks)
            ]
        )
        self.output_projection = nn.Conv2d(hidden_channels, channels, 1)
        self.detail = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.reset_identity()

    def reset_identity(self) -> None:
        # The first optimizer step activates the output interface; deeper
        # residual parameters receive gradients from the following step.
        _zero_module(self.output_projection)
        _identity_depthwise(self.detail)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 5 or values.shape[2] != self.channels:
            raise ValueError(
                f"Trajectory Adapter expects [B,T,{self.channels},H,W], got {tuple(values.shape)}."
            )
        batch, steps, _, height, width = values.shape
        flat = torch.nan_to_num(values).reshape(batch * steps, self.channels, height, width)
        reduced = F.avg_pool2d(flat, kernel_size=2, stride=2, ceil_mode=True)
        encoded = self.input_projection(reduced)
        encoded = encoded.reshape(batch, steps, self.hidden_channels, *encoded.shape[-2:])
        return self.temporal(encoded)

    @staticmethod
    def summary(encoded: torch.Tensor) -> torch.Tensor:
        return encoded.mean(dim=(1, 3, 4))

    def adapt_encoded(
        self,
        original: torch.Tensor,
        encoded: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        batch, steps, _, height, width = original.shape
        values = encoded.reshape(batch * steps, self.hidden_channels, *encoded.shape[-2:])
        repeated_context = context[:, None].expand(-1, steps, -1).reshape(batch * steps, -1)
        for block in self.blocks:
            values = block(values, repeated_context)
        residual = self.output_projection(values)
        residual = F.interpolate(residual, size=(height, width), mode="bilinear", align_corners=False)
        residual = self.detail(residual)
        residual = residual.reshape(batch, steps, self.channels, height, width)
        return original + self.max_correction * torch.tanh(residual)

    def forward(self, values: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.adapt_encoded(values, self.encode(values), context)


class DISTAOceanHistoryAdapter(_TrajectoryAdapter):
    state_channels = 133
    version = "dista_base_v1"

    def __init__(self, context_channels: int = 64, hidden_channels: int = 64) -> None:
        super().__init__(
            self.state_channels,
            hidden_channels,
            context_channels,
            blocks=2,
            max_correction=2.0,
        )

    def forward(
        self,
        history: torch.Tensor,
        context: torch.Tensor,
        observed_history_mask: torch.Tensor,
    ) -> torch.Tensor:
        if history.ndim != 5:
            raise ValueError(f"DISTA ocean history must be [B,L,133,H,W], got {tuple(history.shape)}.")
        mask = observed_history_mask.to(device=history.device, dtype=torch.bool)
        if tuple(mask.shape) != tuple(history.shape[:2]):
            raise ValueError(
                f"observed_history_mask must be {tuple(history.shape[:2])}, got {tuple(mask.shape)}."
            )
        adapted = super().forward(history, context)
        return torch.where(mask[:, :, None, None, None], adapted, history)


class DISTAForcingTrajectoryAdapter(_TrajectoryAdapter):
    forcing_channels = 8
    physical_channels = 6
    adapted_channel_names = ("u10", "v10", "t2m", "d2m", "msl", "precipitation")
    passthrough_channel_names = ("sin_doy", "cos_doy")
    version = "dista_base_v1"

    def __init__(self, context_channels: int = 64, hidden_channels: int = 32) -> None:
        super().__init__(
            self.physical_channels,
            hidden_channels,
            context_channels,
            blocks=2,
            max_correction=2.0,
        )

    def forward(self, forcing: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if forcing.ndim != 5 or forcing.shape[2] != self.forcing_channels:
            raise ValueError(f"DISTA forcing must be [B,T,8,H,W], got {tuple(forcing.shape)}.")
        physical = super().forward(forcing[:, :, : self.physical_channels], context)
        # Calendar channels must remain exactly unchanged.
        return torch.cat([physical, forcing[:, :, self.physical_channels :]], dim=2)


class DISTAOutputTendencyAdapter(nn.Module):
    state_channels = 133
    version = "dista_base_v1"

    def __init__(
        self,
        context_channels: int = 64,
        hidden_channels: int = 64,
        *,
        max_correction: float = 2.0,
    ) -> None:
        super().__init__()
        self.context_channels = int(context_channels)
        self.hidden_channels = int(hidden_channels)
        self.max_correction = float(max_correction)
        self.input_projection = nn.Conv2d(self.state_channels * 2, hidden_channels, 1)
        self.lead_projection = nn.Sequential(
            nn.Linear(4, context_channels),
            nn.GELU(),
            nn.Linear(context_channels, context_channels),
        )
        self.blocks = nn.ModuleList(
            [
                _ConditionedSpatialBlock(
                    hidden_channels,
                    context_channels,
                    dilation=1 if index == 0 else 2,
                )
                for index in range(2)
            ]
        )
        self.output_projection = nn.Conv2d(hidden_channels, self.state_channels, 1)
        self.detail = nn.Conv2d(
            self.state_channels,
            self.state_channels,
            3,
            padding=1,
            groups=self.state_channels,
            bias=False,
        )
        self.reset_identity()

    def reset_identity(self) -> None:
        _zero_module(self.output_projection)
        _identity_depthwise(self.detail)

    @staticmethod
    def lead_features(
        lead_day: int | torch.Tensor,
        *,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        lead = torch.as_tensor(lead_day, device=device, dtype=dtype).reshape(-1)
        if lead.numel() == 1:
            lead = lead.expand(batch)
        if lead.numel() != batch:
            raise ValueError("lead_day batch does not match the forecast batch.")
        progress = ((lead - 1.0) / 9.0).clamp(0.0, 1.0)
        return torch.stack(
            [
                progress,
                progress.square(),
                torch.sin(torch.pi * progress),
                torch.cos(torch.pi * progress),
            ],
            dim=1,
        )

    def forward(
        self,
        delta: torch.Tensor,
        last_state: torch.Tensor,
        context: torch.Tensor,
        lead_day: int | torch.Tensor,
    ) -> torch.Tensor:
        if delta.shape != last_state.shape or delta.ndim != 4 or delta.shape[1] != self.state_channels:
            raise ValueError("DISTA output Adapter expects matching [B,133,H,W] delta and last_state.")
        batch, _, height, width = delta.shape
        values = F.avg_pool2d(
            torch.cat([torch.nan_to_num(delta), torch.nan_to_num(last_state)], dim=1),
            kernel_size=2,
            stride=2,
            ceil_mode=True,
        )
        values = self.input_projection(values)
        lead_context = self.lead_projection(
            self.lead_features(
                lead_day,
                batch=batch,
                device=delta.device,
                dtype=delta.dtype,
            )
        )
        conditioned = context + lead_context
        for block in self.blocks:
            values = block(values, conditioned)
        residual = self.output_projection(values)
        residual = F.interpolate(residual, size=(height, width), mode="bilinear", align_corners=False)
        residual = self.detail(residual)
        return delta + self.max_correction * torch.tanh(residual)


class DualInterfaceSpatiotemporalAdapterWrapper(nn.Module):
    """Universal DISTA Base wrapper for any backbone exposing InputLayout."""

    adapter_version = "dista_base_v1"
    ocean_adapter_history_policy = "observed_product_frames_only"
    forcing_adapter_policy = "full_10day_physical_trajectory"
    output_adapter_policy = "context_conditioned_tendency_residual"

    def __init__(
        self,
        backbone: nn.Module,
        *,
        context_channels: int = 64,
        ocean_hidden_channels: int = 64,
        forcing_hidden_channels: int = 32,
        output_hidden_channels: int = 64,
        layout: InputLayout | None = None,
    ) -> None:
        super().__init__()
        if layout is None and not hasattr(backbone, "layout"):
            raise TypeError("DISTA backbone must expose an InputLayout.")
        layout = layout or backbone.layout
        if (layout.state_channels, layout.forcing_channels, layout.static_channels) != (133, 8, 4):
            raise ValueError("DISTA requires state/forcing/static channels = 133/8/4.")
        self.backbone = backbone
        self._layout = layout
        self.context_channels = int(context_channels)
        self.ocean_adapter = DISTAOceanHistoryAdapter(context_channels, ocean_hidden_channels)
        self.forcing_adapter = DISTAForcingTrajectoryAdapter(context_channels, forcing_hidden_channels)
        self.output_adapter = DISTAOutputTendencyAdapter(
            context_channels,
            output_hidden_channels,
        )
        context_input = ocean_hidden_channels + forcing_hidden_channels + layout.static_channels
        self.context_fusion = nn.Sequential(
            nn.LayerNorm(context_input),
            nn.Linear(context_input, context_channels * 2),
            nn.GELU(),
            nn.Linear(context_channels * 2, context_channels),
        )
        self.configure_adaptation_mode("dual_adapter")

    @property
    def layout(self):
        return self._layout

    def train(self, mode: bool = True):
        super().train(mode)
        # A frozen backbone must retain the exact inference-time dropout/norm
        # behavior while gradients flow through it into the input adapters.
        self.backbone.eval()
        if mode:
            # cuDNN refuses backward through an nn.LSTM left in eval mode,
            # although the frozen Adapter workflow needs gradients with
            # respect to its *input*, not the LSTM parameters.  Keeping only
            # the recurrent module in train mode unlocks that backward path;
            # disabling its internal dropout preserves the frozen eval-time
            # forward semantics. CNN norms and every other backbone module
            # remain in eval mode.
            for module in self.backbone.modules():
                if isinstance(module, nn.LSTM):
                    module.dropout = 0.0
                    module.train(True)
        return self

    def load_backbone_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *,
        strict: bool = True,
    ):
        return self.backbone.load_state_dict(state_dict, strict=strict)

    def configure_adaptation_mode(self, mode: str) -> None:
        if mode not in {"source_only", "ocean_only", "forcing_only", "dual_adapter"}:
            raise ValueError(f"Unsupported Stage5 adaptation mode: {mode}")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.ocean_adapter_enabled = mode in {"ocean_only", "dual_adapter"}
        self.forcing_adapter_enabled = mode in {"forcing_only", "dual_adapter"}
        # The output tendency branch is the second interface of the complete
        # DISTA mechanism. Single-source modes remain pure input ablations.
        self.output_adapter_enabled = mode == "dual_adapter"
        context_enabled = (
            self.ocean_adapter_enabled
            or self.forcing_adapter_enabled
            or self.output_adapter_enabled
        )
        for parameter in self.ocean_adapter.parameters():
            parameter.requires_grad_(self.ocean_adapter_enabled)
        for parameter in self.forcing_adapter.parameters():
            parameter.requires_grad_(self.forcing_adapter_enabled)
        for parameter in self.output_adapter.parameters():
            parameter.requires_grad_(self.output_adapter_enabled)
        for parameter in self.context_fusion.parameters():
            parameter.requires_grad_(context_enabled)
        self.adaptation_mode = mode

    def parameter_inventory(self) -> dict[str, int | float]:
        components = {
            "backbone_parameters": sum(p.numel() for p in self.backbone.parameters()),
            "ocean_adapter_parameters": sum(p.numel() for p in self.ocean_adapter.parameters()),
            "forcing_adapter_parameters": sum(p.numel() for p in self.forcing_adapter.parameters()),
            "output_adapter_parameters": sum(p.numel() for p in self.output_adapter.parameters()),
            "context_adapter_parameters": sum(p.numel() for p in self.context_fusion.parameters()),
        }
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {
            "total_parameters": int(total),
            "trainable_parameters": int(trainable),
            "trainable_ratio": float(trainable / total if total else 0.0),
            **{key: int(value) for key, value in components.items()},
            "wrapper_parameters": 0,
        }

    def _cycle_context(
        self,
        history: torch.Tensor,
        forcing: torch.Tensor,
        static: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ocean_encoded = self.ocean_adapter.encode(history)
        forcing_encoded = self.forcing_adapter.encode(forcing[:, :, :6])
        static_summary = torch.nan_to_num(static).mean(dim=(2, 3))
        context = self.context_fusion(
            torch.cat(
                [
                    self.ocean_adapter.summary(ocean_encoded),
                    self.forcing_adapter.summary(forcing_encoded),
                    static_summary,
                ],
                dim=1,
            )
        )
        return context, forcing_encoded

    def prepare_cycle(
        self,
        history: torch.Tensor,
        forcing_trajectory: torch.Tensor,
        static: torch.Tensor,
        observed_history_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if forcing_trajectory.ndim != 5 or forcing_trajectory.shape[1:3] != (10, 8):
            raise ValueError(
                "DISTA requires the complete [B,10,8,H,W] future forcing trajectory."
            )
        context, forcing_encoded = self._cycle_context(history, forcing_trajectory, static)
        if self.forcing_adapter_enabled:
            physical = self.forcing_adapter.adapt_encoded(
                forcing_trajectory[:, :, :6],
                forcing_encoded,
                context,
            )
            adapted_forcing = torch.cat([physical, forcing_trajectory[:, :, 6:]], dim=2)
        else:
            adapted_forcing = forcing_trajectory
        return adapted_forcing, context

    def adapter_state_dict(self) -> dict[str, Any]:
        return {
            "ocean": self.ocean_adapter.state_dict() if self.ocean_adapter_enabled else None,
            "forcing": self.forcing_adapter.state_dict() if self.forcing_adapter_enabled else None,
            "output": self.output_adapter.state_dict() if self.output_adapter_enabled else None,
            "context": self.context_fusion.state_dict()
            if any(
                (
                    self.ocean_adapter_enabled,
                    self.forcing_adapter_enabled,
                    self.output_adapter_enabled,
                )
            )
            else None,
        }

    def load_adapter_state_dict(self, payload: dict[str, Any]) -> None:
        expected = {
            "ocean": self.ocean_adapter_enabled,
            "forcing": self.forcing_adapter_enabled,
            "output": self.output_adapter_enabled,
            "context": any(
                (
                    self.ocean_adapter_enabled,
                    self.forcing_adapter_enabled,
                    self.output_adapter_enabled,
                )
            ),
        }
        modules = {
            "ocean": self.ocean_adapter,
            "forcing": self.forcing_adapter,
            "output": self.output_adapter,
            "context": self.context_fusion,
        }
        for name, enabled in expected.items():
            if (payload.get(name) is not None) != enabled:
                raise ValueError(f"DISTA checkpoint {name} enablement mismatch.")
            if enabled:
                modules[name].load_state_dict(payload[name], strict=True)

    def diagnostic_values(self) -> dict[str, list[float]]:
        def norm(module: nn.Module) -> list[float]:
            values = [parameter.detach().float().norm() for parameter in module.parameters()]
            return [float(torch.stack(values).norm().cpu())] if values else []

        return {
            "ocean": norm(self.ocean_adapter),
            "forcing": norm(self.forcing_adapter),
            "output": norm(self.output_adapter),
            "context": norm(self.context_fusion),
        }

    def forward(
        self,
        values: torch.Tensor,
        lead_day: int | torch.Tensor | None = None,
        observed_history_mask: torch.Tensor | None = None,
        *,
        cycle_context: torch.Tensor | None = None,
        forcing_pre_adapted: bool = False,
    ) -> torch.Tensor:
        if lead_day is None:
            raise ValueError("DISTA requires an explicit lead_day.")
        history, forcing, static = split_spatiotemporal_input(values, self.layout)
        batch = history.shape[0]
        if observed_history_mask is None:
            observed_history_mask = torch.ones(
                history.shape[:2],
                device=history.device,
                dtype=torch.bool,
            )
        if cycle_context is None:
            # Compatibility fallback for single-step calls. Formal training
            # always supplies the context built from the complete trajectory.
            static_summary = torch.nan_to_num(static).mean(dim=(2, 3))
            ocean_summary = self.ocean_adapter.summary(self.ocean_adapter.encode(history))
            forcing_one = forcing[:, None, :6]
            forcing_summary = self.forcing_adapter.summary(self.forcing_adapter.encode(forcing_one))
            cycle_context = self.context_fusion(
                torch.cat([ocean_summary, forcing_summary, static_summary], dim=1)
            )
        adapted_history = (
            self.ocean_adapter(history, cycle_context, observed_history_mask)
            if self.ocean_adapter_enabled
            else history
        )
        if self.forcing_adapter_enabled and not forcing_pre_adapted:
            adapted_forcing = self.forcing_adapter(forcing[:, None], cycle_context)[:, 0]
        else:
            adapted_forcing = forcing
        adapted_input = torch.cat(
            [
                adapted_history.reshape(
                    batch,
                    self.layout.history_days * self.layout.state_channels,
                    values.shape[-2],
                    values.shape[-1],
                ),
                adapted_forcing,
                static,
            ],
            dim=1,
        )
        delta = self.backbone(adapted_input)
        if self.output_adapter_enabled:
            delta = self.output_adapter(
                delta,
                adapted_history[:, -1],
                cycle_context,
                lead_day,
            )
        return delta
