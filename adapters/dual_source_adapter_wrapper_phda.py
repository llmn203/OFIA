"""PHDA wrapper: provenance-aware ocean and horizon-aware forcing adaptation."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from models.spatiotemporal_utils import split_spatiotemporal_input

from .forcing_adapter_phda import ForcingAdapterPHDA
from .ocean_state_adapter_phda import OceanStateAdapterPHDA


class ProvenanceHorizonDualSourceAdapterWrapper(nn.Module):
    adapter_version = "phda_v1_provenance_horizon"
    ocean_adapter_history_policy = "observed_product_frames_only"
    gate_policy = "smooth_bounded_provenance_horizon"

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        if not hasattr(backbone, "layout"):
            raise TypeError("Backbone must expose an InputLayout.")
        layout = backbone.layout
        if (layout.state_channels, layout.forcing_channels, layout.static_channels) != (133, 8, 4):
            raise ValueError("PHDA requires state/forcing/static channels = 133/8/4.")
        self.backbone = backbone
        self.ocean_adapter = OceanStateAdapterPHDA()
        self.forcing_adapter = ForcingAdapterPHDA(layout.forcing_channels)
        self.configure_adaptation_mode("dual_adapter")

    @property
    def layout(self):
        return self.backbone.layout

    def load_backbone_state_dict(self, state_dict: Mapping[str, torch.Tensor], *, strict: bool = True):
        return self.backbone.load_state_dict(state_dict, strict=strict)

    def configure_adaptation_mode(self, mode: str) -> None:
        if mode not in {"source_only", "ocean_only", "forcing_only", "dual_adapter"}:
            raise ValueError(f"Unsupported Stage5 adaptation mode: {mode}")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.ocean_adapter_enabled = mode in {"ocean_only", "dual_adapter"}
        self.forcing_adapter_enabled = mode in {"forcing_only", "dual_adapter"}
        for parameter in self.ocean_adapter.parameters():
            parameter.requires_grad_(self.ocean_adapter_enabled)
        for parameter in self.forcing_adapter.parameters():
            parameter.requires_grad_(self.forcing_adapter_enabled)
        self.adaptation_mode = mode

    def parameter_inventory(self) -> dict[str, int | float]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        ocean = sum(parameter.numel() for parameter in self.ocean_adapter.parameters())
        forcing = sum(parameter.numel() for parameter in self.forcing_adapter.parameters())
        backbone = sum(parameter.numel() for parameter in self.backbone.parameters())
        return {
            "total_parameters": int(total),
            "trainable_parameters": int(trainable),
            "trainable_ratio": float(trainable / total if total else 0.0),
            "backbone_parameters": int(backbone),
            "ocean_adapter_parameters": int(ocean),
            "forcing_adapter_parameters": int(forcing),
            "wrapper_parameters": int(total - backbone - ocean - forcing),
        }

    def forward(
        self,
        values: torch.Tensor,
        lead_day: int | torch.Tensor | None = None,
        observed_history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if lead_day is None and (self.ocean_adapter_enabled or self.forcing_adapter_enabled):
            raise ValueError("PHDA requires an explicit lead_day.")
        history, forcing, static = split_spatiotemporal_input(values, self.layout)
        if observed_history_mask is None:
            observed_history_mask = torch.ones(history.shape[:2], device=history.device, dtype=torch.bool)
        adapted_history = (
            self.ocean_adapter(
                history,
                observed_history_mask=observed_history_mask,
                lead_day=lead_day,
            )
            if self.ocean_adapter_enabled
            else history
        )
        adapted_forcing = (
            self.forcing_adapter(forcing, lead_day=lead_day)
            if self.forcing_adapter_enabled
            else forcing
        )
        adapted_input = torch.cat(
            [
                adapted_history.reshape(
                    values.shape[0],
                    self.layout.history_days * self.layout.state_channels,
                    values.shape[-2],
                    values.shape[-1],
                ),
                adapted_forcing,
                static,
            ],
            dim=1,
        )
        return self.backbone(adapted_input)
