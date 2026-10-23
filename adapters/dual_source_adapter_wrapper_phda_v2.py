"""PHDA V2 wrapper with frame-age ocean and horizon forcing gates."""

from __future__ import annotations

import torch

from models.spatiotemporal_utils import split_spatiotemporal_input

from .dual_source_adapter_wrapper_phda import ProvenanceHorizonDualSourceAdapterWrapper
from .forcing_adapter_phda import ForcingAdapterPHDA
from .ocean_state_adapter_phda_v2 import OceanStateAdapterPHDAV2


class FrameAgeHorizonDualSourceAdapterWrapper(ProvenanceHorizonDualSourceAdapterWrapper):
    adapter_version = "phda_v2_frame_age_horizon"
    gate_policy = "smooth_bounded_frame_age_horizon"

    def __init__(self, backbone) -> None:
        # Avoid the V1 constructor so the checkpoint architecture is explicit.
        torch.nn.Module.__init__(self)
        if not hasattr(backbone, "layout"):
            raise TypeError("Backbone must expose an InputLayout.")
        layout = backbone.layout
        if (layout.state_channels, layout.forcing_channels, layout.static_channels) != (133, 8, 4):
            raise ValueError("PHDA V2 requires state/forcing/static channels = 133/8/4.")
        self.backbone = backbone
        self.ocean_adapter = OceanStateAdapterPHDAV2()
        self.forcing_adapter = ForcingAdapterPHDA(layout.forcing_channels)
        self.configure_adaptation_mode("dual_adapter")

    def forward(
        self,
        values: torch.Tensor,
        lead_day: int | torch.Tensor | None = None,
        observed_history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if lead_day is None and (self.ocean_adapter_enabled or self.forcing_adapter_enabled):
            raise ValueError("PHDA V2 requires an explicit lead_day.")
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
