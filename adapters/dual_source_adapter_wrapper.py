"""Input-level adapter wrapper that leaves the frozen backbone implementation intact."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
from torch import nn

from models.spatiotemporal_utils import split_spatiotemporal_input

from .forcing_adapter import ForcingAdapter
from .ocean_state_adapter import OceanStateAdapter


class DualSourceAdapterWrapper(nn.Module):
    """Adapt ocean history and lead forcing before calling an existing backbone."""

    def __init__(
        self,
        backbone: nn.Module,
        ocean_adapter: OceanStateAdapter | None = None,
        forcing_adapter: ForcingAdapter | None = None,
    ):
        super().__init__()
        if not hasattr(backbone, "layout"):
            raise TypeError("Backbone must expose the InputLayout used by split_spatiotemporal_input().")
        layout = backbone.layout
        if (layout.state_channels, layout.forcing_channels, layout.static_channels) != (133, 8, 4):
            raise ValueError(
                "Stage5 v1 requires state/forcing/static channels = 133/8/4, "
                f"got {layout.state_channels}/{layout.forcing_channels}/{layout.static_channels}."
            )
        self.backbone = backbone
        self.ocean_adapter = ocean_adapter or OceanStateAdapter()
        self.forcing_adapter = forcing_adapter or ForcingAdapter(layout.forcing_channels)
        self.configure_adaptation_mode("dual_adapter")
        self.ocean_adapter_history_policy = "observed_product_frames_only"

    @property
    def layout(self):
        return self.backbone.layout

    def load_backbone_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *,
        strict: bool = True,
    ):
        """Load a legacy source checkpoint only into the unchanged backbone.

        Old checkpoints are deliberately not expected to contain adapter keys.
        Strictness therefore applies to the backbone, not to the wrapper prefix.
        """

        return self.backbone.load_state_dict(state_dict, strict=strict)

    def configure_trainable(
        self,
        mode: str = "adapter_only",
        selected_layer_prefixes: Iterable[str] = (),
    ) -> None:
        if mode not in {"adapter_only", "adapter_plus_selected_layers"}:
            raise ValueError(f"Unsupported trainable mode: {mode}")
        self.configure_adaptation_mode("dual_adapter")
        if mode == "adapter_plus_selected_layers":
            prefixes = tuple(str(value) for value in selected_layer_prefixes)
            if not prefixes:
                raise ValueError("adapter_plus_selected_layers requires at least one backbone prefix.")
            matched = set()
            for name, parameter in self.backbone.named_parameters():
                for prefix in prefixes:
                    if name == prefix or name.startswith(prefix + "."):
                        parameter.requires_grad_(True)
                        matched.add(prefix)
            missing = sorted(set(prefixes) - matched)
            if missing:
                raise ValueError(f"Selected backbone layer prefixes did not match parameters: {missing}")

    def configure_adaptation_mode(self, mode: str) -> None:
        if mode not in {"source_only", "ocean_only", "forcing_only", "dual_adapter"}:
            raise ValueError(f"Unsupported Stage5 adaptation mode: {mode}")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        ocean_enabled = mode in {"ocean_only", "dual_adapter"}
        forcing_enabled = mode in {"forcing_only", "dual_adapter"}
        for parameter in self.ocean_adapter.parameters():
            parameter.requires_grad_(ocean_enabled)
        for parameter in self.forcing_adapter.parameters():
            parameter.requires_grad_(forcing_enabled)
        self.adaptation_mode = mode
        self.ocean_adapter_enabled = ocean_enabled
        self.forcing_adapter_enabled = forcing_enabled

    def parameter_inventory(self) -> dict[str, int | float]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        ocean = sum(parameter.numel() for parameter in self.ocean_adapter.parameters())
        forcing = sum(parameter.numel() for parameter in self.forcing_adapter.parameters())
        backbone = sum(parameter.numel() for parameter in self.backbone.parameters())
        wrapper_only = total - backbone - ocean - forcing
        return {
            "total_parameters": int(total),
            "trainable_parameters": int(trainable),
            "trainable_ratio": float(trainable / total if total else 0.0),
            "backbone_parameters": int(backbone),
            "ocean_adapter_parameters": int(ocean),
            "forcing_adapter_parameters": int(forcing),
            "wrapper_parameters": int(wrapper_only),
        }

    def forward(
        self,
        values: torch.Tensor,
        lead_day: int | torch.Tensor | None = None,
        observed_history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        history, forcing, static = split_spatiotemporal_input(values, self.layout)
        if observed_history_mask is None:
            observed_history_mask = torch.ones(
                history.shape[:2], device=history.device, dtype=torch.bool
            )
        adapted_history = (
            self.ocean_adapter(history, observed_history_mask=observed_history_mask)
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
