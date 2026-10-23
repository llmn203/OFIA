"""Isolated, prospective path-ablation wrapper for DISTA.

This module is deliberately not imported by the frozen Stage5 trainer or
evaluator while Small/Large capacity runs are active.  It separates context
encoding from applying residual corrections at the three physical interfaces.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch

from models.spatiotemporal_utils import split_spatiotemporal_input
from pipeline.adapters.dual_interface_spatiotemporal_adapter import (
    DualInterfaceSpatiotemporalAdapterWrapper,
)


@dataclass(frozen=True)
class DISTAPathAblationSpec:
    name: str
    ocean_input_correction_applied: bool
    forcing_input_correction_applied: bool
    output_tendency_correction_applied: bool
    context_fusion: str


PATH_ABLATIONS: dict[str, DISTAPathAblationSpec] = {
    "source": DISTAPathAblationSpec("source", False, False, False, "none"),
    "ocean": DISTAPathAblationSpec("ocean", True, False, False, "coupled"),
    "forcing": DISTAPathAblationSpec("forcing", False, True, False, "coupled"),
    "output": DISTAPathAblationSpec("output", False, False, True, "coupled"),
    "ocean_forcing": DISTAPathAblationSpec(
        "ocean_forcing", True, True, False, "coupled"
    ),
    "full": DISTAPathAblationSpec("full", True, True, True, "coupled"),
    "full_no_coupled": DISTAPathAblationSpec(
        "full_no_coupled", True, True, True, "source_separable"
    ),
}


class DISTAPathAblationWrapper(DualInterfaceSpatiotemporalAdapterWrapper):
    """DISTA Base with orthogonal context-training and correction-path switches."""

    adapter_version = "dista_path_ablation_v1"

    def __init__(self, *args, ablation: str = "full", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.configure_path_ablation(ablation)

    def configure_path_ablation(self, ablation: str) -> None:
        if ablation not in PATH_ABLATIONS:
            raise ValueError(f"Unsupported DISTA path ablation: {ablation}")
        spec = PATH_ABLATIONS[ablation]
        for parameter in self.parameters():
            parameter.requires_grad_(False)

        self.path_ablation = ablation
        self.path_ablation_spec = spec
        self.ocean_adapter_enabled = spec.ocean_input_correction_applied
        self.forcing_adapter_enabled = spec.forcing_input_correction_applied
        self.output_adapter_enabled = spec.output_tendency_correction_applied
        self.context_encoders_trainable = ablation != "source"

        if self.context_encoders_trainable:
            # Both context encoders remain trainable because every path
            # ablation receives the same operational regime information.
            # Residual heads are trainable only when their physical correction
            # is actually applied.
            self._configure_trajectory_parameters(
                self.ocean_adapter, self.ocean_adapter_enabled
            )
            self._configure_trajectory_parameters(
                self.forcing_adapter, self.forcing_adapter_enabled
            )
            for parameter in self.context_fusion.parameters():
                parameter.requires_grad_(True)
        for parameter in self.output_adapter.parameters():
            parameter.requires_grad_(self.output_adapter_enabled)
        self.adaptation_mode = f"path_ablation:{ablation}"

    @staticmethod
    def _configure_trajectory_parameters(module, correction_applied: bool) -> None:
        for component in (module.input_projection, module.temporal):
            for parameter in component.parameters():
                parameter.requires_grad_(True)
        for component in (module.blocks, module.output_projection, module.detail):
            for parameter in component.parameters():
                parameter.requires_grad_(correction_applied)

    def ablation_manifest(self) -> dict[str, Any]:
        return {
            **asdict(self.path_ablation_spec),
            "ocean_context_encoder_trainable": self.context_encoders_trainable,
            "forcing_context_encoder_trainable": self.context_encoders_trainable,
            "same_operational_information": self.path_ablation != "source",
        }

    def _source_separable_context(
        self,
        history: torch.Tensor,
        forcing: torch.Tensor,
        static: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ocean_encoded = self.ocean_adapter.encode(history)
        forcing_encoded = self.forcing_adapter.encode(forcing[:, :, :6])
        ocean_summary = self.ocean_adapter.summary(ocean_encoded)
        forcing_summary = self.forcing_adapter.summary(forcing_encoded)
        static_summary = torch.nan_to_num(static).mean(dim=(2, 3))
        zero_o = torch.zeros_like(ocean_summary)
        zero_f = torch.zeros_like(forcing_summary)
        zero_s = torch.zeros_like(static_summary)

        def fuse(o: torch.Tensor, f: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            return self.context_fusion(torch.cat([o, f, s], dim=1))

        origin = fuse(zero_o, zero_f, zero_s)
        context = (
            fuse(ocean_summary, zero_f, zero_s)
            + fuse(zero_o, forcing_summary, zero_s)
            + fuse(zero_o, zero_f, static_summary)
            - 2.0 * origin
        )
        return context, forcing_encoded

    def prepare_cycle(
        self,
        history: torch.Tensor,
        forcing_trajectory: torch.Tensor,
        static: torch.Tensor,
        observed_history_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.path_ablation_spec.context_fusion != "source_separable":
            return super().prepare_cycle(
                history, forcing_trajectory, static, observed_history_mask
            )
        if forcing_trajectory.ndim != 5 or forcing_trajectory.shape[1:3] != (10, 8):
            raise ValueError(
                "DISTA requires the complete [B,10,8,H,W] future forcing trajectory."
            )
        context, forcing_encoded = self._source_separable_context(
            history, forcing_trajectory, static
        )
        physical = self.forcing_adapter.adapt_encoded(
            forcing_trajectory[:, :, :6], forcing_encoded, context
        )
        return torch.cat([physical, forcing_trajectory[:, :, 6:]], dim=2), context

    def adapter_state_dict(self) -> dict[str, Any]:
        if self.path_ablation == "source":
            return {
                "ablation": self.path_ablation,
                "spec": self.ablation_manifest(),
                "ocean": None,
                "forcing": None,
                "output": None,
                "context": None,
            }
        return {
            "ablation": self.path_ablation,
            "spec": self.ablation_manifest(),
            "ocean": self.ocean_adapter.state_dict(),
            "forcing": self.forcing_adapter.state_dict(),
            "output": (
                self.output_adapter.state_dict()
                if self.output_adapter_enabled
                else None
            ),
            "context": self.context_fusion.state_dict(),
        }

    def load_adapter_state_dict(self, payload: Mapping[str, Any]) -> None:
        if payload.get("ablation") != self.path_ablation:
            raise ValueError("DISTA path-ablation checkpoint mode mismatch.")
        if payload.get("spec") != self.ablation_manifest():
            raise ValueError("DISTA path-ablation checkpoint spec mismatch.")
        if self.path_ablation == "source":
            if any(payload.get(name) is not None for name in ("ocean", "forcing", "output", "context")):
                raise ValueError("Source path-ablation checkpoint contains adapter weights.")
            return
        self.ocean_adapter.load_state_dict(payload["ocean"], strict=True)
        self.forcing_adapter.load_state_dict(payload["forcing"], strict=True)
        self.context_fusion.load_state_dict(payload["context"], strict=True)
        if self.output_adapter_enabled:
            if payload.get("output") is None:
                raise ValueError("Output-enabled ablation checkpoint has no output weights.")
            self.output_adapter.load_state_dict(payload["output"], strict=True)
        elif payload.get("output") is not None:
            raise ValueError("Output-disabled ablation checkpoint contains output weights.")

    def forward(
        self,
        values: torch.Tensor,
        lead_day: int | torch.Tensor | None = None,
        observed_history_mask: torch.Tensor | None = None,
        *,
        cycle_context: torch.Tensor | None = None,
        forcing_pre_adapted: bool = False,
    ) -> torch.Tensor:
        # The inherited implementation already applies exactly the three
        # correction booleans. Keeping it here documents that context encoders
        # being trainable is independent of correction application.
        return super().forward(
            values,
            lead_day=lead_day,
            observed_history_mask=observed_history_mask,
            cycle_context=cycle_context,
            forcing_pre_adapted=forcing_pre_adapted,
        )
