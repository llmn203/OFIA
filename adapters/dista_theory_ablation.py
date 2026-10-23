"""Theory-driven DISTA ablations that preserve all three correction interfaces."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch.nn import functional as F

from pipeline.adapters.dual_interface_spatiotemporal_adapter import (
    DualInterfaceSpatiotemporalAdapterWrapper,
)


THEORY_ABLATIONS = {
    "stepwise_forcing_context",
    "all_history_frames",
}


class DISTATheoryAblationWrapper(DualInterfaceSpatiotemporalAdapterWrapper):
    """Full DISTA with one explicitly isolated mechanism removed."""

    adapter_version = "dista_theory_ablation_v1"

    def __init__(self, *args, theory_ablation: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if theory_ablation not in THEORY_ABLATIONS:
            raise ValueError(f"Unsupported DISTA theory ablation: {theory_ablation}")
        self.theory_ablation = theory_ablation
        self.configure_adaptation_mode("dual_adapter")

    def ablation_manifest(self) -> dict[str, Any]:
        return {
            "theory_ablation": self.theory_ablation,
            "ocean_input_correction_applied": True,
            "forcing_input_correction_applied": True,
            "output_tendency_correction_applied": True,
            "context_fusion": "coupled",
            "forcing_context_policy": (
                "current_lead_only_spatial_encoder_no_temporal_mixing"
                if self.theory_ablation == "stepwise_forcing_context"
                else "full_10day_trajectory"
            ),
            "ocean_history_policy": (
                "all_history_frames"
                if self.theory_ablation == "all_history_frames"
                else "observed_product_frames_only"
            ),
        }

    def prepare_cycle(
        self,
        history: torch.Tensor,
        forcing_trajectory: torch.Tensor,
        static: torch.Tensor,
        observed_history_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.theory_ablation != "stepwise_forcing_context":
            return super().prepare_cycle(
                history, forcing_trajectory, static, observed_history_mask
            )
        if forcing_trajectory.ndim != 5 or forcing_trajectory.shape[1:3] != (10, 8):
            raise ValueError(
                "Stepwise forcing ablation requires [B,10,8,H,W] at cycle preparation."
            )

        ocean_summary = self.ocean_adapter.summary(self.ocean_adapter.encode(history))
        static_summary = torch.nan_to_num(static).mean(dim=(2, 3))
        adapted_steps = []
        contexts = []
        for lead_index in range(forcing_trajectory.shape[1]):
            forcing_step = forcing_trajectory[:, lead_index : lead_index + 1]
            # This ablation deliberately has no temporal forcing information.
            # Reusing the T=10 depthwise temporal mixer on a T=1 tensor would
            # both weaken the experimental definition and is numerically
            # unstable in FP16 cuDNN Conv3D on the deployed V100 stack.
            physical_step = torch.nan_to_num(forcing_step[:, 0, :6])
            reduced = F.avg_pool2d(
                physical_step, kernel_size=2, stride=2, ceil_mode=True
            )
            forcing_encoded = self.forcing_adapter.input_projection(reduced)[:, None]
            context = self.context_fusion(
                torch.cat(
                    [
                        ocean_summary,
                        self.forcing_adapter.summary(forcing_encoded),
                        static_summary,
                    ],
                    dim=1,
                )
            )
            physical = self.forcing_adapter.adapt_encoded(
                forcing_step[:, :, :6], forcing_encoded, context
            )
            adapted_steps.append(
                torch.cat([physical, forcing_step[:, :, 6:]], dim=2)
            )
            contexts.append(context)
        return torch.cat(adapted_steps, dim=1), torch.stack(contexts, dim=1)

    @staticmethod
    def _lead_context(
        cycle_context: torch.Tensor | None,
        lead_day: int | torch.Tensor | None,
    ) -> torch.Tensor | None:
        if cycle_context is None or cycle_context.ndim == 2:
            return cycle_context
        if cycle_context.ndim != 3 or cycle_context.shape[1] != 10:
            raise ValueError("Per-lead DISTA context must be [B,10,C].")
        if isinstance(lead_day, torch.Tensor):
            unique = torch.unique(lead_day.detach())
            if unique.numel() != 1:
                raise ValueError("A batch must use one shared lead_day for per-lead context.")
            index = int(unique.item()) - 1
        else:
            index = int(lead_day) - 1
        if not 0 <= index < 10:
            raise ValueError(f"Invalid lead_day for per-lead context: {lead_day}")
        return cycle_context[:, index]

    def forward(
        self,
        values: torch.Tensor,
        lead_day: int | torch.Tensor | None = None,
        observed_history_mask: torch.Tensor | None = None,
        *,
        cycle_context: torch.Tensor | None = None,
        forcing_pre_adapted: bool = False,
    ) -> torch.Tensor:
        if self.theory_ablation == "all_history_frames":
            if observed_history_mask is None:
                raise ValueError("All-history ablation still requires an explicit history mask shape.")
            observed_history_mask = torch.ones_like(
                observed_history_mask, dtype=torch.bool
            )
        return super().forward(
            values,
            lead_day=lead_day,
            observed_history_mask=observed_history_mask,
            cycle_context=self._lead_context(cycle_context, lead_day),
            forcing_pre_adapted=forcing_pre_adapted,
        )

    def adapter_state_dict(self) -> dict[str, Any]:
        return {
            "theory_ablation": self.theory_ablation,
            "spec": self.ablation_manifest(),
            "ocean": self.ocean_adapter.state_dict(),
            "forcing": self.forcing_adapter.state_dict(),
            "output": self.output_adapter.state_dict(),
            "context": self.context_fusion.state_dict(),
        }

    def load_adapter_state_dict(self, payload: Mapping[str, Any]) -> None:
        if payload.get("theory_ablation") != self.theory_ablation:
            raise ValueError("DISTA theory-ablation checkpoint mode mismatch.")
        if payload.get("spec") != self.ablation_manifest():
            raise ValueError("DISTA theory-ablation checkpoint specification mismatch.")
        self.ocean_adapter.load_state_dict(payload["ocean"], strict=True)
        self.forcing_adapter.load_state_dict(payload["forcing"], strict=True)
        self.output_adapter.load_state_dict(payload["output"], strict=True)
        self.context_fusion.load_state_dict(payload["context"], strict=True)
