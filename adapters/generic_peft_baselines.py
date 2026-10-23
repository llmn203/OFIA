"""Backbone-agnostic comparison baselines for the DISTA study.

These modules deliberately avoid DISTA's three-interface, full-trajectory
mechanism.  They provide conventional capacity controls: a generic input
bottleneck Adapter, input FiLM, internal Conv/Linear LoRA, and a low-capacity
leadwise affine output correction.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from models.spatiotemporal_utils import InputLayout, split_spatiotemporal_input


def _freeze(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _count(module: nn.Module, *, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


class _GenericPEFTWrapper(nn.Module):
    baseline_name = "generic_peft"

    def __init__(self, backbone: nn.Module, adapter: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        _freeze(self.backbone)
        self.ocean_adapter = adapter
        self.forcing_adapter = nn.Identity()
        self.output_adapter = nn.Identity()
        self.ocean_adapter_enabled = True
        self.forcing_adapter_enabled = False
        self.output_adapter_enabled = False

    def parameter_inventory(self) -> dict[str, Any]:
        total = _count(self)
        trainable = _count(self, trainable_only=True)
        adapter = _count(self.ocean_adapter, trainable_only=True)
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "trainable_ratio": trainable / max(total, 1),
            "backbone_parameters": _count(self.backbone),
            "ocean_adapter_parameters": adapter,
            "forcing_adapter_parameters": 0,
            "output_adapter_parameters": 0,
            "context_adapter_parameters": 0,
            "generic_baseline": self.baseline_name,
        }

    def diagnostic_values(self) -> dict[str, list[float]]:
        values = [
            float(parameter.detach().float().norm().cpu())
            for parameter in self.ocean_adapter.parameters()
            if parameter.requires_grad
        ]
        return {self.baseline_name: values}

    def adapter_state_dict(self) -> dict[str, Any]:
        return {"baseline_name": self.baseline_name, "adapter": self.ocean_adapter.state_dict()}

    def load_adapter_state_dict(self, payload: dict[str, Any]) -> None:
        if payload.get("baseline_name") != self.baseline_name:
            raise ValueError("Generic baseline checkpoint type mismatch.")
        self.ocean_adapter.load_state_dict(payload["adapter"], strict=True)


class _InputBottleneck(nn.Module):
    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.down = nn.Conv2d(channels, hidden_channels, 1)
        self.up = nn.Conv2d(hidden_channels, channels, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.up(F.gelu(self.down(values)))


class GenericInputBottleneckAdapter(_GenericPEFTWrapper):
    """A conventional residual bottleneck placed at the shared model input."""

    baseline_name = "generic_input_bottleneck"

    def __init__(
        self,
        backbone: nn.Module,
        *,
        input_channels: int,
        target_trainable_parameters: int = 229_288,
    ) -> None:
        denominator = 2 * int(input_channels) + 1
        hidden = max(1, round((target_trainable_parameters - input_channels) / denominator))
        adapter = _InputBottleneck(input_channels, hidden)
        super().__init__(backbone, adapter)
        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden)
        self.target_trainable_parameters = int(target_trainable_parameters)

    def forward(self, values: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.backbone(values + self.ocean_adapter(values))


class _InputFiLM(nn.Module):
    def __init__(self, layout: InputLayout, hidden_channels: int) -> None:
        super().__init__()
        self.layout = layout
        features = layout.state_channels + layout.forcing_channels + layout.static_channels
        channels = (
            layout.history_days * layout.state_channels
            + layout.forcing_channels
            + layout.static_channels
        )
        self.conditioner = nn.Sequential(
            nn.Linear(features, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 2 * channels),
        )
        nn.init.zeros_(self.conditioner[-1].weight)
        nn.init.zeros_(self.conditioner[-1].bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        history, forcing, static = split_spatiotemporal_input(values, self.layout)
        history_summary = history.mean(dim=(1, 3, 4))
        forcing_summary = forcing.mean(dim=(2, 3))
        static_summary = static.mean(dim=(2, 3))
        condition = torch.cat([history_summary, forcing_summary, static_summary], dim=1)
        gamma, beta = self.conditioner(condition).chunk(2, dim=1)
        return values * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]


class GenericInputFiLM(_GenericPEFTWrapper):
    """Feature-wise affine modulation conditioned on pooled current inputs."""

    baseline_name = "generic_input_film"

    def __init__(
        self,
        backbone: nn.Module,
        *,
        layout: InputLayout,
        target_trainable_parameters: int = 229_288,
    ) -> None:
        features = layout.state_channels + layout.forcing_channels + layout.static_channels
        channels = layout.history_days * layout.state_channels + layout.forcing_channels + layout.static_channels
        # Linear(features,h) + Linear(h,2*channels), including biases.
        per_hidden = features + 1 + 2 * channels
        hidden = max(1, round((target_trainable_parameters - 2 * channels) / per_hidden))
        adapter = _InputFiLM(layout, hidden)
        super().__init__(backbone, adapter)
        self.layout = layout
        self.hidden_channels = int(hidden)
        self.target_trainable_parameters = int(target_trainable_parameters)

    def forward(self, values: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.backbone(self.ocean_adapter(values))


class _LoRAConvUpdate(nn.Module):
    def __init__(self, base: nn.Conv2d, rank: int) -> None:
        super().__init__()
        self.rank = int(rank)
        self.a = nn.Conv2d(
            base.in_channels,
            rank,
            base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            bias=False,
        )
        self.b = nn.Conv2d(rank, base.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.b(self.a(values)) / self.rank


class _LoRALinearUpdate(nn.Module):
    def __init__(self, base: nn.Linear, rank: int) -> None:
        super().__init__()
        self.rank = int(rank)
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.b(self.a(values)) / self.rank


def _lora_candidates(backbone: nn.Module) -> list[tuple[str, nn.Module, int]]:
    candidates: list[tuple[str, nn.Module, int]] = []
    for name, module in backbone.named_modules():
        if isinstance(module, nn.Conv2d) and module.groups == 1:
            coefficient = module.in_channels * module.kernel_size[0] * module.kernel_size[1] + module.out_channels
            candidates.append((name, module, coefficient))
        elif isinstance(module, nn.Linear):
            candidates.append((name, module, module.in_features + module.out_features))
    return candidates


def _allocate_lora_ranks(coefficients: Iterable[int], budget: int) -> list[int]:
    values = list(coefficients)
    ranks = [0] * len(values)
    if not values:
        raise ValueError("Backbone has no eligible Conv2d/Linear LoRA targets.")
    total = sum(values)
    if total <= budget:
        base_rank = max(1, budget // total)
        ranks = [base_rank] * len(values)
    else:
        # Cover as many layers as possible at rank one before increasing rank.
        remaining = budget
        for index in sorted(range(len(values)), key=lambda item: values[item]):
            if values[index] <= remaining:
                ranks[index] = 1
                remaining -= values[index]
    used = sum(rank * coefficient for rank, coefficient in zip(ranks, values))
    remaining = budget - used
    while True:
        choices = [index for index, value in enumerate(values) if value <= remaining and ranks[index] > 0]
        if not choices:
            break
        index = max(choices, key=lambda item: values[item])
        ranks[index] += 1
        remaining -= values[index]
    return ranks


class GenericBackboneLoRA(_GenericPEFTWrapper):
    """Low-rank updates attached by hooks to eligible Conv2d/Linear layers."""

    baseline_name = "generic_backbone_lora"

    def __init__(self, backbone: nn.Module, *, target_trainable_parameters: int = 229_288) -> None:
        candidates = _lora_candidates(backbone)
        ranks = _allocate_lora_ranks((item[2] for item in candidates), target_trainable_parameters)
        updates = nn.ModuleDict()
        self._handles: list[Any] = []
        self.target_inventory: list[dict[str, Any]] = []
        for index, ((name, module, coefficient), rank) in enumerate(zip(candidates, ranks)):
            if rank <= 0:
                continue
            update = _LoRAConvUpdate(module, rank) if isinstance(module, nn.Conv2d) else _LoRALinearUpdate(module, rank)
            key = f"layer_{index:03d}"
            updates[key] = update

            def hook(_module, inputs, output, *, update_module=update):
                return output + update_module(inputs[0])

            self._handles.append(module.register_forward_hook(hook))
            self.target_inventory.append(
                {"module": name, "rank": rank, "parameters": rank * coefficient}
            )
        if not updates:
            raise ValueError("LoRA budget did not cover any eligible layer.")
        super().__init__(backbone, updates)
        self.target_trainable_parameters = int(target_trainable_parameters)

    def forward(self, values: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.backbone(values)


class _LeadwiseAffine(nn.Module):
    def __init__(self, output_channels: int, horizon: int) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.zeros(horizon, output_channels))
        self.bias = nn.Parameter(torch.zeros(horizon, output_channels))

    def forward(self, tendency: torch.Tensor, lead_day: int | torch.Tensor) -> torch.Tensor:
        index = int(lead_day) - 1 if not torch.is_tensor(lead_day) else int(lead_day.flatten()[0].item()) - 1
        gain = self.gain[index][None, :, None, None]
        bias = self.bias[index][None, :, None, None]
        return tendency * (1.0 + gain) + bias


class LeadwiseAffineBiasCorrection(_GenericPEFTWrapper):
    """Low-capacity lead/channel affine correction of the frozen tendency."""

    baseline_name = "leadwise_affine_bias_correction"

    def __init__(self, backbone: nn.Module, *, output_channels: int = 133, horizon: int = 10) -> None:
        super().__init__(backbone, _LeadwiseAffine(output_channels, horizon))

    def forward(self, values: torch.Tensor, *, lead_day: int | torch.Tensor, **_: Any) -> torch.Tensor:
        return self.ocean_adapter(self.backbone(values), lead_day)
