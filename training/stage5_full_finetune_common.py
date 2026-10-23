"""Shared, frozen semantics for Stage5 registry-model full-finetune baselines."""

from __future__ import annotations

import hashlib
import json
import math
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.core import (
    architecture_name,
    build_model,
    forcing_names,
    history_days,
    input_feature_parts,
    model_channels_in,
    set_config_path,
    sha256_file,
    validate_checkpoint_interface,
)
from pipeline.run_operational_replay import outer_boundary_mask


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PRETRAINED_SHA256 = "897cec7527da976c4ef783809f88157e31336c13923c963ef63ff2026d4e29e2"
VARIABLE_SLICES = {
    "thetao": slice(0, 33),
    "so": slice(33, 66),
    "uo": slice(66, 99),
    "vo": slice(99, 132),
    "zos": slice(132, 133),
}


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = project_path(path)
    return yaml.safe_load(path.read_text(encoding="utf-8-sig"))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True))


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def validate_contract(
    config: dict[str, Any],
    registry: dict[str, Any],
    *,
    allow_released_holdout_for_development_evaluation: bool = False,
) -> None:
    supplemental_after_holdout = config.get("supplemental_after_holdout") is True
    model_name = str(config.get("model_name", ""))
    if model_name not in registry.get("models", {}):
        raise ValueError(f"Unknown full-finetune registry model: {model_name}")
    if not str(config.get("experiment_id", "")).startswith("STAGE5_"):
        raise ValueError("Full-finetune experiment_id must be explicit and Stage5-scoped.")
    if config.get("condition") != "D_ANFC_GFS":
        raise ValueError("Full-finetune baseline must use D_ANFC_GFS.")
    if config.get("full_finetune") is not True or config.get("adapter_enabled") is not False:
        raise ValueError("Full-finetune must train the bare backbone without adapters.")
    if config.get("trainable_scope") != "all_parameters":
        raise ValueError("Full-finetune trainable_scope must be all_parameters.")
    if config.get("normalization_path") != registry["paths"]["normalization"]:
        raise ValueError("Normalization path must be the frozen registry normalization.")
    if config.get("metric_protocol") != registry["protocol"]:
        raise ValueError("Metric protocol differs from the frozen registry protocol.")
    if config.get("primary_evaluation_domain") != "interior":
        raise ValueError("Primary evaluation domain must be interior.")
    if config.get("guard_2025") is not True:
        raise ValueError("2025 guard must remain enabled.")
    if registry["stage_gates"]["stage4"]["status"] != "passed":
        raise ValueError("Stage4 must remain passed.")
    if supplemental_after_holdout:
        if registry["stage_gates"]["stage5"]["status"] != "passed":
            raise ValueError("Post-holdout supplement requires Stage5 passed.")
        if registry["stage_gates"].get("stage6", {}).get("status") != "passed":
            raise ValueError("Post-holdout supplement requires Stage6 passed.")
        if registry["holdout_release"]["approved"] is not True:
            raise ValueError("Post-holdout supplement requires an already released holdout.")
        supplement = config.get("supplemental_protocol", {})
        if supplement.get("claim_status") != "post_holdout_development_robustness":
            raise ValueError("Post-holdout supplement claim status is not frozen correctly.")
        if supplement.get("data_scope") != "train755_val143_only":
            raise ValueError("Post-holdout supplements are restricted to train755/val143.")
        if supplement.get("test355_forbidden") is not True:
            raise ValueError("Post-holdout supplements must explicitly forbid test355.")
        output_dir = Path(str(config.get("output_dir", ""))).as_posix()
        if not output_dir.startswith("runs/adaptation/post_holdout_supplement/"):
            raise ValueError("Post-holdout supplement output must use its isolated output root.")
    elif allow_released_holdout_for_development_evaluation:
        if registry["stage_gates"]["stage5"]["status"] != "passed":
            raise ValueError("Released-holdout development evaluation requires Stage5 passed.")
        if registry["holdout_release"]["approved"] is not True:
            raise ValueError("Released-holdout development evaluation marker is inconsistent.")
    else:
        if registry["stage_gates"]["stage5"]["status"] != "not_passed":
            raise ValueError("Stage5 gate must remain not_passed during this baseline.")
        if registry["holdout_release"]["approved"] is not False:
            raise ValueError("2025 holdout release must remain false.")
    record = registry["models"][model_name]
    if config.get("pretrained_checkpoint") != record["checkpoint"]:
        raise ValueError("Config checkpoint path does not match the registry model record.")
    actual_checkpoint_sha = sha256_file(project_path(record["checkpoint"]))
    if str(config.get("pretrained_checkpoint_sha256")) != actual_checkpoint_sha:
        raise ValueError("Config checkpoint SHA256 does not match the registry model record.")
    model = config.get("model", {})
    frozen = {
        "learning_rate": 1e-5,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "max_epochs": 100,
        "minimum_epochs": 20,
        "early_stopping_patience": 15,
    }
    if supplemental_after_holdout:
        allowed_lrs = config.get("supplemental_protocol", {}).get("learning_rate_candidates", [])
        actual_lr = float(config.get("model", {}).get("learning_rate", math.nan))
        if actual_lr not in {float(value) for value in allowed_lrs}:
            raise ValueError("Supplemental Full-FT learning rate is outside the documented candidate set.")
        frozen.pop("learning_rate")
    for key, expected in frozen.items():
        actual = model.get(key)
        if float(actual) != float(expected):
            raise ValueError(f"Frozen full-finetune setting {key}={actual} != {expected}.")
    schedule = model.get("progressive_horizon_epochs")
    if schedule != {"lead1_epochs": 10, "lead3_epochs": 15, "lead10_epochs": 75}:
        raise ValueError("逐步延长预报步长训练 schedule must be 10/15/75 epochs.")
    expected_weights = {"thetao": 1.75, "so": 1.25, "uo": 1.0, "vo": 1.0, "zos": 1.0}
    if config.get("loss", {}).get("variable_weights") != expected_weights:
        raise ValueError("Variable loss weights differ from the frozen source-training protocol.")


def build_full_model(
    registry: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    model_name = str(config["model_name"])
    record = registry["models"][model_name]
    source_config_path = project_path(record["config"])
    checkpoint_path = project_path(record["checkpoint"])
    actual_checkpoint_sha = sha256_file(checkpoint_path)
    declared_sha = str(config["pretrained_checkpoint_sha256"])
    if actual_checkpoint_sha != declared_sha:
        raise ValueError(
            f"Frozen {model_name} checkpoint SHA256 mismatch: "
            f"registry file={actual_checkpoint_sha}, config={declared_sha}."
        )
    if config["pretrained_checkpoint"] != record["checkpoint"]:
        raise ValueError("Config checkpoint path does not match the registry model record.")
    source_config = set_config_path(load_yaml(source_config_path), source_config_path)
    force_vars = forcing_names(source_config)
    hdays = history_days(source_config)
    cin = model_channels_in(hdays, force_vars, 0)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_checkpoint_interface(
        checkpoint,
        source_config,
        force_vars,
        cin,
        project_path(registry["paths"]["normalization"]),
    )
    model = build_model(
        architecture_name(source_config),
        cin,
        133,
        source_config["model"]["width"],
        source_config.get("model", {}),
    )
    source_keys = sorted(checkpoint["model"])
    load_result = model.load_state_dict(checkpoint["model"], strict=False)
    missing = list(load_result.missing_keys)
    unexpected = list(load_result.unexpected_keys)
    if missing or unexpected:
        raise ValueError(f"Strict checkpoint key inventory failed: missing={missing}, unexpected={unexpected}")
    loaded_hash = state_dict_sha256(model.state_dict())
    source_state_hash = state_dict_sha256(checkpoint["model"])
    if loaded_hash != source_state_hash:
        raise ValueError("Loaded parameter tensor hash does not equal checkpoint parameter hash.")
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.to(device)
    metadata = {
        "source_config_path": source_config_path,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": actual_checkpoint_sha,
        "source_parameter_sha256": source_state_hash,
        "loaded_parameter_sha256": loaded_hash,
        "source_key_count": len(source_keys),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "forcing_variables": list(force_vars),
        "history_days": hdays,
        "cin": cin,
        "architecture": architecture_name(source_config),
        "model_name": model_name,
        "model_display_name": str(record.get("display_name", model_name)),
    }
    return model, metadata


def tensor_sample(sample: dict[str, Any], device: torch.device) -> dict[str, Any]:
    output: dict[str, Any] = {"metadata": sample["metadata"]}
    for key in (
        "anfc_ocean_history",
        "gfs_future_forcing",
        "anfc_truth",
        "anfc_truth_mask",
        "anfc_initial_mask",
        "static",
    ):
        value = torch.as_tensor(sample[key], device=device)
        value = value.to(torch.bool if value.dtype == torch.bool else torch.float32)
        output[key] = value.unsqueeze(0)
    return output


def variable_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    loss_config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    weights = loss_config["variable_weights"]
    unweighted: dict[str, torch.Tensor] = {}
    contributions: dict[str, torch.Tensor] = {}
    total = prediction.new_zeros(())
    total_weight = float(sum(float(weights[name]) for name in VARIABLE_SLICES))
    for name, channels in VARIABLE_SLICES.items():
        error = (prediction[:, channels] - target[:, channels]) ** 2
        mask = valid_mask[:, channels]
        per_channel = (error * mask).sum(dim=(0, 2, 3)) / mask.sum(dim=(0, 2, 3)).clamp_min(1.0)
        value = per_channel.mean()
        contribution = value * float(weights[name]) / total_weight
        unweighted[name] = value
        contributions[name] = contribution
        total = total + contribution
    return total, unweighted, contributions


def persistence_boundary(
    prediction: torch.Tensor,
    initial: torch.Tensor,
    initial_mask: torch.Tensor,
) -> torch.Tensor:
    boundary = torch.as_tensor(
        outer_boundary_mask(prediction.shape[-2], prediction.shape[-1], 2),
        dtype=torch.bool,
        device=prediction.device,
    )
    return torch.where(boundary[None, None], initial, prediction) * initial_mask


def rollout(
    model: torch.nn.Module,
    sample: dict[str, Any],
    *,
    horizon: int,
    loss_config: dict[str, Any],
    return_predictions: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], list[torch.Tensor]]:
    if horizon not in {1, 3, 10}:
        raise ValueError("Full-finetune horizon must be 1, 3 or 10.")
    history = [sample["anfc_ocean_history"][:, index] for index in range(7)]
    initial = history[-1]
    current = initial
    initial_mask = sample["anfc_initial_mask"].to(torch.float32)
    totals: list[torch.Tensor] = []
    per_variable: dict[str, list[torch.Tensor]] = {name: [] for name in VARIABLE_SLICES}
    per_contribution: dict[str, list[torch.Tensor]] = {name: [] for name in VARIABLE_SLICES}
    predictions: list[torch.Tensor] = []
    for lead_index in range(horizon):
        force = sample["gfs_future_forcing"][:, lead_index]
        inputs = torch.cat(input_feature_parts(history, force, sample["static"]), dim=1)
        delta = model(inputs)
        prediction = persistence_boundary((current + delta) * initial_mask, initial, initial_mask)
        target = sample["anfc_truth"][:, lead_index]
        target_mask = sample["anfc_truth_mask"][:, lead_index].to(torch.float32)
        valid_mask = initial_mask * target_mask
        total, variable, contribution = variable_losses(prediction, target, valid_mask, loss_config)
        totals.append(total)
        for name in VARIABLE_SLICES:
            per_variable[name].append(variable[name])
            per_contribution[name].append(contribution[name])
        if return_predictions:
            predictions.append(prediction)
        history = history[1:] + [prediction]
        current = prediction
    return (
        torch.stack(totals).mean(),
        {name: torch.stack(values).mean() for name, values in per_variable.items()},
        {name: torch.stack(values).mean() for name, values in per_contribution.items()},
        predictions,
    )


def autocast_context(device: torch.device, enabled: bool = True):
    if device.type != "cuda" or not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def make_scaler(device: torch.device, enabled: bool = True):
    active = device.type == "cuda" and enabled
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=active)
    return torch.cuda.amp.GradScaler(enabled=active)


def gradient_module_names(model: torch.nn.Module) -> tuple[str, ...]:
    return tuple(
        name
        for name, module in model.named_children()
        if any(True for _ in module.parameters())
    )


def module_gradient_summary(model: torch.nn.Module) -> dict[str, Any]:
    modules = {
        name: module
        for name, module in model.named_children()
        if any(True for _ in module.parameters())
    }
    output: dict[str, Any] = {}
    for name, module in modules.items():
        gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
        finite = all(bool(torch.isfinite(value).all()) for value in gradients)
        active = sum(int(torch.count_nonzero(value).item() > 0) for value in gradients)
        norm = math.sqrt(sum(float(torch.sum(value.detach().float() ** 2).cpu()) for value in gradients))
        output[f"{name}_gradient_tensors"] = len(gradients)
        output[f"{name}_active_gradient_tensors"] = active
        output[f"{name}_gradient_norm"] = norm
        output[f"{name}_gradients_finite"] = finite
    return output
