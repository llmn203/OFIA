"""Stage5 adapter-only trainer for frozen V1 and bounded V2 experiments.

This entry point deliberately supports forecast loss only.  It rejects 2025,
paired-alignment materialization, full fine-tuning and FH7.  Lead gates are
accepted only by the separately scoped Adapter V2 smoke experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import yaml

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.adapters.dual_source_adapter_wrapper import DualSourceAdapterWrapper
from pipeline.adapters.dual_source_adapter_wrapper_v2 import DualSourceAdapterWrapperV2
from pipeline.adapters.dual_source_adapter_wrapper_phda import (
    ProvenanceHorizonDualSourceAdapterWrapper,
)
from pipeline.adapters.dual_source_adapter_wrapper_phda_v2 import (
    FrameAgeHorizonDualSourceAdapterWrapper,
)
from pipeline.adapters.dual_interface_spatiotemporal_adapter import (
    DualInterfaceSpatiotemporalAdapterWrapper,
)
from pipeline.adapters.dista_path_ablation import DISTAPathAblationWrapper, PATH_ABLATIONS
from pipeline.adapters.dista_theory_ablation import (
    DISTATheoryAblationWrapper,
    THEORY_ABLATIONS,
)
from pipeline.adapters.generic_peft_baselines import (
    GenericBackboneLoRA,
    GenericInputBottleneckAdapter,
    GenericInputFiLM,
    LeadwiseAffineBiasCorrection,
)
from pipeline.core import (
    architecture_name,
    build_model,
    config_sha256,
    forcing_names,
    history_days,
    input_feature_parts,
    load_stats,
    model_channels_in,
    set_config_path,
    sha256_file,
    validate_checkpoint_interface,
)
from pipeline.run_operational_replay import outer_boundary_mask
from pipeline.stage5_paired_domain_dataset import Stage5PairedDomainDataset
from pipeline.train_hindcast_fast import variable_depth_loss
from models.spatiotemporal_utils import infer_input_layout


ROOT = Path(__file__).resolve().parents[1]
MODES = {"source_only", "ocean_only", "forcing_only", "dual_adapter"}
EXECUTION_SCOPES = {
    "stage5_2_smoke3": {
        "work_package": "stage5_2_adapter_only_smoke",
        "max_train_cycles": 3,
        "max_val_cycles": 3,
        "max_epochs": 2,
        "max_optimizer_steps": 4,
        "checkpoint_serialization": "full_wrapper_legacy",
    },
    "stage5_2b_pilot30": {
        "work_package": "stage5_2b_adapter_pilot30",
        "max_train_cycles": 30,
        "max_val_cycles": 30,
        "max_epochs": 3,
        "max_optimizer_steps": 90,
        "checkpoint_serialization": "adapter_only",
    },
    "stage5_2d_formal_train755": {
        "work_package": "stage5_2d_adapter_formal_train755",
        "max_train_cycles": 755,
        "max_val_cycles": 143,
        "max_epochs": 100,
        "max_optimizer_steps": 75500,
        "checkpoint_serialization": "adapter_only",
    },
    "stage5_3a_adapter_v2_smoke3": {
        "work_package": "stage5_3a_lead_conditioned_adapter_v2_smoke",
        "max_train_cycles": 3,
        "max_val_cycles": 3,
        "max_epochs": 2,
        "max_optimizer_steps": 4,
        "checkpoint_serialization": "adapter_only_v2",
    },
    "stage5_3b_phda_smoke3": {
        "work_package": "stage5_3b_provenance_horizon_dual_source_adapter_smoke",
        "max_train_cycles": 3,
        "max_val_cycles": 3,
        "max_epochs": 2,
        "max_optimizer_steps": 4,
        "checkpoint_serialization": "adapter_only_phda_v1",
    },
    "stage5_3c_phda_formal_train755": {
        "work_package": "stage5_3c_phda_frame_age_formal_train755",
        "max_train_cycles": 755,
        "max_val_cycles": 143,
        "max_epochs": 100,
        "max_optimizer_steps": 75500,
        "checkpoint_serialization": "adapter_only_phda_v2",
    },
    "stage5_4a_dista_base_smoke3": {
        "work_package": "stage5_4a_dista_base_smoke3",
        "max_train_cycles": 3,
        "max_val_cycles": 3,
        "max_epochs": 2,
        "max_optimizer_steps": 4,
        "checkpoint_serialization": "adapter_only_dista_base",
    },
    "stage5_4b_dista_base_formal_train755": {
        "work_package": "stage5_4b_dista_base_formal_train755",
        "max_train_cycles": 755,
        "max_val_cycles": 143,
        "max_epochs": 100,
        "max_optimizer_steps": 75500,
        "checkpoint_serialization": "adapter_only_dista_base",
    },
    "stage5_5a_dista_path_ablation_smoke3": {
        "work_package": "stage5_5a_dista_path_ablation_smoke3",
        "max_train_cycles": 3,
        "max_val_cycles": 3,
        "max_epochs": 2,
        "max_optimizer_steps": 4,
        "checkpoint_serialization": "adapter_only_dista_path_ablation",
    },
    "stage5_5b_dista_path_ablation_formal_train755": {
        "work_package": "stage5_5b_dista_path_ablation_formal_train755",
        "max_train_cycles": 755,
        "max_val_cycles": 143,
        "max_epochs": 100,
        "max_optimizer_steps": 75500,
        "checkpoint_serialization": "adapter_only_dista_path_ablation",
    },
    "stage5_5c_dista_theory_ablation_formal_train755": {
        "work_package": "stage5_5c_dista_theory_ablation_formal_train755",
        "max_train_cycles": 755,
        "max_val_cycles": 143,
        "max_epochs": 100,
        "max_optimizer_steps": 75500,
        "checkpoint_serialization": "adapter_only_dista_theory_ablation",
    },
    "stage5_postholdout_generic_baseline_formal_train755": {
        "work_package": "stage5_postholdout_generic_baseline_formal_train755",
        "max_train_cycles": 755,
        "max_val_cycles": 143,
        "max_epochs": 100,
        "max_optimizer_steps": 75500,
        "checkpoint_serialization": "adapter_only_generic_baseline",
    },
}
FORMAL_SCOPES = {
    "stage5_2d_formal_train755",
    "stage5_3c_phda_formal_train755",
    "stage5_4b_dista_base_formal_train755",
    "stage5_5b_dista_path_ablation_formal_train755",
    "stage5_5c_dista_theory_ablation_formal_train755",
    "stage5_postholdout_generic_baseline_formal_train755",
}
GENERIC_BASELINE_VERSIONS = {
    "generic_input_bottleneck",
    "generic_input_film",
    "generic_backbone_lora",
    "leadwise_affine_bias_correction",
}
REQUIRED_FILES = (
    "config_snapshot.yaml",
    "run_manifest.json",
    "data_manifest.json",
    "checkpoint_manifest.json",
    "training_log.csv",
    "validation_log.csv",
    "gradient_audit.csv",
    "runtime_summary.json",
    "checkpoint_best.pt",
    "checkpoint_last.pt",
    "resume_report.json",
    "qc_report.md",
)


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True))


def atomic_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def autocast_context(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def make_scaler(device: torch.device):
    enabled = device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8-sig"))


def validate_stage5_contract(
    config: dict[str, Any],
    registry: dict[str, Any],
    *,
    allow_released_holdout_for_development_evaluation: bool = False,
) -> None:
    mode = str(config.get("mode"))
    scope = str(config.get("execution_scope", "stage5_2_smoke3"))
    adapter_version = str(config.get("adapter_version", "v1"))
    supplemental_after_holdout = config.get("supplemental_after_holdout") is True
    if scope not in EXECUTION_SCOPES:
        raise ValueError(f"Unsupported Stage5 execution_scope: {scope}")
    expected_package = EXECUTION_SCOPES[scope]["work_package"]
    if int(config.get("stage", -1)) != 5 or config.get("work_package") != expected_package:
        raise ValueError(f"execution_scope={scope} requires work_package={expected_package}.")
    if mode not in MODES:
        raise ValueError(f"Unsupported Stage5 mode: {mode}")
    if config.get("materialization_mode") != "target_forecast_only":
        raise ValueError("Stage5.2 training must use materialization_mode=target_forecast_only.")
    policies = config.get("policies", {})
    if policies.get("paired_alignment") or policies.get("fh7"):
        raise ValueError("Stage5 Adapter training forbids alignment and FH7 in this entry point.")
    if scope == "stage5_3a_adapter_v2_smoke3":
        if adapter_version != "v2" or policies.get("lead_conditioned_gate") is not True:
            raise ValueError("Stage5.3a requires adapter_version=v2 and lead_conditioned_gate=true.")
        if policies.get("lead1_adapter_policy") != "hard_bypass":
            raise ValueError("Adapter V2 requires lead1_adapter_policy=hard_bypass.")
        if policies.get("lead_gate_constraint") != "monotone_0_to_1":
            raise ValueError("Adapter V2 requires a monotone_0_to_1 lead gate.")
    elif scope == "stage5_3b_phda_smoke3":
        if adapter_version != "phda_v1" or policies.get("lead_conditioned_gate") is not True:
            raise ValueError("Stage5.3b requires adapter_version=phda_v1 and lead_conditioned_gate=true.")
        if policies.get("gate_type") != "smooth_bounded_provenance_horizon":
            raise ValueError("PHDA requires gate_type=smooth_bounded_provenance_horizon.")
        if policies.get("lead1_adapter_policy") != "learned_bounded_floor":
            raise ValueError("PHDA forbids a hard Day-1 bypass.")
        if policies.get("ocean_gate_inputs") != ["lead_day", "observed_fraction"]:
            raise ValueError("PHDA ocean gate must use lead_day and observed_fraction.")
        if policies.get("forcing_gate_inputs") != ["lead_day"]:
            raise ValueError("PHDA forcing gate must use lead_day only.")
    elif scope == "stage5_3c_phda_formal_train755":
        if adapter_version != "phda_v2" or policies.get("lead_conditioned_gate") is not True:
            raise ValueError("Stage5.3c requires adapter_version=phda_v2 and lead_conditioned_gate=true.")
        if policies.get("gate_type") != "smooth_bounded_frame_age_horizon":
            raise ValueError("PHDA V2 requires gate_type=smooth_bounded_frame_age_horizon.")
        if policies.get("lead1_adapter_policy") != "learned_bounded_floor":
            raise ValueError("PHDA V2 forbids a hard Day-1 bypass.")
        if policies.get("ocean_gate_inputs") != ["lead_day", "frame_age"]:
            raise ValueError("PHDA V2 ocean gate must use lead_day and frame_age.")
        if policies.get("forcing_gate_inputs") != ["lead_day"]:
            raise ValueError("PHDA V2 forcing gate must use lead_day only.")
    elif scope in {
        "stage5_4a_dista_base_smoke3",
        "stage5_4b_dista_base_formal_train755",
    }:
        if adapter_version != "dista_base":
            raise ValueError("Stage5.4 DISTA scopes require adapter_version=dista_base.")
        if policies.get("adapter_placement") != "physical_input_and_tendency_output":
            raise ValueError("DISTA must adapt the shared physical input and tendency-output interfaces.")
        if policies.get("forcing_trajectory_policy") != "full_10day_physical_channels":
            raise ValueError("DISTA requires full 10-day physical forcing trajectory adaptation.")
        if policies.get("calendar_channel_policy") != "bitwise_passthrough":
            raise ValueError("DISTA calendar channels must be bitwise passthrough.")
        if policies.get("output_adapter") is not True:
            raise ValueError("DISTA Base requires the output tendency Adapter.")
    elif scope in {
        "stage5_5a_dista_path_ablation_smoke3",
        "stage5_5b_dista_path_ablation_formal_train755",
    }:
        if adapter_version != "dista_path_ablation":
            raise ValueError("Stage5.5 requires adapter_version=dista_path_ablation.")
        if mode != "dual_adapter":
            raise ValueError("Stage5.5 uses dual_adapter mode with explicit path_ablation switches.")
        ablation = str(config.get("adapter", {}).get("path_ablation", ""))
        if ablation not in {"ocean", "forcing", "output", "ocean_forcing", "full_no_coupled"}:
            raise ValueError("Stage5.5 requires one of ocean/forcing/output/ocean_forcing/full_no_coupled.")
        if policies.get("adapter_placement") != "physical_input_and_tendency_output":
            raise ValueError("Stage5.5 must retain the DISTA physical interface placement.")
        if policies.get("forcing_trajectory_policy") != "full_10day_physical_channels":
            raise ValueError("Stage5.5 requires full 10-day physical forcing trajectory adaptation.")
        if policies.get("calendar_channel_policy") != "bitwise_passthrough":
            raise ValueError("Stage5.5 calendar channels must be bitwise passthrough.")
    elif scope == "stage5_5c_dista_theory_ablation_formal_train755":
        if adapter_version != "dista_theory_ablation" or mode != "dual_adapter":
            raise ValueError("Stage5.5c requires dual_adapter with adapter_version=dista_theory_ablation.")
        ablation = str(config.get("adapter", {}).get("theory_ablation", ""))
        if ablation not in THEORY_ABLATIONS:
            raise ValueError(f"Stage5.5c theory ablation must be one of {sorted(THEORY_ABLATIONS)}.")
        expected_forcing = (
            "stepwise_physical_channels"
            if ablation == "stepwise_forcing_context"
            else "full_10day_physical_channels"
        )
        expected_ocean = (
            "all_history_frames"
            if ablation == "all_history_frames"
            else "observed_product_frames_only"
        )
        if policies.get("forcing_trajectory_policy") != expected_forcing:
            raise ValueError("Stage5.5c forcing policy does not match its theory ablation.")
        if policies.get("ocean_adapter_history_policy") != expected_ocean:
            raise ValueError("Stage5.5c ocean-history policy does not match its theory ablation.")
        if policies.get("adapter_placement") != "physical_input_and_tendency_output":
            raise ValueError("Stage5.5c must retain all three DISTA correction interfaces.")
        if policies.get("calendar_channel_policy") != "bitwise_passthrough":
            raise ValueError("Stage5.5c calendar channels must remain bitwise passthrough.")
        if policies.get("output_adapter") is not True:
            raise ValueError("Stage5.5c must retain the output tendency Adapter.")
        if policies.get("lead_parameterization") != "continuous_progress_features":
            raise ValueError("Stage5.5c must retain the continuous lead parameterization.")
    elif scope == "stage5_postholdout_generic_baseline_formal_train755":
        if not supplemental_after_holdout:
            raise ValueError("Generic comparison baselines are post-holdout supplements only.")
        if adapter_version not in GENERIC_BASELINE_VERSIONS:
            raise ValueError("Unsupported generic comparison baseline.")
        if mode != "dual_adapter":
            raise ValueError("Generic comparison baselines use dual_adapter as the trainer compatibility mode.")
        if policies.get("generic_baseline") != adapter_version:
            raise ValueError("Generic baseline policy does not match adapter_version.")
        if policies.get("dista_mechanism_reused") is not False:
            raise ValueError("Generic baselines must not reuse the DISTA mechanism.")
        if policies.get("ocean_adapter_history_policy") != "generic_full_rolling_history":
            raise ValueError("Generic input baselines must use their explicit rolling-history policy.")
    elif adapter_version != "v1" or policies.get("lead_conditioned_gate"):
        raise ValueError("Frozen Stage5.2 scopes accept only Adapter V1 without lead gates.")
    if policies.get("valid_mask") != "initial_mask AND target_mask":
        raise ValueError("Stage5.2 valid mask must be initial_mask AND target_mask.")
    if (
        scope not in {
            "stage5_5c_dista_theory_ablation_formal_train755",
            "stage5_postholdout_generic_baseline_formal_train755",
        }
        and policies.get("ocean_adapter_history_policy") != "observed_product_frames_only"
    ):
        raise ValueError("Ocean adapter history policy is not frozen correctly.")
    gates = registry.get("stage_gates", {})
    if gates.get("stage4", {}).get("status") != "passed":
        raise ValueError("Stage4 must be passed before Stage5.2 smoke.")
    stage5_status = gates.get("stage5", {}).get("status")
    if supplemental_after_holdout:
        if stage5_status != "passed" or gates.get("stage6", {}).get("status") != "passed":
            raise ValueError("Post-holdout supplements require Stage5 and Stage6 gates to be passed.")
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
    elif stage5_status != "not_passed" and not allow_released_holdout_for_development_evaluation:
        raise ValueError("Stage5 gate must remain not_passed during this work package.")
    holdout_approved = registry.get("holdout_release", {}).get("approved")
    if not isinstance(holdout_approved, bool):
        raise ValueError("holdout_release.approved must be an explicit boolean.")
    if supplemental_after_holdout and holdout_approved is not True:
        raise ValueError("Post-holdout supplement marker requires the holdout to have already been released.")
    if holdout_approved and not (allow_released_holdout_for_development_evaluation or supplemental_after_holdout):
        raise ValueError("Holdout release must remain false.")
    if scope in FORMAL_SCOPES:
        if mode not in {"dual_adapter", "ocean_only", "forcing_only"}:
            raise ValueError(
                "Formal train755/val143 supports only the registered Adapter ablations: "
                "dual_adapter, ocean_only and forcing_only."
            )
        dista_formal = scope in {
            "stage5_4b_dista_base_formal_train755",
            "stage5_5b_dista_path_ablation_formal_train755",
            "stage5_5c_dista_theory_ablation_formal_train755",
            "stage5_postholdout_generic_baseline_formal_train755",
        }
        expected_model = {
            "lr": 3e-4 if dista_formal else 1e-3,
            "weight_decay": 1e-4 if dista_formal else 0.0,
            "gradient_clip_norm": 1.0,
            "max_epochs": 100,
            "minimum_epochs": 20,
            "early_stopping_patience": 15,
            "early_stopping_min_delta": 0.0,
        }
        model = config.get("model", {})
        if supplemental_after_holdout:
            allowed_lrs = config.get("supplemental_protocol", {}).get("learning_rate_candidates", [])
            actual_lr = float(config.get("model", {}).get("lr", math.nan))
            if actual_lr not in {float(value) for value in allowed_lrs}:
                raise ValueError("Supplemental DISTA learning rate is outside the documented candidate set.")
            expected_model.pop("lr")
        for key, expected in expected_model.items():
            if float(model.get(key, math.nan)) != float(expected):
                raise ValueError(f"Frozen formal Adapter setting {key}={model.get(key)} != {expected}.")
        if model.get("progressive_horizon_epochs") != {
            "lead1_epochs": 10,
            "lead3_epochs": 15,
            "lead10_epochs": 75,
        }:
            raise ValueError("Formal Adapter horizon schedule must be 10/15/75 epochs.")
        if config.get("train_manifest") != "configs/cycle_manifests/stage5_full_finetune_train755.csv":
            raise ValueError("Formal Adapter train manifest must reuse the frozen train755 split.")
        if config.get("val_manifest") != "configs/cycle_manifests/stage5_full_finetune_val143.csv":
            raise ValueError("Formal Adapter val manifest must reuse the frozen val143 split.")


def build_wrapper(
    registry: dict[str, Any],
    mode: str,
    device: torch.device,
    adapter_version: str = "v1",
    source_model: str = "temporal_transformer",
    adapter_config: dict[str, Any] | None = None,
) -> tuple[
    DualSourceAdapterWrapper
    | DualSourceAdapterWrapperV2
    | ProvenanceHorizonDualSourceAdapterWrapper
    | FrameAgeHorizonDualSourceAdapterWrapper
    | DualInterfaceSpatiotemporalAdapterWrapper
    | DISTAPathAblationWrapper
    | DISTATheoryAblationWrapper,
    dict[str, Any],
]:
    if source_model not in registry["models"]:
        raise ValueError(f"Unknown source_model in registry: {source_model}")
    model_record = registry["models"][source_model]
    model_config_path = project_path(model_record["config"])
    checkpoint_path = project_path(model_record["checkpoint"])
    model_config = set_config_path(load_yaml(model_config_path), model_config_path)
    force_vars = forcing_names(model_config)
    hdays = history_days(model_config)
    cin = model_channels_in(hdays, force_vars, 0)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_checkpoint_interface(
        checkpoint,
        model_config,
        force_vars,
        cin,
        project_path(registry["paths"]["normalization"]),
    )
    backbone = build_model(
        architecture_name(model_config),
        cin,
        133,
        model_config["model"]["width"],
        model_config.get("model", {}),
    )
    backbone.load_state_dict(checkpoint["model"], strict=True)
    if adapter_version == "v1":
        wrapper = DualSourceAdapterWrapper(backbone)
    elif adapter_version == "v2":
        wrapper = DualSourceAdapterWrapperV2(backbone)
    elif adapter_version == "phda_v1":
        wrapper = ProvenanceHorizonDualSourceAdapterWrapper(backbone)
    elif adapter_version == "phda_v2":
        wrapper = FrameAgeHorizonDualSourceAdapterWrapper(backbone)
    elif adapter_version == "dista_base":
        dista = dict(adapter_config or {})
        wrapper = DualInterfaceSpatiotemporalAdapterWrapper(
            backbone,
            context_channels=int(dista.get("context_channels", 64)),
            ocean_hidden_channels=int(dista.get("ocean_hidden_channels", 64)),
            forcing_hidden_channels=int(dista.get("forcing_hidden_channels", 32)),
            output_hidden_channels=int(dista.get("output_hidden_channels", 64)),
            layout=getattr(backbone, "layout", infer_input_layout(cin, 133)),
        )
    elif adapter_version == "dista_path_ablation":
        dista = dict(adapter_config or {})
        ablation = "source" if mode == "source_only" else str(dista.get("path_ablation", ""))
        if ablation not in PATH_ABLATIONS:
            raise ValueError(f"Unsupported DISTA path ablation: {ablation}")
        wrapper = DISTAPathAblationWrapper(
            backbone,
            context_channels=int(dista.get("context_channels", 64)),
            ocean_hidden_channels=int(dista.get("ocean_hidden_channels", 64)),
            forcing_hidden_channels=int(dista.get("forcing_hidden_channels", 32)),
            output_hidden_channels=int(dista.get("output_hidden_channels", 64)),
            layout=getattr(backbone, "layout", infer_input_layout(cin, 133)),
            ablation=ablation,
        )
    elif adapter_version == "dista_theory_ablation":
        dista = dict(adapter_config or {})
        ablation = str(dista.get("theory_ablation", ""))
        if ablation not in THEORY_ABLATIONS:
            raise ValueError(f"Unsupported DISTA theory ablation: {ablation}")
        common = {
            "context_channels": int(dista.get("context_channels", 64)),
            "ocean_hidden_channels": int(dista.get("ocean_hidden_channels", 64)),
            "forcing_hidden_channels": int(dista.get("forcing_hidden_channels", 32)),
            "output_hidden_channels": int(dista.get("output_hidden_channels", 64)),
            "layout": getattr(backbone, "layout", infer_input_layout(cin, 133)),
        }
        if mode == "source_only":
            wrapper = DualInterfaceSpatiotemporalAdapterWrapper(backbone, **common)
            wrapper.configure_adaptation_mode("source_only")
        else:
            wrapper = DISTATheoryAblationWrapper(
                backbone,
                theory_ablation=ablation,
                **common,
            )
    elif adapter_version == "generic_input_bottleneck":
        generic = dict(adapter_config or {})
        wrapper = GenericInputBottleneckAdapter(
            backbone,
            input_channels=cin,
            target_trainable_parameters=int(generic.get("target_trainable_parameters", 229_288)),
        )
    elif adapter_version == "generic_input_film":
        generic = dict(adapter_config or {})
        wrapper = GenericInputFiLM(
            backbone,
            layout=getattr(backbone, "layout", infer_input_layout(cin, 133)),
            target_trainable_parameters=int(generic.get("target_trainable_parameters", 229_288)),
        )
    elif adapter_version == "generic_backbone_lora":
        generic = dict(adapter_config or {})
        wrapper = GenericBackboneLoRA(
            backbone,
            target_trainable_parameters=int(generic.get("target_trainable_parameters", 229_288)),
        )
    elif adapter_version == "leadwise_affine_bias_correction":
        wrapper = LeadwiseAffineBiasCorrection(backbone, output_channels=133, horizon=10)
    else:
        raise ValueError(f"Unsupported adapter_version: {adapter_version}")
    # Path ablations configure residual application independently from context
    # encoder trainability. Calling the legacy mode switch here would destroy
    # that orthogonality, so this branch deliberately keeps its own switches.
    if (
        adapter_version not in {"dista_path_ablation", "dista_theory_ablation"}
        and hasattr(wrapper, "configure_adaptation_mode")
    ):
        wrapper.configure_adaptation_mode(mode)
    wrapper.to(device)
    metadata = {
        "model_config": model_config,
        "model_config_path": model_config_path,
        "checkpoint_path": checkpoint_path,
        "checkpoint": checkpoint,
        "forcing_variables": list(force_vars),
        "history_days": hdays,
        "cin": cin,
        "adapter_version": adapter_version,
        "source_model": source_model,
        "path_ablation": getattr(wrapper, "path_ablation", None),
        "theory_ablation": getattr(wrapper, "theory_ablation", None),
    }
    return wrapper, metadata


def tensor_sample(sample: dict[str, Any], device: torch.device) -> dict[str, Any]:
    output = {"metadata": sample["metadata"]}
    for key in (
        "anfc_ocean_history",
        "gfs_future_forcing",
        "anfc_truth",
        "anfc_truth_mask",
        "anfc_initial_mask",
        "static",
    ):
        value = torch.as_tensor(sample[key], device=device)
        if value.dtype == torch.bool:
            value = value.to(torch.bool)
        else:
            value = value.to(torch.float32)
        output[key] = value.unsqueeze(0)
    return output


def apply_persistence_boundary(
    prediction: torch.Tensor,
    initial: torch.Tensor,
    initial_mask: torch.Tensor,
    boundary_mask: torch.Tensor,
) -> torch.Tensor:
    output = torch.where(boundary_mask[None, None], initial, prediction)
    return output * initial_mask


def rollout(
    wrapper: DualSourceAdapterWrapper,
    sample: dict[str, Any],
    *,
    horizon: int,
    loss_config: dict[str, Any],
    return_predictions: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor], list[int]]:
    if not 1 <= horizon <= 10:
        raise ValueError("Stage5.2 rollout horizon must be 1..10.")
    history = [sample["anfc_ocean_history"][:, index] for index in range(7)]
    initial = history[-1]
    current = initial
    initial_mask = sample["anfc_initial_mask"].to(torch.float32)
    static = sample["static"]
    observed = torch.ones((initial.shape[0], 7), dtype=torch.bool, device=initial.device)
    dista_cycle_context = None
    forcing_trajectory = sample["gfs_future_forcing"]
    if isinstance(wrapper, DualInterfaceSpatiotemporalAdapterWrapper):
        forcing_trajectory, dista_cycle_context = wrapper.prepare_cycle(
            torch.stack(history, dim=1),
            forcing_trajectory,
            static,
            observed,
        )
    boundary_np = outer_boundary_mask(initial.shape[-2], initial.shape[-1], 2)
    boundary = torch.as_tensor(boundary_np, dtype=torch.bool, device=initial.device)
    losses: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    observed_counts: list[int] = []
    for lead_index in range(horizon):
        force = forcing_trajectory[:, lead_index]
        inputs = torch.cat(input_feature_parts(history, force, static), dim=1)
        if isinstance(wrapper, DualInterfaceSpatiotemporalAdapterWrapper):
            delta = wrapper(
                inputs,
                lead_day=lead_index + 1,
                observed_history_mask=observed,
                cycle_context=dista_cycle_context,
                forcing_pre_adapted=True,
            )
        else:
            delta = wrapper(inputs, lead_day=lead_index + 1, observed_history_mask=observed)
        prediction = (current + delta) * initial_mask
        prediction = apply_persistence_boundary(prediction, initial, initial_mask, boundary)
        target = sample["anfc_truth"][:, lead_index]
        target_mask = sample["anfc_truth_mask"][:, lead_index].to(torch.float32)
        valid_mask = initial_mask * target_mask
        losses.append(variable_depth_loss(prediction, target, valid_mask, loss_config))
        if return_predictions:
            predictions.append(prediction)
        observed_counts.append(int(observed.sum(dim=1)[0].item()))
        history = history[1:] + [prediction]
        current = prediction
        observed = torch.cat(
            [observed[:, 1:], torch.zeros((observed.shape[0], 1), dtype=torch.bool, device=observed.device)],
            dim=1,
        )
    return torch.stack(losses).mean(), predictions, observed_counts


def gradient_counts(wrapper: DualSourceAdapterWrapper) -> dict[str, Any]:
    def component(module: torch.nn.Module) -> tuple[int, float]:
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        active = sum(int(torch.count_nonzero(g).item() > 0) for g in gradients)
        norm = math.sqrt(sum(float(torch.sum(g.detach().float() ** 2).cpu()) for g in gradients))
        return active, norm

    ocean_count, ocean_norm = component(wrapper.ocean_adapter)
    forcing_count, forcing_norm = component(wrapper.forcing_adapter)
    output_count, output_norm = (
        component(wrapper.output_adapter)
        if hasattr(wrapper, "output_adapter")
        else (0, 0.0)
    )
    context_count, context_norm = (
        component(wrapper.context_fusion)
        if hasattr(wrapper, "context_fusion")
        else (0, 0.0)
    )
    backbone_count, backbone_norm = component(wrapper.backbone)
    result = {
        "ocean_gradient_parameter_count": ocean_count,
        "ocean_gradient_norm": ocean_norm,
        "forcing_gradient_parameter_count": forcing_count,
        "forcing_gradient_norm": forcing_norm,
        "output_gradient_parameter_count": output_count,
        "output_gradient_norm": output_norm,
        "context_gradient_parameter_count": context_count,
        "context_gradient_norm": context_norm,
        "backbone_gradient_count": backbone_count,
        "backbone_gradient_norm": backbone_norm,
    }
    if isinstance(wrapper, DISTAPathAblationWrapper):
        def correction(module: torch.nn.Module) -> tuple[int, float]:
            parameters = []
            for name in ("blocks", "output_projection", "detail"):
                parameters.extend(getattr(module, name).parameters())
            return component_parameters(parameters)

        def context_encoder(module: torch.nn.Module) -> tuple[int, float]:
            parameters = list(module.input_projection.parameters()) + list(module.temporal.parameters())
            return component_parameters(parameters)

        def component_parameters(parameters) -> tuple[int, float]:
            gradients = [p.grad for p in parameters if p.grad is not None]
            active = sum(int(torch.count_nonzero(g).item() > 0) for g in gradients)
            norm = math.sqrt(sum(float(torch.sum(g.detach().float() ** 2).cpu()) for g in gradients))
            return active, norm

        ocean_head, _ = correction(wrapper.ocean_adapter)
        forcing_head, _ = correction(wrapper.forcing_adapter)
        ocean_context, _ = context_encoder(wrapper.ocean_adapter)
        forcing_context, _ = context_encoder(wrapper.forcing_adapter)
        result.update(
            {
                "ocean_correction_gradient_parameter_count": ocean_head,
                "forcing_correction_gradient_parameter_count": forcing_head,
                "ocean_context_gradient_parameter_count": ocean_context,
                "forcing_context_gradient_parameter_count": forcing_context,
            }
        )
    return result


def alpha_values(wrapper: DualSourceAdapterWrapper) -> dict[str, list[float]]:
    if hasattr(wrapper, "diagnostic_values"):
        return wrapper.diagnostic_values()
    return {
        "ocean": [
            value
            for branch in wrapper.ocean_adapter.branches.values()
            for value in branch.alpha.detach().cpu().reshape(-1).tolist()
        ],
        "forcing": wrapper.forcing_adapter.alpha.detach().cpu().reshape(-1).tolist(),
    }


def alpha_summary(wrapper: DualSourceAdapterWrapper) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for name, values in alpha_values(wrapper).items():
        array = np.asarray(values, dtype=np.float64)
        output[name] = {
            "mean": float(array.mean()) if array.size else math.nan,
            "abs_mean": float(np.abs(array).mean()) if array.size else math.nan,
            "max_abs": float(np.abs(array).max()) if array.size else math.nan,
        }
    return output


def trainable_parameter_groups(wrapper: DualSourceAdapterWrapper, lr: float, weight_decay: float):
    if isinstance(wrapper, DISTAPathAblationWrapper):
        parameters = [parameter for parameter in wrapper.parameters() if parameter.requires_grad]
        return ([{
            "params": parameters,
            "lr": lr,
            "weight_decay": weight_decay,
            "name": f"dista_path_ablation:{wrapper.path_ablation}",
        }] if parameters else [])
    groups = []
    if wrapper.ocean_adapter_enabled:
        groups.append(
            {
                "params": list(wrapper.ocean_adapter.parameters()),
                "lr": lr,
                "weight_decay": weight_decay,
                "name": "ocean_adapter",
            }
        )
    if wrapper.forcing_adapter_enabled:
        groups.append(
            {
                "params": list(wrapper.forcing_adapter.parameters()),
                "lr": lr,
                "weight_decay": weight_decay,
                "name": "forcing_adapter",
            }
        )
    if getattr(wrapper, "output_adapter_enabled", False):
        groups.append(
            {
                "params": list(wrapper.output_adapter.parameters()),
                "lr": lr,
                "weight_decay": weight_decay,
                "name": "output_adapter",
            }
        )
    if hasattr(wrapper, "context_fusion") and any(
        (
            wrapper.ocean_adapter_enabled,
            wrapper.forcing_adapter_enabled,
            getattr(wrapper, "output_adapter_enabled", False),
        )
    ):
        groups.append(
            {
                "params": list(wrapper.context_fusion.parameters()),
                "lr": lr,
                "weight_decay": weight_decay,
                "name": "context_adapter",
            }
        )
    return groups


def run_gradient_audits(
    wrapper: DualSourceAdapterWrapper,
    sample: dict[str, Any],
    loss_config: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[list[dict[str, Any]], float]:
    rows = []
    wrapper.train()
    for label, horizon in (("single_lead_backward", 1), ("full_10day_backward", 10)):
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with autocast_context(device):
            loss, _, observed_counts = rollout(wrapper, sample, horizon=horizon, loss_config=loss_config)
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        gradients = gradient_counts(wrapper)
        rows.append(
            {
                "audit": label,
                "horizon": horizon,
                "loss": float(loss.detach().cpu()),
                "seconds": elapsed,
                "observed_history_counts": ";".join(map(str, observed_counts)),
                **gradients,
                "finite": bool(torch.isfinite(loss).item()),
            }
        )
    optimizer.zero_grad(set_to_none=True)
    return rows, rows[-1]["seconds"]


def micro_overfit(
    registry: dict[str, Any],
    mode: str,
    sample: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    wrapper, _ = build_wrapper(
        registry,
        mode,
        device,
        str(config.get("adapter_version", "v1")),
        str(config.get("source_model", "temporal_transformer")),
        config.get("adapter"),
    )
    model_cfg = config["model"]
    groups = trainable_parameter_groups(wrapper, float(model_cfg["lr"]), float(model_cfg.get("weight_decay", 0.0)))
    optimizer = torch.optim.AdamW(groups)
    steps = min(int(model_cfg.get("micro_overfit_steps", 10)), 20)
    losses = []
    alpha_before = alpha_values(wrapper)
    micro_horizon = 3 if str(config.get("adapter_version", "v1")) != "v1" else 1
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            loss, _, _ = rollout(
                wrapper,
                sample,
                horizon=micro_horizon,
                loss_config=config.get("loss", {}),
            )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    alpha_after = alpha_values(wrapper)
    return {
        "steps": steps,
        "horizon": micro_horizon,
        "initial_loss": losses[0],
        "best_loss": min(losses),
        "final_loss": losses[-1],
        "finite": bool(np.isfinite(losses).all()),
        "alpha_before": alpha_before,
        "alpha_after": alpha_after,
    }


def prefetched(dataset, indexes: Iterable[int], enabled: bool):
    """Overlap one sample read with GPU work without duplicating NetCDF handles."""

    indexes = list(indexes)
    if not indexes:
        return
    if not enabled:
        for index in indexes:
            started = time.perf_counter()
            yield dataset[index], time.perf_counter() - started
        return
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="stage5-adapter-data") as executor:
        future = executor.submit(dataset.__getitem__, indexes[0])
        for position in range(len(indexes)):
            wait_started = time.perf_counter()
            sample = future.result()
            wait_seconds = time.perf_counter() - wait_started
            if position + 1 < len(indexes):
                future = executor.submit(dataset.__getitem__, indexes[position + 1])
            yield sample, wait_seconds


def horizon_for_epoch(epoch: int, model_config: dict[str, Any]) -> int:
    formal = model_config.get("progressive_horizon_epochs")
    if formal is not None:
        lead1 = int(formal["lead1_epochs"])
        lead3 = int(formal["lead3_epochs"])
        return 1 if epoch < lead1 else (3 if epoch < lead1 + lead3 else 10)
    schedule = list(model_config.get("progressive_horizons", [1, 3, 10]))
    return int(schedule[min(epoch, len(schedule) - 1)])


def evaluate(
    wrapper,
    dataset,
    device,
    loss_config,
    *,
    prefetch: bool = False,
) -> tuple[float, list[dict[str, Any]], float]:
    wrapper.eval()
    rows = []
    total_started = time.perf_counter()
    with torch.no_grad():
        for sample_np, load_seconds in prefetched(dataset, range(len(dataset)), prefetch):
            sample = tensor_sample(sample_np, device)
            with autocast_context(device):
                loss, _, observed = rollout(wrapper, sample, horizon=10, loss_config=loss_config)
            value = float(loss.detach().cpu())
            rows.append(
                {
                    "init_time": sample["metadata"]["init_time"],
                    "loss": value,
                    "horizon": 10,
                    "load_seconds": load_seconds,
                    "observed_history_counts": ";".join(map(str, observed)),
                    "finite": math.isfinite(value),
                }
            )
    return float(np.mean([row["loss"] for row in rows])), rows, time.perf_counter() - total_started


def adapter_state_dict(wrapper: DualSourceAdapterWrapper) -> dict[str, Any]:
    """Return only enabled adapter weights; the frozen backbone is referenced by SHA."""
    if hasattr(wrapper, "adapter_state_dict"):
        return wrapper.adapter_state_dict()
    return {
        "ocean": wrapper.ocean_adapter.state_dict() if wrapper.ocean_adapter_enabled else None,
        "forcing": wrapper.forcing_adapter.state_dict() if wrapper.forcing_adapter_enabled else None,
    }


def load_adapter_state_dict(wrapper: DualSourceAdapterWrapper, payload: dict[str, Any]) -> None:
    if hasattr(wrapper, "load_adapter_state_dict"):
        wrapper.load_adapter_state_dict(payload)
        return
    expected_ocean = wrapper.ocean_adapter_enabled
    expected_forcing = wrapper.forcing_adapter_enabled
    if (payload.get("ocean") is not None) != expected_ocean:
        raise ValueError("Adapter-only checkpoint ocean enablement mismatch.")
    if (payload.get("forcing") is not None) != expected_forcing:
        raise ValueError("Adapter-only checkpoint forcing enablement mismatch.")
    if expected_ocean:
        wrapper.ocean_adapter.load_state_dict(payload["ocean"], strict=True)
    if expected_forcing:
        wrapper.forcing_adapter.load_state_dict(payload["forcing"], strict=True)


def checkpoint_payload(
    wrapper,
    optimizer,
    scaler,
    *,
    epoch,
    completed_steps,
    best_loss,
    config,
    metadata,
    best_epoch: int = -1,
    patience_counter: int = 0,
    rng_state: dict[str, Any] | None = None,
):
    scope = str(config.get("execution_scope", "stage5_2_smoke3"))
    adapter_only = scope in {
        "stage5_2b_pilot30",
        "stage5_2d_formal_train755",
        "stage5_3a_adapter_v2_smoke3",
        "stage5_3b_phda_smoke3",
        "stage5_3c_phda_formal_train755",
        "stage5_4a_dista_base_smoke3",
        "stage5_4b_dista_base_formal_train755",
        "stage5_5a_dista_path_ablation_smoke3",
        "stage5_5b_dista_path_ablation_formal_train755",
        "stage5_5c_dista_theory_ablation_formal_train755",
        "stage5_postholdout_generic_baseline_formal_train755",
    }
    schema_version = {
        "stage5_2b_pilot30": "stage5_adapter_only_v2",
        "stage5_2d_formal_train755": "stage5_adapter_only_v3",
        "stage5_3a_adapter_v2_smoke3": "stage5_lead_conditioned_adapter_v2_smoke_v1",
        "stage5_3b_phda_smoke3": "stage5_phda_adapter_smoke_v1",
        "stage5_3c_phda_formal_train755": "stage5_phda_adapter_formal_v1",
        "stage5_4a_dista_base_smoke3": "stage5_dista_base_smoke_v1",
        "stage5_4b_dista_base_formal_train755": "stage5_dista_base_formal_v1",
        "stage5_5a_dista_path_ablation_smoke3": "stage5_dista_path_ablation_smoke_v1",
        "stage5_5b_dista_path_ablation_formal_train755": "stage5_dista_path_ablation_formal_v1",
        "stage5_5c_dista_theory_ablation_formal_train755": "stage5_dista_theory_ablation_formal_v1",
        "stage5_postholdout_generic_baseline_formal_train755": "stage5_generic_baseline_formal_v1",
    }.get(scope, "stage5_adapter_v1")
    payload = {
        "schema_version": schema_version,
        "execution_scope": scope,
        "mode": config["mode"],
        "adapter_version": str(config.get("adapter_version", "v1")),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": int(epoch),
        "completed_optimizer_steps": int(completed_steps),
        "best_val_loss": float(best_loss),
        "best_epoch": int(best_epoch),
        "patience_counter": int(patience_counter),
        "source_checkpoint_sha256": sha256_file(metadata["checkpoint_path"]),
        "source_config_sha256": config_sha256(metadata["model_config_path"]),
        "saved_at": utc_now(),
    }
    if rng_state is not None:
        payload["numpy_rng_state"] = rng_state
    if adapter_only:
        payload["adapter_state_dict"] = adapter_state_dict(wrapper)
        payload["source_checkpoint"] = str(metadata["checkpoint_path"])
        payload["source_config"] = str(metadata["model_config_path"])
    else:
        payload["model"] = wrapper.state_dict()
    return payload


def restore_training_checkpoint(
    wrapper: DualSourceAdapterWrapper,
    optimizer: torch.optim.Optimizer,
    scaler,
    checkpoint: dict[str, Any],
    source: dict[str, Any],
    mode: str,
    scope: str,
    adapter_version: str = "v1",
) -> None:
    if checkpoint.get("mode") != mode:
        raise ValueError("Resume checkpoint mode mismatch.")
    if checkpoint.get("source_checkpoint_sha256") != sha256_file(source["checkpoint_path"]):
        raise ValueError("Resume checkpoint source checkpoint SHA256 mismatch.")
    if checkpoint.get("source_config_sha256") != config_sha256(source["model_config_path"]):
        raise ValueError("Resume checkpoint source config SHA256 mismatch.")
    adapter_only_schemas = {
        "stage5_2b_pilot30": "stage5_adapter_only_v2",
        "stage5_2d_formal_train755": "stage5_adapter_only_v3",
        "stage5_3a_adapter_v2_smoke3": "stage5_lead_conditioned_adapter_v2_smoke_v1",
        "stage5_3b_phda_smoke3": "stage5_phda_adapter_smoke_v1",
        "stage5_3c_phda_formal_train755": "stage5_phda_adapter_formal_v1",
        "stage5_4a_dista_base_smoke3": "stage5_dista_base_smoke_v1",
        "stage5_4b_dista_base_formal_train755": "stage5_dista_base_formal_v1",
        "stage5_5a_dista_path_ablation_smoke3": "stage5_dista_path_ablation_smoke_v1",
        "stage5_5b_dista_path_ablation_formal_train755": "stage5_dista_path_ablation_formal_v1",
        "stage5_5c_dista_theory_ablation_formal_train755": "stage5_dista_theory_ablation_formal_v1",
    }
    if checkpoint.get("schema_version") in set(adapter_only_schemas.values()):
        expected_schema = adapter_only_schemas.get(scope)
        if checkpoint.get("schema_version") != expected_schema or "model" in checkpoint:
            raise ValueError("Invalid adapter-only checkpoint scope or embedded backbone.")
        if checkpoint.get("adapter_version", "v1") != adapter_version:
            raise ValueError("Resume checkpoint Adapter version mismatch.")
        load_adapter_state_dict(wrapper, checkpoint["adapter_state_dict"])
    elif checkpoint.get("schema_version") == "stage5_adapter_v1":
        if scope != "stage5_2_smoke3":
            raise ValueError("Legacy full-wrapper checkpoint is not accepted for Pilot30.")
        wrapper.load_state_dict(checkpoint["model"], strict=True)
    else:
        raise ValueError(f"Unsupported Stage5 checkpoint schema: {checkpoint.get('schema_version')}")
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint.get("scaler", {}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-cycle-manifest", required=True)
    parser.add_argument("--val-cycle-manifest", required=True)
    parser.add_argument("--max-train-cycles", type=int)
    parser.add_argument("--max-val-cycles", type=int)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--max-optimizer-steps", type=int, default=3)
    parser.add_argument("--resume", nargs="?", const="checkpoint_last.pt")
    parser.add_argument(
        "--authorize-pilot30",
        action="store_true",
        help="Required for non-dry-run Stage5.2b Pilot30 execution.",
    )
    parser.add_argument(
        "--authorize-v2-smoke",
        action="store_true",
        help="Required for the bounded Stage5.3a Adapter V2 Smoke3 execution.",
    )
    parser.add_argument(
        "--authorize-phda-smoke",
        action="store_true",
        help="Required for the bounded Stage5.3b PHDA Smoke3 execution.",
    )
    parser.add_argument(
        "--authorize-phda-formal-train755",
        action="store_true",
        help="Required for registered PHDA V2 train755/val143 execution.",
    )
    parser.add_argument(
        "--authorize-dista-smoke",
        action="store_true",
        help="Required for the bounded DISTA Base Smoke3 execution.",
    )
    parser.add_argument(
        "--authorize-dista-formal-train755",
        action="store_true",
        help="Required for DISTA Base train755/val143 execution.",
    )
    parser.add_argument(
        "--authorize-dista-path-ablation-smoke",
        action="store_true",
        help="Required for bounded Stage5.5 path-ablation Smoke3 execution.",
    )
    parser.add_argument(
        "--authorize-dista-path-ablation-formal-train755",
        action="store_true",
        help="Required for registered Stage5.5 path-ablation train755/val143 execution.",
    )
    parser.add_argument(
        "--authorize-dista-theory-ablation-formal-train755",
        action="store_true",
        help="Required for registered Stage5.5c theory-ablation train755/val143 execution.",
    )
    parser.add_argument(
        "--authorize-formal-train755",
        action="store_true",
        help="Required for the single formal Dual Adapter-v1 train755/val143 run.",
    )
    parser.add_argument(
        "--authorize-generic-baseline-formal-train755",
        action="store_true",
        help="Required for post-holdout generic baseline train755/val143 execution.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml(config_path)
    registry_path = project_path(config["registry"])
    registry = load_yaml(registry_path)
    validate_stage5_contract(config, registry)
    session_started = time.time()
    mode = str(config["mode"])
    scope = str(config.get("execution_scope", "stage5_2_smoke3"))
    adapter_version = str(config.get("adapter_version", "v1"))
    limits = EXECUTION_SCOPES[scope]
    if args.max_train_cycles is None or args.max_val_cycles is None:
        raise ValueError("Stage5.2 requires explicit bounded --max-train-cycles and --max-val-cycles.")
    if args.max_train_cycles > limits["max_train_cycles"] or args.max_val_cycles > limits["max_val_cycles"]:
        raise ValueError(f"{scope} cycle limits are {limits['max_train_cycles']} train + {limits['max_val_cycles']} val.")
    if args.max_epochs > limits["max_epochs"] or args.max_optimizer_steps > limits["max_optimizer_steps"]:
        raise ValueError(
            f"{scope} limits are {limits['max_epochs']} epochs and {limits['max_optimizer_steps']} optimizer steps."
        )
    if scope == "stage5_2b_pilot30" and not args.dry_run and not args.authorize_pilot30:
        raise ValueError("Pilot30 execution is locked. Rerun with --authorize-pilot30 after readiness approval.")
    if scope == "stage5_3a_adapter_v2_smoke3" and not args.dry_run and not args.authorize_v2_smoke:
        raise ValueError("Adapter V2 Smoke3 is locked. Rerun with --authorize-v2-smoke after review.")
    if scope == "stage5_3b_phda_smoke3" and not args.dry_run and not args.authorize_phda_smoke:
        raise ValueError("PHDA Smoke3 is locked. Rerun with --authorize-phda-smoke after review.")
    if scope == "stage5_4a_dista_base_smoke3" and not args.dry_run and not args.authorize_dista_smoke:
        raise ValueError("DISTA Base Smoke3 is locked. Rerun with --authorize-dista-smoke after review.")
    if scope == "stage5_5a_dista_path_ablation_smoke3" and not args.dry_run and not args.authorize_dista_path_ablation_smoke:
        raise ValueError("DISTA path-ablation Smoke3 is locked; pass --authorize-dista-path-ablation-smoke.")
    if scope in FORMAL_SCOPES:
        expected = (755, 143, 100, 75500)
        actual = (
            args.max_train_cycles,
            args.max_val_cycles,
            args.max_epochs,
            args.max_optimizer_steps,
        )
        if actual != expected:
            raise ValueError(
                "Formal Adapter training forbids truncated overrides; expected "
                "755 train, 143 val, 100 epochs and 75500 maximum optimizer steps."
            )
        if scope == "stage5_3c_phda_formal_train755":
            authorized = args.authorize_phda_formal_train755
            flag = "--authorize-phda-formal-train755"
        elif scope == "stage5_4b_dista_base_formal_train755":
            authorized = args.authorize_dista_formal_train755
            flag = "--authorize-dista-formal-train755"
        elif scope == "stage5_5b_dista_path_ablation_formal_train755":
            authorized = args.authorize_dista_path_ablation_formal_train755
            flag = "--authorize-dista-path-ablation-formal-train755"
        elif scope == "stage5_5c_dista_theory_ablation_formal_train755":
            authorized = args.authorize_dista_theory_ablation_formal_train755
            flag = "--authorize-dista-theory-ablation-formal-train755"
        elif scope == "stage5_postholdout_generic_baseline_formal_train755":
            authorized = args.authorize_generic_baseline_formal_train755
            flag = "--authorize-generic-baseline-formal-train755"
        else:
            authorized = args.authorize_formal_train755
            flag = "--authorize-formal-train755"
        if not args.dry_run and not authorized:
            raise ValueError(f"Formal train755/val143 is locked; pass {flag} only after readiness passes.")

    train_manifest = project_path(args.train_cycle_manifest)
    val_manifest = project_path(args.val_cycle_manifest)
    if scope in FORMAL_SCOPES:
        configured_train = project_path(config["train_manifest"]).resolve()
        configured_val = project_path(config["val_manifest"]).resolve()
        if train_manifest.resolve() != configured_train or val_manifest.resolve() != configured_val:
            raise ValueError("Formal CLI manifests differ from the frozen config manifests.")
        if sha256_file(train_manifest) != config["train_manifest_sha256"]:
            raise ValueError("Formal train755 manifest SHA256 mismatch.")
        if sha256_file(val_manifest) != config["val_manifest_sha256"]:
            raise ValueError("Formal val143 manifest SHA256 mismatch.")
    train_dataset = Stage5PairedDomainDataset(
        registry_path,
        split="train",
        cycle_manifest=train_manifest,
        max_cycles=args.max_train_cycles,
        materialization_mode="target_forecast_only",
    )
    val_dataset = Stage5PairedDomainDataset(
        registry_path,
        split="val",
        cycle_manifest=val_manifest,
        max_cycles=args.max_val_cycles,
        materialization_mode="target_forecast_only",
    )
    try:
        train_audit = train_dataset.preaudit_cycles()
        val_audit = val_dataset.preaudit_cycles()
        if scope in FORMAL_SCOPES:
            if len(train_audit) != 755 or len(val_audit) != 143:
                raise ValueError("Formal Adapter split must contain exactly 755 train and 143 val cycles.")
            train_times = {row["init_time"] for row in train_audit}
            val_times = {row["init_time"] for row in val_audit}
            if not train_times.isdisjoint(val_times):
                raise ValueError("Formal Adapter train/val cycles overlap.")
            if any(pd.Timestamp(row["init_time"]).year >= 2025 for row in train_audit + val_audit):
                raise ValueError("Formal Adapter audit reached a 2025 cycle.")
        dry_payload = {
            "status": "passed",
            "mode": mode,
            "adapter_version": adapter_version,
            "execution_scope": scope,
            "pilot30_execution_authorized": bool(args.authorize_pilot30),
            "formal_train755_execution_authorized": bool(args.authorize_formal_train755),
            "adapter_v2_smoke_execution_authorized": bool(args.authorize_v2_smoke),
            "phda_smoke_execution_authorized": bool(args.authorize_phda_smoke),
            "phda_formal_train755_execution_authorized": bool(args.authorize_phda_formal_train755),
            "dista_smoke_execution_authorized": bool(args.authorize_dista_smoke),
            "dista_formal_train755_execution_authorized": bool(args.authorize_dista_formal_train755),
            "dista_path_ablation_smoke_execution_authorized": bool(args.authorize_dista_path_ablation_smoke),
            "dista_path_ablation_formal_train755_execution_authorized": bool(args.authorize_dista_path_ablation_formal_train755),
            "dista_theory_ablation_formal_train755_execution_authorized": bool(args.authorize_dista_theory_ablation_formal_train755),
            "generic_baseline_formal_train755_execution_authorized": bool(args.authorize_generic_baseline_formal_train755),
            "train_cycle_count": len(train_audit),
            "val_cycle_count": len(val_audit),
            "train_cycles": (
                [train_audit[0]["init_time"], train_audit[-1]["init_time"]]
                if scope in FORMAL_SCOPES
                else [row["init_time"] for row in train_audit]
            ),
            "val_cycles": (
                [val_audit[0]["init_time"], val_audit[-1]["init_time"]]
                if scope in FORMAL_SCOPES
                else [row["init_time"] for row in val_audit]
            ),
            "materialization_mode": "target_forecast_only",
            "forecast_loss_only": True,
            "truth_2025_accessed": False,
            "torch_load_called": False,
            "cuda_initialized": False,
            "checkpoint_serialization": limits["checkpoint_serialization"],
            "lead_gate_policy": config.get("policies", {}).get("lead_gate_constraint", "disabled"),
            "lead1_adapter_policy": config.get("policies", {}).get("lead1_adapter_policy", "v1_fixed"),
            "gate_type": config.get("policies", {}).get("gate_type", "none"),
            "ocean_gate_inputs": config.get("policies", {}).get("ocean_gate_inputs", []),
            "forcing_gate_inputs": config.get("policies", {}).get("forcing_gate_inputs", []),
        }
        if args.dry_run:
            print(json.dumps(dry_payload, ensure_ascii=False, indent=2))
            return 0
        if mode == "source_only":
            raise ValueError("source_only does not train; use the dedicated parity command.")

        seed = int(config.get("seed", 20260720))
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if scope in FORMAL_SCOPES and device.type != "cuda":
            raise RuntimeError("Formal Adapter train755/val143 requires CUDA.")
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.cuda.reset_peak_memory_stats()
        wrapper, source = build_wrapper(
            registry,
            mode,
            device,
            adapter_version,
            str(config.get("source_model", "temporal_transformer")),
            config.get("adapter"),
        )
        inventory = wrapper.parameter_inventory()
        model_cfg = config["model"]
        parameter_groups = trainable_parameter_groups(
            wrapper,
            float(model_cfg["lr"]),
            float(model_cfg.get("weight_decay", 0.0)),
        )
        if not parameter_groups:
            raise ValueError("No adapter parameters are trainable.")
        optimizer = torch.optim.AdamW(parameter_groups)
        scaler = make_scaler(device)
        output_dir = project_path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        if (
            scope in FORMAL_SCOPES
            and not args.resume
            and any((output_dir / name).exists() for name in ("checkpoint_best.pt", "checkpoint_last.pt", "training_log.csv"))
        ):
            raise FileExistsError(
                f"Existing formal Adapter artifacts require --resume or a new output directory: {output_dir}"
            )
        previous_runtime = (
            json.loads((output_dir / "runtime_summary.json").read_text(encoding="utf-8-sig"))
            if args.resume and (output_dir / "runtime_summary.json").is_file()
            else {}
        )
        code_snapshot_path = project_path(
            config.get("code_snapshot_manifest", registry["paths"]["code_snapshot_manifest"])
        )
        code_snapshot = json.loads(code_snapshot_path.read_text(encoding="utf-8-sig"))

        # Materialize the first cycle once for bounded gradient and micro-overfit audits.
        load_started = time.perf_counter()
        first_sample = tensor_sample(train_dataset[0], device)
        first_load_seconds = time.perf_counter() - load_started
        resumed = bool(args.resume)
        gradient_rows: list[dict[str, Any]] = []
        micro = previous_runtime.get("micro_overfit")
        full_rollout_seconds = float(
            previous_runtime.get("seconds_per_full_10day_rollout_backward", math.nan)
        )
        if not resumed and scope in {
            "stage5_2_smoke3",
            "stage5_3a_adapter_v2_smoke3",
            "stage5_3b_phda_smoke3",
            "stage5_4a_dista_base_smoke3",
            "stage5_5a_dista_path_ablation_smoke3",
        }:
            gradient_rows, full_rollout_seconds = run_gradient_audits(
                wrapper, first_sample, config.get("loss", {}), optimizer, device
            )
            micro = micro_overfit(registry, mode, first_sample, config, device)
            # Audits must not alter the model used by the 3+3 smoke.
            wrapper, source = build_wrapper(
                registry,
                mode,
                device,
                adapter_version,
                str(config.get("source_model", "temporal_transformer")),
                config.get("adapter"),
            )
            parameter_groups = trainable_parameter_groups(
                wrapper,
                float(model_cfg["lr"]),
                float(model_cfg.get("weight_decay", 0.0)),
            )
            optimizer = torch.optim.AdamW(parameter_groups)
            scaler = make_scaler(device)

        training_path = output_dir / "training_log.csv"
        validation_path = output_dir / "validation_log.csv"
        training_rows = pd.read_csv(training_path).to_dict("records") if resumed and training_path.is_file() else []
        validation_rows = pd.read_csv(validation_path).to_dict("records") if resumed and validation_path.is_file() else []
        completed_steps = 0
        start_epoch = 0
        best_loss = math.inf
        best_epoch = -1
        patience_counter = 0
        rng = np.random.default_rng(seed)
        alpha_before = previous_runtime.get("alpha_values_before", alpha_values(wrapper))
        resume_report = {"attempted": resumed, "passed": not resumed, "resumed_from": None}
        if resumed:
            resume_path = Path(args.resume)
            if not resume_path.is_absolute():
                resume_path = output_dir / resume_path
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            restore_training_checkpoint(
                wrapper,
                optimizer,
                scaler,
                checkpoint,
                source,
                mode,
                scope,
                adapter_version,
            )
            completed_steps = int(checkpoint["completed_optimizer_steps"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_loss = float(checkpoint["best_val_loss"])
            best_epoch = int(checkpoint.get("best_epoch", checkpoint["epoch"]))
            patience_counter = int(checkpoint.get("patience_counter", 0))
            if checkpoint.get("numpy_rng_state") is not None:
                rng.bit_generator.state = checkpoint["numpy_rng_state"]
            resume_report = {
                "attempted": True,
                "passed": True,
                "resumed_from": str(resume_path),
                "checkpoint_sha256": sha256_file(resume_path),
                "restored_optimizer": True,
                "restored_scaler": True,
                "starting_optimizer_steps": completed_steps,
            }

        optimizer_step_seconds = []
        loading_seconds = []
        alpha_at_training_start = alpha_values(wrapper)
        final_epoch = start_epoch - 1
        early_stopped = False
        prefetch_enabled = bool(config.get("prefetch_thread", False))
        for epoch in range(start_epoch, args.max_epochs):
            final_epoch = epoch
            horizon = horizon_for_epoch(epoch, model_cfg)
            if scope in FORMAL_SCOPES and epoch == 25:
                patience_counter = 0
            wrapper.train()
            indexes = (
                rng.permutation(len(train_dataset)).tolist()
                if scope in FORMAL_SCOPES
                else list(range(len(train_dataset)))
            )
            for sample_np, load_seconds in prefetched(train_dataset, indexes, prefetch_enabled):
                if completed_steps >= args.max_optimizer_steps:
                    break
                sample = tensor_sample(sample_np, device)
                loading_seconds.append(load_seconds)
                optimizer.zero_grad(set_to_none=True)
                alpha_step_before = (
                    alpha_summary(wrapper)
                    if scope in FORMAL_SCOPES
                    else alpha_values(wrapper)
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                with autocast_context(device):
                    loss, _, observed_counts = rollout(
                        wrapper, sample, horizon=horizon, loss_config=config.get("loss", {})
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite Stage5 training loss.")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradients = gradient_counts(wrapper)
                gradient_norm = math.sqrt(
                    gradients["ocean_gradient_norm"] ** 2
                    + gradients["forcing_gradient_norm"] ** 2
                    + gradients["output_gradient_norm"] ** 2
                    + gradients["context_gradient_norm"] ** 2
                )
                clip_limit = model_cfg.get("gradient_clip_norm")
                if clip_limit is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [parameter for group in parameter_groups for parameter in group["params"]],
                        float(clip_limit),
                    )
                scaler.step(optimizer)
                scaler.update()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                optimizer_step_seconds.append(elapsed)
                completed_steps += 1
                training_rows.append(
                    {
                        "epoch": epoch,
                        "optimizer_step": completed_steps,
                        "init_time": sample["metadata"]["init_time"],
                        "horizon": horizon,
                        "loss": float(loss.detach().cpu()),
                        "seconds": elapsed,
                        "load_seconds": loading_seconds[-1],
                        "gradient_norm": gradient_norm,
                        "backbone_gradient_count": gradients["backbone_gradient_count"],
                        "ocean_gradient_parameter_count": gradients["ocean_gradient_parameter_count"],
                        "forcing_gradient_parameter_count": gradients["forcing_gradient_parameter_count"],
                        "output_gradient_parameter_count": gradients["output_gradient_parameter_count"],
                        "context_gradient_parameter_count": gradients["context_gradient_parameter_count"],
                        "observed_history_counts": ";".join(map(str, observed_counts)),
                        "alpha_before": json.dumps(alpha_step_before),
                        "alpha_after": json.dumps(
                            alpha_summary(wrapper)
                            if scope in FORMAL_SCOPES
                            else alpha_values(wrapper)
                        ),
                    }
                )
                if scope in FORMAL_SCOPES and (
                    completed_steps == 1 or completed_steps % 50 == 0
                ):
                    print(
                        json.dumps(
                            {
                                "stage": "adapter_train_progress",
                                "epoch": epoch,
                                "optimizer_step": completed_steps,
                                "horizon": horizon,
                                "loss": float(loss.detach().cpu()),
                                "compute_seconds": elapsed,
                                "data_wait_seconds": loading_seconds[-1],
                            }
                        ),
                        flush=True,
                    )
            val_loss, val_rows_epoch, val_seconds = evaluate(
                wrapper,
                val_dataset,
                device,
                config.get("loss", {}),
                prefetch=prefetch_enabled,
            )
            for row in val_rows_epoch:
                row.update({"epoch": epoch, "optimizer_step": completed_steps})
            validation_rows.extend(val_rows_epoch)
            min_delta = float(model_cfg.get("early_stopping_min_delta", 0.0))
            is_best = val_loss < best_loss - min_delta
            if is_best:
                best_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            payload = checkpoint_payload(
                wrapper,
                optimizer,
                scaler,
                epoch=epoch,
                completed_steps=completed_steps,
                best_loss=best_loss,
                config=config,
                metadata=source,
                best_epoch=best_epoch,
                patience_counter=patience_counter,
                rng_state=rng.bit_generator.state,
            )
            atomic_torch_save(output_dir / "checkpoint_last.pt", payload)
            if is_best or not (output_dir / "checkpoint_best.pt").is_file():
                atomic_torch_save(output_dir / "checkpoint_best.pt", payload)
            atomic_csv(output_dir / "training_log.csv", training_rows)
            atomic_csv(output_dir / "validation_log.csv", validation_rows)
            print(
                json.dumps(
                    {
                        "stage": "adapter_epoch_completed",
                        "epoch": epoch,
                        "horizon": horizon,
                        "optimizer_step": completed_steps,
                        "val_loss": val_loss,
                        "best_val_loss": best_loss,
                        "best_epoch": best_epoch,
                        "patience_counter": patience_counter,
                        "val_seconds": val_seconds,
                    }
                ),
                flush=True,
            )
            if completed_steps >= args.max_optimizer_steps:
                break
            if (
                scope in FORMAL_SCOPES
                and epoch >= 25
                and epoch + 1 >= int(model_cfg["minimum_epochs"])
                and patience_counter >= int(model_cfg["early_stopping_patience"])
            ):
                early_stopped = True
                break

        if not (output_dir / "checkpoint_last.pt").is_file():
            raise RuntimeError("Smoke completed without checkpoint_last.pt.")
        reload_check = torch.load(output_dir / "checkpoint_last.pt", map_location="cpu", weights_only=False)
        checkpoint_reload_passed = int(reload_check["completed_optimizer_steps"]) == completed_steps
        expected_adapter_schema = {
            "stage5_2b_pilot30": "stage5_adapter_only_v2",
            "stage5_2d_formal_train755": "stage5_adapter_only_v3",
            "stage5_3a_adapter_v2_smoke3": "stage5_lead_conditioned_adapter_v2_smoke_v1",
            "stage5_3b_phda_smoke3": "stage5_phda_adapter_smoke_v1",
            "stage5_3c_phda_formal_train755": "stage5_phda_adapter_formal_v1",
            "stage5_4a_dista_base_smoke3": "stage5_dista_base_smoke_v1",
            "stage5_4b_dista_base_formal_train755": "stage5_dista_base_formal_v1",
            "stage5_5a_dista_path_ablation_smoke3": "stage5_dista_path_ablation_smoke_v1",
            "stage5_5b_dista_path_ablation_formal_train755": "stage5_dista_path_ablation_formal_v1",
            "stage5_5c_dista_theory_ablation_formal_train755": "stage5_dista_theory_ablation_formal_v1",
            "stage5_postholdout_generic_baseline_formal_train755": "stage5_generic_baseline_formal_v1",
        }.get(scope)
        adapter_only_checkpoint_passed = bool(
            scope not in {
                "stage5_2b_pilot30",
                "stage5_2d_formal_train755",
                "stage5_3a_adapter_v2_smoke3",
                "stage5_3b_phda_smoke3",
                "stage5_3c_phda_formal_train755",
                "stage5_4a_dista_base_smoke3",
                "stage5_4b_dista_base_formal_train755",
                "stage5_5a_dista_path_ablation_smoke3",
                "stage5_5b_dista_path_ablation_formal_train755",
                "stage5_5c_dista_theory_ablation_formal_train755",
                "stage5_postholdout_generic_baseline_formal_train755",
            }
            or (
                reload_check.get("schema_version") == expected_adapter_schema
                and "adapter_state_dict" in reload_check
                and "model" not in reload_check
            )
        )
        resume_report["ending_optimizer_steps"] = completed_steps
        resume_report["additional_optimizer_steps"] = completed_steps - int(resume_report.get("starting_optimizer_steps", 0))
        resume_report["checkpoint_reload_passed"] = checkpoint_reload_passed
        resume_report["atomic_write_passed"] = not any(output_dir.glob("*.tmp"))

        alpha_after = alpha_values(wrapper)
        peak_vram_current = int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
        peak_vram = max(int(previous_runtime.get("peak_vram_bytes", 0)), peak_vram_current)
        all_step_seconds = [float(row["seconds"]) for row in training_rows]
        all_load_seconds = [float(row["load_seconds"]) for row in training_rows]
        runtime = {
            "mode": mode,
            "execution_scope": scope,
            "wall_clock_seconds": time.time() - session_started,
            **inventory,
            "adapter_component_parameters": {
                "ocean": inventory["ocean_adapter_parameters"],
                "forcing": inventory["forcing_adapter_parameters"],
                "output": inventory.get("output_adapter_parameters", 0),
                "context": inventory.get("context_adapter_parameters", 0),
            },
            "optimizer": "AdamW",
            "parameter_groups": [
                {"name": group["name"], "lr": group["lr"], "weight_decay": group["weight_decay"]}
                for group in parameter_groups
            ],
            "device": str(device),
            "gpu_id": torch.cuda.current_device() if device.type == "cuda" else None,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_vram_bytes": peak_vram,
            "mean_seconds_per_optimizer_step": float(np.mean(all_step_seconds)) if all_step_seconds else math.nan,
            "seconds_per_full_10day_rollout_backward": full_rollout_seconds,
            "mean_data_loading_seconds": float(np.mean(all_load_seconds)) if all_load_seconds else first_load_seconds,
            "completed_optimizer_steps": completed_steps,
            "completed_epochs": final_epoch + 1,
            "early_stopped": early_stopped,
            "best_epoch": best_epoch,
            "best_val_loss": best_loss,
            "train_loss_final": float(training_rows[-1]["loss"]) if training_rows else math.nan,
            "val_loss_final": float(validation_rows[-1]["loss"]) if validation_rows else math.nan,
            "alpha_values_before": alpha_before,
            "alpha_values_at_training_start": alpha_at_training_start,
            "alpha_values_after": alpha_after,
            "micro_overfit": micro,
            "parameter_efficient_not_compute_efficient": True,
            "checkpoint_serialization": limits["checkpoint_serialization"],
            "truth_2025_accessed": False,
        }
        enabled_gradient_ok = True
        disabled_gradient_ok = True
        if isinstance(wrapper, DISTAPathAblationWrapper):
            # For Group 2, encoder gradients intentionally remain active even
            # when an input residual path is disabled. Audit correction heads
            # separately so the orthogonal ablation contract is not mistaken
            # for a legacy ocean_only/forcing_only configuration.
            full = gradient_rows[-1] if gradient_rows else training_rows[-1]
            spec = wrapper.path_ablation_spec
            enabled_gradient_ok = bool(
                int(full.get("ocean_context_gradient_parameter_count", 0)) > 0
                and int(full.get("forcing_context_gradient_parameter_count", 0)) > 0
                and (not spec.ocean_input_correction_applied or int(full.get("ocean_correction_gradient_parameter_count", 0)) > 0)
                and (not spec.forcing_input_correction_applied or int(full.get("forcing_correction_gradient_parameter_count", 0)) > 0)
                and (not spec.output_tendency_correction_applied or int(full.get("output_gradient_parameter_count", 0)) > 0)
            )
            disabled_gradient_ok = bool(
                int(full.get("backbone_gradient_count", 0)) == 0
                and (spec.ocean_input_correction_applied or int(full.get("ocean_correction_gradient_parameter_count", 0)) == 0)
                and (spec.forcing_input_correction_applied or int(full.get("forcing_correction_gradient_parameter_count", 0)) == 0)
                and (spec.output_tendency_correction_applied or int(full.get("output_gradient_parameter_count", 0)) == 0)
            )
        elif gradient_rows:
            full = gradient_rows[-1]
            if wrapper.ocean_adapter_enabled:
                enabled_gradient_ok &= full["ocean_gradient_parameter_count"] > 0
            else:
                disabled_gradient_ok &= full["ocean_gradient_parameter_count"] == 0
            if wrapper.forcing_adapter_enabled:
                enabled_gradient_ok &= full["forcing_gradient_parameter_count"] > 0
            else:
                disabled_gradient_ok &= full["forcing_gradient_parameter_count"] == 0
            if getattr(wrapper, "output_adapter_enabled", False):
                enabled_gradient_ok &= full["output_gradient_parameter_count"] > 0
            else:
                disabled_gradient_ok &= full["output_gradient_parameter_count"] == 0
            disabled_gradient_ok &= full["backbone_gradient_count"] == 0
        elif training_rows:
            enabled_gradient_ok = all(
                (not wrapper.ocean_adapter_enabled or int(row.get("ocean_gradient_parameter_count", 0)) > 0)
                and (not wrapper.forcing_adapter_enabled or int(row.get("forcing_gradient_parameter_count", 0)) > 0)
                and (
                    not getattr(wrapper, "output_adapter_enabled", False)
                    or int(row.get("output_gradient_parameter_count", 0)) > 0
                )
                for row in training_rows
            )
            disabled_gradient_ok = all(
                int(row.get("backbone_gradient_count", 0)) == 0
                and (wrapper.ocean_adapter_enabled or int(row.get("ocean_gradient_parameter_count", 0)) == 0)
                and (wrapper.forcing_adapter_enabled or int(row.get("forcing_gradient_parameter_count", 0)) == 0)
                and (
                    getattr(wrapper, "output_adapter_enabled", False)
                    or int(row.get("output_gradient_parameter_count", 0)) == 0
                )
                for row in training_rows
            )

        full_10day_backward_passed = bool(
            gradient_rows[-1]["finite"]
            if gradient_rows
            else any(int(row.get("horizon", 0)) == 10 and math.isfinite(float(row["loss"])) for row in training_rows)
        )

        atomic_text(output_dir / "config_snapshot.yaml", yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        run_manifest = {
            "schema_version": {
                "stage5_2d_formal_train755": "stage5_adapter_formal_train755_v1",
                "stage5_2b_pilot30": "stage5_adapter_pilot30_v1",
                "stage5_3a_adapter_v2_smoke3": "stage5_lead_conditioned_adapter_v2_smoke_v1",
                "stage5_3b_phda_smoke3": "stage5_phda_adapter_smoke_v1",
                "stage5_3c_phda_formal_train755": "stage5_phda_adapter_formal_v1",
                "stage5_4a_dista_base_smoke3": "stage5_dista_base_smoke_v1",
                "stage5_4b_dista_base_formal_train755": "stage5_dista_base_formal_v1",
                "stage5_5a_dista_path_ablation_smoke3": "stage5_dista_path_ablation_smoke_v1",
                "stage5_5b_dista_path_ablation_formal_train755": "stage5_dista_path_ablation_formal_v1",
                "stage5_5c_dista_theory_ablation_formal_train755": "stage5_dista_theory_ablation_formal_v1",
                "stage5_postholdout_generic_baseline_formal_train755": "stage5_generic_baseline_formal_v1",
            }.get(scope, "stage5_adapter_smoke_v1"),
            "status": "passed",
            "mode": mode,
            "stage": 5,
            "work_package": limits["work_package"],
            "execution_scope": scope,
            "adapter_version": adapter_version,
            "pilot30_execution_authorized": bool(args.authorize_pilot30),
            "formal_train755_execution_authorized": bool(args.authorize_formal_train755),
            "adapter_v2_smoke_execution_authorized": bool(args.authorize_v2_smoke),
            "phda_smoke_execution_authorized": bool(args.authorize_phda_smoke),
            "phda_formal_train755_execution_authorized": bool(args.authorize_phda_formal_train755),
            "dista_smoke_execution_authorized": bool(args.authorize_dista_smoke),
            "dista_formal_train755_execution_authorized": bool(args.authorize_dista_formal_train755),
            "dista_path_ablation_smoke_execution_authorized": bool(args.authorize_dista_path_ablation_smoke),
            "dista_path_ablation_formal_train755_execution_authorized": bool(args.authorize_dista_path_ablation_formal_train755),
            "dista_theory_ablation_formal_train755_execution_authorized": bool(args.authorize_dista_theory_ablation_formal_train755),
            "created_at": utc_now(),
            "code_snapshot_id": code_snapshot["code_snapshot_id"],
            "code_snapshot_manifest": str(code_snapshot_path),
            "source_only_parity_required": True,
            "forecast_loss_only": True,
            "ocean_adapter_history_policy": config.get("policies", {}).get(
                "ocean_adapter_history_policy"
            ),
            "forcing_trajectory_policy": config.get("policies", {}).get(
                "forcing_trajectory_policy"
            ),
            "lead_gate_policy": config.get("policies", {}).get("lead_gate_constraint", "disabled"),
            "lead1_adapter_policy": config.get("policies", {}).get("lead1_adapter_policy", "v1_fixed"),
            "gate_type": config.get("policies", {}).get("gate_type", "none"),
            "ocean_gate_inputs": config.get("policies", {}).get("ocean_gate_inputs", []),
            "forcing_gate_inputs": config.get("policies", {}).get("forcing_gate_inputs", []),
            "adapted_forcing_channels": (
                []
                if adapter_version in GENERIC_BASELINE_VERSIONS
                else ["u10", "v10", "t2m", "d2m", "msl", "precipitation"]
            ),
            "passthrough_channels": (
                []
                if adapter_version in GENERIC_BASELINE_VERSIONS
                else ["sin_doy", "cos_doy"]
            ),
            "generic_baseline_contract": (
                {
                    "name": adapter_version,
                    "dista_mechanism_reused": False,
                    "adapter_placement": config.get("policies", {}).get("adapter_placement"),
                    "target_trainable_parameters": config.get("adapter", {}).get(
                        "target_trainable_parameters"
                    ),
                    "compatibility_slot": "ocean_adapter",
                }
                if adapter_version in GENERIC_BASELINE_VERSIONS
                else None
            ),
            "adapter_enabled": {
                "ocean": wrapper.ocean_adapter_enabled,
                "forcing": wrapper.forcing_adapter_enabled,
                "output": getattr(wrapper, "output_adapter_enabled", False),
            },
            "path_ablation": getattr(wrapper, "path_ablation", None),
            "path_ablation_contract": (
                wrapper.ablation_manifest()
                if isinstance(wrapper, DISTAPathAblationWrapper)
                else None
            ),
            "full_10day_backward_passed": full_10day_backward_passed,
            "enabled_adapter_gradients_active": enabled_gradient_ok,
            "disabled_and_backbone_gradients_absent": disabled_gradient_ok,
            "checkpoint_save_load_passed": checkpoint_reload_passed,
            "adapter_only_checkpoint_passed": adapter_only_checkpoint_passed,
            "resume_passed": bool(resume_report["passed"]),
            "no_nan_inf": bool(
                all(math.isfinite(float(row["loss"])) for row in training_rows + validation_rows)
            ),
            "no_oom": True,
            "truth_2025_accessed": False,
            "stage5_gate_status": registry["stage_gates"]["stage5"]["status"],
            "holdout_release_approved": registry["holdout_release"]["approved"],
            "best_epoch": best_epoch,
            "best_val_loss": best_loss,
            "early_stopped": early_stopped,
        }
        data_manifest = {
            "registry": str(registry_path),
            "registry_sha256": sha256_file(registry_path),
            "materialization_mode": "target_forecast_only",
            "returned_fields": [
                "anfc_ocean_history",
                "gfs_future_forcing",
                "anfc_truth",
                "anfc_truth_mask",
                "anfc_initial_mask",
                "static",
                "metadata",
            ],
            "not_materialized": ["my_ocean_history", "era5_future_forcing"],
            "train_cycle_manifest": str(train_manifest),
            "train_cycle_manifest_sha256": sha256_file(train_manifest),
            "val_cycle_manifest": str(val_manifest),
            "val_cycle_manifest_sha256": sha256_file(val_manifest),
            "train_cycles": train_audit,
            "val_cycles": val_audit,
            "normalization": registry["paths"]["normalization"],
            "normalization_sha256": sha256_file(project_path(registry["paths"]["normalization"])),
            "truth_2025_accessed": False,
        }
        checkpoint_manifest = {
            "source_checkpoint": str(source["checkpoint_path"]),
            "source_checkpoint_sha256": sha256_file(source["checkpoint_path"]),
            "source_config": str(source["model_config_path"]),
            "source_config_sha256": config_sha256(source["model_config_path"]),
            "checkpoint_best": "checkpoint_best.pt",
            "checkpoint_best_sha256": sha256_file(output_dir / "checkpoint_best.pt"),
            "checkpoint_last": "checkpoint_last.pt",
            "checkpoint_last_sha256": sha256_file(output_dir / "checkpoint_last.pt"),
            "checkpoint_reload_passed": checkpoint_reload_passed,
            "checkpoint_schema_version": reload_check["schema_version"],
            "checkpoint_serialization": limits["checkpoint_serialization"],
            "adapter_only_checkpoint_passed": adapter_only_checkpoint_passed,
            "optimizer_state_saved": True,
            "scaler_state_saved": True,
        }
        atomic_csv(output_dir / "training_log.csv", training_rows)
        atomic_csv(output_dir / "validation_log.csv", validation_rows)
        if gradient_rows:
            atomic_csv(output_dir / "gradient_audit.csv", gradient_rows)
        elif not (output_dir / "gradient_audit.csv").is_file():
            atomic_csv(output_dir / "gradient_audit.csv", [])
        atomic_json(output_dir / "run_manifest.json", run_manifest)
        atomic_json(output_dir / "data_manifest.json", data_manifest)
        atomic_json(output_dir / "checkpoint_manifest.json", checkpoint_manifest)
        atomic_json(output_dir / "runtime_summary.json", runtime)
        atomic_json(output_dir / "resume_report.json", resume_report)
        qc_lines = [
            f"# Stage5 {scope} {mode} QC",
            "",
            f"- status: {run_manifest['status']}",
            f"- full 10-day backward: {run_manifest['full_10day_backward_passed']}",
            f"- enabled adapter gradients active: {enabled_gradient_ok}",
            f"- disabled adapter and backbone gradients absent: {disabled_gradient_ok}",
            f"- checkpoint reload: {checkpoint_reload_passed}",
            f"- resume: {resume_report['passed']}",
            "- truth_2025_accessed: false",
            "- paired alignment: disabled",
            "- parameter efficient does not imply compute efficient.",
        ]
        atomic_text(output_dir / "qc_report.md", "\n".join(qc_lines) + "\n")
        missing = [name for name in REQUIRED_FILES if not (output_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"Missing Stage5 smoke artifacts: {missing}")
        print(
            json.dumps(
                {
                    "status": "passed",
                    "mode": mode,
                    "run_dir": str(output_dir),
                    "completed_optimizer_steps": completed_steps,
                    "train_cycles": len(train_dataset),
                    "val_cycles": len(val_dataset),
                    "peak_vram_bytes": peak_vram,
                    "truth_2025_accessed": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    finally:
        train_dataset.close()
        val_dataset.close()


if __name__ == "__main__":
    raise SystemExit(main())
