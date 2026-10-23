"""Metrics-only evaluator for Stage5 Adapter Pilot30 or formal val143 runs."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.evaluate_operational_replay_v2 import ensure_climatology
from pipeline.run_operational_replay import append_formal_metrics, aggregate_formal_metrics, outer_boundary_mask
from pipeline.stage5_paired_domain_dataset import Stage5PairedDomainDataset
from pipeline.train_stage5_adapter import (
    build_wrapper,
    load_adapter_state_dict,
    load_yaml,
    project_path,
    rollout,
    sha256_file,
    tensor_sample,
    validate_stage5_contract,
)


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("rmse", "mae", "persistence_skill", "r2", "acc")


def atomic_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    os.replace(temporary, path)


def paired_deltas(adapted: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    keys = [
        key
        for key in ("init_time", "lead_day", "depth", "domain", "variable", "unit")
        if key in adapted.columns and key in source.columns
    ]
    left = adapted[keys + list(METRICS)].copy()
    right = source[keys + list(METRICS)].copy()
    merged = left.merge(right, on=keys, suffixes=("_adapted", "_source"), validate="one_to_one")
    for metric in METRICS:
        raw = merged[f"{metric}_adapted"] - merged[f"{metric}_source"]
        merged[f"{metric}_raw_delta"] = raw
        merged[f"{metric}_degradation_delta"] = raw if metric in {"rmse", "mae"} else -raw
    merged["exploratory_only"] = True
    return merged


def load_evaluation_checkpoint(
    adapted,
    checkpoint: dict,
    *,
    expected_schema: str,
    expected_scope: str,
    expected_mode: str,
    expected_source_checkpoint_sha256: str,
    allow_legacy_path_ablation_wrapper: bool,
) -> str:
    """Load a scope-matching checkpoint and report its serialization format.

    The first Stage5.5 formal path-ablation runs were written as full-wrapper
    payloads while carrying the adapter-only schema. Compatibility is limited
    to that exact scope and version. The frozen embedded backbone must be
    bitwise identical to the registered source checkpoint before the state is
    accepted.
    """
    if checkpoint.get("schema_version") != expected_schema:
        raise ValueError("Adapter evaluator checkpoint schema does not match the execution scope.")
    if checkpoint.get("execution_scope") != expected_scope:
        raise ValueError("Adapter evaluator checkpoint execution scope mismatch.")
    if checkpoint.get("mode") != expected_mode:
        raise ValueError("Adapter evaluator checkpoint mode mismatch.")
    if checkpoint.get("source_checkpoint_sha256") != expected_source_checkpoint_sha256:
        raise ValueError("Adapter evaluator source checkpoint SHA256 mismatch.")

    if "model" not in checkpoint:
        if "adapter_state_dict" not in checkpoint:
            raise ValueError("Adapter-only checkpoint is missing adapter_state_dict.")
        load_adapter_state_dict(adapted, checkpoint["adapter_state_dict"])
        return "adapter_only"

    legacy_allowed = (
        allow_legacy_path_ablation_wrapper
        and expected_scope == "stage5_5b_dista_path_ablation_formal_train755"
        and checkpoint.get("adapter_version") == "dista_path_ablation"
        and "adapter_state_dict" not in checkpoint
    )
    if not legacy_allowed:
        raise ValueError("Adapter evaluator requires the scope-matching adapter-only checkpoint.")

    embedded_state = checkpoint["model"]
    registered_state = adapted.state_dict()
    if set(embedded_state) != set(registered_state):
        raise ValueError("Legacy Stage5.5 wrapper checkpoint state keys do not match the registered wrapper.")
    for name, registered_value in registered_state.items():
        if not name.startswith("backbone."):
            continue
        embedded_value = embedded_state[name]
        if not torch.equal(embedded_value.detach().cpu(), registered_value.detach().cpu()):
            raise ValueError(
                "Legacy Stage5.5 wrapper checkpoint contains a backbone that differs from the registered source."
            )
    adapted.load_state_dict(embedded_state, strict=True)
    return "legacy_full_wrapper_compat"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cycle-manifest", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-cycles", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--authorize-pilot30-evaluation", action="store_true")
    parser.add_argument("--authorize-formal-evaluation", action="store_true")
    parser.add_argument("--authorize-v2-smoke-evaluation", action="store_true")
    parser.add_argument("--authorize-phda-smoke-evaluation", action="store_true")
    parser.add_argument("--authorize-dista-smoke-evaluation", action="store_true")
    parser.add_argument("--authorize-dista-path-ablation-smoke-evaluation", action="store_true")
    parser.add_argument("--authorize-dista-path-ablation-formal-evaluation", action="store_true")
    parser.add_argument("--authorize-dista-theory-ablation-formal-evaluation", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml(config_path)
    registry_path = project_path(config["registry"])
    registry = load_yaml(registry_path)
    # A limited holdout release must not invalidate already-frozen development
    # evaluations.  Stage5PairedDomainDataset still hard-rejects every cycle
    # whose init/target window reaches 2025, so this does not authorize holdout
    # access or broaden the release allowlist.
    validate_stage5_contract(
        config,
        registry,
        allow_released_holdout_for_development_evaluation=True,
    )
    scope = str(config.get("execution_scope"))
    if scope not in {
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
    }:
        raise ValueError("Adapter evaluator accepts registered Pilot30, Smoke3, or formal train755 scopes.")
    if config["mode"] == "source_only":
        raise ValueError("Pilot evaluator requires an Adapter mode; Source-only is evaluated internally.")
    formal = scope in {
        "stage5_2d_formal_train755",
        "stage5_3c_phda_formal_train755",
        "stage5_4b_dista_base_formal_train755",
        "stage5_5b_dista_path_ablation_formal_train755",
        "stage5_5c_dista_theory_ablation_formal_train755",
        "stage5_postholdout_generic_baseline_formal_train755",
    }
    phda_formal = scope == "stage5_3c_phda_formal_train755"
    dista_formal = scope == "stage5_4b_dista_base_formal_train755"
    path_ablation_formal = scope == "stage5_5b_dista_path_ablation_formal_train755"
    theory_ablation_formal = scope == "stage5_5c_dista_theory_ablation_formal_train755"
    generic_baseline_formal = scope == "stage5_postholdout_generic_baseline_formal_train755"
    v2_smoke = scope == "stage5_3a_adapter_v2_smoke3"
    phda_smoke = scope == "stage5_3b_phda_smoke3"
    dista_smoke = scope == "stage5_4a_dista_base_smoke3"
    path_ablation_smoke = scope == "stage5_5a_dista_path_ablation_smoke3"
    if formal and args.max_cycles != 143:
        raise ValueError("Formal Adapter evaluation requires the complete frozen val143 split.")
    if v2_smoke and args.max_cycles != 3:
        raise ValueError("Adapter V2 Smoke3 evaluation requires exactly three validation cycles.")
    if phda_smoke and args.max_cycles != 3:
        raise ValueError("PHDA Smoke3 evaluation requires exactly three validation cycles.")
    if dista_smoke and args.max_cycles != 3:
        raise ValueError("DISTA Base Smoke3 evaluation requires exactly three validation cycles.")
    if path_ablation_smoke and args.max_cycles != 3:
        raise ValueError("DISTA path-ablation Smoke3 evaluation requires exactly three validation cycles.")
    if not formal and not v2_smoke and not phda_smoke and not dista_smoke and not path_ablation_smoke and not 1 <= args.max_cycles <= 30:
        raise ValueError("Pilot evaluation is bounded to 1..30 val cycles.")
    dataset = Stage5PairedDomainDataset(
        registry_path,
        split="val",
        cycle_manifest=project_path(args.cycle_manifest),
        max_cycles=args.max_cycles,
        materialization_mode="target_forecast_only",
    )
    try:
        audit = dataset.preaudit_cycles()
        run_dir = project_path(config["output_dir"])
        checkpoint_path = project_path(args.checkpoint) if args.checkpoint else run_dir / "checkpoint_best.pt"
        output_dir = project_path(args.output_dir) if args.output_dir else run_dir / (
            "evaluation/val143"
            if formal
            else ("evaluation/smoke3" if (v2_smoke or phda_smoke or dista_smoke or path_ablation_smoke) else "evaluation/pilot30")
        )
        dry = {
            "status": "passed",
            "mode": config["mode"],
            "execution_scope": config["execution_scope"],
            "cycles": len(audit),
            "cycle_manifest": str(project_path(args.cycle_manifest)),
            "checkpoint": str(checkpoint_path),
            "checkpoint_status": "available" if checkpoint_path.is_file() else "planned_not_created",
            "output_dir": str(output_dir),
            "prediction_storage_mode": "metrics_only",
            "predictions_written": False,
            "source_only_comparison": True,
            "exploratory_only": True,
            "formal_development_baseline": formal,
            "adapter_version": str(config.get("adapter_version", "v1")),
            "torch_load_called": False,
            "cuda_initialized": False,
            "truth_2025_accessed": False,
        }
        if args.dry_run:
            print(json.dumps(dry, ensure_ascii=False, indent=2))
            return 0
        if (
            formal
            and not args.authorize_formal_evaluation
            and not path_ablation_formal
            and not theory_ablation_formal
        ):
            raise ValueError("Formal val143 evaluation is locked; pass --authorize-formal-evaluation after training.")
        if v2_smoke and not args.authorize_v2_smoke_evaluation:
            raise ValueError("Adapter V2 Smoke3 evaluation is locked; pass --authorize-v2-smoke-evaluation after training.")
        if phda_smoke and not args.authorize_phda_smoke_evaluation:
            raise ValueError("PHDA Smoke3 evaluation is locked; pass --authorize-phda-smoke-evaluation after training.")
        if dista_smoke and not args.authorize_dista_smoke_evaluation:
            raise ValueError("DISTA Base Smoke3 evaluation is locked; pass --authorize-dista-smoke-evaluation.")
        if path_ablation_smoke and not args.authorize_dista_path_ablation_smoke_evaluation:
            raise ValueError("DISTA path-ablation Smoke3 evaluation is locked; pass --authorize-dista-path-ablation-smoke-evaluation.")
        if path_ablation_formal and not args.authorize_dista_path_ablation_formal_evaluation:
            raise ValueError("DISTA path-ablation formal evaluation is locked; pass --authorize-dista-path-ablation-formal-evaluation.")
        if theory_ablation_formal and not args.authorize_dista_theory_ablation_formal_evaluation:
            raise ValueError("DISTA theory-ablation formal evaluation is locked; pass --authorize-dista-theory-ablation-formal-evaluation.")
        if not formal and not args.authorize_pilot30_evaluation:
            if not v2_smoke and not phda_smoke and not dista_smoke and not path_ablation_smoke:
                raise ValueError("Pilot30 evaluation is locked; pass --authorize-pilot30-evaluation after training approval.")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Pilot Adapter checkpoint is missing: {checkpoint_path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        adapter_version = str(config.get("adapter_version", "v1"))
        wrapper_args = (
            adapter_version,
            str(config.get("source_model", "temporal_transformer")),
            config.get("adapter"),
        )
        source_model = wrapper_args[1]
        model_display = str(registry["models"][source_model]["display_name"])
        adapted, source_meta = build_wrapper(registry, config["mode"], device, *wrapper_args)
        source_only, _ = build_wrapper(registry, "source_only", device, *wrapper_args)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        expected_schema = (
            (
                "stage5_generic_baseline_formal_v1"
                if generic_baseline_formal
                else (
                    "stage5_dista_base_formal_v1"
                    if dista_formal
                    else (
                        "stage5_dista_theory_ablation_formal_v1"
                        if theory_ablation_formal
                        else (
                            "stage5_dista_path_ablation_formal_v1"
                            if path_ablation_formal
                            else ("stage5_phda_adapter_formal_v1" if phda_formal else "stage5_adapter_only_v3")
                        )
                    )
                )
            )
            if formal
            else (
                "stage5_lead_conditioned_adapter_v2_smoke_v1"
                if v2_smoke
                else (
                    "stage5_phda_adapter_smoke_v1"
                    if phda_smoke
                    else ("stage5_dista_base_smoke_v1" if dista_smoke else "stage5_adapter_only_v2")
                    if not path_ablation_smoke
                    else "stage5_dista_path_ablation_smoke_v1"
                )
            )
        )
        checkpoint_format = load_evaluation_checkpoint(
            adapted,
            checkpoint,
            expected_schema=expected_schema,
            expected_scope=scope,
            expected_mode=config["mode"],
            expected_source_checkpoint_sha256=sha256_file(source_meta["checkpoint_path"]),
            allow_legacy_path_ablation_wrapper=path_ablation_formal,
        )
        adapted.eval()
        source_only.eval()

        climatology_path, climatology_manifest = ensure_climatology(registry, rebuild=False)
        ocean_path = project_path(registry["paths"]["anfc_era5_dataset"]) / "ocean_state_daily.nc"
        ocean = xr.open_dataset(ocean_path)
        climatology = xr.open_dataset(climatology_path)
        try:
            boundary = outer_boundary_mask(
                ocean.latitude.size,
                ocean.longitude.size,
                int(registry["evaluation_contract"].get("boundary_width", 2)),
            )
            adapted_rows = []
            source_rows = []
            with torch.no_grad():
                for index in range(len(dataset)):
                    sample_np = dataset[index]
                    sample = tensor_sample(sample_np, device)
                    _, adapted_predictions, _ = rollout(
                        adapted,
                        sample,
                        horizon=10,
                        loss_config=config.get("loss", {}),
                        return_predictions=True,
                    )
                    _, source_predictions, _ = rollout(
                        source_only,
                        sample,
                        horizon=10,
                        loss_config=config.get("loss", {}),
                        return_predictions=True,
                    )
                    init_time = pd.Timestamp(sample_np["metadata"]["init_time"])
                    initial = sample_np["anfc_ocean_history"][-1]
                    for lead_index in range(10):
                        common = {
                            "initial": initial,
                            "truth": sample_np["anfc_truth"][lead_index],
                            "initial_mask": sample_np["anfc_initial_mask"],
                            "target_mask": sample_np["anfc_truth_mask"][lead_index],
                            "boundary_2d": boundary,
                            "stats": dataset.stats,
                            "ocean": ocean,
                            "init_time": init_time,
                            "lead_day": lead_index + 1,
                            "boundary_mode": "persistence",
                            "climatology": climatology,
                            "valid_time": init_time + pd.Timedelta(days=lead_index + 1),
                        }
                        append_formal_metrics(
                            adapted_rows,
                            experiment={
                                "experiment_id": config.get(
                                    "experiment_id", f"STAGE5_PILOT30__{config['mode'].upper()}"
                                ),
                                "condition": "D",
                                "period": "val143_dev" if formal else "pilot30_dev",
                            },
                            model_display=model_display,
                            prediction=adapted_predictions[lead_index][0].float().cpu().numpy(),
                            **common,
                        )
                        append_formal_metrics(
                            source_rows,
                            experiment={
                                "experiment_id": "STAGE5_PILOT30__SOURCE_ONLY",
                                "condition": "D",
                                "period": "val143_dev" if formal else "pilot30_dev",
                            },
                            model_display=model_display,
                            prediction=source_predictions[lead_index][0].float().cpu().numpy(),
                            **common,
                        )
            adapted_frames = aggregate_formal_metrics(adapted_rows)
            source_frames = aggregate_formal_metrics(source_rows)
            names = (
                "metrics_by_variable.csv",
                "metrics_by_variable_lead.csv",
                "metrics_by_variable_depth.csv",
                "metrics_by_cycle.csv",
                "overall_summary.csv",
            )
            for name, frame in zip(names, adapted_frames):
                atomic_dataframe(frame, output_dir / name)
            for name, frame in zip(names, source_frames):
                atomic_dataframe(frame, output_dir / f"source_{name}")
            deltas = paired_deltas(adapted_frames[1], source_frames[1])
            atomic_dataframe(deltas, output_dir / "paired_delta_vs_source.csv")
            variable_summary = paired_deltas(adapted_frames[0], source_frames[0])
            variable_summary["relative_rmse_improvement"] = (
                variable_summary["rmse_source"] - variable_summary["rmse_adapted"]
            ) / variable_summary["rmse_source"]
            atomic_dataframe(variable_summary, output_dir / "paired_variable_summary.csv")
            primary = deltas[deltas.domain == "interior"]
            primary_variables = variable_summary[variable_summary.domain == "interior"]
            result = {
                "status": "passed",
                "mode": config["mode"],
                "source_model": source_model,
                "model": model_display,
                "cycles": len(dataset),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "checkpoint_format": checkpoint_format,
                "source_checkpoint_sha256": checkpoint["source_checkpoint_sha256"],
                "climatology_sha256": climatology_manifest["file_sha256"],
                "prediction_storage_mode": "metrics_only",
                "predictions_written": False,
                "paired_rows": len(deltas),
                "primary_interior_mean_degradation_delta": {
                    metric: float(primary[f"{metric}_degradation_delta"].mean())
                    for metric in METRICS
                },
                "five_variable_equal_weight_relative_rmse_gain": float(
                    primary_variables["relative_rmse_improvement"].mean()
                ),
                "variables_with_nonnegative_rmse_gain": int(
                    (primary_variables["relative_rmse_improvement"] >= 0).sum()
                ),
                "exploratory_only": True,
                "formal_development_baseline": formal,
                "truth_2025_accessed": False,
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
            }
            atomic_json(output_dir / "evaluation_manifest.json", result)
            qc = [
                f"# Stage5 {scope} {config['mode']} Evaluation",
                "",
                "- status: passed",
                f"- cycles: {len(dataset)}",
                f"- model: {model_display}",
                f"- checkpoint_format: {checkpoint_format}",
                "- storage: metrics-only; no prediction NetCDF written",
                "- comparison: paired against Source-only on identical val cycles",
                "- inference: development-only; cycle guards reject all 2025 truth access",
                "- truth_2025_accessed: false",
            ]
            output_dir.mkdir(parents=True, exist_ok=True)
            temporary_qc = output_dir / "qc_report.md.tmp"
            temporary_qc.write_text("\n".join(qc) + "\n", encoding="utf-8")
            os.replace(temporary_qc, output_dir / "qc_report.md")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        finally:
            ocean.close()
            climatology.close()
    finally:
        dataset.close()


if __name__ == "__main__":
    raise SystemExit(main())
