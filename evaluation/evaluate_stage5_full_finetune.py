"""Metrics-only val/Pilot evaluator for the Stage5 full-finetune baseline."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.evaluate_operational_replay_v2 import ensure_climatology
from pipeline.run_operational_replay import (
    _combine_metric_frame,
    aggregate_formal_metrics,
    append_formal_metrics,
    outer_boundary_mask,
)
from pipeline.stage5_full_finetune_common import (
    atomic_json,
    atomic_text,
    autocast_context,
    build_full_model,
    load_yaml,
    project_path,
    rollout,
    tensor_sample,
    validate_contract,
)
from pipeline.stage5_paired_domain_dataset import Stage5PairedDomainDataset
from pipeline.core import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def grouped_metrics(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(columns, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(columns, key))
        row.update(_combine_metric_frame(group))
        row["completed_cycles"] = int(group.loc[group.valid_count > 0, "init_time"].nunique())
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_model(model, dataset, device, config, ocean, climatology, label: str):
    model.eval()
    rows: list[dict[str, Any]] = []
    boundary = outer_boundary_mask(ocean.latitude.size, ocean.longitude.size, 2)
    with torch.no_grad():
        for index in range(len(dataset)):
            sample_np = dataset[index]
            sample = tensor_sample(sample_np, device)
            # Formal evaluation follows the frozen Stage3 D FP32 inference
            # semantics for both models. AMP is a training optimization only.
            with autocast_context(device, False):
                _, _, _, predictions = rollout(
                    model, sample, horizon=10, loss_config=config["loss"], return_predictions=True
                )
            init_time = pd.Timestamp(sample_np["metadata"]["init_time"])
            initial = sample_np["anfc_ocean_history"][-1]
            for lead_index in range(10):
                append_formal_metrics(
                    rows,
                    experiment={
                                "experiment_id": config["experiment_id"] if label == "full_finetune" else f"STAGE5_{str(config['model_name']).upper()}_SOURCE_ONLY_D",
                        "condition": "D",
                        "period": "val143" if len(dataset) == 143 else "pilot30",
                    },
                    model_display=(
                        f"{config.get('model_display_name', config['model_name'])} Full Fine-tune"
                        if label == "full_finetune"
                        else f"{config.get('model_display_name', config['model_name'])} Source-only"
                    ),
                    prediction=predictions[lead_index][0].float().cpu().numpy(),
                    initial=initial,
                    truth=sample_np["anfc_truth"][lead_index],
                    initial_mask=sample_np["anfc_initial_mask"],
                    target_mask=sample_np["anfc_truth_mask"][lead_index],
                    boundary_2d=boundary,
                    stats=dataset.stats,
                    ocean=ocean,
                    init_time=init_time,
                    lead_day=lead_index + 1,
                    boundary_mode="persistence",
                    climatology=climatology,
                    valid_time=init_time + pd.Timedelta(days=lead_index + 1),
                )
            if (index + 1) % 10 == 0 or index + 1 == len(dataset):
                print(json.dumps({"stage": "evaluation_progress", "model": label, "done": index + 1, "total": len(dataset)}), flush=True)
    frame = pd.DataFrame(rows)
    standard = aggregate_formal_metrics(rows)
    season = grouped_metrics(
        frame,
        ["experiment_id", "model", "condition", "period", "season", "domain", "variable", "unit"],
    )
    return standard, season


def save_standard(output_dir: Path, prefix: str, standard, season) -> dict[str, str]:
    names = (
        "metrics_by_variable.csv",
        "metrics_by_variable_lead.csv",
        "metrics_by_variable_depth.csv",
        "metrics_by_cycle.csv",
        "overall_summary.csv",
    )
    hashes = {}
    for name, frame in zip(names, standard):
        path = output_dir / f"{prefix}{name}"
        atomic_csv(path, frame)
        hashes[path.name] = sha256_file(path)
    season_path = output_dir / f"{prefix}metrics_by_season.csv"
    atomic_csv(season_path, season)
    hashes[season_path.name] = sha256_file(season_path)
    return hashes


def source_frozen_crosscheck(
    source_cycle: pd.DataFrame, cycles: set[str], model_name: str
) -> dict[str, Any]:
    path = ROOT / "runs/forecast_simulation/d_direct_anfc_gfs_dev_2022_2024" / model_name / "metrics_by_cycle.csv"
    frozen = pd.read_csv(path, dtype={"init_time": str})
    frozen = frozen[frozen.init_time.isin(cycles)].copy()
    keys = ["init_time", "lead_day", "domain", "variable"]
    metrics = ["model_sse", "persistence_sse", "absolute_error_sum", "valid_count", "rmse", "mae", "persistence_skill", "r2", "acc"]
    merged = source_cycle[keys + metrics].merge(
        frozen[keys + metrics], on=keys, suffixes=("_reevaluated", "_frozen"), validate="one_to_one"
    )
    maximum = {}
    for metric in metrics:
        if metric == "valid_count":
            continue
        differences = np.abs(merged[f"{metric}_reevaluated"] - merged[f"{metric}_frozen"])
        maximum[metric] = float(differences.max()) if differences.notna().any() else math.nan
    count_match = bool((merged.valid_count_reevaluated == merged.valid_count_frozen).all())
    # Floating aggregation can differ at very small roundoff; predictions and evaluator are otherwise identical.
    passed = count_match and maximum["rmse"] <= 1e-6 and maximum["mae"] <= 1e-6
    return {
        "status": "passed" if passed else "failed",
        "frozen_path": str(path),
        "frozen_path_sha256": sha256_file(path),
        "rows": len(merged),
        "valid_count_identical": count_match,
        "maximum_absolute_differences": maximum,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--cycle-manifest")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-cycles", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--authorize-evaluation", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    registry = load_yaml(config["registry"])
    validate_contract(
        config,
        registry,
        allow_released_holdout_for_development_evaluation=True,
    )
    model_record = registry["models"][config["model_name"]]
    config["model_display_name"] = str(
        model_record.get("display_name", config["model_name"])
    )
    cycle_manifest = project_path(args.cycle_manifest or config["val_manifest"])
    checkpoint_path = project_path(args.checkpoint or Path(config["output_dir"]) / "checkpoint_best.pt")
    output_dir = project_path(args.output_dir or Path(config["output_dir"]) / "evaluation/val143")
    expected_cycles = args.max_cycles or len(pd.read_csv(cycle_manifest))
    dry = {
        "status": "passed",
        "cycle_manifest": str(cycle_manifest),
        "cycle_manifest_sha256": sha256_file(cycle_manifest),
        "cycles": expected_cycles,
        "checkpoint": str(checkpoint_path),
        "checkpoint_status": "available" if checkpoint_path.is_file() else "planned_not_created",
        "output_dir": str(output_dir),
        "prediction_storage_mode": "metrics_only",
        "torch_load_called": False,
        "cuda_initialized": False,
        "truth_2025_accessed": False,
    }
    if args.dry_run:
        print(json.dumps(dry, ensure_ascii=False, indent=2))
        return 0
    if not args.authorize_evaluation:
        raise ValueError("Full-finetune evaluation is locked; pass --authorize-evaluation.")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    dataset = Stage5PairedDomainDataset(
        config["registry"], split="val", cycle_manifest=cycle_manifest,
        max_cycles=args.max_cycles, materialization_mode="target_forecast_only"
    )
    try:
        audit = dataset.preaudit_cycles()
        if any(pd.Timestamp(row["init_time"]).year >= 2025 for row in audit):
            raise ValueError("2025 cycle rejected during evaluation.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("Evaluation requires CUDA.")
        fine, metadata = build_full_model(registry, config, device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("schema_version") != "stage5_full_finetune_v1":
            raise ValueError("Full-finetune checkpoint schema mismatch.")
        if checkpoint.get("pretrained_checkpoint_sha256") != metadata["checkpoint_sha256"]:
            raise ValueError("Full-finetune source checkpoint SHA mismatch.")
        fine.load_state_dict(checkpoint["model"], strict=True)
        source, _ = build_full_model(registry, config, device)
        climatology_path, climatology_manifest = ensure_climatology(registry, rebuild=False)
        ocean_path = project_path(registry["paths"]["anfc_era5_dataset"]) / "ocean_state_daily.nc"
        ocean = xr.open_dataset(ocean_path)
        climatology = xr.open_dataset(climatology_path)
        try:
            fine_standard, fine_season = evaluate_model(fine, dataset, device, config, ocean, climatology, "full_finetune")
            source_standard, source_season = evaluate_model(source, dataset, device, config, ocean, climatology, "source_only")
        finally:
            ocean.close()
            climatology.close()
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_hashes = {}
        artifact_hashes.update(save_standard(output_dir, "", fine_standard, fine_season))
        artifact_hashes.update(save_standard(output_dir, "source_", source_standard, source_season))
        crosscheck = source_frozen_crosscheck(
            source_standard[3],
            {row["init_time"] for row in audit},
            str(config["model_name"]),
        )
        if crosscheck["status"] != "passed":
            raise RuntimeError(f"Re-evaluated Source-only does not match frozen Stage3 D: {crosscheck}")
        atomic_json(output_dir / "source_frozen_crosscheck.json", crosscheck)
        result = {
            "status": "passed",
            "experiment_id": config["experiment_id"],
            "cycles": len(dataset),
            "cycle_manifest": str(cycle_manifest),
            "cycle_manifest_sha256": sha256_file(cycle_manifest),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "pretrained_checkpoint_sha256": metadata["checkpoint_sha256"],
            "climatology_sha256": climatology_manifest["file_sha256"],
            "prediction_storage_mode": "metrics_only",
            "predictions_written": False,
            "source_frozen_crosscheck": crosscheck,
            "artifact_sha256": artifact_hashes,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "truth_2025_accessed": False,
        }
        atomic_json(output_dir / "evaluation_manifest.json", result)
        atomic_text(output_dir / "qc_report.md", "\n".join([
            "# Stage5 Full Fine-tune Evaluation QC", "", "- status: passed",
            f"- cycles: {len(dataset)}", "- prediction storage: metrics-only",
            "- Source-only frozen Stage3 crosscheck: passed",
            "- primary domain: interior", "- truth_2025_accessed: false", "",
        ]))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    finally:
        dataset.close()


if __name__ == "__main__":
    raise SystemExit(main())
