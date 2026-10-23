"""Metrics-only evaluator for frozen DISTA Base checkpoints on Stage6 2025."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import xarray as xr
import yaml

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.evaluate_operational_replay_v2 import ensure_climatology
from pipeline.evaluate_stage5_adapter_pilot import load_evaluation_checkpoint
from pipeline.run_operational_replay import (
    FAILED_CYCLE_COLUMNS,
    PREDICTION_MANIFEST_COLUMNS,
    append_formal_metrics,
    aggregate_formal_metrics,
    outer_boundary_mask,
)
from pipeline.stage6_holdout_dataset import Stage6HoldoutDomainDataset
from pipeline.train_stage5_adapter import (
    build_wrapper,
    load_yaml,
    rollout,
    sha256_file,
    tensor_sample,
)


ROOT = Path(__file__).resolve().parents[1]
METRIC_FILES = (
    "metrics_by_variable.csv",
    "metrics_by_variable_lead.csv",
    "metrics_by_variable_depth.csv",
    "metrics_by_cycle.csv",
    "overall_summary.csv",
)


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame, columns: tuple[str, ...] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if columns is not None:
        frame = frame.reindex(columns=list(columns))
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def validate_dista_config(config: dict, experiment: dict) -> None:
    if config.get("adapter_version") != "dista_base":
        raise ValueError("Stage6 DISTA holdout requires adapter_version=dista_base.")
    if config.get("execution_scope") != "stage5_4b_dista_base_formal_train755":
        raise ValueError("Stage6 DISTA holdout accepts only the frozen formal train755 scope.")
    if config.get("mode") != "dual_adapter":
        raise ValueError("Stage6 DISTA holdout requires mode=dual_adapter.")
    if config.get("source_model") != experiment.get("model"):
        raise ValueError("DISTA config source_model differs from the registered Stage6 model.")
    policies = config.get("policies", {})
    expected = {
        "adapter_placement": "physical_input_and_tendency_output",
        "forcing_trajectory_policy": "full_10day_physical_channels",
        "ocean_adapter_history_policy": "observed_product_frames_only",
        "calendar_channel_policy": "bitwise_passthrough",
        "output_adapter": True,
        "valid_mask": "initial_mask AND target_mask",
    }
    for key, value in expected.items():
        if policies.get(key) != value:
            raise ValueError(f"Frozen DISTA policy mismatch: {key}={policies.get(key)!r}")


def completed_output(path: Path, expected_cycles: int) -> bool:
    manifest_path = path / "evaluation_manifest.json"
    if not manifest_path.is_file():
        return False
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    required = [
        path / name
        for name in (
            *METRIC_FILES,
            "prediction_manifest.csv",
            "failed_cycles.csv",
            "run_manifest.json",
            "data_manifest.json",
            "checkpoint_manifest.json",
            "config_snapshot.yaml",
            "qc_report.md",
        )
    ]
    return (
        payload.get("status") == "passed"
        and int(payload.get("cycles", -1)) == expected_cycles
        and int(payload.get("failed_cycles", -1)) == 0
        and all(item.is_file() for item in required)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="configs/experiment_registry_2022_2025.yaml")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--cycle-manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--authorize-stage6-holdout", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry_path = project_path(args.registry)
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8-sig"))
    matches = [
        row for row in registry.get("experiments", [])
        if row.get("experiment_id") == args.experiment_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one Stage6 experiment, found {len(matches)}.")
    experiment = matches[0]
    if experiment.get("mode") != "dista_base":
        raise ValueError("Stage6 DISTA evaluator accepts mode=dista_base only.")

    config_path = project_path(args.config)
    checkpoint_path = project_path(args.checkpoint)
    output_dir = project_path(args.output_dir)
    if config_path.resolve() != project_path(experiment["adapter_config"]).resolve():
        raise ValueError("CLI DISTA config differs from the registered frozen config.")
    if checkpoint_path.resolve() != project_path(experiment["adapter_checkpoint"]).resolve():
        raise ValueError("CLI DISTA checkpoint differs from the registered frozen checkpoint.")
    if output_dir.resolve() != project_path(experiment["output_dir"]).resolve():
        raise ValueError("CLI output directory differs from the registered Stage6 output.")
    config = load_yaml(config_path)
    validate_dista_config(config, experiment)

    dataset = Stage6HoldoutDomainDataset(
        registry_path,
        experiment_id=args.experiment_id,
        cycle_manifest=args.cycle_manifest,
    )
    try:
        audits = dataset.preaudit_cycles()
        expected_cycles = int(registry["holdout_release"]["expected_cycles"])
        if len(audits) != expected_cycles:
            raise ValueError("Stage6 preaudit did not cover the complete released holdout.")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        dry = {
            "status": "passed",
            "experiment_id": args.experiment_id,
            "source_model": experiment["model"],
            "cycles": len(audits),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "output_dir": str(output_dir),
            "prediction_storage_mode": "metrics_only",
            "predictions_written": False,
            "torch_load_called": False,
            "cuda_initialized": False,
            "truth_2025_access_authorized": True,
        }
        if args.dry_run:
            print(json.dumps(dry, ensure_ascii=False, indent=2))
            return 0
        if not args.authorize_stage6_holdout:
            raise PermissionError("Pass --authorize-stage6-holdout only for the frozen Stage6 run.")
        if output_dir.exists() and any(output_dir.iterdir()):
            if args.skip_complete and completed_output(output_dir, expected_cycles):
                print(json.dumps({
                    "status": "skipped_complete",
                    "experiment_id": args.experiment_id,
                    "output_dir": str(output_dir),
                    "cycles": expected_cycles,
                }, ensure_ascii=False, indent=2))
                return 0
            raise FileExistsError(
                f"Stage6 DISTA output already exists but is not a validated complete run: {output_dir}"
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        adapted, source_meta = build_wrapper(
            registry,
            "dual_adapter",
            device,
            "dista_base",
            str(config["source_model"]),
            config.get("adapter"),
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_format = load_evaluation_checkpoint(
            adapted,
            checkpoint,
            expected_schema="stage5_dista_base_formal_v1",
            expected_scope="stage5_4b_dista_base_formal_train755",
            expected_mode="dual_adapter",
            expected_source_checkpoint_sha256=sha256_file(source_meta["checkpoint_path"]),
            allow_legacy_path_ablation_wrapper=False,
        )
        adapted.eval()

        climatology_path, climatology_manifest = ensure_climatology(registry, rebuild=False)
        ocean = xr.open_dataset(dataset.ocean_path)
        climatology = xr.open_dataset(climatology_path)
        started_at = datetime.now(timezone.utc)
        metric_rows: list[dict] = []
        prediction_rows: list[dict] = []
        failed_rows: list[dict] = []
        try:
            boundary = outer_boundary_mask(
                ocean.latitude.size,
                ocean.longitude.size,
                int(registry["evaluation_contract"].get("boundary_width", 2)),
            )
            with torch.no_grad():
                for index in range(len(dataset)):
                    audit = audits[index]
                    print(json.dumps({
                        "event": "cycle_started",
                        "experiment_id": args.experiment_id,
                        "sequence": int(audit["sequence"]),
                        "total_cycles": len(dataset),
                        "init_time": audit["init_time"],
                    }, ensure_ascii=False), flush=True)
                    cycle_started = datetime.now(timezone.utc)
                    try:
                        sample_np = dataset[index]
                        sample = tensor_sample(sample_np, device)
                        _, predictions, _ = rollout(
                            adapted,
                            sample,
                            horizon=10,
                            loss_config=config.get("loss", {}),
                            return_predictions=True,
                        )
                        init_time = pd.Timestamp(audit["init_time"])
                        initial = sample_np["anfc_ocean_history"][-1]
                        for lead_index, prediction in enumerate(predictions):
                            append_formal_metrics(
                                metric_rows,
                                experiment={
                                    "experiment_id": args.experiment_id,
                                    "condition": "D",
                                    "period": "test_2025",
                                },
                                model_display=registry["models"][experiment["model"]]["display_name"],
                                prediction=prediction[0].float().cpu().numpy(),
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
                        prediction_rows.append({
                            "sequence": int(audit["sequence"]),
                            "init_time": audit["init_time"],
                            "role": audit["role"],
                            "status": "complete",
                            "prediction_file": "",
                            "prediction_sha256": "",
                            "size_bytes": 0,
                            "gfs_source_file": audit["gfs_source_file"],
                            "gfs_local_index": audit["gfs_local_index"],
                            "gfs_sha256": audit["gfs_sha256"],
                            "started_at": cycle_started.isoformat(),
                            "completed_at": datetime.now(timezone.utc).isoformat(),
                            "error_message": "",
                        })
                        print(json.dumps({
                            "event": "cycle_completed",
                            "experiment_id": args.experiment_id,
                            "sequence": int(audit["sequence"]),
                            "total_cycles": len(dataset),
                            "init_time": audit["init_time"],
                        }, ensure_ascii=False), flush=True)
                    except Exception as exc:
                        failed_rows.append({
                            "sequence": int(audit["sequence"]),
                            "init_time": audit["init_time"],
                            "stage": "evaluation",
                            "error_type": exc.__class__.__name__,
                            "error_message": str(exc),
                            "traceback_file": "",
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        })
                        prediction_rows.append({
                            "sequence": int(audit["sequence"]),
                            "init_time": audit["init_time"],
                            "role": audit["role"],
                            "status": "failed",
                            "prediction_file": "",
                            "prediction_sha256": "",
                            "size_bytes": 0,
                            "gfs_source_file": audit["gfs_source_file"],
                            "gfs_local_index": audit["gfs_local_index"],
                            "gfs_sha256": audit["gfs_sha256"],
                            "started_at": cycle_started.isoformat(),
                            "completed_at": datetime.now(timezone.utc).isoformat(),
                            "error_message": str(exc),
                        })
                        print(json.dumps({
                            "event": "cycle_failed",
                            "experiment_id": args.experiment_id,
                            "sequence": int(audit["sequence"]),
                            "init_time": audit["init_time"],
                            "error_type": exc.__class__.__name__,
                            "error_message": str(exc),
                        }, ensure_ascii=False), flush=True)
        finally:
            ocean.close()
            climatology.close()

        output_dir.mkdir(parents=True, exist_ok=True)
        frames = aggregate_formal_metrics(metric_rows)
        for name, frame in zip(METRIC_FILES, frames):
            atomic_csv(output_dir / name, frame)
        atomic_csv(
            output_dir / "prediction_manifest.csv",
            pd.DataFrame(prediction_rows),
            tuple(PREDICTION_MANIFEST_COLUMNS),
        )
        atomic_csv(
            output_dir / "failed_cycles.csv",
            pd.DataFrame(failed_rows),
            tuple(FAILED_CYCLE_COLUMNS),
        )
        config_snapshot = output_dir / "config_snapshot.yaml"
        temporary_config = config_snapshot.with_name(config_snapshot.name + ".tmp")
        temporary_config.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        os.replace(temporary_config, config_snapshot)
        atomic_json(output_dir / "data_manifest.json", {
            "registry": str(registry_path),
            "cycle_manifest": str(project_path(args.cycle_manifest)),
            "cycles": audits,
            "normalization_sha256": sha256_file(dataset.normalization_path),
            "climatology_sha256": climatology_manifest["file_sha256"],
            "truth_dataset": str(dataset.ocean_path),
            "future_forcing": "GFS",
            "truth_2025_accessed": True,
        })
        atomic_json(output_dir / "checkpoint_manifest.json", {
            "checkpoint_format": checkpoint_format,
            "adapter_checkpoint": str(checkpoint_path),
            "adapter_checkpoint_sha256": sha256_file(checkpoint_path),
            "source_checkpoint": str(source_meta["checkpoint_path"]),
            "source_checkpoint_sha256": sha256_file(source_meta["checkpoint_path"]),
            "source_config": str(source_meta["model_config_path"]),
            "source_config_sha256": sha256_file(source_meta["model_config_path"]),
        })
        completed_at = datetime.now(timezone.utc)
        run_manifest = {
            "runner_version": "stage6_dista_holdout_v1",
            "status": "completed" if not failed_rows else "failed",
            "experiment_id": args.experiment_id,
            "stage": 6,
            "mode": "dista_base",
            "model": experiment["model"],
            "period": "test_2025",
            "cycle_manifest": str(project_path(args.cycle_manifest)),
            "cycle_manifest_sha256": sha256_file(project_path(args.cycle_manifest)),
            "completed_cycles": len(prediction_rows) - len(failed_rows),
            "failed_cycles": len(failed_rows),
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "duration_seconds": (completed_at - started_at).total_seconds(),
            "gpu_id": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "prediction_storage_mode": "metrics_only",
            "predictions_written": False,
            "code_snapshot_id": registry["holdout_release"]["code_snapshot_id"],
            "truth_2025_accessed": True,
        }
        atomic_json(output_dir / "run_manifest.json", run_manifest)
        evaluation_manifest = {
            "status": "passed" if not failed_rows else "failed",
            "experiment_id": args.experiment_id,
            "model": registry["models"][experiment["model"]]["display_name"],
            "method": "DISTA Base",
            "cycles": len(dataset),
            "completed_cycles": run_manifest["completed_cycles"],
            "failed_cycles": len(failed_rows),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "climatology_sha256": climatology_manifest["file_sha256"],
            "prediction_storage_mode": "metrics_only",
            "predictions_written": False,
            "truth_2025_accessed": True,
            "evaluated_at": completed_at.isoformat(),
        }
        atomic_json(output_dir / "evaluation_manifest.json", evaluation_manifest)
        qc = [
            "# Stage6 DISTA Base 2025 Holdout Evaluation",
            "",
            f"- status: {evaluation_manifest['status']}",
            f"- experiment_id: {args.experiment_id}",
            f"- model: {evaluation_manifest['model']}",
            f"- cycles: {len(dataset)}",
            f"- completed_cycles: {run_manifest['completed_cycles']}",
            f"- failed_cycles: {len(failed_rows)}",
            "- prediction_storage_mode: metrics_only",
            "- predictions_written: false",
            "- truth_2025_accessed: true",
            "- method/checkpoint changes after this evaluation: forbidden",
            "",
        ]
        qc_path = output_dir / "qc_report.md"
        temporary_qc = qc_path.with_name(qc_path.name + ".tmp")
        temporary_qc.write_text("\n".join(qc), encoding="utf-8")
        os.replace(temporary_qc, qc_path)
        print(json.dumps(evaluation_manifest, ensure_ascii=False, indent=2))
        return 0 if not failed_rows else 1
    finally:
        dataset.close()


if __name__ == "__main__":
    raise SystemExit(main())

