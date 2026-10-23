"""Train an authorized Stage5 registry-model full-finetune baseline."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.core import denorm, load_stats, sha256_file
from pipeline.stage5_full_finetune_common import (
    VARIABLE_SLICES,
    atomic_json,
    atomic_text,
    atomic_torch_save,
    autocast_context,
    build_full_model,
    gradient_module_names,
    load_yaml,
    make_scaler,
    module_gradient_summary,
    project_path,
    rollout,
    state_dict_sha256,
    tensor_sample,
    validate_contract,
)
from pipeline.stage5_paired_domain_dataset import Stage5PairedDomainDataset


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_OUTPUTS = (
    "config_snapshot.yaml",
    "run_manifest.json",
    "data_manifest.json",
    "checkpoint_manifest.json",
    "training_log.csv",
    "validation_log.csv",
    "per_variable_training_loss.csv",
    "gradient_summary.csv",
    "runtime_summary.json",
    "checkpoint_best.pt",
    "checkpoint_last.pt",
    "resume_report.json",
    "qc_report.md",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    pd.DataFrame(rows, columns=columns).to_csv(temporary, index=False)
    os.replace(temporary, path)


def manifest_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"sequence": int, "init_time": str, "role": str})
    if list(frame.columns) != ["sequence", "init_time", "role"]:
        raise ValueError(f"Unexpected cycle manifest schema: {path}")
    if frame.init_time.duplicated().any() or frame.sequence.tolist() != list(range(1, len(frame) + 1)):
        raise ValueError(f"Cycle manifest uniqueness/order failed: {path}")
    return frame


def validate_manifests(config: dict[str, Any]) -> dict[str, Any]:
    train_path = project_path(config["train_manifest"])
    val_path = project_path(config["val_manifest"])
    train = manifest_frame(train_path)
    val = manifest_frame(val_path)
    if len(train) != 755 or len(val) != 143:
        raise ValueError(f"Formal split must be train755/val143, got {len(train)}/{len(val)}.")
    if sha256_file(train_path) != config["train_manifest_sha256"]:
        raise ValueError("train755 manifest SHA256 mismatch.")
    if sha256_file(val_path) != config["val_manifest_sha256"]:
        raise ValueError("val143 manifest SHA256 mismatch.")
    train_set = set(train.init_time)
    val_set = set(val.init_time)
    if not train_set.isdisjoint(val_set):
        raise ValueError("train/val manifests overlap.")
    for label, frame in (("train", train), ("val", val)):
        valid_max = pd.to_datetime(frame.init_time).max() + pd.Timedelta(days=10)
        if valid_max > pd.Timestamp("2024-12-31"):
            raise ValueError(f"{label} target valid_time enters 2025.")
    return {
        "train_path": train_path,
        "val_path": val_path,
        "train_frame": train,
        "val_frame": val,
        "train_sha256": sha256_file(train_path),
        "val_sha256": sha256_file(val_path),
    }


def prefetched(dataset, indexes: Iterable[int], enabled: bool):
    indexes = list(indexes)
    if not indexes:
        return
    if not enabled:
        for index in indexes:
            started = time.perf_counter()
            yield dataset[index], time.perf_counter() - started
        return
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="stage5-data") as executor:
        future = executor.submit(dataset.__getitem__, indexes[0])
        for position, _ in enumerate(indexes):
            wait_started = time.perf_counter()
            sample = future.result()
            wait_seconds = time.perf_counter() - wait_started
            if position + 1 < len(indexes):
                future = executor.submit(dataset.__getitem__, indexes[position + 1])
            yield sample, wait_seconds


def horizon_for_epoch(epoch: int, canary: bool = False) -> int:
    if canary:
        return 10
    if epoch < 10:
        return 1
    if epoch < 25:
        return 3
    return 10


def evaluate_loss(model, dataset, device, config, *, prefetch: bool) -> tuple[float, list[dict[str, Any]], float]:
    model.eval()
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with torch.no_grad():
        for sample_np, wait_seconds in prefetched(dataset, range(len(dataset)), prefetch):
            sample = tensor_sample(sample_np, device)
            with autocast_context(device, bool(config["amp"])):
                loss, variable, contribution, _ = rollout(
                    model, sample, horizon=10, loss_config=config["loss"]
                )
            value = float(loss.detach().cpu())
            row = {
                "init_time": sample_np["metadata"]["init_time"],
                "loss": value,
                "horizon": 10,
                "data_wait_seconds": wait_seconds,
                "finite": math.isfinite(value),
            }
            row.update({f"{name}_loss": float(variable[name].detach().cpu()) for name in VARIABLE_SLICES})
            row.update(
                {f"{name}_weighted_contribution": float(contribution[name].detach().cpu()) for name in VARIABLE_SLICES}
            )
            rows.append(row)
    return float(np.mean([row["loss"] for row in rows])), rows, time.perf_counter() - started


def checkpoint_payload(
    model,
    optimizer,
    scheduler,
    scaler,
    *,
    epoch: int,
    optimizer_step: int,
    best_metric: float,
    config_path: Path,
    config: dict[str, Any],
    metadata: dict[str, Any],
    manifests: dict[str, Any],
    code_snapshot_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": "stage5_full_finetune_v1",
        "experiment_id": config["experiment_id"],
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": int(epoch),
        "optimizer_step": int(optimizer_step),
        "best_metric": float(best_metric),
        "config_sha256": sha256_file(config_path),
        "code_snapshot_id": code_snapshot_id,
        "train_manifest_sha256": manifests["train_sha256"],
        "val_manifest_sha256": manifests["val_sha256"],
        "normalization_sha256": config["normalization_sha256"],
        "pretrained_checkpoint_sha256": metadata["checkpoint_sha256"],
        "architecture": metadata["architecture"],
        "cin": metadata["cin"],
        "history_days": metadata["history_days"],
        "forcing_history_days": 0,
        "forcing_variables": metadata["forcing_variables"],
        "forecast_horizon": 10,
        "saved_at": utc_now(),
    }


def load_resume(path, model, optimizer, scheduler, scaler, *, config_path, config, manifests, metadata, code_snapshot_id):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "schema_version": "stage5_full_finetune_v1",
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_file(config_path),
        "code_snapshot_id": code_snapshot_id,
        "train_manifest_sha256": manifests["train_sha256"],
        "val_manifest_sha256": manifests["val_sha256"],
        "normalization_sha256": config["normalization_sha256"],
        "pretrained_checkpoint_sha256": metadata["checkpoint_sha256"],
    }
    mismatches = {key: (checkpoint.get(key), value) for key, value in expected.items() if checkpoint.get(key) != value}
    if mismatches:
        raise ValueError(f"Resume checkpoint contract mismatch: {mismatches}")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def parity_only(config, registry, output_dir: Path, tolerance: float) -> int:
    manifest = project_path("configs/cycle_manifests/stage5_adapter_train_smoke3.csv")
    dataset = Stage5PairedDomainDataset(
        config["registry"], split="train", cycle_manifest=manifest, max_cycles=3,
        materialization_mode="target_forecast_only"
    )
    rows = []
    try:
        audit = dataset.preaudit_cycles()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, metadata = build_full_model(registry, config, device)
        model.eval()
        stats = load_stats(project_path(config["normalization_path"]))
        with torch.no_grad():
            for index in range(len(dataset)):
                sample_np = dataset[index]
                sample = tensor_sample(sample_np, device)
                # Frozen Stage3 D replay and the established Stage5 source-only
                # parity path use FP32 inference.  Do not introduce AMP into
                # this zero-step equivalence check.
                with autocast_context(device, False):
                    _, _, _, predictions = rollout(
                        model, sample, horizon=10, loss_config=config["loss"], return_predictions=True
                    )
                init_time = pd.Timestamp(sample_np["metadata"]["init_time"])
                model_name = str(config["model_name"])
                frozen_path = ROOT / "runs/forecast_simulation/d_direct_anfc_gfs_dev_2022_2024" / model_name / f"prediction_{init_time:%Y%m%d}.nc"
                if not frozen_path.is_file():
                    raise FileNotFoundError(f"Frozen Stage3 prediction missing: {frozen_path}")
                with xr.open_dataset(frozen_path) as frozen:
                    expected_valid = pd.date_range(init_time + pd.Timedelta(days=1), periods=10)
                    if not np.array_equal(pd.DatetimeIndex(frozen.valid_time.values), expected_valid):
                        raise ValueError("Frozen valid_time mismatch.")
                    for lead in range(10):
                        for name, channels in VARIABLE_SLICES.items():
                            normalized = predictions[lead][0, channels].float().cpu().numpy()
                            physical = denorm(normalized, stats["ocean"][name])
                            if name == "zos":
                                physical = physical[0]
                            direct = frozen[name].isel(lead_day=lead).values.astype(np.float32)
                            difference = np.abs(physical.astype(np.float64) - direct.astype(np.float64))
                            rows.append({
                                "init_time": init_time.date().isoformat(),
                                "lead_day": lead + 1,
                                "variable": name,
                                "shape_identical": tuple(physical.shape) == tuple(direct.shape),
                                "max_absolute_difference": float(np.nanmax(difference)),
                                "gfs_local_index": sample_np["metadata"]["gfs_local_index"],
                                "valid_time": expected_valid[lead].date().isoformat(),
                            })
        frame = pd.DataFrame(rows)
        maximum = float(frame.max_absolute_difference.max())
        passed = bool(frame.shape_identical.all() and maximum <= tolerance)
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_csv(output_dir / "parity_by_cycle_lead_variable.csv", rows)
        result = {
            "status": "passed" if passed else "failed",
            "cycles": [row["init_time"] for row in audit],
            "rows": len(rows),
            "max_absolute_difference": maximum,
            "tolerance": tolerance,
            "source_checkpoint_sha256": metadata["checkpoint_sha256"],
            "parameter_hash_identical_after_load": metadata["source_parameter_sha256"] == metadata["loaded_parameter_sha256"],
            "truth_2025_accessed": False,
        }
        atomic_json(output_dir / "parity_summary.json", result)
        atomic_text(output_dir / "qc_report.md", f"# Stage5 Full Fine-tune Zero-step Parity\n\n- status: {result['status']}\n- max absolute difference: {maximum:.9g}\n- tolerance: {tolerance:.9g}\n- truth_2025_accessed: false\n")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if passed else 1
    finally:
        dataset.close()


def run_gradient_audit(model, sample, device, config) -> list[dict[str, Any]]:
    rows = []
    model.train()
    for label, horizon in (("single_lead_backward", 1), ("full_10day_backward", 10)):
        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with autocast_context(device, bool(config["amp"])):
            loss, _, _, _ = rollout(model, sample, horizon=horizon, loss_config=config["loss"])
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        row = {
            "audit": label,
            "horizon": horizon,
            "loss": float(loss.detach().cpu()),
            "seconds": time.perf_counter() - started,
            "finite": bool(torch.isfinite(loss).item()),
            **module_gradient_summary(model),
        }
        rows.append(row)
    model.zero_grad(set_to_none=True)
    return rows


def micro_overfit(registry, config, sample_np, device, steps: int = 20) -> dict[str, Any]:
    model, _ = build_full_model(registry, config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["model"]["learning_rate"]),
        weight_decay=float(config["model"]["weight_decay"])
    )
    scaler = make_scaler(device, bool(config["amp"]))
    sample = tensor_sample(sample_np, device)
    losses = []
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, bool(config["amp"])):
            loss, _, _, _ = rollout(model, sample, horizon=1, loss_config=config["loss"])
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["model"]["gradient_clip_norm"]))
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return {
        "steps": steps,
        "initial_loss": losses[0],
        "best_loss": min(losses),
        "final_loss": losses[-1],
        "loss_reduction_fraction": (losses[0] - losses[-1]) / max(abs(losses[0]), 1e-12),
        "finite": bool(np.isfinite(losses).all()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--parity-only", action="store_true")
    parser.add_argument("--parity-tolerance", type=float, default=1e-6)
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--authorize-full-training", action="store_true")
    parser.add_argument("--resume", nargs="?", const="checkpoint_last.pt")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-train-cycles", type=int)
    parser.add_argument("--max-val-cycles", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--max-optimizer-steps", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml(config_path)
    registry = load_yaml(config["registry"])
    validate_contract(config, registry)
    manifests = validate_manifests(config)
    normalization_path = project_path(config["normalization_path"])
    if sha256_file(normalization_path) != config["normalization_sha256"]:
        raise ValueError("Frozen normalization SHA256 mismatch.")
    model_name = str(config["model_name"])
    record = registry["models"][model_name]
    checkpoint_path = project_path(record["checkpoint"])
    if sha256_file(checkpoint_path) != str(config["pretrained_checkpoint_sha256"]):
        raise ValueError(f"Registry {model_name} checkpoint SHA256 changed; stopping.")

    output_dir = project_path(args.output_dir or config["output_dir"])
    parity_dir = ROOT / "runs/adaptation/stage5_full_finetune/source_only_parity"
    dry = {
        "status": "passed",
        "experiment_id": config["experiment_id"],
        "condition": config["condition"],
        "train_cycles": 755,
        "val_cycles": 143,
        "train_manifest_sha256": manifests["train_sha256"],
        "val_manifest_sha256": manifests["val_sha256"],
        "pretrained_checkpoint": record["checkpoint"],
        "pretrained_checkpoint_sha256": sha256_file(checkpoint_path),
        "normalization_sha256": sha256_file(normalization_path),
        "learning_rate": config["model"]["learning_rate"],
        "weight_decay": config["model"]["weight_decay"],
        "horizon_schedule": config["model"]["progressive_horizon_epochs"],
        "torch_load_called": False,
        "cuda_initialized": False,
        "truth_2025_accessed": False,
        "holdout_release_approved": registry["holdout_release"]["approved"],
    }
    if args.dry_run:
        print(json.dumps(dry, ensure_ascii=False, indent=2))
        return 0
    if args.parity_only:
        return parity_only(config, registry, parity_dir, args.parity_tolerance)

    if args.canary:
        train_limit = args.max_train_cycles or 3
        val_limit = args.max_val_cycles or 3
        max_epochs = args.max_epochs or 1
        max_steps = args.max_optimizer_steps or 3
        if train_limit > 3 or val_limit > 3 or max_epochs > 2 or max_steps > 6:
            raise ValueError("Canary is bounded to 3 train, 3 val, 2 epochs and 6 optimizer steps.")
    else:
        if not args.authorize_full_training:
            raise ValueError("Formal train755/val143 is locked; pass --authorize-full-training after canary passes.")
        if any(value is not None for value in (args.max_train_cycles, args.max_val_cycles, args.max_epochs, args.max_optimizer_steps)):
            raise ValueError("Formal training forbids cycle/epoch/step truncation overrides.")
        train_limit, val_limit = 755, 143
        max_epochs = int(config["model"]["max_epochs"])
        max_steps = None

    train_dataset = Stage5PairedDomainDataset(
        config["registry"], split="train", cycle_manifest=config["train_manifest"],
        max_cycles=train_limit, materialization_mode="target_forecast_only"
    )
    val_dataset = Stage5PairedDomainDataset(
        config["registry"], split="val", cycle_manifest=config["val_manifest"],
        max_cycles=val_limit, materialization_mode="target_forecast_only"
    )
    session_started = time.time()
    started_at = utc_now()
    try:
        train_audit = train_dataset.preaudit_cycles()
        val_audit = val_dataset.preaudit_cycles()
        if any(pd.Timestamp(row["init_time"]).year >= 2025 for row in train_audit + val_audit):
            raise ValueError("2025 cycle reached after dataset audit.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("Full fine-tuning requires CUDA.")
        torch.manual_seed(int(config["seed"]))
        np.random.seed(int(config["seed"]))
        torch.cuda.manual_seed_all(int(config["seed"]))
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats()
        model, metadata = build_full_model(registry, config, device)
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        if trainable_parameters != total_parameters:
            raise ValueError("Full fine-tuning did not enable every model parameter.")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(config["model"]["learning_rate"]),
            weight_decay=float(config["model"]["weight_decay"])
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        scaler = make_scaler(device, bool(config["amp"]))
        code_snapshot_path = project_path(
            config.get("code_snapshot_manifest", registry["paths"]["code_snapshot_manifest"])
        )
        code_snapshot = json.loads(code_snapshot_path.read_text(encoding="utf-8-sig"))
        code_snapshot_id = code_snapshot["code_snapshot_id"]
        output_dir.mkdir(parents=True, exist_ok=True)
        if not args.resume and any((output_dir / name).exists() for name in ("checkpoint_best.pt", "checkpoint_last.pt", "training_log.csv")):
            raise FileExistsError(f"Existing training artifacts require --resume or a new output directory: {output_dir}")
        atomic_text(output_dir / "config_snapshot.yaml", yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        atomic_json(output_dir / "data_manifest.json", {
            "train_cycle_manifest": config["train_manifest"],
            "train_cycle_manifest_sha256": manifests["train_sha256"],
            "train_cycles": len(train_dataset),
            "val_cycle_manifest": config["val_manifest"],
            "val_cycle_manifest_sha256": manifests["val_sha256"],
            "val_cycles": len(val_dataset),
            "buffer_selected": False,
            "max_target_valid_time": max(row["valid_times"].split(";")[-1] for row in train_audit + val_audit),
            "gfs_single_cycle_contract": all(row["gfs_single_init_cycle_passed"] for row in train_audit + val_audit),
            "truth_2025_accessed": False,
        })
        run_manifest = {
            "status": "running",
            "experiment_id": config["experiment_id"],
            "mode": "full_finetune",
            "condition": config["condition"],
            "canary": args.canary,
            "started_at": started_at,
            "code_snapshot_id": code_snapshot_id,
            "pretrained_checkpoint_sha256": metadata["checkpoint_sha256"],
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "process_local_cuda_index": torch.cuda.current_device(),
            "physical_gpu_id": (os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"),
            "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device()),
            "truth_2025_accessed": False,
        }
        atomic_json(output_dir / "run_manifest.json", run_manifest)

        training_rows = []
        validation_rows = []
        variable_rows = []
        gradient_rows = []
        resume_report = {"resume_requested": bool(args.resume), "passed": True, "truth_2025_accessed": False}
        start_epoch = 0
        optimizer_step = 0
        best_metric = float("inf")
        best_epoch = -1
        patience_counter = 0
        if args.resume:
            resume_path = Path(args.resume)
            if not resume_path.is_absolute():
                resume_path = output_dir / resume_path
            checkpoint = load_resume(
                resume_path, model, optimizer, scheduler, scaler,
                config_path=config_path, config=config, manifests=manifests,
                metadata=metadata, code_snapshot_id=code_snapshot_id,
            )
            start_epoch = int(checkpoint["epoch"]) + 1
            optimizer_step = int(checkpoint["optimizer_step"])
            best_metric = float(checkpoint["best_metric"])
            best_epoch = int(checkpoint.get("best_epoch", checkpoint["epoch"]))
            if (output_dir / "training_log.csv").is_file():
                training_rows = pd.read_csv(output_dir / "training_log.csv").to_dict("records")
            if (output_dir / "validation_log.csv").is_file():
                validation_rows = pd.read_csv(output_dir / "validation_log.csv").to_dict("records")
            if (output_dir / "per_variable_training_loss.csv").is_file():
                variable_rows = pd.read_csv(output_dir / "per_variable_training_loss.csv").to_dict("records")
            if (output_dir / "gradient_summary.csv").is_file():
                gradient_rows = pd.read_csv(output_dir / "gradient_summary.csv").to_dict("records")
            resume_report.update({
                "checkpoint": str(resume_path), "restored_epoch": checkpoint["epoch"],
                "restored_optimizer_step": optimizer_step, "optimizer_state_restored": True,
                "scheduler_state_restored": True, "scaler_state_restored": True,
            })

        first_np = train_dataset[0]
        if args.canary and not args.resume:
            first = tensor_sample(first_np, device)
            gradient_rows = run_gradient_audit(model, first, device, config)
            required_modules = gradient_module_names(model)
            if not all(
                row["finite"] and all(row[f"{name}_active_gradient_tensors"] > 0 and row[f"{name}_gradients_finite"] for name in required_modules)
                for row in gradient_rows
            ):
                raise RuntimeError("Canary gradient audit failed.")
            micro = micro_overfit(registry, config, first_np, device, steps=20)
            if not micro["finite"] or micro["final_loss"] >= micro["initial_loss"]:
                raise RuntimeError(f"Canary micro-overfit failed: {micro}")
            atomic_json(output_dir / "micro_overfit_report.json", micro)
            model, metadata = build_full_model(registry, config, device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=float(config["model"]["learning_rate"]),
                weight_decay=float(config["model"]["weight_decay"])
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            scaler = make_scaler(device, bool(config["amp"]))

        rng = np.random.default_rng(int(config["seed"]))
        stopped = False
        for epoch in range(start_epoch, max_epochs):
            epoch_started = time.time()
            horizon = horizon_for_epoch(epoch, canary=args.canary)
            if epoch == 25 and not args.canary:
                patience_counter = 0
            model.train()
            order = rng.permutation(len(train_dataset)).tolist()
            epoch_losses = []
            data_wait = 0.0
            compute_seconds = 0.0
            for sample_np, wait_seconds in prefetched(train_dataset, order, bool(config.get("prefetch_thread", True))):
                data_wait += wait_seconds
                sample = tensor_sample(sample_np, device)
                optimizer.zero_grad(set_to_none=True)
                compute_started = time.perf_counter()
                with autocast_context(device, bool(config["amp"])):
                    loss, variable, contribution, _ = rollout(
                        model, sample, horizon=horizon, loss_config=config["loss"]
                    )
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError("Non-finite training loss.")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if not gradient_rows:
                    gradient_rows.append({
                        "audit": "first_training_step",
                        "horizon": horizon,
                        "loss": float(loss.detach().cpu()),
                        "seconds": math.nan,
                        "finite": bool(torch.isfinite(loss).item()),
                        **module_gradient_summary(model),
                    })
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config["model"]["gradient_clip_norm"])
                ).detach().cpu())
                scaler.step(optimizer)
                scaler.update()
                optimizer_step += 1
                compute_seconds += time.perf_counter() - compute_started
                loss_value = float(loss.detach().cpu())
                epoch_losses.append(loss_value)
                row = {
                    "epoch": epoch,
                    "optimizer_step": optimizer_step,
                    "init_time": sample_np["metadata"]["init_time"],
                    "current_horizon": horizon,
                    "train_loss": loss_value,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "gradient_norm_before_clip": grad_norm,
                    "data_wait_seconds": wait_seconds,
                }
                training_rows.append(row)
                for name in VARIABLE_SLICES:
                    variable_rows.append({
                        "epoch": epoch, "optimizer_step": optimizer_step,
                        "init_time": sample_np["metadata"]["init_time"],
                        "horizon": horizon, "variable": name,
                        "unweighted_loss": float(variable[name].detach().cpu()),
                        "weighted_contribution": float(contribution[name].detach().cpu()),
                    })
                if optimizer_step % 50 == 0 or optimizer_step == 1:
                    print(json.dumps({
                        "stage": "train_progress", "epoch": epoch, "optimizer_step": optimizer_step,
                        "horizon": horizon, "loss": loss_value,
                        "data_wait_seconds": round(data_wait, 3), "compute_seconds": round(compute_seconds, 3),
                    }), flush=True)
                if max_steps is not None and optimizer_step >= max_steps:
                    break

            model.eval()
            val_loss, val_cycle_rows, val_seconds = evaluate_loss(
                model, val_dataset, device, config, prefetch=bool(config.get("prefetch_thread", True))
            )
            for row in val_cycle_rows:
                validation_rows.append({"epoch": epoch, **row})
            improved = val_loss < best_metric - float(config["model"]["early_stopping_min_delta"])
            if improved:
                best_metric = val_loss
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            scheduler.step()
            payload = checkpoint_payload(
                model, optimizer, scheduler, scaler, epoch=epoch, optimizer_step=optimizer_step,
                best_metric=best_metric, config_path=config_path, config=config, metadata=metadata,
                manifests=manifests, code_snapshot_id=code_snapshot_id,
            )
            payload["best_epoch"] = best_epoch
            atomic_torch_save(output_dir / "checkpoint_last.pt", payload)
            if improved:
                atomic_torch_save(output_dir / "checkpoint_best.pt", payload)
            atomic_csv(output_dir / "training_log.csv", training_rows)
            atomic_csv(output_dir / "validation_log.csv", validation_rows)
            atomic_csv(output_dir / "per_variable_training_loss.csv", variable_rows)
            atomic_csv(output_dir / "gradient_summary.csv", gradient_rows)
            atomic_json(output_dir / "resume_report.json", resume_report)
            print(json.dumps({
                "stage": "epoch_completed", "epoch": epoch, "horizon": horizon,
                "optimizer_step": optimizer_step, "train_loss": float(np.mean(epoch_losses)),
                "val_loss": val_loss, "best_val_loss": best_metric, "best_epoch": best_epoch,
                "patience_counter": patience_counter, "epoch_seconds": time.time() - epoch_started,
                "val_seconds": val_seconds,
            }), flush=True)
            if max_steps is not None and optimizer_step >= max_steps:
                stopped = True
                break
            if (
                not args.canary and epoch >= 25 and epoch + 1 >= int(config["model"]["minimum_epochs"])
                and patience_counter >= int(config["model"]["early_stopping_patience"])
            ):
                stopped = True
                break

        # Verify final checkpoint can be reopened and exactly reproduces the saved tensors.
        reload_checkpoint = torch.load(output_dir / "checkpoint_last.pt", map_location="cpu", weights_only=False)
        reload_model, _ = build_full_model(registry, config, torch.device("cpu"))
        reload_model.load_state_dict(reload_checkpoint["model"], strict=True)
        checkpoint_reload_passed = state_dict_sha256(reload_model.state_dict()) == state_dict_sha256(reload_checkpoint["model"])
        if not checkpoint_reload_passed:
            raise RuntimeError("Checkpoint save/load tensor hash verification failed.")
        resume_report.update({"checkpoint_save_load_passed": True, "atomic_write": True})
        atomic_json(output_dir / "resume_report.json", resume_report)
        best_path = output_dir / "checkpoint_best.pt"
        last_path = output_dir / "checkpoint_last.pt"
        atomic_json(output_dir / "checkpoint_manifest.json", {
            "schema_version": "stage5_full_finetune_v1",
            "pretrained_checkpoint": str(metadata["checkpoint_path"]),
            "pretrained_checkpoint_sha256": metadata["checkpoint_sha256"],
            "source_parameter_sha256": metadata["source_parameter_sha256"],
            "loaded_parameter_sha256": metadata["loaded_parameter_sha256"],
            "source_key_count": metadata["source_key_count"],
            "missing_keys": metadata["missing_keys"],
            "unexpected_keys": metadata["unexpected_keys"],
            "checkpoint_best_sha256": sha256_file(best_path),
            "checkpoint_last_sha256": sha256_file(last_path),
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "truth_2025_accessed": False,
        })
        completed_at = utc_now()
        runtime = {
            "started_at": started_at, "completed_at": completed_at,
            "wall_clock_seconds": time.time() - session_started,
            "total_parameters": total_parameters, "trainable_parameters": trainable_parameters,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
            "optimizer_steps": optimizer_step, "completed_epochs": max((row["epoch"] for row in training_rows), default=-1) + 1,
            "early_stopped": stopped and max_steps is None,
            "best_epoch": best_epoch, "best_val_loss": best_metric,
            "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device()),
            "physical_gpu_id": run_manifest["physical_gpu_id"],
            "truth_2025_accessed": False,
        }
        atomic_json(output_dir / "runtime_summary.json", runtime)
        run_manifest.update({
            "status": "passed", "completed_at": completed_at,
            "optimizer_steps": optimizer_step, "best_epoch": best_epoch,
            "best_val_loss": best_metric, "failed_cycles": 0,
        })
        atomic_json(output_dir / "run_manifest.json", run_manifest)
        atomic_text(output_dir / "qc_report.md", "\n".join([
            f"# Stage5 {metadata['model_display_name']} Full Fine-tune QC", "",
            "- status: passed", f"- canary: {str(args.canary).lower()}",
            f"- train/val cycles: {len(train_dataset)}/{len(val_dataset)}",
            f"- total/trainable parameters: {total_parameters}/{trainable_parameters}",
            f"- optimizer steps: {optimizer_step}", f"- best epoch: {best_epoch}",
            "- adapters enabled: false", "- buffer selected: false",
            "- truth_2025_accessed: false", "- Stage5.3 started: false", "",
        ]))
        missing_outputs = [name for name in REQUIRED_OUTPUTS if not (output_dir / name).is_file()]
        if missing_outputs:
            raise RuntimeError(f"Formal training artifacts missing: {missing_outputs}")
        print(json.dumps({
            "status": "passed", "output_dir": str(output_dir), "canary": args.canary,
            "optimizer_steps": optimizer_step, "best_epoch": best_epoch,
            "best_val_loss": best_metric, "peak_vram_bytes": runtime["peak_vram_bytes"],
            "truth_2025_accessed": False,
        }, ensure_ascii=False, indent=2))
        return 0
    finally:
        train_dataset.close()
        val_dataset.close()


if __name__ == "__main__":
    raise SystemExit(main())
