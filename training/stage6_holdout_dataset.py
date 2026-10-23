"""Strict target-domain dataset for an explicitly released Stage6 holdout.

This loader is intentionally separate from the Stage5 development loader.
It accepts only the exact chronological support recorded in
``holdout_release`` and only experiments present in the release allowlist.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from torch.utils.data import Dataset

from pipeline.core import STATE_2D, STATE_3D, load_stats, state_array, static_array
from pipeline.operational_gfs_catalog import CORE_FORCE, GFSProcessedCatalog, load_cycle_manifest
from pipeline.run_operational_replay import gfs_forcing_array


ROOT = Path(__file__).resolve().parents[1]
STATE_VARIABLES = tuple(STATE_3D) + tuple(STATE_2D)
FORCING_VARIABLES = tuple(CORE_FORCE)
GRID_COORDINATE_ATOL = 5e-5


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


class Stage6HoldoutDomainDataset(Dataset):
    """Lazy ANFC-history/GFS-forcing/ANFC-truth holdout dataset."""

    def __init__(
        self,
        registry: str | Path,
        *,
        experiment_id: str,
        cycle_manifest: str | Path,
        max_cycles: int | None = None,
    ) -> None:
        self.registry_path = project_path(registry)
        self.registry = yaml.safe_load(self.registry_path.read_text(encoding="utf-8-sig"))
        release = self.registry.get("holdout_release", {})
        if release.get("approved") is not True:
            raise PermissionError("Stage6 holdout access requires holdout_release.approved=true.")
        if experiment_id not in set(release.get("approved_experiments", [])):
            raise PermissionError(f"Experiment is outside the Stage6 holdout allowlist: {experiment_id}")
        matches = [
            row for row in self.registry.get("experiments", [])
            if row.get("experiment_id") == experiment_id
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one registered holdout experiment, found {len(matches)}.")
        experiment = matches[0]
        if (
            int(experiment.get("stage", -1)) != 6
            or experiment.get("period") != "test_2025"
            or experiment.get("cycle_set") != "holdout_2025"
            or experiment.get("holdout") is not True
            or experiment.get("status") not in {"ready", "running", "completed"}
        ):
            raise PermissionError("Registered experiment does not satisfy the released Stage6 contract.")

        self.experiment_id = experiment_id
        self.experiment = experiment
        self.release = release
        self.cycle_manifest_path = project_path(cycle_manifest)
        manifest = load_cycle_manifest(self.cycle_manifest_path)
        expected_count = int(release.get("expected_cycles", -1))
        if len(manifest) != expected_count:
            raise ValueError(f"Holdout manifest must contain {expected_count} cycles.")
        expected_times = pd.date_range(
            pd.Timestamp(release["test_init_start"]),
            pd.Timestamp(release["test_init_end"]),
            freq="D",
        )
        actual_times = pd.to_datetime(manifest["init_time"], errors="raise").dt.normalize()
        if not actual_times.equals(pd.Series(expected_times, index=actual_times.index)):
            raise ValueError("Holdout manifest differs from the frozen chronological support.")
        if actual_times.duplicated().any():
            raise ValueError("Holdout manifest contains duplicate init times.")
        if actual_times.iloc[-1] + pd.Timedelta(days=10) != pd.Timestamp("2025-12-31"):
            raise ValueError("Holdout target support must end exactly on 2025-12-31.")
        if max_cycles is not None:
            if max_cycles < 1 or max_cycles > expected_count:
                raise ValueError("max_cycles is outside the released holdout support.")
            manifest = manifest.iloc[:max_cycles].copy()
        self.cycles = manifest.reset_index(drop=True)

        paths = self.registry["paths"]
        self.anfc_root = project_path(paths["anfc_era5_dataset"])
        self.ocean_path = self.anfc_root / "ocean_state_daily.nc"
        self.static_path = self.anfc_root / "static.nc"
        self.normalization_path = project_path(paths["normalization"])
        self.gfs_catalog_path = project_path(paths["gfs_processed_catalog"])
        missing = [
            str(path)
            for path in (
                self.registry_path,
                self.cycle_manifest_path,
                self.ocean_path,
                self.static_path,
                self.normalization_path,
                self.gfs_catalog_path,
            )
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(f"Stage6 holdout inputs are incomplete: {missing}")

        self.stats = load_stats(self.normalization_path)
        self._catalog = GFSProcessedCatalog(self.gfs_catalog_path)
        self._resolved: dict[str, Any] = {}
        self._gfs: dict[str, xr.Dataset] = {}
        self._ocean: xr.Dataset | None = None
        self._static: xr.Dataset | None = None
        self._static_values: np.ndarray | None = None
        self._time_set: set[pd.Timestamp] | None = None
        self._audit_cache: dict[int, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.cycles)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_ocean"] = None
        state["_static"] = None
        state["_gfs"] = {}
        state["_static_values"] = None
        state["_time_set"] = None
        return state

    def _ensure_open(self) -> None:
        if self._ocean is None:
            self._ocean = xr.open_dataset(self.ocean_path)
            self._static = xr.open_dataset(self.static_path)
            self._validate_interfaces()
            self._time_set = set(pd.DatetimeIndex(self._ocean.time.values).normalize())

    def _validate_interfaces(self) -> None:
        contract = self.registry["data_contract"]
        expected_sizes = {
            "depth": int(contract["grid"]["depth"]),
            "latitude": int(contract["grid"]["latitude"]),
            "longitude": int(contract["grid"]["longitude"]),
        }
        actual = {name: int(self._ocean.sizes.get(name, -1)) for name in expected_sizes}
        if actual != expected_sizes:
            raise ValueError(f"ANFC holdout grid mismatch: {actual} != {expected_sizes}")
        if tuple(contract["ocean_variables"]) != STATE_VARIABLES:
            raise ValueError("Holdout ocean variable order differs from the frozen contract.")
        if tuple(contract["forcing_variables"]) != FORCING_VARIABLES:
            raise ValueError("Holdout forcing variable order differs from the frozen contract.")
        missing_ocean = sorted(set(STATE_VARIABLES) - set(self._ocean.data_vars))
        missing_static = sorted(
            {"wet_mask_3d", "bathymetry", "lon_sin", "lat_sin"} - set(self._static.data_vars)
        )
        if missing_ocean or missing_static:
            raise ValueError(
                f"Holdout interface is incomplete: ocean={missing_ocean}, static={missing_static}"
            )

    @staticmethod
    def _windows(init_time: str) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
        init = pd.Timestamp(init_time).normalize()
        history = pd.date_range(init - pd.Timedelta(days=6), init, freq="D")
        valid = pd.date_range(init + pd.Timedelta(days=1), periods=10, freq="D")
        return history, valid

    def _resolve(self, init_time: str):
        if init_time not in self._resolved:
            self._resolved[init_time] = self._catalog.resolve(init_time, validate_header=True)
        return self._resolved[init_time]

    def audit_cycle(self, index: int) -> dict[str, Any]:
        self._ensure_open()
        row = self.cycles.iloc[index]
        init_time = str(row["init_time"])
        history, valid = self._windows(init_time)
        missing_history = sorted(set(history) - self._time_set)
        missing_truth = sorted(set(valid) - self._time_set)
        if missing_history or missing_truth:
            raise ValueError(
                f"ANFC holdout coverage is incomplete for {init_time}: "
                f"history={missing_history}, truth={missing_truth}"
            )
        resolved = self._resolve(init_time)
        return {
            "sequence": int(row["sequence"]),
            "init_time": init_time,
            "role": str(row["role"]),
            "split": "test",
            "history_dates": ";".join(value.date().isoformat() for value in history),
            "valid_times": ";".join(value.date().isoformat() for value in valid),
            "gfs_source_file": resolved.resolved_file_path.as_posix(),
            "gfs_local_index": int(resolved.local_index),
            "gfs_sha256": resolved.sha256,
            "gfs_lead_count": int(resolved.lead_count),
            "truth_2025_access_authorized": True,
        }

    def preaudit_cycles(self) -> list[dict[str, Any]]:
        records = []
        for index in range(len(self)):
            record = self.audit_cycle(index)
            self._audit_cache[index] = record
            records.append(record)
        return records

    def __getitem__(self, index: int) -> dict[str, Any]:
        self._ensure_open()
        audit = self._audit_cache.get(index) or self.audit_cycle(index)
        self._audit_cache[index] = audit
        history_dates, valid_times = self._windows(audit["init_time"])
        history = []
        initial_mask = None
        for value in history_dates:
            state, mask = state_array(self._ocean, value, self.stats)
            history.append(state)
            initial_mask = mask.astype(bool, copy=False)
        resolved = self._resolve(audit["init_time"])
        gfs_key = str(resolved.resolved_file_path)
        if gfs_key not in self._gfs:
            self._gfs[gfs_key] = xr.open_dataset(resolved.resolved_file_path)
        forcing = [
            gfs_forcing_array(
                self._gfs[gfs_key],
                pd.Timestamp(audit["init_time"]),
                lead_day,
                self._ocean,
                self.stats,
                FORCING_VARIABLES,
            )
            for lead_day in range(1, 11)
        ]
        truth, truth_mask = [], []
        for value in valid_times:
            state, mask = state_array(self._ocean, value, self.stats)
            truth.append(state)
            truth_mask.append(mask.astype(bool, copy=False))
        if self._static_values is None:
            self._static_values = static_array(self._static, self.stats).astype(np.float32, copy=False)
        return {
            "anfc_ocean_history": np.stack(history).astype(np.float32, copy=False),
            "gfs_future_forcing": np.stack(forcing).astype(np.float32, copy=False),
            "anfc_truth": np.stack(truth).astype(np.float32, copy=False),
            "anfc_truth_mask": np.stack(truth_mask).astype(bool, copy=False),
            "anfc_initial_mask": initial_mask,
            "static": self._static_values,
            "metadata": audit,
        }

    def close(self) -> None:
        for dataset in (self._ocean, self._static, *self._gfs.values()):
            if dataset is not None:
                dataset.close()
        self._gfs = {}

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

