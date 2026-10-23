"""Paired MY/ANFC and ERA5/GFS dataset for Stage5 infrastructure validation.

This module references the frozen processed products in the experiment registry.
It does not create another NetCDF product and it hard-rejects buffer and 2025
cycles before any truth field is selected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from torch.utils.data import Dataset

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.core import STATE_2D, STATE_3D, load_stats, state_array, static_array
from pipeline.operational_gfs_catalog import CORE_FORCE, GFSProcessedCatalog, load_cycle_manifest
from pipeline.run_operational_replay import daily_forcing_array, gfs_forcing_array


ROOT = Path(__file__).resolve().parents[1]
STATE_VARIABLES = tuple(STATE_3D) + tuple(STATE_2D)
FORCING_VARIABLES = tuple(CORE_FORCE)
EXPECTED_SPLIT_COUNTS = {"train": 755, "val": 143}
MATERIALIZATION_MODES = {"target_forecast_only", "paired_alignment"}
LAST_DEVELOPMENT_INIT = pd.Timestamp("2024-12-21")
LAST_DEVELOPMENT_TRUTH = pd.Timestamp("2024-12-31")
GRID_COORDINATE_ATOL = 5e-5


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _normalized_samples(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"init_time": str, "split": str})
    required = {"init_time", "split"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{label} samples are missing columns: {sorted(required - set(frame.columns))}")
    if frame["init_time"].duplicated().any():
        raise ValueError(f"{label} samples contain duplicate init_time values.")
    normalized = frame["init_time"].map(lambda value: pd.Timestamp(value).date().isoformat())
    if not normalized.equals(frame["init_time"]):
        raise ValueError(f"{label} samples must use normalized YYYY-MM-DD init_time values.")
    return frame[["init_time", "split"]].copy()


class Stage5PairedDomainDataset(Dataset):
    """Lazy paired dataset with an explicit train/val-only contract."""

    def __init__(
        self,
        registry: str | Path = "configs/experiment_registry_2022_2025.yaml",
        *,
        split: str,
        cycle_manifest: str | Path | None = None,
        max_cycles: int | None = None,
        materialization_mode: str = "paired_alignment",
    ):
        if split not in EXPECTED_SPLIT_COUNTS:
            raise ValueError("Stage5 paired loader only permits split='train' or split='val'.")
        if max_cycles is not None and max_cycles < 1:
            raise ValueError("max_cycles must be positive when provided.")
        if materialization_mode not in MATERIALIZATION_MODES:
            raise ValueError(
                f"materialization_mode must be one of {sorted(MATERIALIZATION_MODES)}, got {materialization_mode}."
            )
        self.registry_path = project_path(registry)
        self.registry = yaml.safe_load(self.registry_path.read_text(encoding="utf-8-sig"))
        self.split = split
        self.materialization_mode = materialization_mode
        self.paths = self.registry["paths"]
        self.my_root = project_path(self.paths["my_era5_stage4_target"])
        self.anfc_root = project_path(self.paths["anfc_era5_dataset"])
        self.normalization_path = project_path(self.paths["normalization"])
        self.gfs_catalog_path = project_path(self.paths["gfs_processed_catalog"])
        self.my_ocean_path = self.my_root / "ocean_state_daily.nc"
        self.anfc_ocean_path = self.anfc_root / "ocean_state_daily.nc"
        self.era5_forcing_path = self.my_root / "forcing_daily.nc"
        self.my_static_path = self.my_root / "static.nc"
        self.anfc_static_path = self.anfc_root / "static.nc"
        self.my_samples_path = self.my_root / "samples.csv"
        self.anfc_samples_path = self.anfc_root / "samples.csv"
        required_paths = [
            self.registry_path,
            self.anfc_ocean_path,
            self.anfc_static_path,
            self.my_samples_path,
            self.anfc_samples_path,
            self.normalization_path,
            self.gfs_catalog_path,
        ]
        if materialization_mode == "paired_alignment":
            required_paths.extend((self.my_ocean_path, self.era5_forcing_path, self.my_static_path))
        missing = [str(path) for path in required_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Stage5 paired loader inputs are incomplete: {missing}")

        my_samples = _normalized_samples(self.my_samples_path, "MY")
        anfc_samples = _normalized_samples(self.anfc_samples_path, "ANFC")
        paired = anfc_samples.merge(my_samples, on="init_time", suffixes=("_anfc", "_my"), how="inner")
        disagreements = paired.loc[paired["split_anfc"] != paired["split_my"]]
        if not disagreements.empty:
            raise ValueError(
                "MY/ANFC sample split disagreement: "
                + ", ".join(disagreements["init_time"].head(10).tolist())
            )
        paired["split"] = paired["split_anfc"]
        self.paired_samples = paired[["init_time", "split"]].copy()
        self.paired_split_counts = {
            name: int((self.paired_samples["split"] == name).sum())
            for name in ("train", "val", "buffer")
        }
        for name, expected in EXPECTED_SPLIT_COUNTS.items():
            actual = self.paired_split_counts[name]
            if actual != expected:
                raise ValueError(f"Paired {name} count must be {expected}, got {actual}.")

        eligible = self.paired_samples.loc[self.paired_samples["split"] == split].copy()
        manifest_path = project_path(cycle_manifest) if cycle_manifest is not None else None
        self.cycle_manifest_path = manifest_path
        if manifest_path is not None:
            manifest = load_cycle_manifest(manifest_path)
            paired_lookup = self.paired_samples.set_index("init_time")["split"].to_dict()
            selected_records = []
            for row in manifest.itertuples(index=False):
                init = pd.Timestamp(row.init_time)
                self._guard_development_date(init)
                actual_split = paired_lookup.get(row.init_time)
                if actual_split not in EXPECTED_SPLIT_COUNTS:
                    if actual_split is None:
                        raise ValueError(f"Cycle {row.init_time} is not in the paired MY/ANFC development support.")
                    raise ValueError(
                        f"Stage5 split guard rejected {actual_split} cycle {row.init_time}; buffer/test are forbidden."
                    )
                if actual_split != split:
                    raise ValueError(
                        f"Cycle {row.init_time} belongs to split={actual_split}, not requested split={split}."
                    )
                selected_records.append(
                    {"sequence": int(row.sequence), "init_time": row.init_time, "role": str(row.role), "split": split}
                )
            selected = pd.DataFrame(selected_records)
        else:
            eligible = eligible.reset_index(drop=True)
            selected = pd.DataFrame(
                {
                    "sequence": np.arange(1, len(eligible) + 1, dtype=np.int64),
                    "init_time": eligible["init_time"],
                    "role": [f"{split}_cycle"] * len(eligible),
                    "split": eligible["split"],
                }
            )
        if max_cycles is not None:
            selected = selected.iloc[:max_cycles].copy()
        self.cycles = selected.reset_index(drop=True)
        self._window_cache = {
            str(row.init_time): self._build_date_windows(str(row.init_time))
            for row in self.cycles.itertuples(index=False)
        }
        self.stats = load_stats(self.normalization_path)
        self._gfs_catalog = GFSProcessedCatalog(self.gfs_catalog_path)
        self._resolved_gfs: dict[str, Any] = {}
        self._gfs_datasets: dict[str, xr.Dataset] = {}
        self._my_ocean: xr.Dataset | None = None
        self._anfc_ocean: xr.Dataset | None = None
        self._era5_forcing: xr.Dataset | None = None
        self._my_static: xr.Dataset | None = None
        self._anfc_static: xr.Dataset | None = None
        self._static_values: np.ndarray | None = None
        self._interface_validated = False
        self._time_sets: dict[str, set[pd.Timestamp]] = {}
        self._audit_cache: dict[int, dict[str, Any]] = {}

    @staticmethod
    def _guard_development_date(init: pd.Timestamp) -> None:
        init = pd.Timestamp(init).normalize()
        if init.year >= 2025 or init > LAST_DEVELOPMENT_INIT:
            raise ValueError(f"2025/beyond-development cycle is forbidden in Stage5: {init.date().isoformat()}")
        if init + pd.Timedelta(days=10) > LAST_DEVELOPMENT_TRUTH:
            raise ValueError(f"Cycle target window enters 2025 and is forbidden: {init.date().isoformat()}")

    def __len__(self) -> int:
        return len(self.cycles)

    def __getstate__(self):
        state = self.__dict__.copy()
        for name in ("_my_ocean", "_anfc_ocean", "_era5_forcing", "_my_static", "_anfc_static"):
            state[name] = None
        state["_gfs_datasets"] = {}
        state["_static_values"] = None
        state["_interface_validated"] = False
        state["_time_sets"] = {}
        return state

    def _ensure_open(self) -> None:
        if self._anfc_ocean is None:
            self._anfc_ocean = xr.open_dataset(self.anfc_ocean_path)
            self._anfc_static = xr.open_dataset(self.anfc_static_path)
            if self.materialization_mode == "paired_alignment":
                self._my_ocean = xr.open_dataset(self.my_ocean_path)
                self._era5_forcing = xr.open_dataset(self.era5_forcing_path)
                self._my_static = xr.open_dataset(self.my_static_path)
        if not self._interface_validated:
            self._validate_interfaces()
            self._time_sets = {
                "anfc_history_truth": set(pd.DatetimeIndex(self._anfc_ocean.time.values).normalize()),
            }
            if self.materialization_mode == "paired_alignment":
                self._time_sets.update(
                    {
                        "my_history": set(pd.DatetimeIndex(self._my_ocean.time.values).normalize()),
                        "era5_future": set(pd.DatetimeIndex(self._era5_forcing.time.values).normalize()),
                    }
                )
            self._interface_validated = True

    def _validate_interfaces(self) -> None:
        contract = self.registry["data_contract"]
        grid = contract["grid"]
        if tuple(contract["ocean_variables"]) != STATE_VARIABLES:
            raise ValueError("Registry ocean variable order does not match the frozen Stage5 interface.")
        if tuple(contract["forcing_variables"]) != FORCING_VARIABLES:
            raise ValueError("Registry forcing variable order does not match the frozen Stage5 interface.")
        expected_sizes = {"depth": int(grid["depth"]), "latitude": int(grid["latitude"]), "longitude": int(grid["longitude"])}
        ocean_datasets = [("ANFC ocean", self._anfc_ocean)]
        if self.materialization_mode == "paired_alignment":
            ocean_datasets.insert(0, ("MY ocean", self._my_ocean))
        for label, dataset in ocean_datasets:
            actual = {name: int(dataset.sizes.get(name, -1)) for name in expected_sizes}
            if actual != expected_sizes:
                raise ValueError(f"{label} grid mismatch: {actual} != {expected_sizes}")
            missing = sorted(set(STATE_VARIABLES) - set(dataset.data_vars))
            if missing:
                raise ValueError(f"{label} is missing variables: {missing}")
        if self.materialization_mode == "paired_alignment":
            for coord in ("depth", "latitude", "longitude"):
                if not np.allclose(
                    self._my_ocean[coord].values,
                    self._anfc_ocean[coord].values,
                    rtol=0.0,
                    atol=GRID_COORDINATE_ATOL,
                ):
                    raise ValueError(
                        f"MY/ANFC {coord} coordinates exceed the index-alignment tolerance "
                        f"{GRID_COORDINATE_ATOL}."
                    )
            forcing_missing = sorted(set(FORCING_VARIABLES) - set(self._era5_forcing.data_vars))
            if forcing_missing:
                raise ValueError(f"ERA5 forcing is missing variables: {forcing_missing}")
            for coord in ("latitude", "longitude"):
                if not np.allclose(
                    self._era5_forcing[coord].values,
                    self._anfc_ocean[coord].values,
                    rtol=0.0,
                    atol=GRID_COORDINATE_ATOL,
                ):
                    raise ValueError(
                        f"ERA5/ANFC {coord} coordinates exceed the index-alignment tolerance "
                        f"{GRID_COORDINATE_ATOL}."
                    )
        required_static = {"wet_mask_3d", "bathymetry", "lon_sin", "lat_sin"}
        static_datasets = [("ANFC static", self._anfc_static)]
        if self.materialization_mode == "paired_alignment":
            static_datasets.insert(0, ("MY static", self._my_static))
        for label, dataset in static_datasets:
            missing = sorted(required_static - set(dataset.data_vars))
            if missing:
                raise ValueError(f"{label} is missing fields: {missing}")
        if sorted(self.stats.get("ocean", {})) != sorted(STATE_VARIABLES):
            raise ValueError("Normalization ocean variables do not match the Stage5 state interface.")
        if sorted(self.stats.get("forcing", {})) != sorted(FORCING_VARIABLES):
            raise ValueError("Normalization forcing variables do not match the Stage5 forcing interface.")

    def _resolve_gfs_cycle(self, init_time: str):
        if init_time not in self._resolved_gfs:
            self._resolved_gfs[init_time] = self._gfs_catalog.resolve(init_time, validate_header=True)
        return self._resolved_gfs[init_time]

    @classmethod
    def _build_date_windows(cls, init_time: str) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
        init = pd.Timestamp(init_time)
        cls._guard_development_date(init)
        history = pd.date_range(init - pd.Timedelta(days=6), init, freq="D")
        valid = pd.date_range(init + pd.Timedelta(days=1), periods=10, freq="D")
        return history, valid

    def _date_windows(self, init_time: str) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
        return self._window_cache[init_time]

    def audit_cycle(self, index: int) -> dict[str, Any]:
        row = self.cycles.iloc[index]
        init_text = str(row["init_time"])
        history, valid = self._date_windows(init_text)
        self._ensure_open()
        missing_my = (
            sorted(set(history) - self._time_sets["my_history"])
            if self.materialization_mode == "paired_alignment"
            else []
        )
        missing_anfc_history = sorted(set(history) - self._time_sets["anfc_history_truth"])
        missing_truth = sorted(set(valid) - self._time_sets["anfc_history_truth"])
        missing_era5 = (
            sorted(set(valid) - self._time_sets["era5_future"])
            if self.materialization_mode == "paired_alignment"
            else []
        )
        if missing_my or missing_anfc_history or missing_truth or missing_era5:
            raise ValueError(
                f"Incomplete paired windows for {init_text}: "
                f"MY={missing_my}, ANFC_history={missing_anfc_history}, truth={missing_truth}, ERA5={missing_era5}"
            )
        resolved = self._resolve_gfs_cycle(init_text)
        return {
            "sequence": int(row["sequence"]),
            "init_time": init_text,
            "role": str(row["role"]),
            "split": self.split,
            "materialization_mode": self.materialization_mode,
            "history_dates": ";".join(value.date().isoformat() for value in history),
            "valid_times": ";".join(value.date().isoformat() for value in valid),
            "history_pairing_passed": True,
            "forcing_valid_time_pairing_passed": True,
            "gfs_single_init_cycle_passed": True,
            "gfs_source_file": resolved.resolved_file_path.as_posix(),
            "gfs_local_index": int(resolved.local_index),
            "gfs_sha256": resolved.sha256,
            "gfs_lead_count": int(resolved.lead_count),
            "grid": "depth=33,latitude=85,longitude=103",
            "grid_coordinate_atol": GRID_COORDINATE_ATOL,
            "truth_2025_accessed": False,
        }

    def preaudit_cycles(self) -> list[dict[str, Any]]:
        records = []
        for index in range(len(self)):
            record = self.audit_cycle(index)
            self._audit_cache[index] = record
            records.append(record)
        return records

    def audit_cycles(self) -> list[dict[str, Any]]:
        return self.preaudit_cycles()

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Worker processes reopen their own xarray handles after __getstate__
        # removes non-picklable file objects.  The cached audit remains valid.
        self._ensure_open()
        audit = self._audit_cache.get(index)
        if audit is None:
            audit = self.audit_cycle(index)
            self._audit_cache[index] = audit
        init_text = audit["init_time"]
        history_dates, valid_times = self._date_windows(init_text)
        anfc_history = []
        anfc_initial_mask = None
        for value in history_dates:
            state_anfc, mask_anfc = state_array(self._anfc_ocean, value, self.stats)
            anfc_history.append(state_anfc)
            anfc_initial_mask = mask_anfc.astype(bool, copy=False)
        resolved = self._resolve_gfs_cycle(init_text)
        gfs_key = str(resolved.resolved_file_path)
        if gfs_key not in self._gfs_datasets:
            self._gfs_datasets[gfs_key] = xr.open_dataset(resolved.resolved_file_path)
        gfs_dataset = self._gfs_datasets[gfs_key]
        gfs_future = [
            gfs_forcing_array(gfs_dataset, pd.Timestamp(init_text), lead_day, self._anfc_ocean, self.stats, FORCING_VARIABLES)
            for lead_day in range(1, 11)
        ]
        truth = []
        truth_mask = []
        for value in valid_times:
            state, mask = state_array(self._anfc_ocean, value, self.stats)
            truth.append(state)
            truth_mask.append(mask.astype(bool, copy=False))
        if self._static_values is None:
            self._static_values = static_array(self._anfc_static, self.stats).astype(np.float32, copy=False)
        result = {
            "anfc_ocean_history": np.stack(anfc_history).astype(np.float32, copy=False),
            "gfs_future_forcing": np.stack(gfs_future).astype(np.float32, copy=False),
            "anfc_truth": np.stack(truth).astype(np.float32, copy=False),
            "anfc_truth_mask": np.stack(truth_mask).astype(bool, copy=False),
            "anfc_initial_mask": anfc_initial_mask,
            "static": self._static_values,
            "metadata": audit,
        }
        if self.materialization_mode == "paired_alignment":
            my_history = [
                state_array(self._my_ocean, value, self.stats)[0]
                for value in history_dates
            ]
            era5_future = [
                daily_forcing_array(
                    self._era5_forcing,
                    value,
                    self._anfc_ocean,
                    self.stats,
                    FORCING_VARIABLES,
                )
                for value in valid_times
            ]
            result.update(
                {
                    "my_ocean_history": np.stack(my_history).astype(np.float32, copy=False),
                    "era5_future_forcing": np.stack(era5_future).astype(np.float32, copy=False),
                }
            )
        return result

    def close(self) -> None:
        for dataset in (
            self._my_ocean,
            self._anfc_ocean,
            self._era5_forcing,
            self._my_static,
            self._anfc_static,
            *self._gfs_datasets.values(),
        ):
            if dataset is not None:
                dataset.close()
        self._gfs_datasets = {}

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="configs/experiment_registry_2022_2025.yaml")
    parser.add_argument("--split", required=True, choices=("train", "val"))
    parser.add_argument("--cycle-manifest")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-cycles", type=int)
    parser.add_argument(
        "--materialization-mode",
        choices=sorted(MATERIALIZATION_MODES),
        default="paired_alignment",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dry_run:
        raise ValueError("The Stage5 infrastructure CLI is dry-run only; no training entry point is enabled.")
    dataset = Stage5PairedDomainDataset(
        args.registry,
        split=args.split,
        cycle_manifest=args.cycle_manifest,
        max_cycles=args.max_cycles,
        materialization_mode=args.materialization_mode,
    )
    try:
        records = dataset.audit_cycles()
        print(
            json.dumps(
                {
                    "status": "passed",
                    "split": args.split,
                    "paired_split_counts": dataset.paired_split_counts,
                    "selected_cycles": len(dataset),
                    "materialization_mode": args.materialization_mode,
                    "cycles": records,
                    "buffer_excluded": True,
                    "truth_2025_accessed": False,
                    "large_netcdf_created": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        dataset.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
