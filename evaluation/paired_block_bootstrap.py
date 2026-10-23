"""Shared paired moving-block bootstrap utilities for formal forecast metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import pandas as pd


METRICS = ("rmse", "mae", "persistence_skill", "r2", "acc")
LOWER_IS_BETTER = {"rmse", "mae"}
PAIR_KEYS = ("init_time", "lead_day", "domain", "variable")
SUFFICIENT_COLUMNS = (
    "model_sse",
    "persistence_sse",
    "absolute_error_sum",
    "truth_sum",
    "truth_sq_sum",
    "valid_count",
    "acc",
)


@dataclass(frozen=True)
class BootstrapSettings:
    replicates: int = 2000
    block_lengths: tuple[int, ...] = (7, 10, 14)
    seed: int = 20260626


def metric_direction(metric: str) -> float:
    if metric not in METRICS:
        raise ValueError(f"Unsupported metric: {metric}")
    return 1.0 if metric in LOWER_IS_BETTER else -1.0


def read_cycle_metrics(path, *, forbid_2025: bool = True) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = set(PAIR_KEYS) | {"season", *METRICS, *SUFFICIENT_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Metric file {path} is missing columns: {missing}")
    frame = frame.copy()
    frame["init_time"] = pd.to_datetime(frame["init_time"])
    frame["lead_day"] = frame["lead_day"].astype(int)
    if forbid_2025:
        valid_time = frame["init_time"] + pd.to_timedelta(frame["lead_day"], unit="D")
        if (valid_time >= pd.Timestamp("2025-01-01")).any():
            raise ValueError(f"Development bootstrap refuses 2025 truth rows in {path}.")
    if frame.duplicated(list(PAIR_KEYS)).any():
        raise ValueError(f"Metric file {path} has duplicate paired keys.")
    return frame


def strata(frame: pd.DataFrame) -> list[tuple[str, str, pd.Series]]:
    output = [("overall", "all", pd.Series(True, index=frame.index))]
    output.extend(
        ("variable", str(value), frame["variable"].eq(value))
        for value in sorted(frame["variable"].dropna().unique())
    )
    lead_groups = {
        "day01_03": frame["lead_day"].between(1, 3),
        "day04_07": frame["lead_day"].between(4, 7),
        "day08_10": frame["lead_day"].between(8, 10),
    }
    output.extend(("lead_group", name, mask) for name, mask in lead_groups.items())
    output.extend(
        ("season", str(value), frame["season"].eq(value))
        for value in sorted(frame["season"].dropna().unique())
    )
    return output


def aligned_cycle_values(
    frames: Mapping[str, pd.DataFrame],
    metric: str,
    masks: Mapping[str, pd.Series],
) -> pd.DataFrame:
    series = []
    for name, frame in frames.items():
        selected = frame.loc[masks[name], ["init_time", metric]].dropna()
        per_cycle = selected.groupby("init_time", sort=True)[metric].mean().rename(name)
        series.append(per_cycle)
    aligned = pd.concat(series, axis=1, join="inner").dropna().sort_index()
    if aligned.empty:
        raise ValueError(f"No aligned cycles remain for metric={metric}.")
    return aligned


def moving_block_indices(
    length: int,
    block_length: int,
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if not 1 <= block_length <= length:
        raise ValueError(f"Invalid block length {block_length} for {length} cycles.")
    blocks = int(np.ceil(length / block_length))
    starts = rng.integers(0, length - block_length + 1, size=(replicates, blocks))
    offsets = np.arange(block_length)[None, None, :]
    return (starts[:, :, None] + offsets).reshape(replicates, -1)[:, :length]


def bootstrap_contrast(
    values: pd.DataFrame,
    contrast: Callable[[Mapping[str, np.ndarray]], np.ndarray],
    *,
    metric: str,
    settings: BootstrapSettings,
    seed_offset: int = 0,
) -> list[dict]:
    arrays = {name: values[name].to_numpy(dtype=np.float64) for name in values.columns}
    point_raw = float(contrast({name: value[None, :] for name, value in arrays.items()})[0])
    direction = metric_direction(metric)
    rows = []
    for block_length in settings.block_lengths:
        rng = np.random.default_rng(settings.seed + seed_offset + block_length * 1009)
        indices = moving_block_indices(len(values), block_length, settings.replicates, rng)
        samples = contrast({name: value[indices] for name, value in arrays.items()})
        degradation = direction * samples
        rows.append(
            {
                "metric": metric,
                "block_length": int(block_length),
                "replicates": int(settings.replicates),
                "n_cycles": int(len(values)),
                "point_raw": point_raw,
                "point_degradation": float(direction * point_raw),
                "raw_ci025": float(np.quantile(samples, 0.025)),
                "raw_ci975": float(np.quantile(samples, 0.975)),
                "degradation_ci025": float(np.quantile(degradation, 0.025)),
                "degradation_ci975": float(np.quantile(degradation, 0.975)),
                "probability_degradation_positive": float(np.mean(degradation > 0.0)),
                "probability_improvement": float(np.mean(degradation < 0.0)),
            }
        )
    return rows


def mean_difference(values: Mapping[str, np.ndarray], candidate: str, reference: str) -> np.ndarray:
    return values[candidate].mean(axis=1) - values[reference].mean(axis=1)


def factorial_contrast(values: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    means = {key: value.mean(axis=1) for key, value in values.items()}
    if name == "ocean_product_regime":
        return means["B"] - means["A"]
    if name == "forcing_product":
        return means["C"] - means["A"]
    if name == "combined":
        return means["D"] - means["A"]
    if name == "interaction":
        return means["D"] - means["B"] - means["C"] + means["A"]
    raise ValueError(f"Unknown factorial contrast: {name}")


def _component_cube(frame: pd.DataFrame, cycles: pd.Index, variables: pd.Index) -> dict[str, np.ndarray]:
    """Return sufficient-statistic arrays shaped [cycle, variable].

    Rows may contain several lead days. Additive sufficient statistics are
    pooled inside each cycle/variable cell. ACC follows the frozen evaluator:
    first average finite ACC groups within the requested stratum, then average
    the resulting variable metrics for the paper-level scalar.
    """

    grouped = frame.groupby(["init_time", "variable"], sort=True, dropna=False)
    index = pd.MultiIndex.from_product([cycles, variables], names=["init_time", "variable"])
    summed = grouped[
        [
            "model_sse",
            "persistence_sse",
            "absolute_error_sum",
            "truth_sum",
            "truth_sq_sum",
            "valid_count",
        ]
    ].sum().reindex(index)
    if summed.isna().any().any():
        raise ValueError("A condition is missing one or more aligned cycle/variable cells.")
    finite_acc = frame.loc[frame["acc"].notna(), ["init_time", "variable", "acc"]]
    acc_sum = finite_acc.groupby(["init_time", "variable"])["acc"].sum().reindex(index, fill_value=0.0)
    acc_count = finite_acc.groupby(["init_time", "variable"])["acc"].count().reindex(index, fill_value=0)
    shape = (len(cycles), len(variables))
    output = {
        name: summed[name].to_numpy(dtype=np.float64).reshape(shape)
        for name in summed.columns
    }
    output["acc_sum"] = acc_sum.to_numpy(dtype=np.float64).reshape(shape)
    output["acc_count"] = acc_count.to_numpy(dtype=np.float64).reshape(shape)
    return output


def aligned_sufficient_statistics(
    frames: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, dict[str, np.ndarray]], pd.Index, pd.Index]:
    """Align conditions without silently changing the paired estimand."""

    cycle_sets = [set(frame["init_time"].unique()) for frame in frames.values()]
    variable_sets = [set(frame["variable"].unique()) for frame in frames.values()]
    cycles = pd.Index(sorted(set.intersection(*cycle_sets)), name="init_time")
    variables = pd.Index(sorted(set.intersection(*variable_sets)), name="variable")
    if cycles.empty or variables.empty:
        raise ValueError("No common cycles or variables remain for paired bootstrap.")
    components = {
        name: _component_cube(frame, cycles, variables)
        for name, frame in frames.items()
    }
    return components, cycles, variables


def _finalize_pooled_metric(
    components: Mapping[str, np.ndarray],
    indices: np.ndarray,
    metric: str,
) -> np.ndarray:
    """Reconstruct the frozen metric and average it across variables."""

    def sampled_sum(name: str) -> np.ndarray:
        return components[name][indices].sum(axis=1)

    valid = sampled_sum("valid_count")
    model_sse = sampled_sum("model_sse")
    with np.errstate(divide="ignore", invalid="ignore"):
        if metric == "rmse":
            values = np.sqrt(model_sse / valid)
        elif metric == "mae":
            values = sampled_sum("absolute_error_sum") / valid
        elif metric == "persistence_skill":
            persistence_rmse = np.sqrt(sampled_sum("persistence_sse") / valid)
            values = 1.0 - np.sqrt(model_sse / valid) / persistence_rmse
            values[persistence_rmse <= 0.0] = np.nan
        elif metric == "r2":
            truth_sum = sampled_sum("truth_sum")
            tss = sampled_sum("truth_sq_sum") - truth_sum**2 / valid
            values = 1.0 - model_sse / tss
            values[tss <= 0.0] = np.nan
        elif metric == "acc":
            count = sampled_sum("acc_count")
            values = sampled_sum("acc_sum") / count
            values[count <= 0.0] = np.nan
        else:
            raise ValueError(f"Unsupported pooled metric: {metric}")
    return np.nanmean(values, axis=1)


def bootstrap_pooled_contrast(
    frames: Mapping[str, pd.DataFrame],
    contrast: Callable[[Mapping[str, np.ndarray]], np.ndarray],
    *,
    metric: str,
    settings: BootstrapSettings,
    seed_offset: int = 0,
    prealigned: tuple[dict[str, dict[str, np.ndarray]], pd.Index, pd.Index] | None = None,
) -> list[dict]:
    """Paired moving-block bootstrap using the evaluator's pooled estimand."""

    components, cycles, variables = (
        prealigned if prealigned is not None else aligned_sufficient_statistics(frames)
    )
    point_indices = np.arange(len(cycles), dtype=int)[None, :]
    point_values = {
        name: _finalize_pooled_metric(component, point_indices, metric)
        for name, component in components.items()
    }
    point_raw = float(contrast(point_values)[0])
    direction = metric_direction(metric)
    rows = []
    for block_length in settings.block_lengths:
        rng = np.random.default_rng(settings.seed + seed_offset + block_length * 1009)
        indices = moving_block_indices(len(cycles), block_length, settings.replicates, rng)
        metric_values = {
            name: _finalize_pooled_metric(component, indices, metric)
            for name, component in components.items()
        }
        samples = contrast(metric_values)
        finite = np.isfinite(samples)
        if not finite.any():
            raise ValueError(f"No finite bootstrap contrasts for metric={metric}.")
        samples = samples[finite]
        degradation = direction * samples
        rows.append(
            {
                "metric": metric,
                "block_length": int(block_length),
                "replicates": int(settings.replicates),
                "finite_replicates": int(len(samples)),
                "n_cycles": int(len(cycles)),
                "n_variables": int(len(variables)),
                "estimand": "pooled_sufficient_stats_then_mean_across_variables",
                "point_raw": point_raw,
                "point_degradation": float(direction * point_raw),
                "raw_ci025": float(np.quantile(samples, 0.025)),
                "raw_ci975": float(np.quantile(samples, 0.975)),
                "degradation_ci025": float(np.quantile(degradation, 0.025)),
                "degradation_ci975": float(np.quantile(degradation, 0.975)),
                "probability_degradation_positive": float(np.mean(degradation > 0.0)),
                "probability_improvement": float(np.mean(degradation < 0.0)),
            }
        )
    return rows


def direct_difference(values: Mapping[str, np.ndarray], candidate: str, reference: str) -> np.ndarray:
    return values[candidate] - values[reference]


def direct_factorial_contrast(values: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    if name == "ocean_product_regime":
        return values["B"] - values["A"]
    if name == "forcing_product":
        return values["C"] - values["A"]
    if name == "combined":
        return values["D"] - values["A"]
    if name == "interaction":
        return values["D"] - values["B"] - values["C"] + values["A"]
    raise ValueError(f"Unknown factorial contrast: {name}")
