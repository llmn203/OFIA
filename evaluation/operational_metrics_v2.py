"""Formal v1.0 metrics for operational replay artifacts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from pipeline.evaluate_hindcast import climatological_acc


@dataclass
class MetricStats:
    model_sse: float
    persistence_sse: float
    absolute_error_sum: float
    truth_sum: float
    truth_sq_sum: float
    valid_count: int
    acc: float = math.nan
    acc_valid_groups: int = 0

    def finalize(self) -> dict[str, Any]:
        if self.valid_count == 0:
            return {
                **self.__dict__,
                "rmse": math.nan,
                "persistence_rmse": math.nan,
                "persistence_skill": math.nan,
                "mae": math.nan,
                "r2": math.nan,
            }
        rmse = math.sqrt(self.model_sse / self.valid_count)
        persistence_rmse = math.sqrt(self.persistence_sse / self.valid_count)
        persistence_skill = 1.0 - rmse / persistence_rmse if persistence_rmse > 0.0 else math.nan
        mae = self.absolute_error_sum / self.valid_count
        tss = self.truth_sq_sum - (self.truth_sum * self.truth_sum) / self.valid_count
        r2 = 1.0 - self.model_sse / tss if tss > 0.0 else math.nan
        return {
            **self.__dict__,
            "rmse": rmse,
            "persistence_rmse": persistence_rmse,
            "persistence_skill": persistence_skill,
            "mae": mae,
            "r2": r2,
        }


def truth_valid_mask(initial_mask, target_mask, domain_mask):
    return np.asarray(initial_mask, dtype=bool) & np.asarray(target_mask, dtype=bool) & np.asarray(domain_mask, dtype=bool)


def stats_from_arrays(prediction, initial, truth, valid, climatology=None) -> MetricStats:
    prediction = np.asarray(prediction, dtype=float)
    initial = np.asarray(initial, dtype=float)
    truth = np.asarray(truth, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if np.any(valid & ~np.isfinite(prediction)):
        raise ValueError("Prediction is non-finite on truth-derived valid support.")
    valid = valid & np.isfinite(truth) & np.isfinite(initial)
    count = int(np.count_nonzero(valid))
    if count == 0:
        return MetricStats(0.0, 0.0, 0.0, 0.0, 0.0, 0, math.nan, 0)
    error = prediction[valid] - truth[valid]
    persistence_error = initial[valid] - truth[valid]
    acc = math.nan
    acc_groups = 0
    if climatology is not None:
        acc = float(climatological_acc(prediction, truth, np.asarray(climatology, dtype=float), valid))
        acc_groups = 0 if math.isnan(acc) else 1
    return MetricStats(
        model_sse=float(np.sum(error**2)),
        persistence_sse=float(np.sum(persistence_error**2)),
        absolute_error_sum=float(np.sum(np.abs(error))),
        truth_sum=float(np.sum(truth[valid])),
        truth_sq_sum=float(np.sum(truth[valid] ** 2)),
        valid_count=count,
        acc=acc,
        acc_valid_groups=acc_groups,
    )


def combine_stats(stats: list[MetricStats]) -> MetricStats:
    valid_acc = [item.acc for item in stats if not math.isnan(item.acc)]
    return MetricStats(
        model_sse=sum(item.model_sse for item in stats),
        persistence_sse=sum(item.persistence_sse for item in stats),
        absolute_error_sum=sum(item.absolute_error_sum for item in stats),
        truth_sum=sum(item.truth_sum for item in stats),
        truth_sq_sum=sum(item.truth_sq_sum for item in stats),
        valid_count=sum(item.valid_count for item in stats),
        acc=float(np.mean(valid_acc)) if valid_acc else math.nan,
        acc_valid_groups=len(valid_acc),
    )


def formal_metric_row(base: dict[str, Any], stats: MetricStats) -> dict[str, Any]:
    return {**base, **stats.finalize()}
