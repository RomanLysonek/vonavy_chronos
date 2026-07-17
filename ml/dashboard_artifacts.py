"""Tier C6 dashboard diagnostics and static-site publication helpers."""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from framework import compute_metrics, prediction_columns_for_strategy


from static_site import DASHBOARD_SCHEMA_VERSION


def summarize_sanity_baseline(
    oof: pd.DataFrame,
    pred_columns: Mapping[str, str],
) -> pd.DataFrame:
    """Compare both contenders and the seasonal anchor on identical rows."""
    if oof.empty or "baseline" not in oof:
        return pd.DataFrame()
    rows: list[dict] = []
    for (origin_type, strategy), split in oof.groupby(
        ["origin_type", "strategy"], sort=False
    ):
        columns = prediction_columns_for_strategy(dict(pred_columns), str(strategy))
        columns = {name: col for name, col in columns.items() if col in split}
        all_columns = {**columns, "SeasonalWeekdayNaive": "baseline"}
        finite_columns = list(all_columns.values())
        common = (
            split["ProductAvailable"].astype("boolean").fillna(False).astype(bool)
            & np.isfinite(pd.to_numeric(split["actual"], errors="coerce"))
            & split[finite_columns].apply(np.isfinite).all(axis=1)
        )
        scored = split.loc[common]
        for estimator, column in all_columns.items():
            metrics = compute_metrics(scored["actual"], scored[column])
            rows.append({
                "origin_type": str(origin_type),
                "strategy": str(strategy),
                "estimator": estimator,
                "role": (
                    "sanity_baseline"
                    if estimator == "SeasonalWeekdayNaive"
                    else "head_to_head"
                ),
                **metrics,
            })
    return pd.DataFrame(rows)


def summarize_probabilistic_oof(oof: pd.DataFrame) -> pd.DataFrame:
    """Score Chronos q10/q50/q90 only when all real OOF columns exist."""
    required = {
        "actual",
        "ProductAvailable",
        "pred_Chronos2_q10",
        "pred_Chronos2_q50",
        "pred_Chronos2_q90",
    }
    if oof.empty or not required.issubset(oof.columns):
        return pd.DataFrame()
    rows: list[dict] = []
    for (origin_type, strategy), split in oof.groupby(
        ["origin_type", "strategy"], sort=False
    ):
        actual = pd.to_numeric(split["actual"], errors="coerce").to_numpy(dtype=float)
        available = (
            split["ProductAvailable"].astype("boolean").fillna(False).to_numpy(dtype=bool)
        )
        quantiles = {
            0.1: pd.to_numeric(
                split["pred_Chronos2_q10"], errors="coerce"
            ).to_numpy(dtype=float),
            0.5: pd.to_numeric(
                split["pred_Chronos2_q50"], errors="coerce"
            ).to_numpy(dtype=float),
            0.9: pd.to_numeric(
                split["pred_Chronos2_q90"], errors="coerce"
            ).to_numpy(dtype=float),
        }
        mask = available & np.isfinite(actual)
        for values in quantiles.values():
            mask &= np.isfinite(values)
        if not mask.any():
            continue
        y = actual[mask]
        scored = {level: values[mask] for level, values in quantiles.items()}
        record: dict = {
            "origin_type": str(origin_type),
            "strategy": str(strategy),
            "model": "Chronos2",
            "n": int(mask.sum()),
            "nominal_interval_coverage": 0.8,
            "interval_coverage": float(
                np.mean((y >= scored[0.1]) & (y <= scored[0.9]))
            ),
            "interval_mean_width": float(np.mean(scored[0.9] - scored[0.1])),
        }
        mean_actual = float(np.mean(np.abs(y)))
        record["interval_normalized_width"] = (
            record["interval_mean_width"] / mean_actual
            if mean_actual > 0
            else np.nan
        )
        for level, values in scored.items():
            error = y - values
            suffix = f"q{int(level * 100)}"
            record[f"pinball_{suffix}"] = float(
                np.mean(np.maximum(level * error, (level - 1.0) * error))
            )
            empirical = float(np.mean(y <= values))
            record[f"empirical_{suffix}"] = empirical
            record[f"calibration_error_{suffix}"] = empirical - level
        rows.append(record)
    return pd.DataFrame(rows)


def summarize_per_product_oof(
    oof: pd.DataFrame,
    pred_columns: Mapping[str, str],
) -> pd.DataFrame:
    """Conditional/common product-level error and bias diagnostics."""
    if oof.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for (origin_type, strategy), split in oof.groupby(["origin_type", "strategy"], sort=False):
        columns = prediction_columns_for_strategy(dict(pred_columns), str(strategy))
        columns = {model: col for model, col in columns.items() if col in split.columns}
        if not columns:
            continue
        common = (
            split["ProductAvailable"].astype("boolean").fillna(False).astype(bool)
            & np.isfinite(pd.to_numeric(split["actual"], errors="coerce"))
            & split[list(columns.values())].apply(np.isfinite).all(axis=1)
        )
        work = split.loc[common].copy()
        for product_id, product in work.groupby("ProductId", sort=True):
            actual = pd.to_numeric(product["actual"], errors="coerce").to_numpy(dtype=float)
            for model, column in columns.items():
                prediction = pd.to_numeric(product[column], errors="coerce").to_numpy(dtype=float)
                metrics = compute_metrics(actual, prediction)
                rows.append({
                    "origin_type": str(origin_type),
                    "strategy": str(strategy),
                    "ProductId": int(product_id),
                    "model": model,
                    "n": int(len(product)),
                    "actual_total": float(np.sum(actual)),
                    "prediction_total": float(np.sum(prediction)),
                    **metrics,
                })
    return pd.DataFrame(rows)


def summarize_top_deciles(
    oof: pd.DataFrame,
    pred_columns: Mapping[str, str],
    *,
    quantile: float = 0.90,
    max_error_rows: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return top-demand metrics and the largest row-level errors.

    The high-demand population is defined from actual demand once per
    split/strategy and therefore remains comparable across models.  Largest
    error rows are model-specific explanatory diagnostics, not a selection
    population.
    """
    if not 0.5 <= quantile < 1.0:
        raise ValueError("quantile must be in [0.5, 1.0)")
    summary_rows: list[dict] = []
    error_rows: list[pd.DataFrame] = []
    for (origin_type, strategy), split in oof.groupby(["origin_type", "strategy"], sort=False):
        columns = prediction_columns_for_strategy(dict(pred_columns), str(strategy))
        columns = {model: col for model, col in columns.items() if col in split.columns}
        if not columns:
            continue
        common = (
            split["ProductAvailable"].astype("boolean").fillna(False).astype(bool)
            & np.isfinite(pd.to_numeric(split["actual"], errors="coerce"))
            & split[list(columns.values())].apply(np.isfinite).all(axis=1)
        )
        work = split.loc[common].copy()
        if work.empty:
            continue
        actual = pd.to_numeric(work["actual"], errors="coerce")
        threshold = float(actual.quantile(quantile))
        high = work.loc[actual.ge(threshold)]
        for model, column in columns.items():
            metrics = compute_metrics(high["actual"], high[column])
            summary_rows.append({
                "origin_type": str(origin_type),
                "strategy": str(strategy),
                "model": model,
                "quantile": float(quantile),
                "actual_threshold": threshold,
                "n": int(len(high)),
                **metrics,
            })
            detail_columns = [
                col for col in (
                    "origin_type", "strategy", "origin", "validation_stratum",
                    "ProductId", "DateKey", "horizon", "actual"
                ) if col in work.columns
            ]
            detail = work[detail_columns].copy()
            detail["model"] = model
            detail["prediction"] = pd.to_numeric(work[column], errors="coerce")
            detail["absolute_error"] = np.abs(detail["prediction"] - detail["actual"])
            detail["signed_error"] = detail["prediction"] - detail["actual"]
            detail = detail.nlargest(max_error_rows, "absolute_error")
            error_rows.append(detail)
    return (
        pd.DataFrame(summary_rows),
        pd.concat(error_rows, ignore_index=True) if error_rows else pd.DataFrame(),
    )
