"""Chronos-2 zero-shot adapter for the direct seven-day forecasting path.

The adapter keeps the foundation model isolated behind a lazy import so the
stable project remains runnable without downloading Chronos-2.  It translates
the repository's product/day panel into Chronos' ``predict_df`` contract,
masks censored stockout/gap demand as missing rather than zero, passes only
future-known covariates, and realigns outputs by explicit product/date keys.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable

import numpy as np
import pandas as pd

from framework import Config, compute_baseline, prepare_features, product_reference_dates


CHRONOS2_ID_COLUMN = "item_id"
CHRONOS2_TIMESTAMP_COLUMN = "timestamp"
CHRONOS2_TARGET_COLUMN = "target"

# Every column below is deterministic at the target date and available in the
# supplied test panel.  Availability is intentionally absent: it is used to
# mask historical demand and define evaluation populations, but is not assumed
# to be a known future predictor.
CHRONOS2_KNOWN_COVARIATES = (
    "campaign_web",
    "campaign_app",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "week_of_year_sin",
    "week_of_year_cos",
    "day_of_month",
    "is_weekend",
    "discount_web",
    "discount_app",
    "discount_max",
    "is_sale",
    "price",
    "price_rel",
    "effective_price_web",
    "effective_price_app",
)
CHRONOS2_PAST_COVARIATES = (
    "was_observed",
    "was_available",
)


def _normalise_quantiles(levels: Iterable[float]) -> tuple[float, ...]:
    quantiles = tuple(float(level) for level in levels)
    if not quantiles:
        raise ValueError("Chronos-2 requires at least one quantile level")
    if any(not 0.0 < level < 1.0 for level in quantiles):
        raise ValueError("Chronos-2 quantiles must lie strictly between 0 and 1")
    if tuple(sorted(set(quantiles))) != quantiles:
        raise ValueError("Chronos-2 quantiles must be unique and sorted")
    return quantiles


def resolve_chronos2_device(requested: str = "auto") -> str:
    """Resolve ``auto`` without importing Chronos itself."""
    requested = str(requested).lower()
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_torch_dtype(name: str):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Chronos-2 requires PyTorch. Install the optional Chronos runtime "
            "with `uv pip install -r requirements-chronos.txt`."
        ) from exc
    dtype_name = str(name).lower()
    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    if dtype_name == "auto":
        return torch.float32
    if dtype_name not in mapping:
        raise ValueError(
            f"Unknown Chronos-2 dtype {name!r}; expected auto, float32, "
            "bfloat16, or float16"
        )
    return mapping[dtype_name]


@lru_cache(maxsize=4)
def load_chronos2_pipeline(
    model_id: str,
    model_revision: str,
    device: str,
    dtype: str,
):
    """Load and cache the expensive foundation-model weights once per run."""
    try:
        from chronos import BaseChronosPipeline
    except ImportError as exc:
        raise RuntimeError(
            "Chronos-2 is enabled but `chronos-forecasting` is not installed. "
            "Run `uv pip install -r requirements-chronos.txt` (or "
            "`pip install -r requirements-chronos.txt`) and rerun the pipeline."
        ) from exc

    resolved_device = resolve_chronos2_device(device)
    return BaseChronosPipeline.from_pretrained(
        model_id,
        revision=model_revision,
        device_map=resolved_device,
        torch_dtype=_resolve_torch_dtype(dtype),
    )


def clear_chronos2_pipeline_cache() -> None:
    load_chronos2_pipeline.cache_clear()


def _validate_unique_keys(frame: pd.DataFrame, name: str) -> None:
    required = {"ProductId", "DateKey"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
    if frame.duplicated(["ProductId", "DateKey"]).any():
        examples = frame.loc[
            frame.duplicated(["ProductId", "DateKey"], keep=False),
            ["ProductId", "DateKey"],
        ].head(5)
        raise ValueError(
            f"{name} contains duplicate product/date keys: "
            f"{examples.to_dict(orient='records')}"
        )


def _regularise_context(
    history_raw: pd.DataFrame,
    product_ids: Iterable,
) -> pd.DataFrame:
    """Create one daily context ending at the common forecast origin.

    Existing calendar gaps retain missing targets.  Products that lack rows at
    the very end of the shared history receive explicit missing-target days so
    Chronos sees a consistent daily frequency rather than an accidental jump.
    """
    history = history_raw.copy()
    history["DateKey"] = pd.to_datetime(history["DateKey"])
    cutoff = history["DateKey"].max()
    carry = [
        "CampaignSubTypeWeb",
        "CampaignSubTypeApp",
        "DiscountValueWebRelative",
        "DiscountValueAppRelative",
        "IsSaleOrPromo",
        "PriceLocalVat",
    ]
    frames: list[pd.DataFrame] = []
    for pid in product_ids:
        sub = history[history["ProductId"].eq(pid)].sort_values("DateKey").copy()
        if sub.empty:
            continue
        full_dates = pd.date_range(sub["DateKey"].min(), cutoff, freq="D")
        original_dates = pd.Index(sub["DateKey"])
        regular = sub.set_index("DateKey").reindex(full_dates)
        regular.index.name = "DateKey"
        inserted = ~regular.index.isin(original_dates)
        prior_gap = regular.get(
            "is_gap_filled", pd.Series(False, index=regular.index)
        ).astype("boolean").fillna(False).to_numpy(dtype=bool)
        regular["is_gap_filled"] = inserted | prior_gap
        regular["ProductId"] = pid
        for column in carry:
            if column in regular.columns:
                regular[column] = regular[column].ffill().bfill()
        if "ProductAvailable" in regular.columns:
            regular["ProductAvailable"] = regular["ProductAvailable"].astype("boolean")
        if "Quantity" in regular.columns:
            regular["Quantity"] = pd.to_numeric(regular["Quantity"], errors="coerce")
        frames.append(regular.reset_index())
    if not frames:
        return history.iloc[0:0].copy()
    return pd.concat(frames, ignore_index=True).sort_values(
        ["ProductId", "DateKey"]
    ).reset_index(drop=True)


def _eligible_product_ids(
    history_raw: pd.DataFrame,
    future_raw: pd.DataFrame,
) -> tuple[Any, ...]:
    """Return products with enough non-censored context for model inference.

    Chronos validates each item independently.  A newly introduced product, or
    a product whose entire history is censored by unavailability/calendar gaps,
    must therefore be handled by the repository's established fallback rather
    than aborting a complete rolling-origin fold.
    """
    history = history_raw.copy()
    history["DateKey"] = pd.to_datetime(history["DateKey"])
    cutoff = history["DateKey"].max()
    eligible: list[Any] = []
    for pid in pd.unique(future_raw["ProductId"]):
        sub = history.loc[history["ProductId"].eq(pid)].copy()
        if sub.empty:
            continue
        calendar_length = int((cutoff - sub["DateKey"].min()).days) + 1
        if calendar_length < 3:
            continue
        available = sub["ProductAvailable"].astype("boolean").fillna(False).astype(bool)
        gap = sub.get(
            "is_gap_filled", pd.Series(False, index=sub.index)
        ).astype("boolean").fillna(False).astype(bool)
        target = pd.to_numeric(sub["Quantity"], errors="coerce")
        if (target.notna() & available & ~gap).any():
            eligible.append(pid)
    return tuple(eligible)


def _empty_feature_frames(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    context_columns = [
        CHRONOS2_ID_COLUMN, CHRONOS2_TIMESTAMP_COLUMN, CHRONOS2_TARGET_COLUMN
    ]
    future_columns = [CHRONOS2_ID_COLUMN, CHRONOS2_TIMESTAMP_COLUMN]
    if cfg.chronos2_covariates:
        context_columns.extend(CHRONOS2_KNOWN_COVARIATES)
        context_columns.extend(CHRONOS2_PAST_COVARIATES)
        future_columns.extend(CHRONOS2_KNOWN_COVARIATES)
    return pd.DataFrame(columns=context_columns), pd.DataFrame(columns=future_columns)


def _validate_future_grid(future_raw: pd.DataFrame, horizon: int) -> None:
    _validate_unique_keys(future_raw, "Chronos-2 future panel")
    future = future_raw.copy()
    future["DateKey"] = pd.to_datetime(future["DateKey"])
    counts = future.groupby("ProductId", sort=False).size()
    invalid = counts[counts.ne(horizon)]
    if not invalid.empty:
        raise ValueError(
            "Chronos-2 expects exactly one full forecast horizon per product; "
            f"invalid row counts: {invalid.to_dict()}"
        )
    for pid, group in future.groupby("ProductId", sort=False):
        dates = group["DateKey"].sort_values().reset_index(drop=True)
        expected = pd.date_range(dates.iloc[0], periods=horizon, freq="D")
        if not dates.equals(pd.Series(expected)):
            raise ValueError(
                f"Chronos-2 future dates are not a contiguous daily horizon for "
                f"ProductId={pid!r}"
            )


def _feature_frames(
    history_raw: pd.DataFrame,
    future_raw: pd.DataFrame,
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate_unique_keys(history_raw, "Chronos-2 history")
    _validate_future_grid(future_raw, cfg.horizon)
    product_ids = _eligible_product_ids(history_raw, future_raw)
    if not product_ids:
        return _empty_feature_frames(cfg)
    context_raw = _regularise_context(history_raw, product_ids)
    future = future_raw.loc[future_raw["ProductId"].isin(product_ids)].copy()
    future["DateKey"] = pd.to_datetime(future["DateKey"])

    context_products = set(context_raw["ProductId"].unique())
    future_products = set(future["ProductId"].unique())
    if context_products != future_products:
        raise ValueError(
            "Chronos-2 context/future product sets differ: "
            f"context_only={sorted(context_products - future_products)}, "
            f"future_only={sorted(future_products - context_products)}"
        )

    context_end = context_raw.groupby("ProductId")["DateKey"].max()
    future_start = future.groupby("ProductId")["DateKey"].min()
    gaps = (future_start - context_end).dt.days
    if not gaps.eq(1).all():
        raise ValueError(
            "Chronos-2 requires the future horizon to begin one day after the "
            f"context origin; day gaps={gaps.to_dict()}"
        )

    price_ref = context_raw.groupby("ProductId")["PriceLocalVat"].median()
    first_seen, first_available = product_reference_dates(context_raw)
    context_features = prepare_features(
        context_raw, price_ref, first_seen, first_available, cfg
    )
    future_features = prepare_features(
        future, price_ref, first_seen, first_available, cfg
    )

    gap = context_features.get(
        "is_gap_filled", pd.Series(False, index=context_features.index)
    ).astype("boolean").fillna(False).astype(bool)
    available = context_features["ProductAvailable"].astype("boolean")
    observed = available.notna() & ~gap
    available_bool = available.fillna(False).astype(bool) & ~gap
    target = pd.to_numeric(context_features["Quantity"], errors="coerce").where(
        available_bool
    )

    context = pd.DataFrame({
        CHRONOS2_ID_COLUMN: context_features["ProductId"].astype(str),
        CHRONOS2_TIMESTAMP_COLUMN: pd.to_datetime(context_features["DateKey"]),
        CHRONOS2_TARGET_COLUMN: target.astype(float),
    })
    future_frame = pd.DataFrame({
        CHRONOS2_ID_COLUMN: future_features["ProductId"].astype(str),
        CHRONOS2_TIMESTAMP_COLUMN: pd.to_datetime(future_features["DateKey"]),
    })

    if cfg.chronos2_covariates:
        context["campaign_web"] = (
            pd.to_numeric(context_features["CampaignSubTypeWeb"], errors="coerce")
            .fillna(-1).astype(int).astype(str)
        )
        context["campaign_app"] = (
            pd.to_numeric(context_features["CampaignSubTypeApp"], errors="coerce")
            .fillna(-1).astype(int).astype(str)
        )
        future_frame["campaign_web"] = (
            pd.to_numeric(future_features["CampaignSubTypeWeb"], errors="coerce")
            .fillna(-1).astype(int).astype(str)
        )
        future_frame["campaign_app"] = (
            pd.to_numeric(future_features["CampaignSubTypeApp"], errors="coerce")
            .fillna(-1).astype(int).astype(str)
        )
        numeric_known = [
            column for column in CHRONOS2_KNOWN_COVARIATES
            if column not in {"campaign_web", "campaign_app"}
        ]
        for column in numeric_known:
            context[column] = pd.to_numeric(
                context_features[column], errors="coerce"
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
            future_frame[column] = pd.to_numeric(
                future_features[column], errors="coerce"
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        context["was_observed"] = observed.astype(float)
        context["was_available"] = available_bool.astype(float)

    return context, future_frame


def build_chronos2_frames(
    history_raw: pd.DataFrame,
    future_raw: pd.DataFrame,
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Public, testable translation into the Chronos dataframe contract."""
    return _feature_frames(history_raw, future_raw, cfg)


def _find_quantile_column(columns: Iterable[Any], quantile: float):
    candidates = {
        quantile,
        str(quantile),
        f"{quantile:g}",
        f"quantile_{quantile:g}",
        f"q{quantile:g}",
    }
    for column in columns:
        if column in candidates:
            return column
        try:
            if np.isclose(float(str(column)), quantile, atol=1e-12):
                return column
        except (TypeError, ValueError):
            continue
    return None


def _robust_fallback(
    history_raw: pd.DataFrame,
    future_raw: pd.DataFrame,
    baseline_variant: str,
) -> np.ndarray:
    baseline = compute_baseline(future_raw, history_raw, baseline_variant)
    history = history_raw.copy()
    available = history["ProductAvailable"].astype("boolean").fillna(False).astype(bool)
    gap = history.get(
        "is_gap_filled", pd.Series(False, index=history.index)
    ).astype("boolean").fillna(False).astype(bool)
    quantity = pd.to_numeric(history["Quantity"], errors="coerce").where(available & ~gap)
    product_level = quantity.groupby(history["ProductId"]).median()
    global_level = float(quantity.median()) if quantity.notna().any() else 0.0
    product_fallback = future_raw["ProductId"].map(product_level).fillna(global_level)
    return np.where(
        np.isfinite(baseline), baseline, product_fallback.to_numpy(dtype=float)
    )


def forecast_chronos2(
    history_raw: pd.DataFrame,
    future_raw: pd.DataFrame,
    cfg: Config,
    *,
    pipeline=None,
) -> pd.DataFrame:
    """Run Chronos-2 and return a key-aligned prediction/diagnostic frame."""
    quantiles = _normalise_quantiles(cfg.chronos2_quantile_levels)
    context, future = build_chronos2_frames(history_raw, future_raw, cfg)
    if future.empty:
        predicted = pd.DataFrame({
            CHRONOS2_ID_COLUMN: pd.Series(dtype="object"),
            CHRONOS2_TIMESTAMP_COLUMN: pd.Series(dtype="datetime64[ns]"),
            "predictions": pd.Series(dtype=float),
            **{f"{quantile:g}": pd.Series(dtype=float) for quantile in quantiles},
        })
    else:
        model = pipeline or load_chronos2_pipeline(
            cfg.chronos2_model_id,
            cfg.chronos2_model_revision,
            cfg.chronos2_device,
            cfg.chronos2_dtype,
        )
        kwargs: dict[str, Any] = {
            "df": context,
            "future_df": future,
            "id_column": CHRONOS2_ID_COLUMN,
            "timestamp_column": CHRONOS2_TIMESTAMP_COLUMN,
            "target": CHRONOS2_TARGET_COLUMN,
            "prediction_length": cfg.horizon,
            "quantile_levels": list(quantiles),
            "batch_size": cfg.chronos2_batch_size,
            "cross_learning": cfg.chronos2_cross_learning,
            "validate_inputs": True,
            "freq": "D",
        }
        if cfg.chronos2_context_length is not None:
            kwargs["context_length"] = cfg.chronos2_context_length
        predicted = model.predict_df(**kwargs).copy()
    required = {
        CHRONOS2_ID_COLUMN,
        CHRONOS2_TIMESTAMP_COLUMN,
        "predictions",
    }
    missing = sorted(required - set(predicted.columns))
    if missing:
        raise RuntimeError(
            f"Chronos-2 returned an unexpected dataframe; missing columns: {missing}"
        )
    if "target_name" in predicted.columns:
        target_rows = predicted["target_name"].astype(str).eq(CHRONOS2_TARGET_COLUMN)
        if not predicted.empty and not target_rows.any():
            raise RuntimeError(
                "Chronos-2 output contains no rows for the requested target column"
            )
        predicted = predicted.loc[target_rows].copy()

    predicted[CHRONOS2_ID_COLUMN] = predicted[CHRONOS2_ID_COLUMN].astype(str)
    predicted[CHRONOS2_TIMESTAMP_COLUMN] = pd.to_datetime(
        predicted[CHRONOS2_TIMESTAMP_COLUMN]
    )
    if predicted.duplicated(
        [CHRONOS2_ID_COLUMN, CHRONOS2_TIMESTAMP_COLUMN]
    ).any():
        raise RuntimeError("Chronos-2 returned duplicate product/date predictions")

    lookup_columns = [
        CHRONOS2_ID_COLUMN,
        CHRONOS2_TIMESTAMP_COLUMN,
        "predictions",
    ]
    quantile_lookup: dict[float, Any] = {}
    for quantile in quantiles:
        column = _find_quantile_column(predicted.columns, quantile)
        if column is None:
            raise RuntimeError(
                f"Chronos-2 output is missing requested quantile {quantile:g}; "
                f"returned columns={list(predicted.columns)!r}"
            )
        quantile_lookup[quantile] = column
        if column not in lookup_columns:
            lookup_columns.append(column)

    keys = future_raw[["ProductId", "DateKey"]].copy()
    keys["_chronos_id"] = keys["ProductId"].astype(str)
    keys["_chronos_timestamp"] = pd.to_datetime(keys["DateKey"])
    renamed = predicted[lookup_columns].rename(columns={
        CHRONOS2_ID_COLUMN: "_chronos_id",
        CHRONOS2_TIMESTAMP_COLUMN: "_chronos_timestamp",
        "predictions": "prediction_raw_model",
        **{
            column: f"quantile_{quantile:g}"
            for quantile, column in quantile_lookup.items()
        },
    })
    aligned = keys.merge(
        renamed,
        on=["_chronos_id", "_chronos_timestamp"],
        how="left",
        validate="one_to_one",
    )
    if not context.empty and aligned["prediction_raw_model"].isna().all():
        raise RuntimeError("Chronos-2 predictions did not align to any eligible future keys")
    eligible_ids = set(context[CHRONOS2_ID_COLUMN].astype(str))
    no_context = ~aligned["_chronos_id"].isin(eligible_ids).to_numpy(dtype=bool)

    if 0.5 not in quantile_lookup:
        raise RuntimeError("Chronos-2 publication requires the q50 median")
    raw_point = pd.to_numeric(
        aligned["quantile_0.5"], errors="coerce"
    ).to_numpy(dtype=float)
    nonfinite = ~np.isfinite(raw_point)
    fallback = _robust_fallback(history_raw, future_raw, cfg.baseline_variant)
    point = np.where(nonfinite, fallback, raw_point)
    point = np.clip(point, 0.0, None)

    quantile_matrix = []
    for quantile in quantiles:
        values = pd.to_numeric(
            aligned[f"quantile_{quantile:g}"], errors="coerce"
        ).to_numpy(dtype=float)
        values = np.where(np.isfinite(values), values, point)
        values = np.where(nonfinite, point, values)
        quantile_matrix.append(np.clip(values, 0.0, None))
    quantile_values = np.maximum.accumulate(np.column_stack(quantile_matrix), axis=1)

    result = aligned[["ProductId", "DateKey"]].copy()
    result["prediction"] = point
    result["prediction_raw_model"] = raw_point
    result["fallback_used"] = nonfinite
    result["nonfinite_raw"] = nonfinite
    result["no_context"] = no_context
    result["catastrophic_guard"] = False
    result["residual_guard"] = False
    result["residual_nonfinite"] = False
    result["residual_raw_min"] = np.nan
    result["residual_raw_max"] = np.nan
    result["safety_limit"] = np.nan
    for index, quantile in enumerate(quantiles):
        result[f"quantile_{quantile:g}"] = quantile_values[:, index]
    return result
