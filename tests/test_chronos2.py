import numpy as np
import pandas as pd

from framework import Config, MODEL_META, MODEL_ORDER, MODEL_STRATEGY_SUPPORT
from models.chronos2_model import build_chronos2_frames, forecast_chronos2
from pipeline import SubmissionModel, configure_chronos2_runtime, parse_args


def _raw(periods: int = 35, horizon: int = 3):
    dates = pd.date_range("2025-01-01", periods=periods + horizon, freq="D")
    rows = []
    for pid in (1, 2):
        for i, date in enumerate(dates):
            rows.append({
                "ProductId": pid,
                "DateKey": date,
                "Quantity": float(10 * pid + i),
                "QuantityApp": float(6 * pid + i / 2),
                "QuantityWeb": float(4 * pid + i / 2),
                "ProductAvailable": True,
                "CampaignSubTypeWeb": 16 if i % 9 == 0 else -1,
                "CampaignSubTypeApp": 3 if i % 11 == 0 else -1,
                "DiscountValueWebRelative": 15.0 if i % 9 == 0 else 0.0,
                "DiscountValueAppRelative": 10.0 if i % 11 == 0 else 0.0,
                "IsSaleOrPromo": i % 7 == 0,
                "PriceLocalVat": 100.0 + pid,
                "is_gap_filled": False,
            })
    full = pd.DataFrame(rows)
    split = dates[periods - 1]
    history = full[full["DateKey"].le(split)].copy()
    future = full[full["DateKey"].gt(split)].copy()
    return history, future


class FakeChronosPipeline:
    def __init__(self, *, make_first_nonfinite: bool = False):
        self.make_first_nonfinite = make_first_nonfinite
        self.calls = []

    def predict_df(self, **kwargs):
        self.calls.append(kwargs)
        future = kwargs["future_df"][["item_id", "timestamp"]].copy()
        first_date = future["timestamp"].min()
        day = (future["timestamp"] - first_date).dt.days.to_numpy(dtype=float)
        item = future["item_id"].astype(int).to_numpy(dtype=float)
        base_point = item * 100.0 + day
        point = base_point.copy()
        if self.make_first_nonfinite:
            point[0] = np.nan
        output = future.copy()
        output["target_name"] = "target"
        output["predictions"] = point
        output["0.1"] = base_point - 5.0
        output["0.5"] = base_point
        output["0.9"] = base_point + 5.0
        if self.make_first_nonfinite:
            output.iloc[0, output.columns.get_loc("0.5")] = np.nan
        return output.iloc[::-1].reset_index(drop=True)


def test_chronos2_frames_mask_censored_targets_and_separate_covariates():
    history, future = _raw()
    unavailable_idx = history.index[5]
    history.loc[unavailable_idx, "ProductAvailable"] = False
    history.loc[unavailable_idx, "Quantity"] = 9999.0
    cfg = Config(horizon=3, chronos2_covariates=True)

    context, future_frame = build_chronos2_frames(history, future, cfg)

    unavailable_date = history.loc[unavailable_idx, "DateKey"]
    masked = context[
        context["item_id"].eq("1") & context["timestamp"].eq(unavailable_date)
    ]["target"]
    assert len(masked) == 1
    assert masked.isna().all()
    assert "was_available" in context.columns
    assert "was_available" not in future_frame.columns
    assert "ProductAvailable" not in future_frame.columns
    for column in ("campaign_web", "discount_web", "price", "is_weekend"):
        assert column in context.columns
        assert column in future_frame.columns


def test_chronos2_forecast_realigns_shuffled_output_and_passes_contract():
    history, future = _raw()
    cfg = Config(
        horizon=3,
        chronos2_batch_size=17,
        chronos2_context_length=28,
        chronos2_cross_learning=True,
        chronos2_quantile_levels=(0.1, 0.5, 0.9),
    )
    fake = FakeChronosPipeline()

    result = forecast_chronos2(history, future, cfg, pipeline=fake)

    expected = (
        future["ProductId"].to_numpy(dtype=float) * 100.0
        + (future["DateKey"] - future["DateKey"].min()).dt.days.to_numpy(dtype=float)
    )
    np.testing.assert_allclose(result["prediction"], expected)
    np.testing.assert_allclose(result["quantile_0.5"], expected)
    assert result[["ProductId", "DateKey"]].equals(
        future[["ProductId", "DateKey"]].reset_index(drop=True)
    )
    call = fake.calls[0]
    assert call["prediction_length"] == 3
    assert call["batch_size"] == 17
    assert call["context_length"] == 28
    assert call["cross_learning"] is True
    assert call["freq"] == "D"


def test_chronos2_nonfinite_point_uses_finite_nonnegative_fallback():
    history, future = _raw()
    cfg = Config(horizon=3)
    result = forecast_chronos2(
        history,
        future,
        cfg,
        pipeline=FakeChronosPipeline(make_first_nonfinite=True),
    )

    assert bool(result.loc[0, "fallback_used"])
    assert bool(result.loc[0, "nonfinite_raw"])
    assert np.isfinite(result.loc[0, "prediction"])
    assert result.loc[0, "prediction"] >= 0.0
    assert result.loc[0, "quantile_0.1"] == result.loc[0, "prediction"]
    assert result.loc[0, "quantile_0.5"] == result.loc[0, "prediction"]
    assert result.loc[0, "quantile_0.9"] == result.loc[0, "prediction"]
    assert (result["quantile_0.1"] <= result["quantile_0.5"]).all()
    assert (result["quantile_0.5"] <= result["quantile_0.9"]).all()


def test_chronos2_cli_and_registry_are_direct_only():
    options = parse_args([
        "--chronos2", "on",
        "--chronos2-device", "cpu",
        "--chronos2-profile", "target-only",
        "--chronos2-batch-size", "24",
        "--submission-model", "Chronos2",
    ])
    cfg = Config()
    runtime = configure_chronos2_runtime(cfg, options)

    assert options.submission_model is SubmissionModel.CHRONOS2
    assert runtime["enabled"] is True
    assert cfg.chronos2_context_length is None
    assert cfg.chronos2_batch_size == 24
    assert cfg.chronos2_cross_learning is False
    assert cfg.chronos2_covariates is False
    assert cfg.chronos2_model_revision == "29ec3766d36d6f73f0696f85560a422f50e8498c"
    assert "Chronos2" in MODEL_ORDER
    assert MODEL_STRATEGY_SUPPORT["Chronos2"] == {"direct"}
    assert MODEL_META["Chronos2"]["source_url"].endswith("amazon/chronos-2")


def test_direct_cv_merges_chronos_predictions_and_diagnostics(monkeypatch):
    import pipeline as pipeline_module

    history, future = _raw(periods=35, horizon=3)
    full = pd.concat([history, future], ignore_index=True)
    origin = history["DateKey"].max()
    cfg = Config(horizon=3, enable_chronos2=True, num_products=2)

    def fake_forecast(fold_history, fold_future, passed_cfg):
        result = fold_future[["ProductId", "DateKey"]].copy()
        result["prediction"] = result["ProductId"].astype(float) * 10.0
        result["fallback_used"] = False
        result["nonfinite_raw"] = False
        result["no_context"] = False
        result["catastrophic_guard"] = False
        result["residual_guard"] = False
        result["residual_nonfinite"] = False
        result["residual_raw_min"] = np.nan
        result["residual_raw_max"] = np.nan
        result["safety_limit"] = np.nan
        result["quantile_0.1"] = result["prediction"] - 1.0
        result["quantile_0.5"] = result["prediction"]
        result["quantile_0.9"] = result["prediction"] + 1.0
        return result

    monkeypatch.setattr(pipeline_module, "forecast_chronos2", fake_forecast)
    oof = pipeline_module.run_walk_forward_cv_direct(
        full,
        [origin],
        "development",
        cfg,
        run_neural=False,
    )

    assert len(oof) == 6
    assert "pred_Chronos2" in oof
    assert "pred_Chronos2_q10" in oof
    assert "fallback_Chronos2" in oof
    assert "no_context_Chronos2" in oof
    np.testing.assert_allclose(
        oof["pred_Chronos2"], oof["ProductId"].to_numpy(dtype=float) * 10.0
    )

    diagnostics = pipeline_module.summarize_prediction_diagnostics(oof)
    chronos = diagnostics.loc[diagnostics["model"].eq("Chronos2")].iloc[0]
    assert chronos["no_context_count"] == 0
    assert chronos["no_context_rate"] == 0.0



def test_chronos2_falls_back_only_for_products_without_usable_context():
    history, future = _raw()
    template = future.loc[future["ProductId"].eq(1)].copy()
    template["ProductId"] = 3
    template["Quantity"] = 0.0
    template["QuantityApp"] = 0.0
    template["QuantityWeb"] = 0.0
    future = pd.concat([future, template], ignore_index=True).sort_values(
        ["ProductId", "DateKey"]
    ).reset_index(drop=True)
    cfg = Config(horizon=3)
    fake = FakeChronosPipeline()

    result = forecast_chronos2(history, future, cfg, pipeline=fake)

    assert len(fake.calls) == 1
    assert set(fake.calls[0]["future_df"]["item_id"]) == {"1", "2"}
    missing_history = result["ProductId"].eq(3)
    assert result.loc[missing_history, "fallback_used"].all()
    assert result.loc[missing_history, "no_context"].all()
    assert np.isfinite(result.loc[missing_history, "prediction"]).all()
    assert (~result.loc[~missing_history, "no_context"]).all()


def test_chronos2_skips_model_when_every_product_lacks_usable_context():
    history, future = _raw()
    history["ProductAvailable"] = False
    fake = FakeChronosPipeline()
    cfg = Config(horizon=3)

    result = forecast_chronos2(history, future, cfg, pipeline=fake)

    assert fake.calls == []
    assert result["fallback_used"].all()
    assert result["no_context"].all()
    assert np.isfinite(result["prediction"]).all()


def test_resume_augments_only_chronos_and_reuses_base_checkpoint(
    monkeypatch, tmp_path
):
    import pipeline as pipeline_module

    history, future = _raw(periods=35, horizon=3)
    full = pd.concat([history, future], ignore_index=True)
    origin = history["DateKey"].max()
    checkpoint_dir = tmp_path / "checkpoints"

    base_cfg = Config(horizon=3, enable_chronos2=False, num_products=2)
    base = pipeline_module.run_walk_forward_cv_direct(
        full,
        [origin],
        "development",
        base_cfg,
        run_neural=False,
        checkpoint_dir=str(checkpoint_dir),
        resume=False,
    )
    assert "pred_Chronos2" not in base

    calls = []

    def fake_forecast(fold_history, fold_future, passed_cfg):
        calls.append((len(fold_history), len(fold_future)))
        result = fold_future[["ProductId", "DateKey"]].copy()
        result["prediction"] = 42.0
        result["fallback_used"] = False
        result["nonfinite_raw"] = False
        result["no_context"] = False
        result["catastrophic_guard"] = False
        result["residual_guard"] = False
        result["residual_nonfinite"] = False
        result["residual_raw_min"] = np.nan
        result["residual_raw_max"] = np.nan
        result["safety_limit"] = np.nan
        result["quantile_0.1"] = 40.0
        result["quantile_0.5"] = 42.0
        result["quantile_0.9"] = 44.0
        return result

    monkeypatch.setattr(pipeline_module, "forecast_chronos2", fake_forecast)
    chronos_cfg = Config(horizon=3, enable_chronos2=True, num_products=2)
    augmented = pipeline_module.run_walk_forward_cv_direct(
        full,
        [origin],
        "development",
        chronos_cfg,
        run_neural=False,
        checkpoint_dir=str(checkpoint_dir),
        resume=True,
    )
    assert calls == [(len(history), len(future))]
    assert (augmented["pred_Chronos2"] == 42.0).all()

    calls.clear()
    reused = pipeline_module.run_walk_forward_cv_direct(
        full,
        [origin],
        "development",
        chronos_cfg,
        run_neural=False,
        checkpoint_dir=str(checkpoint_dir),
        resume=True,
    )
    assert calls == []
    assert (reused["pred_Chronos2"] == 42.0).all()

    disabled = pipeline_module.run_walk_forward_cv_direct(
        full,
        [origin],
        "development",
        base_cfg,
        run_neural=False,
        checkpoint_dir=str(checkpoint_dir),
        resume=True,
    )
    assert "pred_Chronos2" not in disabled
    assert not any(column.endswith("_Chronos2") for column in disabled.columns)
