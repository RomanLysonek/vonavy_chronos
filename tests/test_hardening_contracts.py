import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dashboard_artifacts import (
    summarize_probabilistic_oof,
    summarize_sanity_baseline,
)
from export_results import export_from_artifacts
from pipeline import (
    OOF_MODEL_COLUMNS,
    RuntimeOptions,
    validate_final_audit_policy,
)
from provenance import sha256_file, sha256_json
from static_site import check_static_dashboard, publish_static_dashboard


def _summary() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "model": model,
            "strategy": "direct",
            "evaluation_regime": "conditional",
            "comparison_population": "common",
            "aggregation": "global",
            "n_folds": 1,
            "n_expected": 2,
            "n_actual": 2,
            "n_predicted": 2,
            "n_scored": 2,
            "coverage": 1.0,
            "MAE": 1.0 + index,
            "RMSE": 1.0 + index,
            "MAPE": 10.0,
            "WAPE": 0.1 + index / 10,
            "sMAPE": 0.1,
            "RMSLE": 0.1,
            "Bias": float(index),
            "BiasRatio": 0.01 * index,
            "n": 2,
        }
        for index, model in enumerate(("NeuralNet", "Chronos2"))
    ])


def test_supporting_evidence_uses_common_rows_and_real_quantiles():
    oof = pd.DataFrame({
        "origin_type": ["development"] * 3,
        "strategy": ["direct"] * 3,
        "ProductAvailable": [True, True, False],
        "actual": [10.0, 20.0, 999.0],
        "pred_NeuralNet": [11.0, 18.0, 999.0],
        "pred_Chronos2": [12.0, 17.0, 999.0],
        "baseline": [9.0, 19.0, np.nan],
        "pred_Chronos2_q10": [8.0, 14.0, 0.0],
        "pred_Chronos2_q50": [12.0, 17.0, 1.0],
        "pred_Chronos2_q90": [15.0, 24.0, 2.0],
    })
    baseline = summarize_sanity_baseline(oof, OOF_MODEL_COLUMNS)
    assert set(baseline["estimator"]) == {
        "NeuralNet", "Chronos2", "SeasonalWeekdayNaive"
    }
    assert set(baseline["n"]) == {2}

    probability = summarize_probabilistic_oof(oof)
    assert len(probability) == 1
    assert probability.iloc[0]["n"] == 2
    assert probability.iloc[0]["interval_coverage"] == 1.0
    assert probability.iloc[0]["interval_mean_width"] == 8.5
    assert probability.iloc[0]["pinball_q50"] > 0


def test_final_audit_publication_is_single_use(tmp_path):
    marker = tmp_path / "consumed.json"
    options = RuntimeOptions(run_kind="publication", include_final_audit=True)
    validate_final_audit_policy(options, str(marker))
    marker.write_text("{}")
    with pytest.raises(RuntimeError, match="already consumed"):
        validate_final_audit_policy(options, str(marker))
    validate_final_audit_policy(
        RuntimeOptions(run_kind="reproduction", include_final_audit=True),
        str(marker),
    )


def test_hash_helpers_are_stable(tmp_path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("chronos\n")
    assert len(sha256_file(artifact)) == 64
    assert sha256_json({"b": 2, "a": 1}) == sha256_json({"a": 1, "b": 2})


def test_static_publisher_round_trip(tmp_path):
    static = tmp_path / "webapp" / "static"
    outputs = tmp_path / "outputs"
    static.mkdir(parents=True)
    outputs.mkdir()
    (static / "index.html").write_text(
        '<script src="/static/common.js"></script>', encoding="utf-8"
    )
    (static / "common.js").write_text("const ok = true;\n", encoding="utf-8")
    (outputs / "results.json").write_text('{"ok": true}\n', encoding="utf-8")
    publish_static_dashboard(tmp_path, outputs / "results.json")
    check_static_dashboard(tmp_path)
    assert "window.STATIC_DASHBOARD" in (
        tmp_path / "docs" / "index.html"
    ).read_text()


def test_artifact_only_exporter_end_to_end(tmp_path):
    data_dir = tmp_path / "data"
    out = tmp_path / "outputs"
    static = tmp_path / "webapp" / "static"
    for directory in (data_dir, out / "runs", static):
        directory.mkdir(parents=True, exist_ok=True)

    train_dates = pd.date_range("2025-01-01", periods=5, freq="D")
    test_dates = pd.date_range("2025-01-06", periods=2, freq="D")
    train = pd.DataFrame({
        "ProductId": [1] * 5,
        "DateKey": train_dates,
        "QuantityApp": [2.0] * 5,
        "QuantityWeb": [3.0] * 5,
        "ProductAvailable": [True] * 5,
    })
    test = pd.DataFrame({"ProductId": [1, 1], "DateKey": test_dates})
    train.to_parquet(data_dir / "train_data.parquet", index=False)
    test.to_parquet(data_dir / "test_data.parquet", index=False)

    submission = test.assign(Quantity=[5, 6])
    submission.to_csv(out / "submission.csv", index=False)
    summary = _summary()
    summary.to_csv(out / "dev_summary.csv", index=False)
    summary.to_csv(out / "benchmark_summary.csv", index=False)
    pd.DataFrame([
        {
            "fold": 0,
            "model": model,
            "regime": "conditional",
            "comparison_population": "common",
            "MAE": 1.0,
            "RMSE": 1.0,
            "WAPE": 0.1 + index / 10,
            "Bias": 0.0,
            "BiasRatio": 0.0,
        }
        for index, model in enumerate(("NeuralNet", "Chronos2"))
    ]).to_csv(out / "cv_results.csv", index=False)

    final_rows = []
    for model, predictions in {
        "NeuralNet": [5.0, 6.0],
        "Chronos2": [6.0, 7.0],
    }.items():
        for date, prediction in zip(test_dates, predictions):
            final_rows.append({
                "strategy": "direct",
                "model": model,
                "ProductId": 1,
                "DateKey": date,
                "prediction_raw": prediction,
                "prediction_q10": prediction - 1 if model == "Chronos2" else np.nan,
                "prediction_q50": prediction if model == "Chronos2" else np.nan,
                "prediction_q90": prediction + 1 if model == "Chronos2" else np.nan,
            })
    pd.DataFrame(final_rows).to_parquet(
        out / "final_forecasts.parquet", index=False
    )

    provenance = {
        "schema_version": "chronos-run-v1",
        "run_id": "fixture-run",
        "generated_at": "2026-07-17T00:00:00+00:00",
        "run_kind": "reproduction",
        "source": {"revision": "abc", "tree": "def", "dirty": False},
        "chronos": {
            "model_id": "amazon/chronos-2",
            "model_revision": "29ec3766d36d6f73f0696f85560a422f50e8498c",
        },
        "runtime": {"os": "test", "machine": "test", "torch": {"device": "cpu"}},
        "run_manifest": "outputs/runs/fixture-run.json",
        "output_hashes": {
            "status": "recorded_in_run_manifest",
            "manifest": "outputs/runs/fixture-run.json",
        },
    }
    (out / "runs" / "fixture-run.json").write_text(json.dumps(provenance))
    existing = {
        "config": {
            "selection_metric": "WAPE",
            "selection_protocol": "test-aligned",
            "chronos2_profile": "published",
            "chronos2_model_id": "amazon/chronos-2",
            "chronos2_model_revision": "29ec3766d36d6f73f0696f85560a422f50e8498c",
            "chronos2_device": "cpu",
            "chronos2_dtype": "float32",
            "chronos2_batch_size": 100,
        },
        "selection": {"canonical_model": "NeuralNet"},
        "provenance": provenance,
        "evaluation_origins": [],
    }
    (out / "results.json").write_text(json.dumps(existing))

    payload = export_from_artifacts(
        tmp_path,
        destination=tmp_path / "rebuilt.json",
        publish_static=False,
    )
    assert payload["schema_version"] == "vonavy-chronos-v2"
    assert payload["project"]["status"] == "complete"
    assert set(payload["forecasts"]) == {"NeuralNet", "Chronos2"}
    assert payload["generated_at"] == provenance["generated_at"]
