import json
import pickle
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dashboard_artifacts import (
    summarize_probabilistic_oof,
    summarize_sanity_baseline,
)
from export_results import REQUIRED_MODEL_ARTIFACTS, export_from_artifacts
from finalize_publication import checkpoint_metadata
from pipeline import (
    OOF_MODEL_COLUMNS,
    RuntimeOptions,
    _fold_checkpoint_signature,
    _load_fold_checkpoint,
    actual_execution_matches,
    build_actual_execution,
    build_checkpoint_run_identity,
    load_authenticated_checkpoint_index,
    model_output_artifact_paths,
    reserve_final_audit,
    validate_final_audit_policy,
)
from framework import Config
from provenance import (
    build_run_provenance,
    output_hashes,
    sha256_file,
    sha256_json,
)
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
    reservation = reserve_final_audit(
        options,
        run_id="publication-1",
        source_revision="abc",
        generated_at="2026-07-17T00:00:00+00:00",
        marker_path=str(marker),
    )
    reserved = json.loads(marker.read_text())
    assert reserved["status"] == "consumed"
    assert reserved["fresh_evidence_claim_consumed"] is True
    with pytest.raises(RuntimeError, match="already reserved or consumed"):
        reserve_final_audit(
            options,
            run_id="publication-2",
            source_revision="def",
            generated_at="2026-07-17T00:01:00+00:00",
            marker_path=str(marker),
        )
    before_reproduction = marker.read_bytes()
    reproduced = reserve_final_audit(
        RuntimeOptions(run_kind="reproduction", include_final_audit=True),
        run_id="reproduction-1",
        source_revision="def",
        generated_at="2026-07-17T00:02:00+00:00",
        marker_path=str(marker),
    )
    assert reproduced["fresh"] is False
    assert marker.read_bytes() == before_reproduction
    assert reservation["mode"] == "publication"
    assert json.loads(marker.read_text())["status"] == "consumed"


def test_final_audit_crash_reservation_remains_consumed(tmp_path):
    marker = tmp_path / "crashed.json"
    options = RuntimeOptions(run_kind="publication", include_final_audit=True)
    reserve_final_audit(
        options,
        run_id="crashed-publication",
        source_revision="abc",
        generated_at="2026-07-17T00:00:00+00:00",
        marker_path=str(marker),
    )
    with pytest.raises(RuntimeError, match="already reserved or consumed"):
        reserve_final_audit(
            options,
            run_id="second-publication",
            source_revision="abc",
            generated_at="2026-07-17T00:01:00+00:00",
            marker_path=str(marker),
        )
    assert json.loads(marker.read_text())["status"] == "consumed"


def test_hash_helpers_are_stable(tmp_path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("chronos\n")
    assert len(sha256_file(artifact)) == 64
    assert sha256_json({"b": 2, "a": 1}) == sha256_json({"a": 1, "b": 2})


def test_checkpoint_signature_binds_run_identity():
    cfg = Config()
    origin = pd.Timestamp("2025-01-01")
    first = _fold_checkpoint_signature(
        cfg,
        "direct",
        "development",
        origin,
        {"source_tree": "tree-a", "input": "hash-a", "device": "cpu"},
    )
    second = _fold_checkpoint_signature(
        cfg,
        "direct",
        "development",
        origin,
        {"source_tree": "tree-b", "input": "hash-a", "device": "cpu"},
    )
    assert first != second
    assert first["run_identity"]["source_tree"] == "tree-a"


def test_checkpoint_run_identity_covers_source_inputs_dependencies_and_backends():
    identity = build_checkpoint_run_identity(
        {
            "source": {"tree": "tree"},
            "inputs_sha256": {"train": "train-hash", "test": "test-hash"},
            "lock": {"sha256": "lock-hash"},
            "chronos": {
                "package": {"value": "2.3.1"},
                "model_revision": "model-sha",
            },
            "runtime": {"torch": {"device": "mps", "package": {"value": "2.13.0"}}},
        },
        {
            "resolved_device": "mps",
            "dtype": "float32",
            "batch_size": 100,
            "enabled": True,
        },
        {
            "training_backend": "device_resident",
            "batch_size": 512,
            "device": "cuda",
        },
    )
    assert identity["source_tree"] == "tree"
    assert identity["inputs_sha256"]["train"] == "train-hash"
    assert identity["dependency_lock"]["sha256"] == "lock-hash"
    assert identity["chronos"]["model_revision"] == "model-sha"
    assert identity["resolved_device"] == "mps"
    assert identity["nn_training_backend"] == "device_resident"
    assert identity["expected_actual_execution"]["nn"]["device"] == "cuda"
    assert identity["expected_actual_execution"]["chronos2"]["device"] == "mps"


def test_checkpoint_requires_prior_hash_and_rejects_tampered_bytes(tmp_path):
    cfg = Config(output_dir=str(tmp_path))
    origin = pd.Timestamp("2025-01-01")
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_path = (
        checkpoint_dir / "direct" / "development" / f"{origin.date().isoformat()}.pkl"
    )
    checkpoint_path.parent.mkdir(parents=True)
    payload = {
        "signature": _fold_checkpoint_signature(
            cfg, "direct", "development", origin
        ),
        "oof": pd.DataFrame({"actual": [1.0]}),
        "actual_execution": None,
    }
    checkpoint_path.write_bytes(pickle.dumps(payload))
    assert _load_fold_checkpoint(
        str(checkpoint_dir),
        "direct",
        "development",
        origin,
        cfg,
    ) is None
    trusted_hash = sha256_file(checkpoint_path)
    checkpoint_path.write_bytes(checkpoint_path.read_bytes() + b"tampered")
    assert _load_fold_checkpoint(
        str(checkpoint_dir),
        "direct",
        "development",
        origin,
        cfg,
        expected_checkpoint_sha256=trusted_hash,
    ) is None


def test_checkpoint_trust_index_is_authenticated_by_prior_publication(tmp_path):
    runs = tmp_path / "outputs" / "runs"
    publications = tmp_path / "outputs" / "publications"
    runs.mkdir(parents=True)
    publications.mkdir()
    model_path = runs / "prior-run.json"
    model_path.write_text(json.dumps({
        "run_id": "prior-run",
        "checkpoints": {
            "status": "authenticated",
            "files_sha256": {"outputs/checkpoints/fold.pkl": "checkpoint-hash"},
        },
    }))
    publication_path = publications / "prior-publication.json"
    publication_path.write_text(json.dumps({
        "model_run_id": "prior-run",
        "artifact_sha256": {
            "outputs/runs/prior-run.json": sha256_file(model_path),
        },
    }))
    trust = load_authenticated_checkpoint_index(
        tmp_path, "outputs/publications/prior-publication.json"
    )
    assert trust["status"] == "authenticated"
    assert trust["checkpoint_sha256"]["outputs/checkpoints/fold.pkl"] == (
        "checkpoint-hash"
    )

    model_path.write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_authenticated_checkpoint_index(
            tmp_path, "outputs/publications/prior-publication.json"
        )

def test_backend_fallback_identity_is_not_reusable_as_device_resident():
    actual = build_actual_execution(
        [{"device": "mps", "backend": "dataloader_fallback", "batch_size": 512}],
        Config(enable_chronos2=False),
    )
    expected_identity = {
        "expected_actual_execution": {
            "nn": {
                "device": "mps",
                "backend": "device_resident",
                "batch_size": 512,
            },
            "chronos2": None,
        }
    }
    assert actual["nn"]["backend"] == "dataloader_fallback"
    assert actual_execution_matches(actual, expected_identity) is False


def test_dirty_reproduction_cannot_build_canonical_provenance(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path,
        check=True,
    )
    data = tmp_path / "data"
    data.mkdir()
    train = data / "train.parquet"
    test = data / "test.parquet"
    lock = tmp_path / "requirements.lock"
    source = tmp_path / "source.py"
    for path, content in (
        (train, b"train"),
        (test, b"test"),
        (lock, b"lock"),
        (source, b"clean"),
    ):
        path.write_bytes(content)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=tmp_path, check=True)
    source.write_bytes(b"dirty")
    with pytest.raises(RuntimeError, match="clean source tree"):
        build_run_provenance(
            repository_root=tmp_path,
            command=["python", "pipeline.py"],
            run_kind="reproduction",
            config={"test": True},
            input_paths=[train, test],
            lock_path=lock,
            model_id="amazon/chronos-2",
            model_revision="revision",
            resolved_device="cpu",
        )
    development = build_run_provenance(
        repository_root=tmp_path,
        command=["python", "pipeline.py"],
        run_kind="development",
        config={"test": True},
        input_paths=[train, test],
        lock_path=lock,
        model_id="amazon/chronos-2",
        model_revision="revision",
        resolved_device="cpu",
    )
    assert development["source"]["dirty"] is True


def test_noncanonical_experiment_outputs_are_gitignored():
    root = Path(__file__).resolve().parents[1]
    assert "outputs/experiments/" in (root / ".gitignore").read_text()


def test_experiment_tree_cannot_change_canonical_artifact_bytes(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "results.json").write_text('{"canonical": true}\n')
    (outputs / "dev_summary.csv").write_text("model,WAPE\nNeuralNet,0.1\n")

    def snapshot():
        payload = (outputs / "results.json").read_bytes()
        hashes = output_hashes(
            tmp_path,
            model_output_artifact_paths(tmp_path, "outputs"),
        )
        manifest = json.dumps(
            {"output_sha256": hashes}, sort_keys=True, separators=(",", ":")
        ).encode()
        checksums = "".join(
            f"{digest}  {path}\n" for path, digest in sorted(hashes.items())
        ).encode()
        return payload, manifest, checksums, hashes

    before = snapshot()
    experiment = outputs / "experiments" / "dirty-run"
    experiment.mkdir(parents=True)
    (experiment / "results.json").write_text('{"tampered": true}\n')
    (experiment / "SHA256SUMS").write_text("untrusted\n")
    after = snapshot()

    assert before[:3] == after[:3]
    assert before[3] == after[3]
    assert all("outputs/experiments/" not in path for path in after[3])


def test_publication_finalizer_preserves_authenticated_checkpoint_metadata(tmp_path):
    authenticated = {
        "status": "authenticated",
        "reused_folds": 2,
        "trained_folds": 3,
        "files_sha256": {"outputs/checkpoints/fold.pkl": "hash"},
    }
    verification, checkpoint_provenance = checkpoint_metadata(
        tmp_path,
        {"run_id": "future-run", "checkpoints": authenticated},
    )
    assert verification["status"] == "complete"
    assert verification["checkpoint_identity_at_run"] is True
    assert checkpoint_provenance == authenticated


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
    pd.read_csv(out / "cv_results.csv").assign(strategy="direct").to_csv(
        out / "cv_results_all.csv", index=False
    )

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
    pd.DataFrame({"origin": ["2025-01-05"], "actual": [5.0]}).to_parquet(
        out / "oof_predictions.parquet", index=False
    )
    summary.assign(horizon=1, origin_type="development").to_csv(
        out / "strategy_by_horizon.csv", index=False
    )
    summary.assign(
        validation_stratum="regular", origin_type="development"
    ).to_csv(out / "validation_strata_summary.csv", index=False)
    pd.DataFrame([
        {
            "strategy": "direct",
            "model": model,
            "metric": "WAPE",
            "test_aligned_score": 0.1 + index / 10,
        }
        for index, model in enumerate(("NeuralNet", "Chronos2"))
    ]).to_csv(out / "test_aligned_scores.csv", index=False)
    pd.DataFrame([
        {"origin_type": "development", "strategy": "direct", "model": model}
        for model in ("NeuralNet", "Chronos2")
    ]).to_csv(out / "prediction_diagnostics.csv", index=False)
    pd.DataFrame([
        {
            "origin_type": "development",
            "strategy": "direct",
            "origin": "2025-01-05",
            "model": model,
        }
        for model in ("NeuralNet", "Chronos2")
    ]).to_csv(out / "prediction_diagnostics_by_origin.csv", index=False)
    (out / "channel_share_summary.csv").write_text("\n")
    summary.assign(ProductId=1, origin_type="development").to_csv(
        out / "per_product_summary.csv", index=False
    )
    summary.assign(origin_type="development", quantile=0.9).to_csv(
        out / "top_decile_summary.csv", index=False
    )
    pd.DataFrame([
        {
            "origin": "2025-01-05",
            "DateKey": "2025-01-06",
            "model": model,
            "absolute_error": 1.0,
        }
        for model in ("NeuralNet", "Chronos2")
    ]).to_csv(out / "top_error_rows.csv", index=False)
    pd.DataFrame([
        {
            "origin_type": "development",
            "strategy": "direct",
            "estimator": estimator,
            "WAPE": 0.1,
        }
        for estimator in ("NeuralNet", "Chronos2", "SeasonalWeekdayNaive")
    ]).to_csv(out / "sanity_baseline.csv", index=False)
    pd.DataFrame([{
        "origin_type": "development",
        "strategy": "direct",
        "model": "Chronos2",
        "interval_coverage": 0.8,
    }]).to_csv(out / "probabilistic_summary.csv", index=False)
    pd.DataFrame([
        {
            "scheme": "frozen_test_aligned",
            "strategy": "direct",
            "model": model,
            "test_aligned_score": 0.1 + index / 10,
            "winner": "NeuralNet",
        }
        for index, model in enumerate(("NeuralNet", "Chronos2"))
    ]).to_csv(out / "weight_sensitivity.csv", index=False)
    summary.to_csv(out / "final_audit_summary.csv", index=False)

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
    }
    model_manifest = {
        **provenance,
        "inputs_sha256": {
            "data/train_data.parquet": sha256_file(data_dir / "train_data.parquet"),
            "data/test_data.parquet": sha256_file(data_dir / "test_data.parquet"),
        },
        "model_artifact_sha256": {
            relative: sha256_file(tmp_path / relative)
            for relative in REQUIRED_MODEL_ARTIFACTS
        },
    }
    model_manifest_path = out / "runs" / "fixture-run.json"
    model_manifest_path.write_text(json.dumps(model_manifest))
    publication_provenance = {
        "schema_version": "chronos-publication-v1",
        "status": "authenticated",
        "publication_id": "fixture-publication",
        "manifest": "outputs/publications/fixture-publication.json",
    }
    (out / "publications").mkdir()
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
        "publication_provenance": publication_provenance,
        "evaluation_origins": [],
    }
    results_path = out / "results.json"
    results_path.write_text(json.dumps(existing))
    publication_manifest = {
        **publication_provenance,
        "artifact_sha256": {
            "outputs/results.json": sha256_file(results_path),
            "outputs/runs/fixture-run.json": sha256_file(model_manifest_path),
        },
    }
    (out / "publications" / "fixture-publication.json").write_text(
        json.dumps(publication_manifest)
    )

    payload = export_from_artifacts(
        tmp_path,
        destination=tmp_path / "rebuilt.json",
        publish_static=False,
    )
    assert payload["schema_version"] == "vonavy-chronos-v2"
    assert payload["project"]["status"] == "complete"
    assert set(payload["forecasts"]) == {"NeuralNet", "Chronos2"}
    assert payload["generated_at"] == provenance["generated_at"]

    dev_path = out / "dev_summary.csv"
    original_dev = dev_path.read_bytes()
    dev_path.write_bytes(original_dev + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        export_from_artifacts(
            tmp_path,
            destination=tmp_path / "tampered.json",
            publish_static=False,
        )
    dev_path.write_bytes(original_dev)

    (out / "probabilistic_summary.csv").unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        export_from_artifacts(
            tmp_path,
            destination=tmp_path / "missing.json",
            publish_static=False,
        )
