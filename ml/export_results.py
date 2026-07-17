"""Rebuild the dashboard contract exclusively from persisted run artifacts."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from provenance import model_provenance_summary, sha256_file
from static_site import publish_static_dashboard
from pipeline import (
    Config,
    RuntimeOptions,
    SubmissionModel,
    _chronos_quantiles_to_json,
    configure_c1_runtime,
    configure_c2_runtime,
    configure_c34_runtime,
    configure_chronos2_runtime,
    configure_nn_runtime,
    export_results_json,
    load_raw,
)

REQUIRED_MODEL_ARTIFACTS = (
    "outputs/submission.csv",
    "outputs/cv_results.csv",
    "outputs/cv_results_all.csv",
    "outputs/dev_summary.csv",
    "outputs/benchmark_summary.csv",
    "outputs/final_forecasts.parquet",
    "outputs/oof_predictions.parquet",
    "outputs/strategy_by_horizon.csv",
    "outputs/validation_strata_summary.csv",
    "outputs/test_aligned_scores.csv",
    "outputs/prediction_diagnostics.csv",
    "outputs/prediction_diagnostics_by_origin.csv",
    "outputs/channel_share_summary.csv",
    "outputs/per_product_summary.csv",
    "outputs/top_decile_summary.csv",
    "outputs/top_error_rows.csv",
    "outputs/sanity_baseline.csv",
    "outputs/probabilistic_summary.csv",
    "outputs/weight_sensitivity.csv",
    "outputs/final_audit_summary.csv",
)


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Required artifact is missing: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _read_csv(
    path: Path,
    *,
    required: bool = False,
    allow_empty: bool = False,
    **kwargs,
) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required artifact is missing: {path}")
        return pd.DataFrame()
    if path.stat().st_size == 0:
        if required:
            raise ValueError(f"Required CSV artifact is empty: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        if required and not allow_empty:
            raise ValueError(f"Required CSV artifact is empty: {path}") from None
        return pd.DataFrame()


def _runtime_options(results: dict) -> RuntimeOptions:
    config = results.get("config")
    if not isinstance(config, dict):
        raise ValueError("Existing results.json has no configuration object")
    return RuntimeOptions(
        submission_model=SubmissionModel.AUTO,
        selection_metric=str(config.get("selection_metric", "WAPE")),
        selection_protocol=str(config.get("selection_protocol", "test-aligned")),
        chronos2_profile=str(config.get("chronos2_profile", "published")),
        chronos2_model_id=str(config.get("chronos2_model_id", "amazon/chronos-2")),
        chronos2_model_revision=str(config["chronos2_model_revision"]),
        chronos2_device=str(config.get("chronos2_device", "auto")),
        chronos2_dtype=str(config.get("chronos2_dtype", "float32")),
        chronos2_batch_size=int(config.get("chronos2_batch_size", 100)),
        run_kind=str(results.get("provenance", {}).get("run_kind", "reproduction")),
    )


def _manifest_path(root: Path, relative: str, directory: str) -> Path:
    candidate = (root / relative).resolve()
    allowed = (root / "outputs" / directory).resolve()
    if not candidate.is_relative_to(allowed):
        raise ValueError(f"Manifest path escapes outputs/{directory}: {relative}")
    return candidate


def _verify_hash(path: Path, expected: str, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required authenticated artifact is missing: {label}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"Authenticated artifact hash mismatch for {label}: "
            f"expected {expected}, got {actual}"
        )


def _load_manifests(root: Path, results: dict) -> tuple[dict, dict, dict, dict]:
    provenance = results.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Existing results.json has no immutable provenance")
    relative = provenance.get("run_manifest")
    if not relative:
        raise ValueError("Existing results.json does not identify its run manifest")
    publication_provenance = results.get("publication_provenance")
    if not isinstance(publication_provenance, dict):
        raise ValueError("Existing results.json has no publication provenance")
    publication_relative = publication_provenance.get("manifest")
    if not publication_relative:
        raise ValueError("Existing results.json does not identify its publication manifest")
    publication_path = _manifest_path(root, publication_relative, "publications")
    publication_manifest = _read_json(publication_path)
    if publication_manifest.get("publication_id") != publication_provenance.get(
        "publication_id"
    ):
        raise ValueError("Publication manifest and results.json have different IDs")
    publication_hashes = publication_manifest.get("artifact_sha256")
    if not isinstance(publication_hashes, dict):
        raise ValueError("Publication manifest has no artifact hash map")
    results_label = "outputs/results.json"
    if results_label not in publication_hashes:
        raise ValueError("Publication manifest does not authenticate outputs/results.json")
    _verify_hash(root / results_label, publication_hashes[results_label], results_label)

    manifest_path = _manifest_path(root, relative, "runs")
    manifest_label = str(manifest_path.relative_to(root))
    if manifest_label not in publication_hashes:
        raise ValueError("Publication manifest does not authenticate the model run manifest")
    _verify_hash(manifest_path, publication_hashes[manifest_label], manifest_label)
    manifest = _read_json(manifest_path)
    if manifest.get("run_id") != provenance.get("run_id"):
        raise ValueError("Run manifest and results.json have different run IDs")
    model_hashes = manifest.get("model_artifact_sha256") or manifest.get("output_sha256")
    if not isinstance(model_hashes, dict):
        raise ValueError("Model run manifest has no artifact hash map")
    for relative_path in REQUIRED_MODEL_ARTIFACTS:
        expected = model_hashes.get(relative_path)
        if not expected:
            raise ValueError(
                f"Model run manifest does not authenticate {relative_path}"
            )
        _verify_hash(root / relative_path, expected, relative_path)
    input_hashes = manifest.get("inputs_sha256")
    if not isinstance(input_hashes, dict):
        raise ValueError("Model run manifest has no input hash map")
    for relative_path in ("data/train_data.parquet", "data/test_data.parquet"):
        expected = input_hashes.get(relative_path)
        if not expected:
            raise ValueError(f"Model run manifest does not authenticate {relative_path}")
        _verify_hash(root / relative_path, expected, relative_path)

    published_provenance = model_provenance_summary(
        manifest,
        manifest_path=relative,
        manifest_sha256=sha256_file(manifest_path),
        authenticated_artifact_count=len(REQUIRED_MODEL_ARTIFACTS) + 2,
    )
    return (
        manifest,
        published_provenance,
        publication_manifest,
        publication_provenance,
    )


def export_from_artifacts(
    repository_root: str | Path,
    *,
    destination: str | Path | None = None,
    publish_static: bool = True,
) -> dict:
    root = Path(repository_root).resolve()
    out = root / "outputs"
    existing_path = out / "results.json"
    existing = _read_json(existing_path)
    _, provenance, _, publication_provenance = _load_manifests(root, existing)
    options = _runtime_options(existing)

    cfg = Config(
        output_dir=str(out),
        train_path=str(root / "data" / "train_data.parquet"),
        test_path=str(root / "data" / "test_data.parquet"),
    )
    train_raw, test_raw = load_raw(cfg)
    cfg.num_products = int(
        max(train_raw["ProductId"].max(), test_raw["ProductId"].max())
    )
    configure_c1_runtime(cfg, options)
    configure_c2_runtime(cfg, options)
    configure_c34_runtime(cfg, options)
    configure_chronos2_runtime(cfg, options)
    configure_nn_runtime(cfg, options)

    submission = _read_csv(
        out / "submission.csv", required=True, parse_dates=["DateKey"]
    )
    cv_results = _read_csv(out / "cv_results.csv", required=True)
    cv_results_all = _read_csv(out / "cv_results_all.csv", required=True)
    dev_summary = _read_csv(out / "dev_summary.csv", required=True)
    benchmark_summary = _read_csv(out / "benchmark_summary.csv", required=True)
    final_df = pd.read_parquet(out / "final_forecasts.parquet")
    final_df["DateKey"] = pd.to_datetime(final_df["DateKey"])

    canonical_strategy = "direct"
    canonical_model = str(existing.get("selection", {}).get("canonical_model"))
    try:
        parsed_canonical = SubmissionModel(canonical_model)
    except ValueError as exc:
        raise ValueError(
            f"Invalid canonical model in results.json: {canonical_model!r}"
        ) from exc
    if parsed_canonical is SubmissionModel.AUTO:
        raise ValueError("Published canonical model cannot be auto")
    forecasts_by_strategy: dict[str, dict] = {}
    raw_forecasts: dict[str, dict] = {}
    for strategy, strategy_df in final_df.groupby("strategy"):
        strategy_name = str(strategy)
        raw_forecasts[strategy_name] = {}
        strategy_json = {}
        for model, model_df in strategy_df.groupby("model"):
            aligned = test_raw[["ProductId", "DateKey"]].merge(
                model_df[["ProductId", "DateKey", "prediction_raw"]],
                on=["ProductId", "DateKey"],
                how="left",
                validate="one_to_one",
            )
            if aligned["prediction_raw"].isna().any():
                raise ValueError(f"Persisted {model} final forecast is incomplete")
            raw_forecasts[strategy_name][str(model)] = aligned[
                "prediction_raw"
            ].to_numpy(dtype=float)
            per_product = {}
            for product_id, product in aligned.groupby("ProductId"):
                product = product.sort_values("DateKey")
                per_product[str(int(product_id))] = {
                    "dates": product["DateKey"].dt.strftime("%Y-%m-%d").tolist(),
                    "quantity": product["prediction_raw"].astype(float).tolist(),
                }
            strategy_json[str(model)] = per_product
        forecasts_by_strategy[strategy_name] = strategy_json

    if canonical_model not in raw_forecasts.get(canonical_strategy, {}):
        raise RuntimeError(
            f"No persisted {canonical_model} forecast for {canonical_strategy}"
        )

    frames = {
        "strategy_by_horizon": _read_csv(
            out / "strategy_by_horizon.csv", required=True
        ),
        "validation_strata_summary": _read_csv(
            out / "validation_strata_summary.csv", required=True
        ),
        "test_aligned_scores": _read_csv(
            out / "test_aligned_scores.csv", required=True
        ),
        "prediction_diagnostics": _read_csv(
            out / "prediction_diagnostics.csv", required=True
        ),
        "prediction_diagnostics_by_origin": _read_csv(
            out / "prediction_diagnostics_by_origin.csv",
            required=True,
            parse_dates=["origin"],
        ),
        "channel_share_summary": _read_csv(
            out / "channel_share_summary.csv",
            required=True,
            allow_empty=True,
        ),
        "per_product_summary": _read_csv(
            out / "per_product_summary.csv", required=True
        ),
        "top_decile_summary": _read_csv(
            out / "top_decile_summary.csv", required=True
        ),
        "top_error_rows": _read_csv(
            out / "top_error_rows.csv",
            required=True,
            parse_dates=["origin", "DateKey"],
        ),
        "sanity_baseline": _read_csv(
            out / "sanity_baseline.csv", required=True
        ),
        "probabilistic_summary": _read_csv(
            out / "probabilistic_summary.csv", required=True
        ),
        "weight_sensitivity": _read_csv(
            out / "weight_sensitivity.csv", required=True
        ),
        "audit_summary": _read_csv(
            out / "final_audit_summary.csv", required=True
        ),
    }
    destination_path = Path(destination) if destination else existing_path
    payload = export_results_json(
        train_raw,
        test_raw,
        submission,
        raw_forecasts[canonical_strategy],
        cv_results,
        cfg,
        path=str(destination_path),
        dev_summary=dev_summary,
        benchmark_summary=benchmark_summary,
        runtime_options=options,
        forecasts_by_strategy=forecasts_by_strategy,
        canonical_strategy=canonical_strategy,
        canonical_model=canonical_model,
        cv_results_all=cv_results_all,
        final_quantile_forecasts=_chronos_quantiles_to_json(final_df),
        origin_registry=existing.get("evaluation_origins", []),
        provenance=provenance,
        publication_provenance=publication_provenance,
        **frames,
    )
    if publish_static:
        publish_static_dashboard(root, destination_path)
    return payload


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--repository-root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args(argv)
    root = Path(args.repository_root).resolve()
    if args.check:
        existing = _read_json(root / "outputs" / "results.json")
        with tempfile.TemporaryDirectory() as temporary:
            candidate_path = Path(temporary) / "results.json"
            candidate = export_from_artifacts(
                root, destination=candidate_path, publish_static=False
            )
        if candidate != existing:
            raise RuntimeError(
                "Artifact-only export differs from outputs/results.json; rerun "
                "without --check and regenerate the static site"
            )
        print("Artifact-only export matches the published results.")
        return
    export_from_artifacts(root)
    print("Artifact-only export complete; no model training was performed.")


if __name__ == "__main__":
    main()
