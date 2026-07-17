"""Rebuild the dashboard contract exclusively from persisted run artifacts."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

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


def _read_csv(path: Path, *, required: bool = False, **kwargs) -> pd.DataFrame:
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
        if required:
            raise ValueError(f"Required CSV artifact is empty: {path}") from None
        return pd.DataFrame()


def _runtime_options(results: dict) -> RuntimeOptions:
    config = results.get("config")
    if not isinstance(config, dict):
        raise ValueError("Existing results.json has no configuration object")
    canonical = results.get("selection", {}).get("canonical_model")
    try:
        submission_model = SubmissionModel(canonical)
    except ValueError as exc:
        raise ValueError(f"Invalid canonical model in results.json: {canonical!r}") from exc
    return RuntimeOptions(
        submission_model=submission_model,
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


def _load_manifest(root: Path, results: dict) -> tuple[dict, dict]:
    provenance = results.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Existing results.json has no immutable provenance")
    relative = provenance.get("run_manifest")
    if not relative:
        raise ValueError("Existing results.json does not identify its run manifest")
    manifest = _read_json(root / relative)
    if manifest.get("run_id") != provenance.get("run_id"):
        raise ValueError("Run manifest and results.json have different run IDs")
    published_provenance = {
        key: value for key, value in manifest.items() if key != "output_sha256"
    }
    published_provenance["output_hashes"] = provenance.get("output_hashes")
    return manifest, published_provenance


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
    _, provenance = _load_manifest(root, existing)
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
    cv_results_all = _read_csv(out / "cv_results_all.csv")
    dev_summary = _read_csv(out / "dev_summary.csv", required=True)
    benchmark_summary = _read_csv(out / "benchmark_summary.csv", required=True)
    final_df = pd.read_parquet(out / "final_forecasts.parquet")
    final_df["DateKey"] = pd.to_datetime(final_df["DateKey"])

    canonical_strategy = "direct"
    canonical_model = options.submission_model.value
    if cv_results_all.empty:
        cv_results_all = cv_results.assign(strategy=canonical_strategy)

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
        "strategy_by_horizon": _read_csv(out / "strategy_by_horizon.csv"),
        "validation_strata_summary": _read_csv(out / "validation_strata_summary.csv"),
        "test_aligned_scores": _read_csv(out / "test_aligned_scores.csv"),
        "prediction_diagnostics": _read_csv(out / "prediction_diagnostics.csv"),
        "prediction_diagnostics_by_origin": _read_csv(
            out / "prediction_diagnostics_by_origin.csv", parse_dates=["origin"]
        ),
        "channel_share_summary": _read_csv(out / "channel_share_summary.csv"),
        "per_product_summary": _read_csv(out / "per_product_summary.csv"),
        "top_decile_summary": _read_csv(out / "top_decile_summary.csv"),
        "top_error_rows": _read_csv(
            out / "top_error_rows.csv", parse_dates=["origin", "DateKey"]
        ),
        "sanity_baseline": _read_csv(out / "sanity_baseline.csv"),
        "probabilistic_summary": _read_csv(out / "probabilistic_summary.csv"),
        "weight_sensitivity": _read_csv(out / "weight_sensitivity.csv"),
        "audit_summary": _read_csv(out / "final_audit_summary.csv"),
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
