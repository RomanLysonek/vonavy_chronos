"""Two-contender retail forecasting challenge.

The active entry point evaluates exactly two direct seven-day forecasters:
the frozen neural incumbent (``Best NN``) and Amazon Chronos-2. Both models
share the same walk-forward origins, information cutoffs, target keys and
common scoring population. Winner selection uses development OOF only; the
recent benchmark is confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from enum import Enum
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from framework import (
    BASELINE_VARIANTS,
    CFG,
    CHALLENGE_MODELS,
    MODEL_META,
    MODEL_SLUGS,
    Config,
    add_train_lags,
    build_direct_panel,
    compute_baseline,
    compute_metrics,
    load_raw,
    order_models,
    prediction_columns_for_strategy,
    prepare_features,
    product_reference_dates,
    select_trainable_panel_rows,
    normalize_c2_feature_groups,
)
from models.chronos2_model import forecast_chronos2, resolve_chronos2_device
from models.neural_net import (
    DEVICE, effective_learning_rate, make_numeric_preprocessor, make_tensors,
    neural_training_target, nn_performance_signature, predict_direct,
    resolve_training_backend, train_model,
)
from dashboard_artifacts import (
    summarize_per_product_oof,
    summarize_probabilistic_oof,
    summarize_sanity_baseline,
    summarize_top_deciles,
)
from static_site import publish_static_dashboard
from provenance import (
    build_run_provenance,
    output_hashes,
    sha256_file,
    write_json_atomic,
    write_json_exclusive,
)

np.random.seed(CFG.seed)


class ForecastStrategy(str, Enum):
    DIRECT = "direct"
    RECURSIVE = "recursive"
    BOTH = "both"


class PrimaryStrategy(str, Enum):
    AUTO = "auto"
    DIRECT = "direct"
    RECURSIVE = "recursive"


class SubmissionModel(str, Enum):
    NEURAL_NET = "NeuralNet"
    CHRONOS2 = "Chronos2"
    AUTO = "auto"


@dataclass(frozen=True)
class RuntimeOptions:
    forecast_strategy: ForecastStrategy = ForecastStrategy.DIRECT
    primary_strategy: PrimaryStrategy = PrimaryStrategy.AUTO
    submission_model: SubmissionModel = SubmissionModel.AUTO
    selection_metric: str = "WAPE"
    selection_protocol: str = "test-aligned"
    resume: bool = False
    reset_checkpoints: bool = False
    checkpoint_dir: str = "outputs/checkpoints"
    checkpoint_trust_publication: str | None = None
    nn_batch_size: str = "auto"
    nn_lr_scaling: str = "auto"
    nn_training_backend: str = "auto"
    nn_benchmark_file: str = "outputs/nn_batch_benchmark.json"
    c1_config: str | None = None
    training_window_days: str | None = None
    recency_half_life_days: str | None = None
    baseline_variant: str | None = None
    trend_features: str | None = None
    c2_config: str | None = None
    c2_feature_groups: str | None = None
    c34_config: str | None = None
    nn_loss: str | None = None
    nn_target_mode: str | None = None
    nn_combined_mse_weight: float | None = None
    channel_history_features: str | None = None
    channel_aux_weight: float | None = None
    channel_share_smoothing: float | None = None
    chronos2: str = "on"
    chronos2_profile: str = "published"
    chronos2_model_id: str = "amazon/chronos-2"
    chronos2_model_revision: str = "29ec3766d36d6f73f0696f85560a422f50e8498c"
    chronos2_device: str = "auto"
    chronos2_dtype: str = "float32"
    chronos2_batch_size: int = 100
    run_kind: str = "development"
    include_final_audit: bool = False
    plot: bool = False


def resolve_strategies(strategy: ForecastStrategy) -> tuple[ForecastStrategy, ...]:
    """The challenge is intentionally direct-vs-direct.

    The incumbent was frozen under the direct seven-horizon contract and
    Chronos-2 is direct-only.  Keeping a single strategy removes an irrelevant
    axis from both computation and presentation.
    """
    if strategy is not ForecastStrategy.DIRECT:
        raise ValueError("vonavy_chronos supports only --forecast-strategy direct")
    return (ForecastStrategy.DIRECT,)


def parse_args(argv=None) -> RuntimeOptions:
    """Parse only challenge-level controls.

    Incumbent architecture, features, objective and training hyperparameters are
    frozen in ``Config``. Exposing the old ablation knobs here would turn the
    benchmark back into a tuning exercise after the challenger was introduced.
    """
    parser = argparse.ArgumentParser(
        description="vonavy_chronos: frozen incumbent vs Amazon Chronos-2"
    )
    parser.add_argument("--forecast-strategy", choices=["direct"], default="direct")
    parser.add_argument("--primary-strategy", choices=["auto", "direct"], default="auto")
    parser.add_argument(
        "--submission-model",
        choices=[item.value for item in SubmissionModel],
        default="auto",
    )
    parser.add_argument(
        "--selection-metric", choices=["WAPE", "MAE", "RMSE"], default="WAPE"
    )
    parser.add_argument(
        "--selection-protocol",
        choices=["global", "test-aligned"],
        default="test-aligned",
        help=(
            "Choose the winner from development OOF. Test-aligned applies the "
            "frozen January-like stratum weights; global pools all scored rows."
        ),
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Reuse compatible direct-fold checkpoints",
    )
    parser.add_argument(
        "--reset-checkpoints", action="store_true",
        help="Delete existing challenge checkpoints before starting",
    )
    parser.add_argument(
        "--checkpoint-dir", default="outputs/checkpoints",
        help="Directory for atomic per-fold challenge checkpoints",
    )
    parser.add_argument(
        "--checkpoint-trust-publication",
        help=(
            "Prior authenticated outputs/publications/*.json manifest whose "
            "model-run checkpoint hashes may be reused"
        ),
    )
    parser.add_argument(
        "--chronos2", choices=["on", "off"], default="on",
        help="Run Chronos-2; off is retained only for incumbent smoke tests",
    )
    parser.add_argument(
        "--chronos2-profile",
        choices=[
            "published", "target-only", "no-cross-learning",
            "context-64", "context-128", "context-256",
        ],
        default="published",
        help="Bounded zero-shot inference profile; no profile fine-tunes Chronos",
    )
    parser.add_argument("--chronos2-model-id", default="amazon/chronos-2")
    parser.add_argument(
        "--chronos2-model-revision",
        default="29ec3766d36d6f73f0696f85560a422f50e8498c",
    )
    parser.add_argument(
        "--chronos2-device", choices=["auto", "cpu", "cuda", "mps"], default="auto"
    )
    parser.add_argument(
        "--chronos2-dtype",
        choices=["auto", "float32", "bfloat16", "float16"],
        default="float32",
    )
    parser.add_argument("--chronos2-batch-size", type=int, default=100)
    parser.add_argument(
        "--run-kind",
        choices=["development", "publication", "reproduction"],
        default="development",
    )
    parser.add_argument(
        "--include-final-audit",
        action="store_true",
        help="Run reserved final-audit origins; publication consumes them",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate the optional matplotlib forecast plot and fail if it cannot be written",
    )
    args = parser.parse_args(argv)

    if args.chronos2_batch_size <= 0:
        parser.error("--chronos2-batch-size must be positive")
    if args.include_final_audit and args.run_kind == "development":
        parser.error("--include-final-audit requires publication or reproduction")
    if args.run_kind == "publication" and args.chronos2_profile != "published":
        parser.error("Publication runs require --chronos2-profile published")
    if args.submission_model != "auto":
        parser.error(
            "Canonical pipeline outputs require --submission-model auto; "
            "manual contenders must use lower-level experimental APIs"
        )

    return RuntimeOptions(
        forecast_strategy=ForecastStrategy(args.forecast_strategy),
        primary_strategy=PrimaryStrategy(args.primary_strategy),
        submission_model=SubmissionModel(args.submission_model),
        selection_metric=args.selection_metric,
        selection_protocol=args.selection_protocol,
        resume=args.resume,
        reset_checkpoints=args.reset_checkpoints,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_trust_publication=args.checkpoint_trust_publication,
        nn_batch_size="512",
        chronos2=args.chronos2,
        chronos2_profile=args.chronos2_profile,
        chronos2_model_id=args.chronos2_model_id,
        chronos2_model_revision=args.chronos2_model_revision,
        chronos2_device=args.chronos2_device,
        chronos2_dtype=args.chronos2_dtype,
        chronos2_batch_size=args.chronos2_batch_size,
        run_kind=args.run_kind,
        include_final_audit=args.include_final_audit,
        plot=args.plot,
    )

def _parse_optional_days(value, *, none_token: str, cast):
    if value is None:
        return None
    if isinstance(value, str) and value.lower() == none_token:
        return None
    parsed = cast(value)
    if parsed <= 0:
        raise ValueError(f"Expected positive value or {none_token!r}, got {value!r}")
    return parsed


def configure_c1_runtime(cfg: Config, options: RuntimeOptions) -> dict:
    """Apply a C1 recommendation plus explicit CLI overrides."""
    source = "C0 defaults"
    recommendation = {}
    if options.c1_config is not None:
        with open(options.c1_config, encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid C1 recommendation in {options.c1_config}")
        candidate = payload.get("recommendation", payload)
        if isinstance(candidate, dict) and "config" in candidate:
            candidate = candidate["config"]
        elif "config" in payload:
            candidate = payload["config"]
        recommendation = candidate
        if not isinstance(recommendation, dict):
            raise ValueError(f"Invalid C1 recommendation in {options.c1_config}")
        source = options.c1_config

    def value(name, explicit, default):
        if explicit is not None:
            return explicit, "CLI override"
        if name in recommendation:
            return recommendation[name], source
        return default, "C0 default"

    window_raw, window_source = value(
        "training_window_days", options.training_window_days, cfg.training_window_days
    )
    half_life_raw, half_life_source = value(
        "recency_half_life_days", options.recency_half_life_days,
        cfg.recency_half_life_days,
    )
    baseline, baseline_source = value(
        "baseline_variant", options.baseline_variant, cfg.baseline_variant
    )
    trend_raw, trend_source = value(
        "enable_trend_features", options.trend_features, cfg.enable_trend_features
    )

    cfg.training_window_days = _parse_optional_days(
        window_raw, none_token="all", cast=int
    )
    cfg.recency_half_life_days = _parse_optional_days(
        half_life_raw, none_token="none", cast=float
    )
    if baseline not in BASELINE_VARIANTS:
        raise ValueError(
            f"Unknown baseline_variant={baseline!r}; expected {sorted(BASELINE_VARIANTS)}"
        )
    cfg.baseline_variant = str(baseline)
    if isinstance(trend_raw, str):
        if trend_raw not in {"on", "off"}:
            raise ValueError("trend_features must be 'on' or 'off'")
        cfg.enable_trend_features = trend_raw == "on"
    else:
        cfg.enable_trend_features = bool(trend_raw)

    return {
        "training_window_days": cfg.training_window_days,
        "recency_half_life_days": cfg.recency_half_life_days,
        "baseline_variant": cfg.baseline_variant,
        "enable_trend_features": cfg.enable_trend_features,
        "sources": {
            "training_window_days": window_source,
            "recency_half_life_days": half_life_source,
            "baseline_variant": baseline_source,
            "enable_trend_features": trend_source,
        },
        "config_file": options.c1_config,
    }


def configure_c2_runtime(cfg: Config, options: RuntimeOptions) -> dict:
    """Apply C2 semantic feature-group recommendations and CLI overrides."""
    recommendation = {}
    source = "C1 baseline"
    if options.c2_config is not None:
        with open(options.c2_config, encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid C2 recommendation in {options.c2_config}")
        candidate = payload.get("recommendation", payload)
        if isinstance(candidate, dict) and "config" in candidate:
            candidate = candidate["config"]
        elif "config" in payload:
            candidate = payload["config"]
        recommendation = candidate if isinstance(candidate, dict) else {}
        source = options.c2_config

    if options.c2_feature_groups is not None:
        raw_groups = options.c2_feature_groups
        group_source = "CLI override"
    elif "c2_feature_groups" in recommendation:
        raw_groups = recommendation["c2_feature_groups"]
        group_source = source
    else:
        raw_groups = cfg.c2_feature_groups
        group_source = "C1 baseline"

    cfg.c2_feature_groups = normalize_c2_feature_groups(raw_groups)
    return {
        "c2_feature_groups": list(cfg.c2_feature_groups),
        "source": group_source,
        "config_file": options.c2_config,
    }


def configure_c34_runtime(cfg: Config, options: RuntimeOptions) -> dict:
    """Return the frozen incumbent objective and channel configuration."""
    if options.c34_config is not None:
        raise ValueError("The incumbent is frozen; C3/C4 config overrides are disabled")
    if options.nn_loss is not None or options.nn_target_mode is not None:
        raise ValueError("The incumbent objective is frozen in vonavy_chronos")
    if options.nn_combined_mse_weight is not None:
        raise ValueError("The incumbent objective is frozen in vonavy_chronos")
    if (
        options.channel_history_features is not None
        or options.channel_aux_weight is not None
        or options.channel_share_smoothing is not None
    ):
        raise ValueError("The incumbent channel-head configuration is frozen")
    return {
        "nn_loss": cfg.nn_loss,
        "nn_target_mode": cfg.nn_target_mode,
        "nn_combined_mse_weight": cfg.nn_combined_mse_weight,
        "enable_channel_history_features": cfg.enable_channel_history_features,
        "channel_aux_weight": cfg.channel_aux_weight,
        "channel_share_smoothing": cfg.channel_share_smoothing,
        "source": "frozen incumbent",
    }


def chronos2_quantile_suffix(quantile: float) -> str:
    """Stable column suffix; 0.1 -> q10 and 0.125 -> q12p5."""
    percent = f"{float(quantile) * 100:g}".replace(".", "p")
    return f"q{percent}"


def configure_chronos2_runtime(cfg: Config, options: RuntimeOptions) -> dict:
    """Configure the optional zero-shot foundation-model challenger."""
    profile_path = Path(__file__).with_name("chronos2_profiles.json")
    profile_payload = json.loads(profile_path.read_text(encoding="utf-8"))
    profiles = profile_payload.get("profiles")
    if not isinstance(profiles, dict) or options.chronos2_profile not in profiles:
        raise ValueError(f"Unknown Chronos-2 profile {options.chronos2_profile!r}")
    profile = profiles[options.chronos2_profile]
    required = {
        "cross_learning", "covariates", "context_length",
        "quantile_levels", "publication_eligible",
    }
    if not isinstance(profile, dict) or not required.issubset(profile):
        raise ValueError(f"Invalid Chronos-2 profile {options.chronos2_profile!r}")
    if options.run_kind == "publication" and not profile["publication_eligible"]:
        raise ValueError("Publication requires a publication-eligible Chronos profile")
    cfg.enable_chronos2 = options.chronos2 == "on"
    cfg.chronos2_model_id = str(options.chronos2_model_id)
    cfg.chronos2_model_revision = str(options.chronos2_model_revision)
    cfg.chronos2_device = str(options.chronos2_device)
    cfg.chronos2_dtype = str(options.chronos2_dtype)
    cfg.chronos2_batch_size = int(options.chronos2_batch_size)
    cfg.chronos2_context_length = profile["context_length"]
    cfg.chronos2_cross_learning = bool(profile["cross_learning"])
    cfg.chronos2_covariates = bool(profile["covariates"])
    cfg.chronos2_quantile_levels = tuple(
        float(level) for level in profile["quantile_levels"]
    )
    if cfg.chronos2_quantile_levels != (0.1, 0.5, 0.9):
        raise ValueError("Bounded Chronos profiles must request q10/q50/q90")
    return {
        "enabled": cfg.enable_chronos2,
        "profile": options.chronos2_profile,
        "profile_schema_version": profile_payload.get("schema_version"),
        "profile_path": str(profile_path),
        "model_id": cfg.chronos2_model_id,
        "model_revision": cfg.chronos2_model_revision,
        "requested_device": cfg.chronos2_device,
        "resolved_device": resolve_chronos2_device(cfg.chronos2_device),
        "dtype": cfg.chronos2_dtype,
        "batch_size": cfg.chronos2_batch_size,
        "context_length": cfg.chronos2_context_length,
        "cross_learning": cfg.chronos2_cross_learning,
        "covariates": cfg.chronos2_covariates,
        "quantile_levels": list(cfg.chronos2_quantile_levels),
    }


def configure_nn_runtime(cfg: Config, options: RuntimeOptions) -> dict:
    """Resolve batch/LR/backend without guessing away model quality.

    Auto mode consumes the recommendation produced by
    ``ml/benchmark_nn_batch_size.py`` only when it was measured on the same
    accelerator type.  Without that artifact the historical 512/fixed policy
    is preserved.
    """
    recommendation = None
    if os.path.exists(options.nn_benchmark_file):
        try:
            with open(options.nn_benchmark_file, encoding="utf-8") as f:
                payload = json.load(f)
            candidate = payload.get("recommendation") or {}
            measured_device = payload.get("environment", {}).get("device")
            measured_signature = payload.get("model_signature")
            current_signature = nn_performance_signature(cfg)
            # JSON converts tuples to lists, so compare through a JSON-normalised
            # representation rather than Python container types.
            signature_matches = (
                json.dumps(measured_signature, sort_keys=True)
                == json.dumps(current_signature, sort_keys=True)
            )
            if (
                payload.get("schema_version") == "nn-batch-v1"
                and measured_device == DEVICE.type
                and signature_matches
                and candidate.get("batch_size")
            ):
                recommendation = candidate
        except (OSError, ValueError, TypeError) as exc:
            print(f"Ignoring unreadable NN benchmark {options.nn_benchmark_file}: {exc}")

    if options.nn_batch_size == "auto":
        if recommendation is not None:
            batch_size = int(recommendation["batch_size"])
            batch_source = options.nn_benchmark_file
        else:
            batch_size = int(cfg.reference_batch_size)
            batch_source = "historical safe fallback"
    else:
        batch_size = int(options.nn_batch_size)
        batch_source = "CLI override"

    if options.nn_lr_scaling == "auto":
        if recommendation is not None and options.nn_batch_size == "auto":
            lr_scaling = str(recommendation.get("lr_scaling", "sqrt"))
        elif batch_size == cfg.reference_batch_size:
            lr_scaling = "fixed"
        else:
            lr_scaling = "sqrt"
    else:
        lr_scaling = options.nn_lr_scaling

    cfg.batch_size = batch_size
    cfg.nn_lr_scaling = lr_scaling
    cfg.nn_training_backend = options.nn_training_backend
    return {
        "batch_size": batch_size,
        "batch_source": batch_source,
        "lr_scaling": lr_scaling,
        "effective_learning_rate": effective_learning_rate(cfg),
        "training_backend": resolve_training_backend(cfg),
        "device": DEVICE.type,
        "benchmark_file": options.nn_benchmark_file,
    }


CHECKPOINT_SCHEMA_VERSION = "publication-identity-v2"
CHRONOS2_CHECKPOINT_SCHEMA_VERSION = "chronos2-publication-identity-v2"
MODEL_OUTPUT_FILENAMES = (
    "submission.csv",
    "submission.parquet",
    "submission_best_nn.csv",
    "submission_best_nn.parquet",
    "submission_chronos2.csv",
    "submission_chronos2.parquet",
    "challenge_comparison.csv",
    "oof_predictions.parquet",
    "final_forecasts.parquet",
    "dev_summary.csv",
    "benchmark_summary.csv",
    "validation_strata_summary.csv",
    "test_aligned_scores.csv",
    "prediction_diagnostics.csv",
    "prediction_diagnostics_by_origin.csv",
    "channel_share_summary.csv",
    "per_product_summary.csv",
    "top_decile_summary.csv",
    "top_error_rows.csv",
    "sanity_baseline.csv",
    "probabilistic_summary.csv",
    "weight_sensitivity.csv",
    "final_audit_summary.csv",
    "strategy_by_horizon.csv",
    "cv_results.csv",
    "cv_results_all.csv",
    "timings.json",
    "results.json",
)


def model_output_artifact_paths(
    repository_root: str | Path,
    output_dir: str | Path,
    *,
    include_plot: bool = False,
) -> list[Path]:
    """Return only declared model outputs; never discover scratch trees."""
    root = Path(repository_root).resolve()
    directory = Path(output_dir)
    if not directory.is_absolute():
        directory = root / directory
    names = list(MODEL_OUTPUT_FILENAMES)
    if include_plot:
        names.append("forecast_plot.png")
    return [
        (directory / name).resolve()
        for name in names
        if (directory / name).is_file()
    ]


def build_checkpoint_run_identity(
    provenance: dict,
    chronos2_runtime: dict,
    nn_runtime: dict,
) -> dict:
    """Bind every environment/input identity that can affect cached predictions."""
    return {
        "schema_version": "checkpoint-run-identity-v1",
        "source_tree": provenance["source"]["tree"],
        "inputs_sha256": provenance["inputs_sha256"],
        "dependency_lock": provenance["lock"],
        "chronos": provenance["chronos"],
        "resolved_device": chronos2_runtime["resolved_device"],
        "torch": provenance["runtime"]["torch"],
        "nn_training_backend": nn_runtime["training_backend"],
        "expected_actual_execution": {
            "nn": {
                "device": nn_runtime["device"],
                "backend": nn_runtime["training_backend"],
                "batch_size": nn_runtime["batch_size"],
            },
            "chronos2": {
                "device": chronos2_runtime["resolved_device"],
                "dtype": chronos2_runtime["dtype"],
                "batch_size": chronos2_runtime["batch_size"],
            } if chronos2_runtime["enabled"] else None,
        },
    }


def load_authenticated_checkpoint_index(
    repository_root: str | Path,
    publication_manifest: str | None,
) -> dict:
    """Load hashes only through a prior publication-authenticated run manifest."""
    if not publication_manifest:
        return {
            "status": "absent",
            "checkpoint_sha256": {},
            "reason": "No authenticated prior publication index was supplied.",
        }
    root = Path(repository_root).resolve()
    publication_path = (root / publication_manifest).resolve()
    allowed = (root / "outputs" / "publications").resolve()
    if not publication_path.is_relative_to(allowed):
        raise ValueError("Checkpoint trust manifest must be under outputs/publications")
    publication = json.loads(publication_path.read_text(encoding="utf-8"))
    model_run_id = publication.get("model_run_id")
    if not model_run_id:
        raise ValueError("Checkpoint trust publication has no model_run_id")
    model_relative = f"outputs/runs/{model_run_id}.json"
    expected_model_hash = (publication.get("artifact_sha256") or {}).get(model_relative)
    if not expected_model_hash:
        raise ValueError("Checkpoint trust publication does not authenticate its model run")
    model_path = root / model_relative
    if sha256_file(model_path) != expected_model_hash:
        raise ValueError("Checkpoint trust model-run manifest hash mismatch")
    model_manifest = json.loads(model_path.read_text(encoding="utf-8"))
    checkpoints = model_manifest.get("checkpoints")
    if not isinstance(checkpoints, dict) or checkpoints.get("status") != "authenticated":
        raise ValueError("Prior model run has no authenticated checkpoint index")
    hashes = checkpoints.get("files_sha256")
    if not isinstance(hashes, dict):
        raise ValueError("Prior checkpoint index has no file hashes")
    return {
        "status": "authenticated",
        "publication_manifest": str(publication_path.relative_to(root)),
        "publication_manifest_sha256": sha256_file(publication_path),
        "model_run_manifest": model_relative,
        "model_run_manifest_sha256": expected_model_hash,
        "checkpoint_sha256": dict(hashes),
    }


def actual_execution_matches(
    actual: dict | None,
    checkpoint_identity: dict | None,
) -> bool:
    if checkpoint_identity is None:
        return True
    expected = checkpoint_identity.get("expected_actual_execution")
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        return False
    if actual.get("nn") != expected.get("nn"):
        return False
    actual_chronos = actual.get("chronos2")
    return actual_chronos is None or actual_chronos == expected.get("chronos2")


def build_actual_execution(
    nn_seed_stats: list[dict],
    cfg: Config,
) -> dict:
    nn_execution = None
    if nn_seed_stats:
        identities = {
            (
                str(stats["device"]),
                str(stats["backend"]),
                int(stats["batch_size"]),
            )
            for stats in nn_seed_stats
        }
        if len(identities) != 1:
            raise RuntimeError(
                f"Neural seeds used inconsistent execution identities: {sorted(identities)}"
            )
        device, backend, batch_size = identities.pop()
        nn_execution = {
            "device": device,
            "backend": backend,
            "batch_size": batch_size,
        }
    return {
        "nn": nn_execution,
        "chronos2": (
            {
                "device": resolve_chronos2_device(cfg.chronos2_device),
                "dtype": cfg.chronos2_dtype,
                "batch_size": cfg.chronos2_batch_size,
            }
            if cfg.enable_chronos2
            else None
        ),
    }


def _chronos2_checkpoint_signature(
    cfg: Config,
    checkpoint_identity: dict | None = None,
) -> dict:
    """Fingerprint only the optional foundation-model prediction contract."""
    return {
        "schema_version": CHRONOS2_CHECKPOINT_SCHEMA_VERSION,
        "model_id": cfg.chronos2_model_id,
        "model_revision": cfg.chronos2_model_revision,
        "requested_device": cfg.chronos2_device,
        "resolved_device": resolve_chronos2_device(cfg.chronos2_device),
        "dtype": cfg.chronos2_dtype,
        "batch_size": cfg.chronos2_batch_size,
        "context_length": cfg.chronos2_context_length,
        "cross_learning": cfg.chronos2_cross_learning,
        "covariates": cfg.chronos2_covariates,
        "quantile_levels": tuple(cfg.chronos2_quantile_levels),
        "run_identity": checkpoint_identity or {"status": "unbound"},
    }


def _chronos2_oof_mapping(cfg: Config) -> dict[str, str]:
    columns = {
        "prediction": "pred_Chronos2",
        "fallback_used": "fallback_Chronos2",
        "nonfinite_raw": "nonfinite_Chronos2",
        "catastrophic_guard": "catastrophic_Chronos2",
        "residual_guard": "residual_guard_Chronos2",
        "residual_nonfinite": "residual_nonfinite_Chronos2",
        "residual_raw_min": "residual_raw_min_Chronos2",
        "residual_raw_max": "residual_raw_max_Chronos2",
        "safety_limit": "safety_limit_Chronos2",
        "no_context": "no_context_Chronos2",
    }
    for quantile in cfg.chronos2_quantile_levels:
        columns[f"quantile_{quantile:g}"] = (
            f"pred_Chronos2_{chronos2_quantile_suffix(quantile)}"
        )
    return columns


def _drop_chronos2_oof_columns(oof: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column for column in oof.columns
        if column == "pred_Chronos2"
        or column.startswith("pred_Chronos2_")
        or column.endswith("_Chronos2")
    ]
    return oof.drop(columns=columns, errors="ignore")


def _merge_chronos2_oof(
    fold_oof: pd.DataFrame,
    result: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    mapping = _chronos2_oof_mapping(cfg)
    base = _drop_chronos2_oof_columns(fold_oof)
    return base.merge(
        result[["ProductId", "DateKey", *mapping.keys()]].rename(columns=mapping),
        on=["ProductId", "DateKey"],
        how="left",
        validate="one_to_one",
    )


def _fold_checkpoint_path(
    checkpoint_dir: str | None,
    strategy: str,
    origin_type: str,
    origin: pd.Timestamp,
) -> str | None:
    if not checkpoint_dir:
        return None
    filename = f"{pd.Timestamp(origin).date().isoformat()}.pkl"
    return os.path.join(checkpoint_dir, strategy, origin_type, filename)


def _trusted_checkpoint_hash(
    checkpoint_path: str | None,
    trusted_checkpoint_hashes: dict[str, str] | None,
) -> str | None:
    if checkpoint_path is None or not trusted_checkpoint_hashes:
        return None
    candidates = {
        checkpoint_path,
        str(Path(checkpoint_path)),
        str(Path(checkpoint_path).resolve()),
    }
    try:
        candidates.add(str(Path(checkpoint_path).resolve().relative_to(Path.cwd())))
    except ValueError:
        pass
    for candidate in candidates:
        if candidate in trusted_checkpoint_hashes:
            return trusted_checkpoint_hashes[candidate]
    return None


def _fold_checkpoint_signature(
    cfg: Config,
    strategy: str,
    origin_type: str,
    origin: pd.Timestamp,
    checkpoint_identity: dict | None = None,
) -> dict:
    cfg_signature = asdict(cfg)
    # The execution backend changes throughput, not the estimator definition.
    cfg_signature.pop("nn_training_backend", None)
    # Chronos-2 is an independently cached direct-only augmentation. Its
    # settings must not invalidate expensive incumbent or recursive folds.
    for name in tuple(cfg_signature):
        if name == "enable_chronos2" or name.startswith("chronos2_"):
            cfg_signature.pop(name, None)
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "strategy": strategy,
        "origin_type": origin_type,
        "origin": pd.Timestamp(origin).isoformat(),
        "cfg": cfg_signature,
        "run_identity": checkpoint_identity or {"status": "unbound"},
    }


def _checkpoint_signature_compatible(actual: dict, expected: dict) -> bool:
    """Require an exact semantic/training-policy checkpoint signature."""
    return actual == expected


def _load_fold_checkpoint(
    checkpoint_dir: str | None,
    strategy: str,
    origin_type: str,
    origin: pd.Timestamp,
    cfg: Config,
    checkpoint_identity: dict | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> dict | None:
    path = _fold_checkpoint_path(checkpoint_dir, strategy, origin_type, origin)
    if path is None or not os.path.exists(path):
        return None
    if not expected_checkpoint_sha256:
        print(f"    [checkpoint] no authenticated prior hash for {path}; not reusing")
        return None
    try:
        checkpoint_bytes = Path(path).read_bytes()
        actual_checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
        if actual_checkpoint_sha256 != expected_checkpoint_sha256:
            print(f"    [checkpoint] authenticated hash mismatch for {path}; not reusing")
            return None
        payload = pickle.loads(checkpoint_bytes)
    except (
        OSError,
        EOFError,
        pickle.UnpicklingError,
        AttributeError,
        ImportError,
        IndexError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"    [checkpoint] ignoring unreadable {path}: {exc}")
        return None
    expected = _fold_checkpoint_signature(
        cfg, strategy, origin_type, origin, checkpoint_identity
    )
    if not _checkpoint_signature_compatible(payload.get("signature") or {}, expected):
        print(f"    [checkpoint] ignoring stale checkpoint {path}")
        return None
    if not actual_execution_matches(payload.get("actual_execution"), checkpoint_identity):
        print(f"    [checkpoint] actual execution identity differs for {path}")
        return None
    frame = payload.get("oof")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        print(f"    [checkpoint] ignoring invalid checkpoint {path}")
        return None
    payload["_authenticated_checkpoint_sha256"] = actual_checkpoint_sha256
    return payload


def _save_fold_checkpoint(
    checkpoint_dir: str | None,
    strategy: str,
    origin_type: str,
    origin: pd.Timestamp,
    cfg: Config,
    oof: pd.DataFrame,
    timing: dict,
    checkpoint_identity: dict | None = None,
) -> None:
    path = _fold_checkpoint_path(checkpoint_dir, strategy, origin_type, origin)
    if path is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "signature": _fold_checkpoint_signature(
            cfg, strategy, origin_type, origin, checkpoint_identity
        ),
        "oof": oof,
        "timing": timing,
        "actual_execution": timing.get("actual_execution"),
    }
    if strategy == "direct" and "pred_Chronos2" in oof.columns:
        payload["chronos2_signature"] = _chronos2_checkpoint_signature(
            cfg, checkpoint_identity
        )
    tmp_path = f"{path}.tmp-{os.getpid()}"
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


# Broader, seasonally-scattered origins used to make modeling/feature
# decisions (spring/summer lulls, several Januaries, Black Friday windows,
# pre/post-Christmas, a Valentine's-adjacent week -- relevant for a
# cosmetics retailer). Deliberately disjoint from `recent_benchmark_origins`
# below: these are for iteration, the benchmark is for a pseudo-test check,
# and mixing the two would let repeated tuning quietly overfit to the
# benchmark the same way a single reused test set would.
DEVELOPMENT_ORIGINS = pd.to_datetime([
    "2022-02-01", "2022-06-15", "2022-11-20",
    "2023-01-10", "2023-07-01", "2023-11-24", "2023-12-18",
    "2024-02-14", "2024-06-20", "2024-11-29", "2024-12-20",
    "2025-02-10",
])


# Frozen source-level audit origins, disjoint from the current development
# windows and normal recent-benchmark protocol. They are intentionally not
# executed by the normal pipeline or Tier-C screening commands. Run them only
# once after C1-C5 decisions are frozen.
FINAL_AUDIT_ORIGINS = pd.to_datetime([
    "2024-01-17",  # winter/test-like regular week
    "2024-05-15",  # ordinary week
    "2024-11-14",  # pre-Black-Friday stress week
])

FINAL_AUDIT_MARKER = "outputs/final_audit_consumed.json"

VALIDATION_STRATUM_WEIGHTS = {
    "winter_test_like": 0.60,
    "regular": 0.25,
    "holiday_event": 0.15,
}

WEIGHT_SENSITIVITY_SCHEMES = {
    "frozen_test_aligned": VALIDATION_STRATUM_WEIGHTS,
    "equal_strata": {
        "winter_test_like": 1 / 3,
        "regular": 1 / 3,
        "holiday_event": 1 / 3,
    },
}


def classify_validation_stratum(
    origin: pd.Timestamp,
    horizon: int = 7,
) -> str:
    """Classify a forecast window for test-aligned reporting.

    January/February regular weeks provide a larger but still test-like
    winter sample.  Late-November/December windows are event stress tests;
    all remaining development windows are ordinary regular periods.
    """
    target_dates = pd.date_range(
        pd.Timestamp(origin) + pd.Timedelta(days=1), periods=horizon, freq="D"
    )
    if target_dates.month.isin([1, 2]).all():
        return "winter_test_like"
    if ((target_dates.month == 11) & (target_dates.day >= 20)).any() or (
        target_dates.month == 12
    ).any():
        return "holiday_event"
    return "regular"


def recent_benchmark_origins(hist_df: pd.DataFrame, cfg: Config = CFG) -> pd.DatetimeIndex:
    """Last `cfg.n_cv_folds` non-overlapping `cfg.horizon`-day origins ending
    at the most recent training data -- the closest pseudo-test periods to
    the actual forecast. Meant as a final model-selection check (a benchmark
    of recent performance), not something to repeatedly re-tune against."""
    max_date = hist_df["DateKey"].max()
    return pd.DatetimeIndex([max_date - pd.Timedelta(days=(i + 1) * cfg.horizon) for i in range(cfg.n_cv_folds)])


def evaluation_origin_registry(
    recent_origins: pd.DatetimeIndex,
    *,
    include_final_audit: bool,
    run_kind: str,
) -> list[dict]:
    """Describe selection eligibility and inspection state without overclaiming."""
    registry = [
        {
            "key": "development",
            "label": "Development",
            "role": "model_selection",
            "selection_eligible": True,
            "inspection_state": "iterative",
            "origins": [date.date().isoformat() for date in DEVELOPMENT_ORIGINS],
        },
        {
            "key": "recent_diagnostic",
            "artifact_key": "recent_benchmark",
            "label": "Recent diagnostic",
            "role": "non_selection_diagnostic",
            "selection_eligible": False,
            "inspection_state": "previously_inspected",
            "origins": [date.date().isoformat() for date in recent_origins],
        },
    ]
    registry.append({
        "key": "final_audit",
        "label": "Final audit",
        "role": "non_selection_audit",
        "selection_eligible": False,
        "inspection_state": (
            "consumed_by_publication"
            if include_final_audit and run_kind == "publication"
            else "reproduced"
            if include_final_audit
            else "reserved_not_run"
        ),
        "origins": (
            [date.date().isoformat() for date in FINAL_AUDIT_ORIGINS]
            if include_final_audit
            else []
        ),
    })
    return registry


def validate_final_audit_policy(
    options: RuntimeOptions,
    marker_path: str = FINAL_AUDIT_MARKER,
) -> None:
    if not options.include_final_audit:
        return
    if options.run_kind not in {"publication", "reproduction"}:
        raise RuntimeError("Final-audit origins require publication or reproduction")


def reserve_final_audit(
    options: RuntimeOptions,
    *,
    run_id: str,
    source_revision: str,
    generated_at: str,
    marker_path: str = FINAL_AUDIT_MARKER,
) -> dict | None:
    """Consume the fresh-audit claim durably before any model evaluation."""
    validate_final_audit_policy(options, marker_path)
    if not options.include_final_audit:
        return None
    path = Path(marker_path)
    if options.run_kind == "reproduction":
        if not path.exists():
            raise RuntimeError(
                "Final-audit reproduction requires an existing consumed marker"
            )
        return {
            "mode": "reproduction",
            "fresh": False,
            "marker": str(path),
        }
    reservation = {
        "schema_version": "final-audit-consumption-v2",
        "status": "consumed",
        "run_id": run_id,
        "source_revision": source_revision,
        "consumed_at": generated_at,
        "fresh_evidence_claim_consumed": True,
        "origins": [date.date().isoformat() for date in FINAL_AUDIT_ORIGINS],
    }
    try:
        write_json_exclusive(path, reservation)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Final-audit origins are already reserved or consumed in {path}; "
            "use --run-kind reproduction for a labelled rerun"
        ) from exc
    return {"mode": "publication", "fresh": True, "marker": str(path)}


def _reindex_predictions(panel: pd.DataFrame, preds: np.ndarray, date_col: str,
                          keys: pd.DataFrame) -> np.ndarray:
    """Realign `preds` (computed in `panel`'s own row order) to exactly
    `keys`'s (ProductId, DateKey) row order, via an explicit key-based
    merge -- two independently-constructed frames should never be assumed
    to share a row order."""
    lookup = panel[["ProductId", date_col]].rename(columns={date_col: "DateKey"}).copy()
    lookup["_pred"] = preds
    aligned = keys[["ProductId", "DateKey"]].merge(lookup, on=["ProductId", "DateKey"], how="left")
    return aligned["_pred"].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Walk-forward cross-validation
# ---------------------------------------------------------------------------
def run_walk_forward_cv_direct(
    hist_df: pd.DataFrame, origins, origin_type: str, cfg: Config = CFG,
    timings: list[dict] | None = None, *, checkpoint_dir: str | None = None,
    resume: bool = False, checkpoint_identity: dict | None = None,
    trusted_checkpoint_hashes: dict[str, str] | None = None,
    run_neural: bool = True,
) -> pd.DataFrame:
    """Evaluate at each `origin` date (the last training day): trains only
    on data up to and including `origin` (no leakage) and predicts all
    `cfg.horizon` days directly from the multi-horizon panel (see
    `framework.build_direct_panel`) -- no recursion, since every horizon's
    features are already lookups into observed data, never a value that
    would first need to be predicted.

    Trains the same `cfg.seeds`-sized neural incumbent used for the final
    forecast -- CV must score the actual estimator, not a cheaper
    single-seed stand-in. `cv_epochs` vs `final_epochs` remains a
    deliberate, disclosed compute/accuracy trade-off (cheaper proxy
    training while iterating; the one-time final artifact trains longer)
    -- unlike the seed count, that's not a hidden inconsistency, since
    it's applied identically across every model/fold.

    Returns row-level out-of-fold predictions -- one row per (origin,
    product, date), with the frozen incumbent, Chronos-2, incumbent seed
    diagnostics and Chronos uncertainty/fallback fields.  Both contenders are
    therefore evaluated from the same exact rows without carrying historical
    baseline models into the new repository.

    If `timings` is given, one {origin_type, origin, incumbent_seconds,
    chronos2_seconds, fold_seconds} dict is appended per fold -- lets `main()`
    build an `outputs/timings.json` breakdown without this function owning
    the file write itself.
    """
    horizons = range(1, cfg.horizon + 1)
    fold_frames = []

    for origin in origins:
        fold_start = time.perf_counter()
        eval_start = origin + pd.Timedelta(days=1)
        eval_end = origin + pd.Timedelta(days=cfg.horizon)
        fold_train_raw = hist_df[hist_df["DateKey"] <= origin].copy()
        fold_eval_raw = hist_df[(hist_df["DateKey"] >= eval_start) & (hist_df["DateKey"] <= eval_end)].copy()
        if fold_train_raw.empty or fold_eval_raw.empty:
            continue

        if resume:
            checkpoint_path = _fold_checkpoint_path(
                checkpoint_dir, "direct", origin_type, origin
            )
            cached = _load_fold_checkpoint(
                checkpoint_dir,
                "direct",
                origin_type,
                origin,
                cfg,
                checkpoint_identity,
                _trusted_checkpoint_hash(
                    checkpoint_path, trusted_checkpoint_hashes
                ),
            )
            if cached is not None:
                cached_oof = cached["oof"].copy()
                timing_record = dict(cached.get("timing") or {})
                timing_record["checkpoint_source"] = "reused"
                timing_record["checkpoint_path"] = checkpoint_path
                timing_record["consumed_checkpoint_sha256"] = cached[
                    "_authenticated_checkpoint_sha256"
                ]
                timing_record["consumed_checkpoint_path"] = checkpoint_path
                cached_chronos_signature = cached.get("chronos2_signature")
                expected_chronos_signature = _chronos2_checkpoint_signature(
                    cfg, checkpoint_identity
                )
                has_current_chronos = (
                    "pred_Chronos2" in cached_oof.columns
                    and cached_chronos_signature == expected_chronos_signature
                )
                if cfg.enable_chronos2 and not has_current_chronos:
                    print(
                        f"  [{origin_type}] origin {origin.date()}: "
                        "augmenting cached direct fold with Chronos-2"
                    )
                    chronos_start = time.perf_counter()
                    chronos_result = forecast_chronos2(
                        fold_train_raw, fold_eval_raw, cfg
                    )
                    chronos_seconds = time.perf_counter() - chronos_start
                    cached_oof = _merge_chronos2_oof(
                        cached_oof, chronos_result, cfg
                    )
                    prior_chronos_seconds = float(
                        timing_record.get("chronos2_seconds", 0.0) or 0.0
                    )
                    base_fold_seconds = max(
                        0.0,
                        float(timing_record.get("fold_seconds", 0.0) or 0.0)
                        - prior_chronos_seconds,
                    )
                    timing_record["chronos2_seconds"] = round(
                        chronos_seconds, 2
                    )
                    timing_record["fold_seconds"] = round(
                        base_fold_seconds + chronos_seconds, 2
                    )
                    prior_execution = dict(
                        cached.get("actual_execution") or {
                            "nn": None,
                            "chronos2": None,
                        }
                    )
                    prior_execution["chronos2"] = {
                        "device": resolve_chronos2_device(cfg.chronos2_device),
                        "dtype": cfg.chronos2_dtype,
                        "batch_size": cfg.chronos2_batch_size,
                    }
                    timing_record["actual_execution"] = prior_execution
                    _save_fold_checkpoint(
                        checkpoint_dir,
                        "direct",
                        origin_type,
                        origin,
                        cfg,
                        cached_oof,
                        timing_record,
                        checkpoint_identity,
                    )
                elif not cfg.enable_chronos2:
                    cached_oof = _drop_chronos2_oof_columns(cached_oof)
                    prior_chronos_seconds = float(
                        timing_record.pop("chronos2_seconds", 0.0) or 0.0
                    )
                    if "fold_seconds" in timing_record:
                        timing_record["fold_seconds"] = round(
                            max(
                                0.0,
                                float(timing_record["fold_seconds"])
                                - prior_chronos_seconds,
                            ),
                            2,
                        )
                print(
                    f"  [{origin_type}] origin {origin.date()}: "
                    "loaded completed direct fold checkpoint"
                )
                fold_frames.append(cached_oof)
                if timings is not None and timing_record:
                    timings.append(timing_record)
                continue

        print(f"  [{origin_type}] origin {origin.date()}: eval {eval_start.date()}..{eval_end.date()}")

        price_ref = fold_train_raw.groupby("ProductId")["PriceLocalVat"].median()
        first_seen, first_available = product_reference_dates(fold_train_raw)

        fold_train_feat = prepare_features(
            fold_train_raw, price_ref, first_seen, first_available, cfg
        )
        fold_train_feat = add_train_lags(
            fold_train_feat, cfg.lag_windows,
            baseline_variant=cfg.baseline_variant,
        )
        fold_eval_feat = prepare_features(
            fold_eval_raw, price_ref, first_seen, first_available, cfg
        ).reset_index(drop=True)

        panel = build_direct_panel(fold_train_feat, horizons, cfg=cfg, future_covariates=fold_eval_feat)
        # Leakage-safe training slice: a training row's own target must
        # already be observable as of `origin` -- an origin close to the
        # fold's own cutoff combined with a large horizon would otherwise
        # land on a target date this fold isn't allowed to have seen yet.
        train_panel = select_trainable_panel_rows(
            panel, cutoff=origin, available_only=True, cfg=cfg
        )
        eval_panel = panel[panel["OriginDateKey"] == origin].reset_index(drop=True)

        seed_preds: dict[int, np.ndarray] = {}
        nn_seed_stats: list[dict] = []
        ensemble_output = None
        nn_seconds = 0.0
        if run_neural:
            scaler = make_numeric_preprocessor()
            tensors = make_tensors(train_panel, scaler, fit=True, cfg=cfg)
            y_target = neural_training_target(train_panel, cfg)
            nn_start = time.perf_counter()
            seed_models = []
            for seed in cfg.seeds:
                seed_stats: dict = {}
                seed_models.append(
                    train_model(
                        tensors,
                        y_target,
                        cfg,
                        epochs=cfg.cv_epochs,
                        seed=seed,
                        stats_out=seed_stats,
                    )
                )
                nn_seed_stats.append(seed_stats)
            nn_seconds = time.perf_counter() - nn_start
            seed_preds = {
                seed: predict_direct([model], scaler, eval_panel, cfg)
                for seed, model in zip(cfg.seeds, seed_models)
            }
            ensemble_output = predict_direct(
                seed_models, scaler, eval_panel, cfg, return_diagnostics=True
            )

        chronos2_result = None
        chronos2_seconds = 0.0
        if cfg.enable_chronos2:
            chronos2_start = time.perf_counter()
            chronos2_result = forecast_chronos2(
                fold_train_raw, fold_eval_raw, cfg
            )
            chronos2_seconds = time.perf_counter() - chronos2_start

        baseline_pred = compute_baseline(
            fold_eval_feat, fold_train_raw, cfg.baseline_variant
        )

        # The actual target, availability state and incumbent seasonal anchor
        # are joined by explicit keys.  No historical comparison model is
        # emitted into the challenge OOF table.
        evaluation_df = fold_eval_feat[
            ["ProductId", "DateKey", "Quantity", "ProductAvailable"]
        ].copy()
        evaluation_df["baseline"] = baseline_pred

        fold_oof = eval_panel[["ProductId", "horizon", "TargetDateKey"]].rename(columns={"TargetDateKey": "DateKey"})
        fold_oof["origin"] = origin
        fold_oof["origin_type"] = origin_type
        fold_oof["strategy"] = "direct"
        fold_oof["validation_stratum"] = classify_validation_stratum(
            origin, cfg.horizon
        )
        model_names: list[str] = []
        if ensemble_output is not None:
            fold_oof["pred_NeuralNet"] = ensemble_output["prediction"]
            model_names.append("NeuralNet")
            if "app_share" in ensemble_output:
                fold_oof["pred_AppShare_NeuralNet"] = ensemble_output["app_share"]
                fold_oof["pred_QuantityApp_NeuralNet"] = ensemble_output["prediction_app"]
                fold_oof["pred_QuantityWeb_NeuralNet"] = ensemble_output["prediction_web"]
                actual_total = pd.to_numeric(
                    eval_panel.get("target", pd.Series(np.nan, index=eval_panel.index)),
                    errors="coerce",
                ).to_numpy(dtype=float)
                actual_app = pd.to_numeric(
                    eval_panel.get("target_app", pd.Series(np.nan, index=eval_panel.index)),
                    errors="coerce",
                ).to_numpy(dtype=float)
                actual_share = np.divide(
                    actual_app,
                    actual_total,
                    out=np.full(len(eval_panel), np.nan, dtype=float),
                    where=np.isfinite(actual_total) & (actual_total > 0),
                )
                fold_oof["actual_AppShare"] = actual_share
        if chronos2_result is not None:
            fold_oof = _merge_chronos2_oof(fold_oof, chronos2_result, cfg)
        for name in model_names:
            fold_oof[f"fallback_{name}"] = False
            fold_oof[f"nonfinite_{name}"] = False
            fold_oof[f"catastrophic_{name}"] = False
            fold_oof[f"residual_guard_{name}"] = False
            fold_oof[f"residual_nonfinite_{name}"] = False
            fold_oof[f"residual_raw_min_{name}"] = np.nan
            fold_oof[f"residual_raw_max_{name}"] = np.nan
            fold_oof[f"safety_limit_{name}"] = np.nan
        for seed, predictions in seed_preds.items():
            fold_oof[f"pred_NeuralNet_seed{seed}"] = predictions
        fold_oof = fold_oof.merge(evaluation_df, on=["ProductId", "DateKey"], how="left")
        fold_oof = fold_oof.rename(columns={"Quantity": "actual"})
        fold_frames.append(fold_oof)

        fold_seconds = time.perf_counter() - fold_start
        timing_record = {
            "strategy": "direct", "origin_type": origin_type,
            "origin": str(origin.date()),
            "checkpoint_source": "trained",
            "checkpoint_path": _fold_checkpoint_path(
                checkpoint_dir, "direct", origin_type, origin
            ),
            "actual_execution": build_actual_execution(nn_seed_stats, cfg),
            "incumbent_seconds": round(nn_seconds, 2),
            "chronos2_seconds": round(chronos2_seconds, 2),
            "fold_seconds": round(fold_seconds, 2),
        }
        print(
            f"    [timing] {origin_type} {origin.date()}: "
            f"Best NN {nn_seconds:.1f}s | Chronos-2 {chronos2_seconds:.1f}s | "
            f"fold total {fold_seconds:.1f}s"
        )
        if timings is not None:
            timings.append(timing_record)
        _save_fold_checkpoint(
            checkpoint_dir,
            "direct",
            origin_type,
            origin,
            cfg,
            fold_oof,
            timing_record,
            checkpoint_identity,
        )

    return pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()


OOF_MODEL_COLUMNS = {
    "NeuralNet": "pred_NeuralNet",
    "Chronos2": "pred_Chronos2",
}


def summarize_oof(oof: pd.DataFrame, pred_columns: dict = None) -> pd.DataFrame:
    """B4: Refactored to support common-population evaluation and detailed metrics.
    Produces combinations of:
      - evaluation_regime: 'realized' (all days) vs 'conditional' (available only)
      - comparison_population: 'common' (same rows for all models) vs 'model_specific'
      - aggregation: 'global' (micro) vs 'mean_fold' (macro)
    """
    pred_columns = pred_columns or OOF_MODEL_COLUMNS
    pred_columns = {
        model: column
        for model, column in pred_columns.items()
        if column in oof.columns
    }
    pred_cols = list(pred_columns.values())
    if not pred_cols:
        return pd.DataFrame()

    # Base masks for regimes
    regime_masks = {
        "realized": oof["actual"].notna(),
        "conditional": (
            oof["actual"].notna()
            & oof["ProductAvailable"].fillna(False)
        ),
    }

    rows = []

    for regime_name, regime_mask in regime_masks.items():
        # Rows where ALL models have finite predictions
        common_mask = regime_mask & oof[pred_cols].apply(np.isfinite).all(axis=1)

        populations = ["common", "model_specific"]
        for pop_name in populations:
            for model_name, pred_col in pred_columns.items():
                if pred_col not in oof.columns:
                    continue

                # Rows for THIS model and THIS population
                if pop_name == "common":
                    mask = common_mask
                else:
                    mask = regime_mask & np.isfinite(oof[pred_col])

                scored_df = oof[mask]

                # Diagnostics (always regime-relative)
                n_expected = int(regime_mask.sum())
                n_actual = int((regime_mask & oof["actual"].notna()).sum())
                n_predicted = int((regime_mask & np.isfinite(oof[pred_col])).sum())
                n_scored = int(mask.sum())
                coverage = n_predicted / n_expected if n_expected > 0 else 0.0

                def add_row(df, agg_name):
                    if df.empty:
                        metrics = {k: np.nan for k in ["MAE", "RMSE", "WAPE", "sMAPE", "RMSLE", "Bias", "BiasRatio", "MAPE"]}
                        n_folds = 0
                    else:
                        if agg_name == "global":
                            metrics = compute_metrics(df["actual"], df[pred_col])
                            n_folds = df["origin"].nunique()
                        else:  # mean_fold
                            fold_metrics = [compute_metrics(g["actual"], g[pred_col]) for _, g in df.groupby("origin")]
                            metrics = pd.DataFrame(fold_metrics).mean(numeric_only=True).to_dict()
                            n_folds = len(fold_metrics)

                    rows.append({
                        "model": model_name,
                        "evaluation_regime": regime_name,
                        "comparison_population": pop_name,
                        "aggregation": agg_name,
                        "n_folds": n_folds,
                        "n_expected": n_expected,
                        "n_actual": n_actual,
                        "n_predicted": n_predicted,
                        "n_scored": n_scored,
                        "coverage": coverage,
                        **metrics
                    })

                add_row(scored_df, "global")
                add_row(scored_df, "mean_fold")

    return pd.DataFrame(rows)


def summarize_oof_by_strategy(oof: pd.DataFrame, pred_columns: dict = None) -> pd.DataFrame:
    if "strategy" not in oof.columns:
        out = summarize_oof(oof, pred_columns)
        out["strategy"] = "direct"
        return out
    frames = []
    for strategy, group in oof.groupby("strategy", sort=False):
        strategy_columns = prediction_columns_for_strategy(
            pred_columns or OOF_MODEL_COLUMNS, strategy
        )
        summary = summarize_oof(group, strategy_columns)
        summary["strategy"] = strategy
        frames.append(summary)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_validation_strata(
    oof: pd.DataFrame,
    pred_columns: dict | None = None,
) -> pd.DataFrame:
    """Metric summaries by strategy and data-generating-process stratum."""
    if oof.empty:
        return pd.DataFrame()
    if "validation_stratum" not in oof.columns:
        work = oof.copy()
        work["validation_stratum"] = [
            classify_validation_stratum(origin)
            for origin in work["origin"]
        ]
    else:
        work = oof
    frames = []
    group_keys = ["strategy", "validation_stratum"]
    if "origin_type" in work.columns:
        group_keys = ["origin_type"] + group_keys
    for keys, group in work.groupby(group_keys, sort=False):
        if "origin_type" in work.columns:
            origin_type, strategy, stratum = keys
        else:
            strategy, stratum = keys
            origin_type = "development"
        columns = prediction_columns_for_strategy(
            pred_columns or OOF_MODEL_COLUMNS, strategy
        )
        summary = summarize_oof(group, columns)
        if summary.empty:
            continue
        summary["origin_type"] = origin_type
        summary["strategy"] = strategy
        summary["validation_stratum"] = stratum
        frames.append(summary)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compute_test_aligned_scores(
    stratum_summary: pd.DataFrame,
    metric: str = "WAPE",
    weights: dict[str, float] | None = None,
    *,
    origin_type: str = "development",
) -> pd.DataFrame:
    """Weighted stratum score for one explicitly named evaluation split."""
    if stratum_summary.empty:
        return pd.DataFrame()
    weights = weights or VALIDATION_STRATUM_WEIGHTS
    selected = stratum_summary[
        stratum_summary["evaluation_regime"].eq("conditional")
        & stratum_summary["comparison_population"].eq("common")
        & stratum_summary["aggregation"].eq("global")
    ].copy()
    if "origin_type" in selected.columns:
        selected = selected[selected["origin_type"].eq(origin_type)]
    rows = []
    for (strategy, model), group in selected.groupby(["strategy", "model"]):
        available = group[group["validation_stratum"].isin(weights)].copy()
        available = available[np.isfinite(available[metric])]
        if available.empty:
            continue
        available["stratum_weight"] = available["validation_stratum"].map(weights)
        total_weight = float(available["stratum_weight"].sum())
        if total_weight <= 0:
            continue
        score = float(
            np.average(available[metric], weights=available["stratum_weight"])
        )
        rows.append({
            "strategy": strategy,
            "model": model,
            "metric": metric,
            "test_aligned_score": score,
            "weight_sum": total_weight,
            "strata_present": ",".join(sorted(available["validation_stratum"].unique())),
        })
    return pd.DataFrame(rows)


def summarize_weight_sensitivity(
    stratum_summary: pd.DataFrame,
    development_summary: pd.DataFrame,
    metric: str = "WAPE",
) -> pd.DataFrame:
    """Report predeclared weighting views without changing winner selection."""
    frames: list[pd.DataFrame] = []
    for scheme, weights in WEIGHT_SENSITIVITY_SCHEMES.items():
        scores = compute_test_aligned_scores(
            stratum_summary,
            metric=metric,
            weights=weights,
            origin_type="development",
        )
        if scores.empty:
            continue
        scores.insert(0, "scheme", scheme)
        scores["selection_eligible"] = scheme == "frozen_test_aligned"
        frames.append(scores)

    global_rows = development_summary[
        development_summary["evaluation_regime"].eq("conditional")
        & development_summary["comparison_population"].eq("common")
        & development_summary["aggregation"].eq("global")
        & development_summary["model"].isin(CHALLENGE_MODELS)
    ][["strategy", "model", metric]].copy()
    if not global_rows.empty:
        global_rows.insert(0, "scheme", "global_pooled")
        global_rows["metric"] = metric
        global_rows["test_aligned_score"] = global_rows.pop(metric)
        global_rows["weight_sum"] = np.nan
        global_rows["strata_present"] = "pooled_rows"
        global_rows["selection_eligible"] = False
        frames.append(global_rows)

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if result.empty:
        return result
    winners = (
        result.sort_values("test_aligned_score")
        .groupby("scheme", as_index=False)
        .first()[["scheme", "model"]]
        .rename(columns={"model": "winner"})
    )
    return result.merge(winners, on="scheme", how="left", validate="many_to_one")


def summarize_channel_share_oof(oof: pd.DataFrame) -> pd.DataFrame:
    """C4 app-share diagnostics by split and strategy.

    Total demand remains the submitted target. This table verifies whether the
    auxiliary head learned a meaningful channel composition without allowing
    share quality to conceal a deterioration in total-demand WAPE.
    """
    required = {"actual_AppShare", "pred_AppShare_NeuralNet", "actual"}
    if not required.issubset(oof.columns):
        return pd.DataFrame()
    rows = []
    group_columns = [column for column in ("origin_type", "strategy") if column in oof]
    grouped = oof.groupby(group_columns, sort=False) if group_columns else [((), oof)]
    for keys, group in grouped:
        if group_columns and not isinstance(keys, tuple):
            keys = (keys,)
        context = dict(zip(group_columns, keys if group_columns else ()))
        actual = pd.to_numeric(group["actual_AppShare"], errors="coerce").to_numpy(dtype=float)
        predicted = pd.to_numeric(
            group["pred_AppShare_NeuralNet"], errors="coerce"
        ).to_numpy(dtype=float)
        total = pd.to_numeric(group["actual"], errors="coerce").to_numpy(dtype=float)
        mask = (
            np.isfinite(actual) & np.isfinite(predicted)
            & np.isfinite(total) & (total > 0.0)
        )
        n_expected = int((np.isfinite(actual) & np.isfinite(total) & (total > 0.0)).sum())
        if mask.any():
            error = predicted[mask] - actual[mask]
            absolute = np.abs(error)
            weights = total[mask]
            weighted_mae = (
                float(np.average(absolute, weights=weights))
                if weights.sum() > 0 else np.nan
            )
            rows.append({
                **context,
                "model": "NeuralNet",
                "n_expected": n_expected,
                "n_scored": int(mask.sum()),
                "coverage": float(mask.sum() / n_expected) if n_expected else np.nan,
                "app_share_MAE": float(absolute.mean()),
                "app_share_weighted_MAE": weighted_mae,
                "app_share_bias": float(error.mean()),
                "actual_app_share_mean": float(actual[mask].mean()),
                "predicted_app_share_mean": float(predicted[mask].mean()),
            })
        else:
            rows.append({
                **context,
                "model": "NeuralNet",
                "n_expected": n_expected,
                "n_scored": 0,
                "coverage": 0.0 if n_expected else np.nan,
                "app_share_MAE": np.nan,
                "app_share_weighted_MAE": np.nan,
                "app_share_bias": np.nan,
                "actual_app_share_mean": np.nan,
                "predicted_app_share_mean": np.nan,
            })
    return pd.DataFrame(rows)


def _summarize_prediction_diagnostics_grouped(
    oof: pd.DataFrame,
    pred_columns: dict,
    group_columns: list[str],
) -> pd.DataFrame:
    rows = []
    for keys, group in oof.groupby(group_columns, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        context = dict(zip(group_columns, keys))
        strategy = context["strategy"]
        columns = prediction_columns_for_strategy(pred_columns, strategy)
        observed_max = pd.to_numeric(group["actual"], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).max()
        for model, column in columns.items():
            if column not in group.columns:
                continue
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(values)
            finite_values = values[finite]

            def bool_values(prefix: str) -> np.ndarray:
                name = f"{prefix}_{model}"
                if name not in group:
                    return np.zeros(len(group), dtype=bool)
                return group[name].fillna(False).astype(bool).to_numpy()

            fallback = bool_values("fallback")
            nonfinite_raw = bool_values("nonfinite")
            no_context = bool_values("no_context")
            catastrophic = bool_values("catastrophic")
            residual_guard = bool_values("residual_guard")
            residual_nonfinite = bool_values("residual_nonfinite")

            residual_min_col = f"residual_raw_min_{model}"
            residual_max_col = f"residual_raw_max_{model}"
            safety_limit_col = f"safety_limit_{model}"
            residual_min = (
                pd.to_numeric(group[residual_min_col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan).min()
                if residual_min_col in group else np.nan
            )
            residual_max = (
                pd.to_numeric(group[residual_max_col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan).max()
                if residual_max_col in group else np.nan
            )
            safety_limit_min = (
                pd.to_numeric(group[safety_limit_col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan).min()
                if safety_limit_col in group else np.nan
            )
            prediction_max = float(np.max(finite_values)) if finite_values.size else np.nan
            prediction_p99 = float(np.quantile(finite_values, 0.99)) if finite_values.size else np.nan
            rows.append({
                **context,
                "model": model,
                "n_rows": int(len(group)),
                "n_finite": int(finite.sum()),
                "coverage": float(finite.mean()) if len(group) else np.nan,
                "fallback_count": int(fallback.sum()),
                "fallback_rate": float(fallback.mean()) if len(group) else np.nan,
                "nonfinite_raw_count": int(nonfinite_raw.sum()),
                "no_context_count": int(no_context.sum()),
                "no_context_rate": float(no_context.mean()) if len(group) else np.nan,
                "catastrophic_guard_count": int(catastrophic.sum()),
                "residual_guard_count": int(residual_guard.sum()),
                "residual_nonfinite_count": int(residual_nonfinite.sum()),
                "residual_guard_rate": (
                    float(residual_guard.mean()) if len(group) else np.nan
                ),
                "residual_raw_min": (
                    float(residual_min) if np.isfinite(residual_min) else np.nan
                ),
                "residual_raw_max": (
                    float(residual_max) if np.isfinite(residual_max) else np.nan
                ),
                "safety_limit_min": (
                    float(safety_limit_min) if np.isfinite(safety_limit_min) else np.nan
                ),
                "prediction_max": prediction_max,
                "prediction_p99": prediction_p99,
                "observed_max": float(observed_max) if np.isfinite(observed_max) else np.nan,
                "prediction_to_observed_max_ratio": (
                    prediction_max / observed_max
                    if np.isfinite(prediction_max) and np.isfinite(observed_max) and observed_max > 0
                    else np.nan
                ),
            })
    return pd.DataFrame(rows)


def summarize_prediction_diagnostics(
    oof: pd.DataFrame,
    pred_columns: dict | None = None,
) -> pd.DataFrame:
    """Aggregate fallback, support-guard and extreme behavior by split."""
    return _summarize_prediction_diagnostics_grouped(
        oof, pred_columns or OOF_MODEL_COLUMNS, ["origin_type", "strategy"]
    )


def summarize_prediction_diagnostics_by_origin(
    oof: pd.DataFrame,
    pred_columns: dict | None = None,
) -> pd.DataFrame:
    """Per-origin diagnostics so isolated recursive explosions stay visible."""
    return _summarize_prediction_diagnostics_grouped(
        oof, pred_columns or OOF_MODEL_COLUMNS,
        ["origin_type", "strategy", "origin"],
    )


def oof_to_legacy_cv_results(oof: pd.DataFrame, pred_columns: dict = None) -> pd.DataFrame:
    """Reshape row-level OOF predictions back into the older
    fold/model/MAE/RMSE/WAPE/Bias/BiasRatio shape.
    B4/Fix: Use common populations per fold/regime for fair comparison."""
    pred_columns = pred_columns or OOF_MODEL_COLUMNS
    strategies = oof.get("strategy", pd.Series("direct", index=oof.index)).dropna().unique()
    if len(strategies) == 1:
        pred_columns = prediction_columns_for_strategy(
            pred_columns, str(strategies[0])
        )
    pred_columns = {
        model: column
        for model, column in pred_columns.items()
        if column in oof.columns
    }
    pred_cols = list(pred_columns.values())

    origins_sorted = sorted(oof["origin"].unique(), reverse=True)
    fold_of_origin = {origin: i for i, origin in enumerate(origins_sorted)}

    regime_masks_base = {
        "realized": oof["actual"].notna(),
        "conditional": (
            oof["actual"].notna()
            & oof["ProductAvailable"].fillna(False)
        ),
    }

    rows = []
    for origin, fold_df in oof.groupby("origin"):
        for regime_name, regime_mask_all in regime_masks_base.items():
            # Regime mask for THIS fold
            regime_mask = regime_mask_all.loc[fold_df.index]

            # Common population: rows where ALL models have finite predictions
            common_mask = regime_mask & fold_df[pred_cols].apply(np.isfinite).all(axis=1)

            for model_name, col in pred_columns.items():
                if col not in fold_df.columns:
                    continue

                scored_df = fold_df[common_mask]
                if scored_df.empty:
                    continue

                rows.append({
                    "fold": fold_of_origin[origin],
                    "model": model_name,
                    "regime": regime_name,
                    "comparison_population": "common",
                    **compute_metrics(scored_df["actual"], scored_df[col])
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Final incumbent and Chronos-2 forecasts
# ---------------------------------------------------------------------------
def _prepare_final_direct_panel(train_raw: pd.DataFrame, test_raw: pd.DataFrame, cfg: Config = CFG):
    """Build the frozen incumbent's final direct training/evaluation panel.

    This helper builds
    the direct multi-horizon panel for the real forecast -- origin = the
    last training day, targets = the actual test week (covariates from
    `test_raw` itself, since nothing later exists in `train_raw` to look
    up)."""
    price_ref = train_raw.groupby("ProductId")["PriceLocalVat"].median()
    first_seen, first_available = product_reference_dates(train_raw)

    train_feat = prepare_features(
        train_raw, price_ref, first_seen, first_available, cfg
    )
    train_feat = add_train_lags(
        train_feat, cfg.lag_windows, baseline_variant=cfg.baseline_variant
    )
    test_feat = prepare_features(
        test_raw, price_ref, first_seen, first_available, cfg
    ).reset_index(drop=True)

    horizons = range(1, cfg.horizon + 1)
    panel = build_direct_panel(train_feat, horizons, cfg=cfg, future_covariates=test_feat)

    last_train_date = train_raw["DateKey"].max()
    train_panel = select_trainable_panel_rows(
        panel, cutoff=last_train_date, available_only=True, cfg=cfg
    )
    eval_panel = panel[panel["OriginDateKey"] == last_train_date].reset_index(drop=True)
    return train_panel, eval_panel


def run_final_forecast_direct(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    cfg: Config = CFG,
    *,
    return_diagnostics: bool = False,
):
    train_panel, eval_panel = _prepare_final_direct_panel(train_raw, test_raw, cfg)

    scaler = make_numeric_preprocessor()
    tensors = make_tensors(train_panel, scaler, fit=True, cfg=cfg)
    y_target = neural_training_target(train_panel, cfg)

    models = []
    for seed in cfg.seeds:
        seed_start = time.perf_counter()
        print(f"    seed {seed}")
        models.append(train_model(tensors, y_target, cfg, epochs=cfg.final_epochs, seed=seed))
        print(f"      [timing] seed {seed}: {time.perf_counter() - seed_start:.1f}s")

    output = predict_direct(
        models, scaler, eval_panel, cfg, return_diagnostics=True
    )
    preds = output["prediction"]
    preds_aligned = _reindex_predictions(eval_panel, preds, "TargetDateKey", test_raw)

    submission = test_raw[["ProductId", "DateKey"]].copy()
    submission["Quantity"] = np.round(preds_aligned).astype(int)
    if not return_diagnostics:
        return submission, preds_aligned

    diagnostics = {"prediction": preds_aligned}
    for key in ("app_share", "prediction_app", "prediction_web"):
        if key in output:
            diagnostics[key] = _reindex_predictions(
                eval_panel, output[key], "TargetDateKey", test_raw
            )
    return submission, preds_aligned, diagnostics


def run_final_chronos2_forecast_direct(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    cfg: Config = CFG,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Zero-shot Chronos-2 forecast on all history and the supplied test week."""
    start = time.perf_counter()
    details = forecast_chronos2(train_raw, test_raw, cfg)
    print(f"    [timing] Chronos-2 final: {time.perf_counter() - start:.1f}s")
    return details["prediction"].to_numpy(dtype=float), details


def plot_forecast(train_raw: pd.DataFrame, submission: pd.DataFrame,
                   product_ids: tuple = (1, 5, 16), lookback_days: int = 60,
                   cfg: Config = CFG) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(product_ids), 1, figsize=(9, 3 * len(product_ids)))
    axes = np.atleast_1d(axes)
    for ax, pid in zip(axes, product_ids):
        hist = train_raw[train_raw["ProductId"] == pid].sort_values("DateKey").tail(lookback_days)
        fut = submission[submission["ProductId"] == pid].sort_values("DateKey")
        ax.plot(hist["DateKey"], hist["Quantity"], label="history", color="steelblue")
        ax.plot(fut["DateKey"], fut["Quantity"], label="forecast", color="darkorange", marker="o")
        ax.axvline(hist["DateKey"].max(), color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"Product {pid}")
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    os.makedirs(cfg.output_dir, exist_ok=True)
    out_path = os.path.join(cfg.output_dir, "forecast_plot.png")
    fig.savefig(out_path, dpi=130)
    print(f"Saved: {out_path}")


def _json_safe(obj):
    """Convert pipeline payload values into strict JSON-compatible scalars.

    DataFrame ``to_dict`` preserves pandas/NumPy scalar types, including
    ``Timestamp`` values in per-origin diagnostics.  The JSON encoder does
    not know how to serialize those objects.  Normalize them once at the
    artifact boundary and reject any remaining non-standard NaN/Infinity
    tokens when writing the file.
    """
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if obj is pd.NA or obj is pd.NaT:
        return None
    if isinstance(obj, (pd.Timestamp, np.datetime64, datetime)):
        timestamp = pd.Timestamp(obj)
        return None if pd.isna(timestamp) else timestamp.isoformat()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def select_primary_summary(
    summary: pd.DataFrame,
    *,
    evaluation_regime: str = "conditional",
    comparison_population: str = "common",
    aggregation: str = "global",
) -> pd.DataFrame:
    """Helper to select a canonical slice of the expanded OOF summary (Tier B Fix)."""
    selected = summary[
        (summary["evaluation_regime"] == evaluation_regime)
        & (summary["comparison_population"] == comparison_population)
        & (summary["aggregation"] == aggregation)
    ].copy()

    if selected.empty:
        raise RuntimeError(
            f"Primary evaluation summary is empty for "
            f"{evaluation_regime}/{comparison_population}/{aggregation}"
        )

    if selected["model"].duplicated().any():
        raise RuntimeError(
            f"Primary evaluation summary contains duplicate model rows for "
            f"{evaluation_regime}/{comparison_population}/{aggregation}"
        )

    return selected


def export_results_json(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    submission: pd.DataFrame,
    final_forecasts: dict,
    cv_results: pd.DataFrame,
    cfg: Config = CFG,
    history_lookback: int = 90,
    path: str | None = None,
    dev_summary: pd.DataFrame = None,
    benchmark_summary: pd.DataFrame = None,
    runtime_options: RuntimeOptions | None = None,
    forecasts_by_strategy: dict | None = None,
    canonical_strategy: str = "direct",
    canonical_model: str = "NeuralNet",
    cv_results_all: pd.DataFrame | None = None,
    strategy_by_horizon: pd.DataFrame | None = None,
    validation_strata_summary: pd.DataFrame | None = None,
    test_aligned_scores: pd.DataFrame | None = None,
    prediction_diagnostics: pd.DataFrame | None = None,
    prediction_diagnostics_by_origin: pd.DataFrame | None = None,
    channel_share_summary: pd.DataFrame | None = None,
    per_product_summary: pd.DataFrame | None = None,
    top_decile_summary: pd.DataFrame | None = None,
    top_error_rows: pd.DataFrame | None = None,
    sanity_baseline: pd.DataFrame | None = None,
    probabilistic_summary: pd.DataFrame | None = None,
    weight_sensitivity: pd.DataFrame | None = None,
    audit_summary: pd.DataFrame | None = None,
    final_quantile_forecasts: dict | None = None,
    origin_registry: list[dict] | None = None,
    provenance: dict | None = None,
    publication_provenance: dict | None = None,
) -> dict:
    """Write the strict two-model dashboard contract.

    Every model-bearing table is filtered at the artifact boundary.  This is
    deliberately stronger than hiding legacy rows in JavaScript: consumers of
    ``results.json`` can only observe the frozen incumbent and Chronos-2.
    """

    def challenge_only(frame: pd.DataFrame | None) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame() if frame is None else frame.copy()
        result = frame.copy()
        if "model" in result.columns:
            result = result[result["model"].isin(CHALLENGE_MODELS)]
        return result.reset_index(drop=True)

    def records(frame: pd.DataFrame | None, digits: int = 6) -> list[dict]:
        filtered = challenge_only(frame)
        if filtered.empty:
            return []
        if "model" in filtered.columns:
            filtered = order_models(filtered)
        numeric_columns = filtered.select_dtypes(include=[np.number]).columns
        filtered.loc[:, numeric_columns] = filtered[numeric_columns].round(digits)
        return filtered.to_dict(orient="records")

    dev_summary = challenge_only(dev_summary)
    benchmark_summary = challenge_only(benchmark_summary)
    cv_results = challenge_only(cv_results)
    cv_results_all = challenge_only(cv_results_all)
    strategy_by_horizon = challenge_only(strategy_by_horizon)
    validation_strata_summary = challenge_only(validation_strata_summary)
    test_aligned_scores = challenge_only(test_aligned_scores)
    prediction_diagnostics = challenge_only(prediction_diagnostics)
    prediction_diagnostics_by_origin = challenge_only(
        prediction_diagnostics_by_origin
    )
    per_product_summary = challenge_only(per_product_summary)
    top_decile_summary = challenge_only(top_decile_summary)
    top_error_rows = challenge_only(top_error_rows)
    probabilistic_summary = challenge_only(probabilistic_summary)
    audit_summary = challenge_only(audit_summary)

    if benchmark_summary is not None and not benchmark_summary.empty:
        summary = select_primary_summary(benchmark_summary).copy()
    elif not cv_results.empty:
        summary_source = cv_results
        if "regime" in summary_source.columns:
            conditional = summary_source[summary_source["regime"].eq("conditional")]
            summary_source = conditional if not conditional.empty else summary_source
        metric_columns = [
            column
            for column in ("MAE", "RMSE", "WAPE", "Bias", "BiasRatio")
            if column in summary_source.columns
        ]
        summary = (
            summary_source.groupby("model")[metric_columns]
            .mean(numeric_only=True)
            .reset_index()
        )
    else:
        summary = pd.DataFrame(columns=["model", "MAE", "RMSE", "WAPE", "Bias"])
    summary = challenge_only(summary)
    if not summary.empty:
        summary = order_models(summary)

    history: dict[str, dict] = {}
    for pid in sorted(train_raw["ProductId"].unique()):
        hist = (
            train_raw[train_raw["ProductId"].eq(pid)]
            .sort_values("DateKey")
            .tail(history_lookback)
        )
        history[str(int(pid))] = {
            "dates": hist["DateKey"].dt.strftime("%Y-%m-%d").tolist(),
            "quantity": hist["Quantity"].astype(float).tolist(),
        }

    test_keys = test_raw[["ProductId", "DateKey"]].reset_index(drop=True)
    forecasts: dict[str, dict] = {}
    for model_name in CHALLENGE_MODELS:
        if model_name not in final_forecasts:
            continue
        frame = test_keys.copy()
        frame["Quantity"] = np.asarray(final_forecasts[model_name], dtype=float)
        per_product: dict[str, dict] = {}
        for pid, product in frame.groupby("ProductId"):
            product = product.sort_values("DateKey")
            per_product[str(int(pid))] = {
                "dates": product["DateKey"].dt.strftime("%Y-%m-%d").tolist(),
                "quantity": product["Quantity"].astype(float).tolist(),
            }
        forecasts[model_name] = per_product

    normalized_forecasts_by_strategy: dict[str, dict] = {}
    for strategy, strategy_forecasts in (forecasts_by_strategy or {}).items():
        normalized_forecasts_by_strategy[str(strategy)] = {
            model: value
            for model, value in strategy_forecasts.items()
            if model in CHALLENGE_MODELS
        }
    if not normalized_forecasts_by_strategy:
        normalized_forecasts_by_strategy = {"direct": forecasts}

    available = set(forecasts)
    models_meta = [
        {
            "key": name,
            "slug": MODEL_SLUGS[name],
            "strategies": ["direct"],
            "available": name in available,
            **MODEL_META[name],
        }
        for name in CHALLENGE_MODELS
    ]

    selection_metric = (
        runtime_options.selection_metric if runtime_options else "WAPE"
    )

    def primary_rows(frame: pd.DataFrame | None) -> pd.DataFrame:
        frame = challenge_only(frame)
        if frame.empty:
            return frame
        mask = pd.Series(True, index=frame.index)
        if "evaluation_regime" in frame.columns:
            mask &= frame["evaluation_regime"].eq("conditional")
        if "comparison_population" in frame.columns:
            mask &= frame["comparison_population"].eq("common")
        if "aggregation" in frame.columns:
            mask &= frame["aggregation"].eq("global")
        if "strategy" in frame.columns:
            mask &= frame["strategy"].eq("direct")
        return frame[mask].copy()

    def split_result(frame: pd.DataFrame | None) -> dict:
        rows = primary_rows(frame)
        if rows.empty or selection_metric not in rows.columns:
            return {"winner": None, "rows": [], "chronos_relative_to_incumbent": None}
        winner = str(rows.sort_values(selection_metric).iloc[0]["model"])
        indexed = rows.set_index("model")
        relative = None
        if {"NeuralNet", "Chronos2"}.issubset(indexed.index):
            incumbent = float(indexed.loc["NeuralNet", selection_metric])
            challenger = float(indexed.loc["Chronos2", selection_metric])
            if np.isfinite(incumbent) and incumbent != 0 and np.isfinite(challenger):
                relative = round(challenger / incumbent - 1.0, 12)
        return {
            "winner": winner,
            "rows": records(rows),
            "chronos_relative_to_incumbent": relative,
        }

    development_result = split_result(dev_summary)
    benchmark_result = split_result(benchmark_summary)
    audit_result = split_result(audit_summary)
    challenge_complete = set(CHALLENGE_MODELS).issubset(available)
    probabilistic_evaluated = (
        probabilistic_summary is not None
        and not probabilistic_summary.empty
        and bool(final_quantile_forecasts)
    )

    payload = {
        "schema_version": "vonavy-chronos-v2",
        "generated_at": (
            provenance.get("generated_at")
            if provenance
            else datetime.now(timezone.utc).isoformat()
        ),
        "project": {
            "name": "vonavy_chronos",
            "title": "Best NN vs Chronos-2",
            "status": "complete" if challenge_complete else "awaiting_chronos",
        },
        "challenge": {
            "incumbent_model": "NeuralNet",
            "challenger_model": "Chronos2",
            "selection_metric": selection_metric,
            "selection_source": "development OOF only",
            "development": development_result,
            "recent_benchmark": benchmark_result,
            "final_audit": audit_result,
            "benchmark_confirms_development": (
                development_result["winner"] == benchmark_result["winner"]
                if challenge_complete
                and development_result["winner"] is not None
                and benchmark_result["winner"] is not None
                else None
            ),
            "lineage": [
                {
                    "step": "Evaluation contract",
                    "decision": "Conditional demand, common rows, global WAPE",
                    "reason": "Every candidate is scored on the same purchasable product-days.",
                },
                {
                    "step": "Forecast formulation",
                    "decision": "Direct seven-horizon panel",
                    "reason": "It beat the recursive route and avoids feedback compounding.",
                },
                {
                    "step": "Target design",
                    "decision": "Guarded log-residual around a 4:3:2:1 weekday anchor",
                    "reason": "The anchor carries stable weekly level; the network learns departures.",
                },
                {
                    "step": "Frozen incumbent",
                    "decision": "Three-seed PyTorch network with product and campaign embeddings",
                    "reason": "This was the strongest defensible non-tree specification before Chronos-2 was introduced.",
                },
            ],
        },
        "config": {
            "forecast_strategy": "direct",
            "primary_strategy": "direct",
            "submission_model": canonical_model,
            "selection_metric": selection_metric,
            "selection_protocol": (
                runtime_options.selection_protocol if runtime_options else "test-aligned"
            ),
            "primary_evaluation_regime": "conditional",
            "primary_comparison_population": "common",
            "primary_aggregation": "global",
            "horizon": cfg.horizon,
            "n_cv_folds": cfg.n_cv_folds,
            "n_dev_origins": len(DEVELOPMENT_ORIGINS),
            "cv_epochs": cfg.cv_epochs,
            "final_epochs": cfg.final_epochs,
            "seeds": list(cfg.seeds),
            "num_products": cfg.num_products,
            "validation_stratum_weights": VALIDATION_STRATUM_WEIGHTS,
            "nn_batch_size": cfg.batch_size,
            "nn_learning_rate": effective_learning_rate(cfg),
            "nn_training_backend": resolve_training_backend(cfg),
            "training_window_days": cfg.training_window_days,
            "recency_half_life_days": cfg.recency_half_life_days,
            "baseline_variant": cfg.baseline_variant,
            "c2_feature_groups": list(cfg.c2_feature_groups),
            "nn_loss": cfg.nn_loss,
            "nn_target_mode": cfg.nn_target_mode,
            "enable_channel_history_features": cfg.enable_channel_history_features,
            "channel_aux_weight": cfg.channel_aux_weight,
            "enable_chronos2": cfg.enable_chronos2,
            "chronos2_model_id": cfg.chronos2_model_id,
            "chronos2_model_revision": cfg.chronos2_model_revision,
            "chronos2_profile": (
                runtime_options.chronos2_profile if runtime_options else "published"
            ),
            "chronos2_device": cfg.chronos2_device,
            "chronos2_dtype": cfg.chronos2_dtype,
            "chronos2_batch_size": cfg.chronos2_batch_size,
            "chronos2_context_length": cfg.chronos2_context_length,
            "chronos2_cross_learning": cfg.chronos2_cross_learning,
            "chronos2_covariates": cfg.chronos2_covariates,
            "chronos2_quantile_levels": list(cfg.chronos2_quantile_levels),
        },
        "models": models_meta,
        "cv_results": records(cv_results, 3),
        "cv_results_all": records(
            cv_results_all if cv_results_all is not None else cv_results, 3
        ),
        "cv_summary": records(summary, 6),
        "benchmark_summary_all": records(benchmark_summary),
        "dev_summary_all": records(dev_summary),
        "benchmark_summary": records(primary_rows(benchmark_summary), 6),
        "dev_summary": records(primary_rows(dev_summary), 6),
        "submission": submission.assign(
            DateKey=pd.to_datetime(submission["DateKey"]).dt.strftime("%Y-%m-%d")
        ).to_dict(orient="records"),
        "history": history,
        "forecasts": forecasts,
        "forecasts_by_strategy": normalized_forecasts_by_strategy,
        "strategy_by_horizon": records(strategy_by_horizon),
        "validation_strata_summary": records(validation_strata_summary),
        "test_aligned_scores": records(test_aligned_scores),
        "prediction_diagnostics": records(prediction_diagnostics),
        "prediction_diagnostics_by_origin": records(
            prediction_diagnostics_by_origin
        ),
        "channel_share_summary": (
            channel_share_summary.round(6).to_dict(orient="records")
            if channel_share_summary is not None and not channel_share_summary.empty
            else []
        ),
        "per_product_summary": records(per_product_summary),
        "top_decile_summary": records(top_decile_summary),
        "top_error_rows": records(top_error_rows),
        "sanity_baseline": (
            sanity_baseline.round(6).to_dict(orient="records")
            if sanity_baseline is not None and not sanity_baseline.empty
            else []
        ),
        "weight_sensitivity": records(weight_sensitivity),
        "probabilistic_evaluation": {
            "status": "evaluated" if probabilistic_evaluated else "not_evaluated",
            "reason": (
                None
                if probabilistic_evaluated
                else "Complete q10/q50/q90 OOF metrics and final forecasts are required."
            ),
            "metrics": records(probabilistic_summary),
            "forecasts": final_quantile_forecasts or {},
        },
        "evaluation_origins": origin_registry or [],
        "provenance": provenance or {
            "status": "unknown",
            "reason": "This artifact predates immutable run provenance.",
        },
        "publication_provenance": publication_provenance or {
            "status": "not_published",
            "reason": "No authenticated static publication manifest is attached.",
        },
        "selection": {
            "canonical_model": canonical_model,
            "canonical_strategy": "direct",
            "selected_from": "development",
            "development_winner": development_result["winner"],
            "benchmark_winner": benchmark_result["winner"],
            "recent_benchmark_confirmation": (
                development_result["winner"] == benchmark_result["winner"]
                if challenge_complete
                and development_result["winner"] is not None
                and benchmark_result["winner"] is not None
                else None
            ),
            "final_audit_winner": audit_result["winner"],
        },
    }

    payload = _json_safe(payload)
    out_path = path or os.path.join(cfg.output_dir, "results.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp_path = f"{out_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    os.replace(tmp_path, out_path)
    print(f"Saved: {out_path}")
    return payload

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _choose_canonical_model_strategy(
    options: RuntimeOptions,
    dev_summary: pd.DataFrame,
    test_aligned_scores: pd.DataFrame | None = None,
) -> tuple[str, str]:
    """Select the head-to-head winner using development data only."""
    requested_model = options.submission_model.value
    available = set(dev_summary.get("model", pd.Series(dtype=str)).astype(str))

    if options.submission_model is not SubmissionModel.AUTO:
        if requested_model not in CHALLENGE_MODELS:
            raise RuntimeError(f"Unsupported challenge model: {requested_model}")
        if requested_model not in available:
            raise RuntimeError(
                f"No development OOF predictions are available for {requested_model}"
            )
        return requested_model, "direct"

    if options.selection_protocol == "test-aligned":
        candidates = (
            test_aligned_scores.copy()
            if test_aligned_scores is not None
            else pd.DataFrame()
        )
        candidates = candidates[candidates["model"].isin(CHALLENGE_MODELS)]
        if not candidates.empty:
            row = candidates.sort_values("test_aligned_score").iloc[0]
            return str(row["model"]), "direct"

    candidates = dev_summary[
        dev_summary["evaluation_regime"].eq("conditional")
        & dev_summary["comparison_population"].eq("common")
        & dev_summary["aggregation"].eq("global")
        & dev_summary["model"].isin(CHALLENGE_MODELS)
    ]
    if candidates.empty:
        raise RuntimeError("No comparable development rows for the two-model challenge")
    row = candidates.sort_values(options.selection_metric).iloc[0]
    return str(row["model"]), "direct"


def _forecast_dict_to_json(test_raw: pd.DataFrame, forecasts: dict) -> dict:
    keys = test_raw[["ProductId", "DateKey"]].reset_index(drop=True)
    result = {}
    for model, preds in forecasts.items():
        frame = keys.copy()
        frame["Quantity"] = np.asarray(preds, dtype=float)
        per_product = {}
        for pid, sub in frame.groupby("ProductId"):
            sub = sub.sort_values("DateKey")
            per_product[str(int(pid))] = {
                "dates": sub["DateKey"].dt.strftime("%Y-%m-%d").tolist(),
                "quantity": sub["Quantity"].tolist(),
            }
        result[model] = per_product
    return result


def _chronos_quantiles_to_json(final_forecasts: pd.DataFrame) -> dict:
    required = {
        "ProductId", "DateKey", "model",
        "prediction_q10", "prediction_q50", "prediction_q90",
    }
    if final_forecasts.empty or not required.issubset(final_forecasts.columns):
        return {}
    chronos = final_forecasts[final_forecasts["model"].eq("Chronos2")].copy()
    if chronos.empty or chronos[
        ["prediction_q10", "prediction_q50", "prediction_q90"]
    ].isna().any().any():
        return {}
    per_product = {}
    for product_id, product in chronos.groupby("ProductId", sort=True):
        product = product.sort_values("DateKey")
        per_product[str(int(product_id))] = {
            "dates": pd.to_datetime(product["DateKey"]).dt.strftime("%Y-%m-%d").tolist(),
            "q10": product["prediction_q10"].astype(float).tolist(),
            "q50": product["prediction_q50"].astype(float).tolist(),
            "q90": product["prediction_q90"].astype(float).tolist(),
        }
    return {"Chronos2": per_product}


def main(argv=None) -> None:
    options = parse_args(argv)
    validate_final_audit_policy(options)
    if options.forecast_strategy is not ForecastStrategy.DIRECT:
        raise RuntimeError("vonavy_chronos is a direct-vs-direct challenge")
    cfg = CFG
    c1_runtime = configure_c1_runtime(cfg, options)
    c2_runtime = configure_c2_runtime(cfg, options)
    c34_runtime = configure_c34_runtime(cfg, options)
    chronos2_runtime = configure_chronos2_runtime(cfg, options)
    nn_runtime = configure_nn_runtime(cfg, options)
    if options.run_kind in {"publication", "reproduction"}:
        # Canonical runs fail on OOM instead of silently changing estimators.
        cfg.nn_training_backend = nn_runtime["training_backend"]
        nn_runtime["fallback_policy"] = "disabled"

    if options.submission_model is SubmissionModel.CHRONOS2 and not cfg.enable_chronos2:
        raise RuntimeError("--submission-model Chronos2 requires --chronos2 on")

    print("Repository: vonavy_chronos")
    print("Challenge: Best NN (frozen direct NeuralNet) vs Amazon Chronos-2")
    print(f"Incumbent device: {DEVICE}")
    print(
        f"Selection: development OOF / {options.selection_protocol} / "
        f"{options.selection_metric}"
    )
    print(
        "Frozen incumbent: "
        f"window={c1_runtime['training_window_days'] or 'all'}, "
        f"half_life={c1_runtime['recency_half_life_days'] or 'none'}, "
        f"baseline={c1_runtime['baseline_variant']}, "
        f"features={','.join(c2_runtime['c2_feature_groups']) or 'core'}, "
        f"loss={c34_runtime['nn_loss']}, "
        f"target={c34_runtime['nn_target_mode']}"
    )
    print(
        "Chronos-2: "
        f"enabled={chronos2_runtime['enabled']}, "
        f"model={chronos2_runtime['model_id']}@"
        f"{chronos2_runtime['model_revision'][:12]}, "
        f"profile={chronos2_runtime['profile']}, "
        f"device={chronos2_runtime['resolved_device']}, "
        f"covariates={chronos2_runtime['covariates']}, "
        f"cross_learning={chronos2_runtime['cross_learning']}"
    )
    print(
        "Incumbent runtime: "
        f"batch={nn_runtime['batch_size']} ({nn_runtime['batch_source']}), "
        f"lr={nn_runtime['effective_learning_rate']:.6g}, "
        f"backend={nn_runtime['training_backend']}"
    )

    run_start = time.perf_counter()
    timings: dict = {
        "cv_folds": [],
        "nn_runtime": nn_runtime,
        "c1_runtime": c1_runtime,
        "c2_runtime": c2_runtime,
        "c34_runtime": c34_runtime,
        "chronos2_runtime": chronos2_runtime,
    }

    train_raw, test_raw = load_raw(cfg)
    cfg.num_products = int(max(train_raw["ProductId"].max(), test_raw["ProductId"].max()))
    benchmark_origins = recent_benchmark_origins(train_raw, cfg)
    origin_registry = evaluation_origin_registry(
        benchmark_origins,
        include_final_audit=options.include_final_audit,
        run_kind=options.run_kind,
    )
    option_config = {
        key: (value.value if isinstance(value, Enum) else value)
        for key, value in asdict(options).items()
    }
    repository_root = Path(__file__).resolve().parents[1]
    provenance = build_run_provenance(
        repository_root=repository_root,
        command=["python", "ml/pipeline.py", *(sys.argv[1:] if argv is None else argv)],
        run_kind=options.run_kind,
        config={
            "config": asdict(cfg),
            "runtime_options": option_config,
            "evaluation_origins": origin_registry,
        },
        input_paths=[cfg.train_path, cfg.test_path],
        lock_path="requirements-chronos.txt",
        model_id=cfg.chronos2_model_id,
        model_revision=cfg.chronos2_model_revision,
        resolved_device=chronos2_runtime["resolved_device"],
    )
    canonical_output = not provenance["source"]["dirty"]
    effective_checkpoint_dir = options.checkpoint_dir
    effective_resume = options.resume
    if canonical_output:
        provenance["scope"] = "canonical"
        provenance["run_manifest"] = f"outputs/runs/{provenance['run_id']}.json"
    else:
        cfg.output_dir = f"outputs/experiments/{provenance['run_id']}"
        effective_checkpoint_dir = None
        effective_resume = False
        provenance["scope"] = "noncanonical_dirty_experiment"
        provenance["canonical_outputs_written"] = False
        provenance["run_manifest"] = f"{cfg.output_dir}/run.json"
        print(
            "Dirty development source detected: writing only to "
            f"{cfg.output_dir}; canonical caches/static outputs are disabled."
        )
    provenance["output_hashes"] = {
        "status": "recorded_in_run_manifest",
        "manifest": provenance["run_manifest"],
    }
    if (
        effective_checkpoint_dir
        and options.reset_checkpoints
        and os.path.exists(effective_checkpoint_dir)
    ):
        shutil.rmtree(effective_checkpoint_dir)
        print(f"Removed checkpoints: {effective_checkpoint_dir}")
    if effective_resume:
        print(f"CV resume enabled: {effective_checkpoint_dir}")
    audit_reservation = reserve_final_audit(
        options,
        run_id=provenance["run_id"],
        source_revision=provenance["source"]["revision"],
        generated_at=provenance["generated_at"],
        marker_path=str(repository_root / FINAL_AUDIT_MARKER),
    )
    if audit_reservation is not None:
        provenance["final_audit_consumption"] = audit_reservation
    checkpoint_identity = build_checkpoint_run_identity(
        provenance, chronos2_runtime, nn_runtime
    )
    checkpoint_trust = (
        load_authenticated_checkpoint_index(
            repository_root, options.checkpoint_trust_publication
        )
        if canonical_output
        else {
            "status": "disabled_noncanonical_dirty_experiment",
            "checkpoint_sha256": {},
        }
    )
    provenance["checkpoint_trust"] = {
        key: value
        for key, value in checkpoint_trust.items()
        if key != "checkpoint_sha256"
    }
    trusted_checkpoint_hashes = checkpoint_trust["checkpoint_sha256"]

    print("\n=== Development head-to-head CV ===")
    dev_oof = run_walk_forward_cv_direct(
        train_raw,
        DEVELOPMENT_ORIGINS,
        "development",
        cfg,
        timings=timings["cv_folds"],
        checkpoint_dir=effective_checkpoint_dir,
        resume=effective_resume,
        checkpoint_identity=checkpoint_identity,
        trusted_checkpoint_hashes=trusted_checkpoint_hashes,
    )
    print("\n=== Previously inspected recent diagnostic ===")
    benchmark_oof = run_walk_forward_cv_direct(
        train_raw,
        benchmark_origins,
        "recent_benchmark",
        cfg,
        timings=timings["cv_folds"],
        checkpoint_dir=effective_checkpoint_dir,
        resume=effective_resume,
        checkpoint_identity=checkpoint_identity,
        trusted_checkpoint_hashes=trusted_checkpoint_hashes,
    )

    audit_oof = pd.DataFrame()
    if options.include_final_audit:
        print("\n=== Final audit (non-selection evidence) ===")
        audit_oof = run_walk_forward_cv_direct(
            train_raw,
            FINAL_AUDIT_ORIGINS,
            "final_audit",
            cfg,
            timings=timings["cv_folds"],
            checkpoint_dir=effective_checkpoint_dir,
            resume=effective_resume,
            checkpoint_identity=checkpoint_identity,
            trusted_checkpoint_hashes=trusted_checkpoint_hashes,
        )

    checkpoint_paths = sorted({
        Path(timing["checkpoint_path"])
        for timing in timings["cv_folds"]
        if timing.get("checkpoint_path")
    })
    provenance["checkpoints"] = {
        "status": "authenticated",
        "identity": checkpoint_identity,
        "files_sha256": output_hashes(repository_root, checkpoint_paths),
        "consumed_sha256": {
            str(Path(timing["consumed_checkpoint_path"])): timing[
                "consumed_checkpoint_sha256"
            ]
            for timing in timings["cv_folds"]
            if timing.get("consumed_checkpoint_sha256")
        },
        "reused_folds": sum(
            timing.get("checkpoint_source") == "reused"
            for timing in timings["cv_folds"]
        ),
        "trained_folds": sum(
            timing.get("checkpoint_source") == "trained"
            for timing in timings["cv_folds"]
        ),
    }

    oof_frames = [dev_oof, benchmark_oof]
    if not audit_oof.empty:
        oof_frames.append(audit_oof)
    oof = pd.concat(oof_frames, ignore_index=True)
    dev_summary = summarize_oof_by_strategy(dev_oof, OOF_MODEL_COLUMNS)
    benchmark_summary = summarize_oof_by_strategy(benchmark_oof, OOF_MODEL_COLUMNS)
    audit_summary = summarize_oof_by_strategy(audit_oof, OOF_MODEL_COLUMNS)
    validation_strata_summary = summarize_validation_strata(oof, OOF_MODEL_COLUMNS)
    test_aligned_scores = compute_test_aligned_scores(
        validation_strata_summary, metric=options.selection_metric
    )
    prediction_diagnostics = summarize_prediction_diagnostics(oof, OOF_MODEL_COLUMNS)
    prediction_diagnostics_by_origin = summarize_prediction_diagnostics_by_origin(
        oof, OOF_MODEL_COLUMNS
    )
    channel_share_summary = summarize_channel_share_oof(oof)
    per_product_summary = summarize_per_product_oof(oof, OOF_MODEL_COLUMNS)
    top_decile_summary, top_error_rows = summarize_top_deciles(
        oof, OOF_MODEL_COLUMNS
    )
    sanity_baseline = summarize_sanity_baseline(oof, OOF_MODEL_COLUMNS)
    probabilistic_summary = summarize_probabilistic_oof(oof)
    weight_sensitivity = summarize_weight_sensitivity(
        validation_strata_summary,
        dev_summary,
        metric=options.selection_metric,
    )

    canonical_model, canonical_strategy = _choose_canonical_model_strategy(
        options, dev_summary, test_aligned_scores
    )
    print(f"\nDevelopment-selected challenge winner: {canonical_model}")

    print("\n=== Final seven-day forecasts ===")
    _, nn_preds, nn_details = run_final_forecast_direct(
        train_raw, test_raw, cfg, return_diagnostics=True
    )
    forecasts: dict[str, np.ndarray] = {"NeuralNet": nn_preds}
    chronos2_details = None
    if cfg.enable_chronos2:
        chronos2_preds, chronos2_details = run_final_chronos2_forecast_direct(
            train_raw, test_raw, cfg
        )
        forecasts["Chronos2"] = chronos2_preds

    if canonical_model not in forecasts:
        raise RuntimeError(f"Final forecast missing selected model {canonical_model}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    test_keys = test_raw[["ProductId", "DateKey"]].reset_index(drop=True)

    def write_submission(model: str, filename_stem: str) -> pd.DataFrame:
        frame = test_keys.copy()
        frame["Quantity"] = np.round(
            np.clip(np.asarray(forecasts[model], dtype=float), 0, None)
        ).astype(int)
        frame.to_csv(os.path.join(cfg.output_dir, f"{filename_stem}.csv"), index=False)
        frame.to_parquet(
            os.path.join(cfg.output_dir, f"{filename_stem}.parquet"), index=False
        )
        return frame

    incumbent_submission = write_submission("NeuralNet", "submission_best_nn")
    chronos_submission = (
        write_submission("Chronos2", "submission_chronos2")
        if "Chronos2" in forecasts
        else None
    )
    submission = write_submission(canonical_model, "submission")

    raw_rows: list[dict] = []
    for model, predictions in forecasts.items():
        for row_index, ((pid, date), pred) in enumerate(
            zip(
                test_keys.itertuples(index=False, name=None),
                np.asarray(predictions, dtype=float),
            )
        ):
            chronos_row = (
                chronos2_details.iloc[row_index]
                if model == "Chronos2" and chronos2_details is not None
                else None
            )
            record = {
                "strategy": "direct",
                "model": model,
                "ProductId": pid,
                "DateKey": date,
                "prediction_raw": float(pred),
                "prediction_submission": int(round(max(float(pred), 0.0))),
                "fallback_used": bool(chronos_row["fallback_used"]) if chronos_row is not None else False,
                "nonfinite_raw": bool(chronos_row["nonfinite_raw"]) if chronos_row is not None else False,
                "no_context": bool(chronos_row.get("no_context", False)) if chronos_row is not None else False,
                "catastrophic_guard": False,
                "residual_guard": False,
                "residual_nonfinite": False,
                "residual_raw_min": np.nan,
                "residual_raw_max": np.nan,
                "safety_limit": np.nan,
                "predicted_app_share": (
                    float(nn_details.get("app_share", np.full(len(test_raw), np.nan))[row_index])
                    if model == "NeuralNet"
                    else np.nan
                ),
                "prediction_app": (
                    float(nn_details.get("prediction_app", np.full(len(test_raw), np.nan))[row_index])
                    if model == "NeuralNet"
                    else np.nan
                ),
                "prediction_web": (
                    float(nn_details.get("prediction_web", np.full(len(test_raw), np.nan))[row_index])
                    if model == "NeuralNet"
                    else np.nan
                ),
            }
            if chronos_row is not None:
                for quantile in cfg.chronos2_quantile_levels:
                    record[f"prediction_{chronos2_quantile_suffix(quantile)}"] = float(
                        chronos_row.get(f"quantile_{quantile:g}", np.nan)
                    )
            raw_rows.append(record)

    final_forecast_df = pd.DataFrame(raw_rows)
    final_forecast_df.to_parquet(
        os.path.join(cfg.output_dir, "final_forecasts.parquet"), index=False
    )
    oof.to_parquet(os.path.join(cfg.output_dir, "oof_predictions.parquet"), index=False)
    dev_summary.to_csv(os.path.join(cfg.output_dir, "dev_summary.csv"), index=False)
    benchmark_summary.to_csv(
        os.path.join(cfg.output_dir, "benchmark_summary.csv"), index=False
    )
    validation_strata_summary.to_csv(
        os.path.join(cfg.output_dir, "validation_strata_summary.csv"), index=False
    )
    test_aligned_scores.to_csv(
        os.path.join(cfg.output_dir, "test_aligned_scores.csv"), index=False
    )
    prediction_diagnostics.to_csv(
        os.path.join(cfg.output_dir, "prediction_diagnostics.csv"), index=False
    )
    prediction_diagnostics_by_origin.to_csv(
        os.path.join(cfg.output_dir, "prediction_diagnostics_by_origin.csv"),
        index=False,
    )
    channel_share_summary.to_csv(
        os.path.join(cfg.output_dir, "channel_share_summary.csv"), index=False
    )
    per_product_summary.to_csv(
        os.path.join(cfg.output_dir, "per_product_summary.csv"), index=False
    )
    top_decile_summary.to_csv(
        os.path.join(cfg.output_dir, "top_decile_summary.csv"), index=False
    )
    top_error_rows.to_csv(
        os.path.join(cfg.output_dir, "top_error_rows.csv"), index=False
    )
    sanity_baseline.to_csv(
        os.path.join(cfg.output_dir, "sanity_baseline.csv"), index=False
    )
    probabilistic_summary.to_csv(
        os.path.join(cfg.output_dir, "probabilistic_summary.csv"), index=False
    )
    weight_sensitivity.to_csv(
        os.path.join(cfg.output_dir, "weight_sensitivity.csv"), index=False
    )
    audit_summary.to_csv(
        os.path.join(cfg.output_dir, "final_audit_summary.csv"), index=False
    )

    by_horizon_frames: list[pd.DataFrame] = []
    for (origin_type, strategy), group in oof.groupby(
        ["origin_type", "strategy"], sort=False
    ):
        for horizon, hgroup in group.groupby("horizon"):
            summary = summarize_oof(hgroup, OOF_MODEL_COLUMNS)
            summary["strategy"] = strategy
            summary["horizon"] = horizon
            summary["origin_type"] = origin_type
            by_horizon_frames.append(summary)
    strategy_by_horizon = (
        pd.concat(by_horizon_frames, ignore_index=True)
        if by_horizon_frames
        else pd.DataFrame()
    )
    strategy_by_horizon.to_csv(
        os.path.join(cfg.output_dir, "strategy_by_horizon.csv"), index=False
    )

    cv_results = oof_to_legacy_cv_results(benchmark_oof, OOF_MODEL_COLUMNS)
    cv_results_all = cv_results.assign(strategy="direct")
    cv_results.to_csv(os.path.join(cfg.output_dir, "cv_results.csv"), index=False)
    cv_results_all.to_csv(
        os.path.join(cfg.output_dir, "cv_results_all.csv"), index=False
    )

    # A compact, presentation-friendly comparison artifact.
    comparison_rows: list[pd.DataFrame] = []
    for split_name, summary in (
        ("development", dev_summary),
        ("recent_benchmark", benchmark_summary),
        ("final_audit", audit_summary),
    ):
        if summary.empty:
            continue
        primary = summary[
            summary["evaluation_regime"].eq("conditional")
            & summary["comparison_population"].eq("common")
            & summary["aggregation"].eq("global")
            & summary["model"].isin(CHALLENGE_MODELS)
        ].copy()
        primary.insert(0, "split", split_name)
        comparison_rows.append(primary)
    challenge_comparison = pd.concat(comparison_rows, ignore_index=True)
    challenge_comparison.to_csv(
        os.path.join(cfg.output_dir, "challenge_comparison.csv"), index=False
    )

    forecasts_by_strategy = {
        "direct": _forecast_dict_to_json(test_raw, forecasts)
    }
    final_quantile_forecasts = _chronos_quantiles_to_json(final_forecast_df)
    payload = export_results_json(
        train_raw,
        test_raw,
        submission,
        forecasts,
        cv_results,
        cfg,
        dev_summary=dev_summary,
        benchmark_summary=benchmark_summary,
        runtime_options=options,
        forecasts_by_strategy=forecasts_by_strategy,
        canonical_strategy="direct",
        canonical_model=canonical_model,
        cv_results_all=cv_results_all,
        strategy_by_horizon=strategy_by_horizon,
        validation_strata_summary=validation_strata_summary,
        test_aligned_scores=test_aligned_scores,
        prediction_diagnostics=prediction_diagnostics,
        prediction_diagnostics_by_origin=prediction_diagnostics_by_origin,
        channel_share_summary=channel_share_summary,
        per_product_summary=per_product_summary,
        top_decile_summary=top_decile_summary,
        top_error_rows=top_error_rows,
        sanity_baseline=sanity_baseline,
        probabilistic_summary=probabilistic_summary,
        weight_sensitivity=weight_sensitivity,
        audit_summary=audit_summary,
        final_quantile_forecasts=final_quantile_forecasts,
        origin_registry=origin_registry,
        provenance=provenance,
    )
    if canonical_output:
        publish_static_dashboard(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            os.path.join(cfg.output_dir, "results.json"),
        )

    if options.plot:
        plot_forecast(train_raw, submission, cfg=cfg)

    timings["total_seconds"] = round(time.perf_counter() - run_start, 2)
    timings["winner"] = canonical_model
    timings["artifacts"] = {
        "incumbent_submission": "submission_best_nn.csv",
        "chronos_submission": (
            "submission_chronos2.csv" if chronos_submission is not None else None
        ),
        "selected_submission": "submission.csv",
    }
    write_json_atomic(os.path.join(cfg.output_dir, "timings.json"), _json_safe(timings))

    output_paths = model_output_artifact_paths(
        repository_root,
        cfg.output_dir,
        include_plot=options.plot,
    )
    hashes = output_hashes(repository_root, output_paths)
    run_record = {**provenance, "output_sha256": hashes}
    run_record_path = repository_root / provenance["run_manifest"]
    write_json_atomic(run_record_path, _json_safe(run_record))
    checksums = "\n".join(
        f"{digest}  {path}" for path, digest in hashes.items()
    ) + "\n"
    checksum_path = (
        repository_root / "outputs" / "SHA256SUMS"
        if canonical_output
        else repository_root / cfg.output_dir / "SHA256SUMS"
    )
    checksum_path.write_text(
        checksums, encoding="utf-8"
    )
    print(f"\nSaved challenge winner submission: {canonical_model} / direct")
    print(f"Saved incumbent forecast: {cfg.output_dir}/submission_best_nn.csv")
    if chronos_submission is not None:
        print(f"Saved Chronos-2 forecast: {cfg.output_dir}/submission_chronos2.csv")
    print(f"Total runtime: {timings['total_seconds'] / 60:.1f} min")

if __name__ == "__main__":
    main()
