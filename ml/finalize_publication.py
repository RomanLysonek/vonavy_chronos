"""Create or verify post-run publication provenance without model inference."""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from provenance import (
    model_provenance_summary,
    output_hashes,
    sha256_file,
    sha256_json,
    write_json_atomic,
)
from static_site import publish_static_dashboard


MODEL_ARTIFACT_PATHS = (
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
COMPACT_OUTPUTS = (
    "outputs/README.md",
    "outputs/results.json",
    "outputs/challenge_comparison.csv",
    "outputs/final_audit_consumed.json",
    "outputs/final_audit_summary.csv",
    "outputs/probabilistic_summary.csv",
    "outputs/sanity_baseline.csv",
    "outputs/weight_sensitivity.csv",
)


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _source_paths(root: Path) -> list[Path]:
    explicit = [
        root / ".gitignore",
        root / "README.md",
        root / "pyproject.toml",
        root / "uv.lock",
        root / "requirements-chronos.in",
        root / "requirements-chronos.txt",
    ]
    patterns = (
        ".github/workflows/*",
        "ml/**/*.py",
        "ml/**/*.json",
        "tests/**/*",
        "webapp/server.py",
        "webapp/static/*",
    )
    paths = {path.resolve() for path in explicit if path.is_file()}
    for pattern in patterns:
        paths.update(
            path.resolve()
            for path in root.glob(pattern)
            if path.is_file()
            and path.name != "results.json"
            and "__pycache__" not in path.parts
        )
    return sorted(paths)


def _publication_artifact_paths(root: Path, model_manifest_path: Path) -> list[Path]:
    paths = [root / relative for relative in COMPACT_OUTPUTS]
    paths.append(model_manifest_path)
    for base in (root / "webapp" / "static", root / "docs"):
        paths.extend(path for path in base.rglob("*") if path.is_file())
    return sorted({path.resolve() for path in paths})


def _git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _migrate_audit_marker(path: Path) -> None:
    marker = _read_json(path)
    if marker.get("schema_version") == "final-audit-consumption-v2":
        return
    consumed_at = marker.get("consumed_at")
    migrated = {
        "schema_version": "final-audit-consumption-v2",
        "status": "completed",
        "run_id": marker["run_id"],
        "source_revision": marker["source_revision"],
        "reserved_at": consumed_at,
        "completed_at": consumed_at,
        "fresh_evidence_claim_consumed": True,
        "reservation_mode": "legacy_marker_migrated_without_rerun",
        "origins": marker["origins"],
    }
    write_json_atomic(path, migrated)


def checkpoint_metadata(root: Path, model_manifest: dict) -> tuple[dict, dict]:
    authenticated = model_manifest.get("checkpoints")
    if isinstance(authenticated, dict) and authenticated.get("status") == "authenticated":
        return (
            {
                "status": "complete",
                "reason": (
                    "Checkpoint identities, reuse counts, and file hashes were "
                    "bound at run time."
                ),
                "checkpoint_identity_at_run": True,
            },
            authenticated,
        )
    is_known_legacy_run = model_manifest.get("run_id") == "941bbd3a1dd0cf23"
    checkpoints = sorted((root / "outputs" / "checkpoints").rglob("*.pkl"))
    return (
        {
            "status": "incomplete",
            "reason": (
                "The run enabled --resume before checkpoint source/input/dependency/"
                "device identities and checkpoint hashes were bound into the immutable "
                "manifest. The local execution log reports zero reused folds and 19 "
                "trained folds, but that observation was not authenticated at run time."
                if is_known_legacy_run
                else
                "This legacy run did not bind checkpoint identities, reuse counts, "
                "and checkpoint hashes into its immutable run record."
            ),
            "pinned_components": {
                "source_revision": True,
                "input_hashes": True,
                "dependency_lock_hash": True,
                "chronos_package_and_model_revision": True,
            },
            "checkpoint_identity_at_run": False,
        },
        {
            "status": "post_run_observation_not_immutable_at_run",
            "observed_reused_folds": 0 if is_known_legacy_run else None,
            "observed_trained_folds": 19 if is_known_legacy_run else None,
            "post_run_files_sha256": output_hashes(root, checkpoints),
        },
    )


def finalize(root: Path) -> dict:
    results_path = root / "outputs" / "results.json"
    results = _read_json(results_path)
    model_relative = results["provenance"]["run_manifest"]
    model_manifest_path = (root / model_relative).resolve()
    model_manifest = _read_json(model_manifest_path)
    legacy_hashes = (
        model_manifest.get("model_artifact_sha256")
        or model_manifest.get("output_sha256")
        or {}
    )
    missing = [path for path in MODEL_ARTIFACT_PATHS if path not in legacy_hashes]
    if missing:
        raise ValueError(f"Existing model manifest lacks artifact hashes: {missing}")
    model_manifest["schema_version"] = "chronos-run-v2-migrated"
    model_manifest["scope"] = "model_evidence"
    model_manifest["model_artifact_sha256"] = {
        path: legacy_hashes[path] for path in MODEL_ARTIFACT_PATHS
    }
    verification, checkpoint_provenance = checkpoint_metadata(root, model_manifest)
    model_manifest["verification"] = verification
    model_manifest["checkpoint_provenance"] = checkpoint_provenance
    model_manifest["output_hashes"] = {
        "status": "post_run_migrated",
        "scope": "model_artifacts",
        "reason": (
            "Artifact hashes authenticate the retained files now but were not "
            "all committed immutably at model-run completion."
        ),
    }
    model_manifest.pop("output_sha256", None)
    write_json_atomic(model_manifest_path, model_manifest)

    marker_path = root / "outputs" / "final_audit_consumed.json"
    _migrate_audit_marker(marker_path)

    source_hashes = output_hashes(root, _source_paths(root))
    source_content_sha = sha256_json(source_hashes)
    publication_id = sha256_json({
        "model_run_id": model_manifest["run_id"],
        "source_content_sha256": source_content_sha,
    })[:16]
    publication_relative = f"outputs/publications/{publication_id}.json"
    generated_at = datetime.now(timezone.utc).isoformat()
    publication_provenance = {
        "schema_version": "chronos-publication-v1",
        "status": "authenticated",
        "publication_id": publication_id,
        "generated_at": generated_at,
        "model_run_id": model_manifest["run_id"],
        "manifest": publication_relative,
        "source": {
            "status": "content_addressed_worktree",
            "base_revision": _git_head(root),
            "content_sha256": source_content_sha,
            "reason": (
                "Publication-only hardening occurred after model inference. "
                "These file hashes bind the exact publication source without "
                "claiming that the later source produced the model forecasts."
            ),
        },
    }
    model_manifest_sha = sha256_file(model_manifest_path)
    results["provenance"] = model_provenance_summary(
        model_manifest,
        manifest_path=model_relative,
        manifest_sha256=model_manifest_sha,
        authenticated_artifact_count=len(MODEL_ARTIFACT_PATHS) + 2,
    )
    results["publication_provenance"] = publication_provenance
    write_json_atomic(results_path, results)
    publish_static_dashboard(root, results_path)

    artifact_hashes = output_hashes(
        root, _publication_artifact_paths(root, model_manifest_path)
    )
    publication_manifest = {
        **publication_provenance,
        "source_files_sha256": source_hashes,
        "artifact_sha256": artifact_hashes,
    }
    publication_manifest_path = root / publication_relative
    write_json_atomic(publication_manifest_path, publication_manifest)
    for stale in publication_manifest_path.parent.glob("*.json"):
        if stale != publication_manifest_path:
            stale.unlink()
    checksum_hashes = {
        **artifact_hashes,
        publication_relative: sha256_file(publication_manifest_path),
    }
    (root / "outputs" / "SHA256SUMS").write_text(
        "".join(
            f"{digest}  {path}\n"
            for path, digest in sorted(checksum_hashes.items())
        ),
        encoding="utf-8",
    )
    return publication_manifest


def check(root: Path) -> None:
    results = _read_json(root / "outputs" / "results.json")
    publication = results.get("publication_provenance")
    if not isinstance(publication, dict) or not publication.get("manifest"):
        raise ValueError("results.json has no publication manifest")
    manifest = _read_json(root / publication["manifest"])
    if manifest.get("publication_id") != publication.get("publication_id"):
        raise ValueError("Publication IDs differ")
    for relative, expected in manifest["source_files_sha256"].items():
        path = root / relative
        if not path.exists() or sha256_file(path) != expected:
            raise ValueError(f"Publication source mismatch: {relative}")
    for relative, expected in manifest["artifact_sha256"].items():
        path = root / relative
        if not path.exists() or sha256_file(path) != expected:
            raise ValueError(f"Publication artifact mismatch: {relative}")
    for line in (root / "outputs" / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.exists() or sha256_file(path) != expected:
            raise ValueError(f"Checksum mismatch: {relative}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    if args.check:
        check(root)
        print("Publication provenance and committed checksums are valid.")
    else:
        manifest = finalize(root)
        print(f"Finalized publication {manifest['publication_id']} without inference.")


if __name__ == "__main__":
    main()
