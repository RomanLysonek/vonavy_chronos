"""Immutable run provenance for published forecasting evidence."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: str | Path, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp-{os.getpid()}")
    encoded = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    _fsync_directory(destination.parent)


def write_json_exclusive(path: str | Path, payload: dict) -> None:
    """Durably create a JSON file exactly once, refusing an existing path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(destination, flags, 0o644)
    except FileExistsError:
        raise FileExistsError(f"Exclusive provenance file already exists: {destination}") from None
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            destination.unlink()
        except FileNotFoundError:
            pass
        raise
    _fsync_directory(destination.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def source_identity(root: str | Path) -> dict:
    repository = Path(root).resolve()
    revision = _git(repository, "rev-parse", "HEAD")
    tree = _git(repository, "rev-parse", "HEAD^{tree}")
    dirty_lines = _git(
        repository, "status", "--porcelain", "--untracked-files=all"
    ).splitlines()
    return {
        "revision": revision,
        "tree": tree,
        "dirty": bool(dirty_lines),
        "dirty_paths": [line[3:] for line in dirty_lines],
    }


def package_version(name: str) -> dict:
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return {"status": "unknown", "value": None, "reason": "not installed"}
    return {"status": "known", "value": version}


def runtime_identity(resolved_device: str) -> dict:
    torch_details: dict[str, Any] = {
        "package": package_version("torch"),
        "device": resolved_device,
    }
    try:
        import torch
    except ImportError:
        torch_details["backend"] = {
            "status": "unknown",
            "value": None,
            "reason": "torch not installed",
        }
    else:
        torch_details["backend"] = {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "mps_available": bool(
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ),
        }
    processor = platform.processor() or None
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "processor": {
            "status": "known" if processor else "unknown",
            "value": processor,
            **({} if processor else {"reason": "platform did not report a processor"}),
        },
        "torch": torch_details,
    }


def build_run_provenance(
    *,
    repository_root: str | Path,
    command: Iterable[str],
    run_kind: str,
    config: dict,
    input_paths: Iterable[str | Path],
    lock_path: str | Path,
    model_id: str,
    model_revision: str,
    resolved_device: str,
) -> dict:
    root = Path(repository_root).resolve()
    source = source_identity(root)
    if run_kind in {"publication", "reproduction"} and source["dirty"]:
        raise RuntimeError(
            "Canonical publication/reproduction requires a clean source tree, "
            "including untracked files; dirty paths: "
            + ", ".join(source["dirty_paths"])
        )
    inputs = {
        str(Path(path).resolve().relative_to(root)): sha256_file(path)
        for path in input_paths
    }
    lock = Path(lock_path).resolve()
    if not lock.exists():
        raise FileNotFoundError(lock)
    generated_at = datetime.now(timezone.utc).isoformat()
    identity = {
        "schema_version": "chronos-run-v1",
        "generated_at": generated_at,
        "run_kind": run_kind,
        "source": source,
        "inputs_sha256": inputs,
        "config_sha256": sha256_json(config),
        "lock": {
            "path": str(lock.relative_to(root)),
            "sha256": sha256_file(lock),
        },
        "command": list(command),
        "chronos": {
            "package": package_version("chronos-forecasting"),
            "model_id": model_id,
            "model_revision": model_revision,
        },
        "runtime": runtime_identity(resolved_device),
    }
    identity["run_id"] = sha256_json(identity)[:16]
    return identity


def output_hashes(
    repository_root: str | Path,
    paths: Iterable[str | Path],
) -> dict[str, str]:
    root = Path(repository_root).resolve()
    result = {}
    for path in paths:
        candidate = Path(path).resolve()
        if candidate.exists() and candidate.is_file():
            result[str(candidate.relative_to(root))] = sha256_file(candidate)
    return dict(sorted(result.items()))


def model_provenance_summary(
    manifest: dict,
    *,
    manifest_path: str,
    manifest_sha256: str,
    authenticated_artifact_count: int,
) -> dict:
    summary = {
        key: value
        for key, value in manifest.items()
        if key not in {
            "output_sha256",
            "legacy_output_sha256",
            "model_artifact_sha256",
        }
    }
    summary["artifact_authentication"] = {
        "status": "verified",
        "manifest": manifest_path,
        "manifest_sha256": manifest_sha256,
        "authenticated_artifact_count": authenticated_artifact_count,
    }
    return summary
