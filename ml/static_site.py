"""Standard-library-only static dashboard publication and parity checks."""
from __future__ import annotations

import shutil
from pathlib import Path


DASHBOARD_SCHEMA_VERSION = "vonavy-chronos-v2"
GENERATED_README = (
    "# Static Notino interview dashboard\n\n"
    "Generated from `webapp/static/` and `outputs/results.json`. "
    "Do not edit `docs/` directly.\n"
)


def _static_html(source: str) -> str:
    result = source.replace("/static/", "./")
    result = result.replace('href="/dataset"', 'href="./dataset.html"')
    result = result.replace('href="/evaluation"', 'href="./evaluation.html"')
    result = result.replace('href="/"', 'href="./index.html"')
    marker = '<script src="./common.js'
    if marker in result and "window.STATIC_DASHBOARD" not in result:
        result = result.replace(
            marker,
            '<script>window.STATIC_DASHBOARD = true;</script>\n  ' + marker,
            1,
        )
    return result


def _expected_files(root: Path) -> dict[str, bytes]:
    static_dir = root / "webapp" / "static"
    expected: dict[str, bytes] = {}
    for source in static_dir.iterdir():
        if not source.is_file():
            continue
        if source.suffix.lower() == ".html":
            expected[source.name] = _static_html(
                source.read_text(encoding="utf-8")
            ).encode("utf-8")
        else:
            expected[source.name] = source.read_bytes()
    expected[".nojekyll"] = b""
    expected["README.md"] = GENERATED_README.encode("utf-8")
    return expected


def publish_static_dashboard(
    repository_root: str | Path,
    results_path: str | Path,
) -> dict:
    root = Path(repository_root).resolve()
    results = Path(results_path).resolve()
    static_dir = root / "webapp" / "static"
    docs_dir = root / "docs"
    if not results.exists():
        raise FileNotFoundError(results)
    static_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(results, static_dir / "results.json")
    expected = _expected_files(root)
    if docs_dir.exists():
        shutil.rmtree(docs_dir)
    docs_dir.mkdir(parents=True)
    for name, content in expected.items():
        (docs_dir / name).write_bytes(content)
    return {
        "schema_version": DASHBOARD_SCHEMA_VERSION,
        "source": "webapp/static",
        "results": str(results.relative_to(root)),
        "entrypoint": "docs/index.html",
    }


def check_static_dashboard(repository_root: str | Path) -> None:
    root = Path(repository_root).resolve()
    canonical_results = root / "outputs" / "results.json"
    static_results = root / "webapp" / "static" / "results.json"
    if canonical_results.read_bytes() != static_results.read_bytes():
        raise RuntimeError("webapp/static/results.json differs from outputs/results.json")
    expected = _expected_files(root)
    docs_dir = root / "docs"
    actual_names = {
        path.name for path in docs_dir.iterdir() if path.is_file()
    }
    if actual_names != set(expected):
        missing = sorted(set(expected) - actual_names)
        extra = sorted(actual_names - set(expected))
        raise RuntimeError(f"Generated docs file set differs: missing={missing}, extra={extra}")
    mismatches = [
        name for name, content in expected.items()
        if (docs_dir / name).read_bytes() != content
    ]
    if mismatches:
        raise RuntimeError(f"Generated docs differ from authored source: {mismatches}")
