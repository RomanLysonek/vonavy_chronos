import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from framework import CHALLENGE_MODELS, MODEL_ORDER, MODEL_STRATEGY_SUPPORT, Config
from pipeline import (
    ForecastStrategy,
    RuntimeOptions,
    SubmissionModel,
    export_results_json,
    parse_args,
    resolve_strategies,
)


PROMO_FACTS = (
    "30 Product Time Series",
    "7-Day Direct Forecast",
    "2 Contenders",
    "Same Walk-Forward Test",
)
CHROME_CANDIDATES = (
    "google-chrome",
    "chromium",
    "chromium-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
)


def _promo_markup(dataset_href: str, evaluation_href: str) -> str:
    return (
        '<div class="promo-bar">\n'
        f'    <a class="promo-dataset-link" data-dataset-link href="{dataset_href}">'
        f"{PROMO_FACTS[0]}</a>\n"
        f'    <span id="promo-strategy">{PROMO_FACTS[1]}</span>\n'
        f'    <span id="promo-model-count">{PROMO_FACTS[2]}</span>\n'
        f'    <a class="promo-evaluation-link" data-evaluation-link '
        f'href="{evaluation_href}">{PROMO_FACTS[3]}</a>\n'
        "  </div>"
    )


def _chrome_binary() -> str | None:
    for candidate in CHROME_CANDIDATES:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
        if Path(candidate).is_file():
            return candidate
    return None


def _render_promo_geometry(
    tmp_path: Path,
    chrome: str,
    viewport_width: int,
) -> dict:
    root = Path(__file__).resolve().parents[1]
    shutil.copy2(root / "webapp" / "static" / "styles.css", tmp_path / "styles.css")
    frame = tmp_path / f"promo-frame-{viewport_width}.html"
    frame.write_text(
        f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="./styles.css">
</head>
<body>
  <div class="promo-bar">
    <a>{PROMO_FACTS[0]}</a>
    <span>{PROMO_FACTS[1]}</span>
    <span>{PROMO_FACTS[2]}</span>
    <a>{PROMO_FACTS[3]}</a>
  </div>
</body>
</html>
""",
        encoding="utf-8",
    )
    probe = tmp_path / f"promo-probe-{viewport_width}.html"
    probe.write_text(
        f"""<!DOCTYPE html>
<html>
<body>
  <pre id="result"></pre>
  <script>
    async function measure(frame) {{
      const view = frame.contentWindow;
      await Promise.race([
        view.document.fonts.ready,
        new Promise((resolve) => setTimeout(resolve, 1000)),
      ]);
      const bar = view.document.querySelector(".promo-bar");
      const children = [...bar.children];
      const textRects = children.map((child) => {{
        const range = view.document.createRange();
        range.selectNodeContents(child);
        const rect = range.getBoundingClientRect();
        return {{ left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom }};
      }});
      const overlaps = [];
      for (let left = 0; left < textRects.length; left += 1) {{
        for (let right = left + 1; right < textRects.length; right += 1) {{
          const a = textRects[left];
          const b = textRects[right];
          const sameRow = a.top < b.bottom && b.top < a.bottom;
          const intersects = a.left < b.right && b.left < a.right;
          if (sameRow && intersects) overlaps.push([left + 1, right + 1]);
        }}
      }}
      const style = view.getComputedStyle(bar);
      document.getElementById("result").textContent = JSON.stringify({{
        viewportWidth: view.innerWidth,
        columns: style.gridTemplateColumns.split(" ").length,
        minHeight: style.minHeight,
        rowGap: style.rowGap,
        alignments: children.map((child) => view.getComputedStyle(child).textAlign),
        overlaps,
      }});
    }}
  </script>
  <iframe src="{frame.name}" onload="measure(this)"
          style="width:{viewport_width}px;height:180px;border:0"></iframe>
</body>
</html>
""",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--no-sandbox",
            "--allow-file-access-from-files",
            "--force-device-scale-factor=1",
            "--window-size=900,300",
            "--virtual-time-budget=3000",
            "--dump-dom",
            probe.as_uri(),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    match = re.search(r'<pre id="result">(\{.*?\})</pre>', completed.stdout)
    assert match, completed.stdout
    return json.loads(match.group(1))


def _summary(models):
    return pd.DataFrame([
        {
            "model": model,
            "strategy": "direct",
            "evaluation_regime": "conditional",
            "comparison_population": "common",
            "aggregation": "global",
            "n_folds": 2,
            "n_scored": 4,
            "WAPE": 0.2 + index / 100,
            "MAE": 2.0 + index,
            "RMSE": 3.0 + index,
            "Bias": 0.1,
            "BiasRatio": 0.01,
        }
        for index, model in enumerate(models)
    ])


def _raw_frames():
    history_dates = pd.date_range("2025-01-01", periods=5, freq="D")
    future_dates = pd.date_range("2025-01-06", periods=2, freq="D")
    train = pd.DataFrame({
        "ProductId": [1] * len(history_dates),
        "DateKey": history_dates,
        "Quantity": np.arange(1, len(history_dates) + 1, dtype=float),
    })
    test = pd.DataFrame({
        "ProductId": [1] * len(future_dates),
        "DateKey": future_dates,
    })
    return train, test


def test_registry_and_cli_define_exactly_two_direct_contenders():
    assert CHALLENGE_MODELS == ("NeuralNet", "Chronos2")
    assert MODEL_ORDER == ["NeuralNet", "Chronos2"]
    assert set(MODEL_STRATEGY_SUPPORT) == set(CHALLENGE_MODELS)
    assert all(MODEL_STRATEGY_SUPPORT[name] == {"direct"} for name in CHALLENGE_MODELS)
    assert resolve_strategies(ForecastStrategy.DIRECT) == (ForecastStrategy.DIRECT,)
    with pytest.raises(ValueError, match="only --forecast-strategy direct"):
        resolve_strategies(ForecastStrategy.RECURSIVE)

    options = parse_args([])
    assert options.forecast_strategy is ForecastStrategy.DIRECT
    assert options.submission_model is SubmissionModel.AUTO
    assert options.chronos2 == "on"
    with pytest.raises(SystemExit):
        parse_args(["--forecast-strategy", "recursive"])
    with pytest.raises(SystemExit):
        parse_args([
            "--run-kind", "publication",
            "--submission-model", "Chronos2",
        ])
    with pytest.raises(SystemExit):
        parse_args([
            "--run-kind", "reproduction",
            "--submission-model", "NeuralNet",
        ])


def test_export_boundary_removes_every_legacy_model(tmp_path):
    train, test = _raw_frames()
    models = ["NeuralNet", "Chronos2", "XGBoost", "LightGBM", "MovingAvg28"]
    summary = _summary(models)
    cv = pd.DataFrame([
        {"fold": 1, "model": model, "regime": "conditional", "WAPE": 0.2}
        for model in models
    ])
    submission = test.copy()
    submission["Quantity"] = [10, 11]
    forecasts = {
        "NeuralNet": np.array([10.0, 11.0]),
        "Chronos2": np.array([9.0, 12.0]),
        "XGBoost": np.array([999.0, 999.0]),
    }
    out = tmp_path / "results.json"
    cfg = Config(output_dir=str(tmp_path), num_products=1, horizon=2)
    payload = export_results_json(
        train,
        test,
        submission,
        forecasts,
        cv,
        cfg,
        path=str(out),
        dev_summary=summary,
        benchmark_summary=summary,
        runtime_options=RuntimeOptions(),
        forecasts_by_strategy={"direct": forecasts},
        cv_results_all=cv.assign(strategy="direct"),
        strategy_by_horizon=summary.assign(horizon=1),
        validation_strata_summary=summary.assign(validation_stratum="regular"),
        test_aligned_scores=pd.DataFrame([
            {"strategy": "direct", "model": model, "metric": "WAPE", "test_aligned_score": 0.2}
            for model in models
        ]),
        prediction_diagnostics=pd.DataFrame([
            {"model": model, "coverage": 1.0} for model in models
        ]),
        per_product_summary=pd.DataFrame([
            {"model": model, "ProductId": 1, "WAPE": 0.2} for model in models
        ]),
        top_decile_summary=pd.DataFrame([
            {"model": model, "WAPE": 0.2} for model in models
        ]),
        top_error_rows=pd.DataFrame([
            {"model": model, "absolute_error": 1.0} for model in models
        ]),
        canonical_model="NeuralNet",
    )

    assert payload["schema_version"] == "vonavy-chronos-v2"
    assert [model["key"] for model in payload["models"]] == list(CHALLENGE_MODELS)
    assert set(payload["forecasts"]) == set(CHALLENGE_MODELS)
    assert set(payload["forecasts_by_strategy"]["direct"]) == set(CHALLENGE_MODELS)
    assert out.exists()

    def assert_no_legacy(value):
        if isinstance(value, dict):
            if "model" in value:
                assert value["model"] in CHALLENGE_MODELS
            for child in value.values():
                assert_no_legacy(child)
        elif isinstance(value, list):
            for child in value:
                assert_no_legacy(child)

    assert_no_legacy(payload)


def test_checked_in_dashboard_snapshot_obeys_challenge_schema():
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "outputs" / "results.json").read_text())
    assert data["project"]["name"] == "vonavy_chronos"
    assert data["schema_version"] == "vonavy-chronos-v2"
    assert data["project"]["status"] == "complete"
    assert [model["key"] for model in data["models"]] == list(CHALLENGE_MODELS)
    assert data["models"][0]["label"] == "Best NN"
    assert data["models"][0]["available"] is True
    assert data["models"][1]["available"] is True
    assert set(data["forecasts"]) == set(CHALLENGE_MODELS)
    assert data["selection"]["canonical_model"] == "NeuralNet"
    assert data["provenance"]["source"]["revision"]
    assert data["provenance"]["verification"]["status"] == "incomplete"
    assert data["publication_provenance"]["status"] == "authenticated"
    assert data["probabilistic_evaluation"]["status"] == "evaluated"


def test_branding_and_local_port_are_frozen():
    root = Path(__file__).resolve().parents[1]
    server_source = (root / "webapp" / "server.py").read_text()
    readme = (root / "README.md").read_text()
    pyproject = (root / "pyproject.toml").read_text()

    assert "Best NN vs Chronos-2" in server_source
    assert "PORT = 8998" in server_source
    assert "port=PORT" in server_source
    assert "http://127.0.0.1:8998" in readme
    assert "uv run python webapp/server.py" in readme
    assert "uv run python -m webapp.server" in readme
    assert '"fastapi>=0.139.0"' in pyproject
    assert '"uvicorn[standard]>=0.51.0"' in pyproject
    assert "preview = [" not in pyproject
    assert "Our Best" not in server_source + readme
    assert "8999" not in server_source + readme


def test_authored_presentation_is_standalone_and_two_contender_focused():
    root = Path(__file__).resolve().parents[1]
    presentation_paths = [
        root / "README.md",
        root / "webapp" / "static" / "common.js",
        root / "webapp" / "static" / "styles.css",
        *sorted((root / "webapp" / "static").glob("*.html")),
        *sorted((root / "webapp" / "static").glob("*.js")),
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in presentation_paths)
    forbidden = [
        "SUITE" + "_APPS",
        "suite-" + "switcher",
        "Classical" + " Forecasting",
        "Anomaly" + " Research",
        "vonava_" + "predikce",
        "vonave_" + "anomalie",
    ]
    for fragment in forbidden:
        assert fragment not in combined

    common = (root / "webapp" / "static" / "common.js").read_text(encoding="utf-8")
    for label in ("Challenge", "Data Story", "Evaluation", "Best NN", "Chronos-2"):
        assert label in combined
    assert 'label: "Data Story"' in common
    assert "Why Chronos-2 likely lost" in combined
    assert "What would justify another attempt" in combined
    assert "consumed final audit" in combined.lower()
    assert "incomplete" in combined.lower()


def test_every_authored_and_generated_page_has_one_shared_description_strip():
    root = Path(__file__).resolve().parents[1]
    title = "<title>NOTINO - chronos</title>"
    page_names = ("index.html", "dataset.html", "evaluation.html", "model.html")
    route_inventory = {
        "/": "index.html",
        "/dataset": "dataset.html",
        "/evaluation": "evaluation.html",
        "/model/neuralnet": "model.html",
        "/model/chronos2": "model.html",
    }
    assert len(route_inventory) == 5

    for directory in (root / "webapp" / "static", root / "docs"):
        for page_name in page_names:
            source = (directory / page_name).read_text(encoding="utf-8")
            assert source.count("<title") == 1
            assert source.count(title) == 1
            assert len(
                re.findall(r'class="description-strip model-hero\b[^"]*"', source)
            ) == 1
            assert len(re.findall(r'class="[^"]*\bmodel-hero\b[^"]*"', source)) == 1
            assert re.search(
                r'<header class="hero[^"]*">.*?</header>\s*'
                r'<header class="description-strip model-hero[^"]*"[^>]*>',
                source,
                flags=re.DOTALL,
            )
            assert source.index('class="description-strip') < source.index(
                '<main id="app">'
            )

    overview = (root / "webapp" / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    assert "Final challenge" in overview
    assert "Best NN vs Chronos-2" in overview
    assert "controlled negative-result experiment" in overview


def test_every_authored_and_generated_page_has_the_same_stable_promo_bar():
    root = Path(__file__).resolve().parents[1]
    page_names = ("index.html", "dataset.html", "evaluation.html", "model.html")
    directories = (
        (root / "webapp" / "static", "/dataset", "/evaluation"),
        (root / "docs", "./dataset.html", "./evaluation.html"),
    )

    for directory, dataset_href, evaluation_href in directories:
        expected = _promo_markup(dataset_href, evaluation_href)
        for page_name in page_names:
            source = (directory / page_name).read_text(encoding="utf-8")
            promo = re.search(r'<div class="promo-bar">.*?</div>', source, re.DOTALL)
            assert promo
            assert promo.group(0) == expected
            assert source.count('class="promo-bar"') == 1

    common = (root / "webapp" / "static" / "common.js").read_text(encoding="utf-8")
    for selector in (
        "promo-dataset-link",
        "promo-strategy",
        "promo-model-count",
        "promo-evaluation-link",
    ):
        assert selector not in common


def test_chronos_promo_has_safe_computed_geometry_at_responsive_boundaries(tmp_path):
    chrome = _chrome_binary()
    if not chrome:
        pytest.skip("Chrome/Chromium is required for rendered promo geometry")

    at_800 = _render_promo_geometry(tmp_path, chrome, 800)
    assert at_800 == {
        "viewportWidth": 800,
        "columns": 2,
        "minHeight": "57px",
        "rowGap": "8px",
        "alignments": ["left", "right", "left", "right"],
        "overlaps": [],
    }

    at_801 = _render_promo_geometry(tmp_path, chrome, 801)
    assert at_801["viewportWidth"] == 801
    assert at_801["columns"] == 4
    assert at_801["minHeight"] == "40px"
    assert at_801["alignments"] == ["left", "center", "center", "right"]
    assert at_801["overlaps"] == []

    at_480 = _render_promo_geometry(tmp_path, chrome, 480)
    assert at_480["columns"] == 1
    assert at_480["minHeight"] == "89px"
    assert at_480["alignments"] == ["left", "left", "left", "left"]
    assert at_480["overlaps"] == []


def test_description_strip_geometry_and_title_are_single_source_contracts():
    root = Path(__file__).resolve().parents[1]
    styles = (root / "webapp" / "static" / "styles.css").read_text(
        encoding="utf-8"
    )
    for declaration in (
        "--page-padding-inline: 56px;",
        "--description-strip-padding-block: 40px;",
        "--description-strip-border-width: 6px;",
        "--description-strip-min-height: 300px;",
    ):
        assert declaration in styles

    base_rules = re.findall(r"(?m)^\.description-strip\s*\{([^}]*)\}", styles)
    assert len(base_rules) == 1
    base = base_rules[0]
    for declaration in (
        "box-sizing: border-box;",
        "width: 100%;",
        "max-width: none;",
        "min-height: var(--description-strip-min-height);",
        "margin: 0;",
        "padding: var(--description-strip-padding-block) var(--page-padding-inline);",
        "border-bottom: var(--description-strip-border-width) solid var(--mc);",
    ):
        assert declaration in base

    assert re.search(
        r"@media \(max-width: 900px\)\s*\{\s*"
        r":root\s*\{\s*--page-padding-inline: 24px;\s*\}",
        styles,
    )
    assert not re.search(r"(?m)^\.model-hero\s*\{", styles)

    forbidden_geometry = re.compile(
        r"\b(?:box-sizing|width|max-width|margin|padding|padding-left|"
        r"padding-right|min-height|text-align)\s*:"
    )
    for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", styles):
        if any(
            class_name in selector
            for class_name in (
                ".model-hero",
                ".overview-hero",
                ".dataset-hero",
                ".evaluation-hero",
            )
        ):
            assert not forbidden_geometry.search(body)

    authored_js = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((root / "webapp" / "static").glob("*.js"))
    )
    generated_js = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((root / "docs").glob("*.js"))
    )
    for javascript in (authored_js, generated_js):
        assert "document.title" not in javascript
        assert "page-title" not in javascript
