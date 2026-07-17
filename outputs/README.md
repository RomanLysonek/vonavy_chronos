# Published outputs

`results.json` is the canonical complete dashboard payload. It is copied to `webapp/static/results.json` and then into generated `docs/results.json`.

Compact committed evidence includes the same-row seasonal baseline, probabilistic diagnostics, weight sensitivity, final-audit summary, audit-consumption marker, immutable run record, and SHA-256 list. Heavy parquet/checkpoint artifacts are reproducible but gitignored.

Run `python ml/publish_static.py --check` to verify static parity. `ml/export_results.py` rebuilds the payload only when the full persisted artifacts from a pipeline run are available.
