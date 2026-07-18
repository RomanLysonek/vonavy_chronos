# Published outputs

`results.json` is the canonical complete dashboard payload. It is copied to `webapp/static/results.json` and then into generated `docs/results.json`.

Compact committed evidence includes the same-row seasonal baseline, probabilistic diagnostics, weight sensitivity, final-audit summary, audit-consumption marker, model-run record, publication manifest, and SHA-256 list. Heavy parquet/checkpoint artifacts are gitignored.

The current model-run record explicitly marks checkpoint provenance incomplete: checkpoint identities/hashes were not captured immutably before its `--resume` execution. The content-addressed publication manifest authenticates the later static publication without claiming that its later source produced the forecasts.

Run `python ml/publish_static.py --check` to verify static parity. `ml/export_results.py --check` authenticates the committed snapshot in a fresh clone; a byte-for-byte rebuild is additionally performed when the full persisted artifacts from a pipeline run are available. Running `ml/export_results.py` without `--check` still requires those full artifacts.
