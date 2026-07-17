# NOTINO / Interview Assignment — Chronos-2 challenge

This repository asks one deliberately narrow question:

> Can zero-shot Amazon Chronos-2 beat the frozen custom neural network developed for the original retail forecasting assignment?

The published answer is **no**. Chronos-2 is technically suitable as a same-split foundation-model benchmark, but the custom network wins materially. The project does not fine-tune Chronos, blend the contenders, or manufacture a larger leaderboard.

| Role | Published label | Internal key | Contract |
|---|---|---|---|
| Frozen incumbent | **Best NN** | `NeuralNet` | Direct seven-day multi-horizon forecast |
| Zero-shot challenger | **Chronos-2** | `Chronos2` | Direct q10/q50/q90 forecast; q50 is the point estimate |

## Portfolio suite

- [Classical Forecasting](https://romanlysonek.github.io/vonava_predikce/)
- [Anomaly Research](https://romanlysonek.github.io/vonave_anomalie/)
- [Chronos-2 Challenger](https://romanlysonek.github.io/vonavy_chronos/)

All three applications are static GitHub Pages sites. In this repository, `webapp/static/` is authored source and `docs/` is generated-only.

## Honest evaluation contract

- Development walk-forward origins are the only selection evidence.
- The recent diagnostic has already been inspected. It is reported as non-selection evidence, not described as untouched or independent.
- `FINAL_AUDIT_ORIGINS` are executed once by a publication run. A consumption marker prevents a second run from being labelled fresh; explicit reruns are labelled reproductions.
- Both contenders use the same forecast origins, target keys, information cut-offs, availability-aware scoring population, and primary global WAPE.
- The same-row seasonal weekday naive appears only in a compact sanity table. It is not a third contender.
- The frozen 60/25/15 winter/regular/event weighting remains the selection rule. Equal-strata and global views report sensitivity only and never retune those weights.

## Probabilistic contract

Chronos requests q10/q50/q90 and uses q50 as its point forecast. Quantiles are published as evaluated only when real OOF artifacts support:

- pinball loss at q10, q50, and q90;
- empirical quantile calibration;
- 80% interval coverage;
- mean and normalized interval width.

The UI renders an interval band only in that evaluated state. Missing artifacts produce an explicit `not_evaluated` state; intervals are never inferred or fabricated.

## Reproducibility and provenance

The publication profile pins:

- `chronos-forecasting==2.3.1` through the generated transitive `requirements-chronos.txt`;
- `amazon/chronos-2` revision `29ec3766d36d6f73f0696f85560a422f50e8498c`;
- the bounded profile in `ml/chronos2_profiles.json`.

Every publication records the exact source revision/tree, dirty state, input/config/lock hashes, canonical command, package/model revisions, resolved device/runtime/hardware, UTC generation time, run ID, and output hashes. Unknown runtime fields are labelled rather than guessed.

The core `uv.lock` intentionally excludes Chronos. Static Pages requires no inference environment, and FastAPI is an optional local preview.

## Install

Core development and tests:

```bash
uv sync --group dev
```

Optional FastAPI preview:

```bash
uv sync --group preview
uv run --group preview python webapp/server.py
```

Optional Chronos publication overlay:

```bash
uv run --locked \
  --with-requirements requirements-chronos.txt \
  python ml/pipeline.py \
  --run-kind publication \
  --include-final-audit \
  --chronos2-profile published \
  --chronos2-device auto \
  --resume
```

No target-only, cross-learning, covariate, or context-length ablation is executed by that command. Those bounded profiles exist only to make future plumbing explicit and reproducible.

## Static and local use

The checked-in complete demo needs only a file server:

```bash
python -m http.server 8998 --directory docs
```

Open <http://127.0.0.1:8998/>.

The optional FastAPI preview also uses <http://127.0.0.1:8998/> and reads `outputs/results.json` on each request.

Regenerate or verify Pages:

```bash
uv run python ml/publish_static.py
uv run python ml/publish_static.py --check
```

Rebuild presentation JSON from retained run artifacts without training:

```bash
uv run python ml/export_results.py
uv run python ml/export_results.py --check
```

## Outputs

The full pipeline writes submissions, aligned OOF/final parquet files, diagnostics, compact evidence CSVs, `results.json`, a run record, and SHA-256 checksums. Heavy reproducible artifacts remain gitignored; the complete cached dashboard, compact evidence, run manifest, and checksums are committed.

The headline remains exactly two models. Supporting evidence is exported under dedicated `sanity_baseline`, `weight_sensitivity`, and `probabilistic_evaluation` keys.

## Validation

```bash
python -m compileall -q ml webapp
uv run pytest -q
node tests/webapp_smoke_test.js
uv run python ml/publish_static.py --check
```

Model-weight inference is intentionally separate from unit and Pages checks.
