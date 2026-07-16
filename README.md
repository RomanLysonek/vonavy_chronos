# vonavy_chronos

A deliberately narrow forecasting challenge:

> **Can Amazon Chronos-2 beat the strongest model developed in the original Notino project?**

This repository is not another broad model leaderboard. Its executable and presentation contract contains exactly two contenders:

| Role | Dashboard label | Internal key | Forecast contract |
|---|---|---|---|
| Frozen incumbent | **Best NN** | `NeuralNet` | Direct 7-day multi-horizon forecast |
| Foundation-model challenger | **Chronos-2** | `Chronos2` | Direct 7-day zero-shot quantile forecast |

The checked-in dashboard preserves the last stable incumbent results and marks Chronos-2 as **pending**. Running the pipeline downloads the Chronos weights, evaluates both models on identical walk-forward rows, writes both forecast files, and selects the final `submission.csv` using development out-of-fold performance only.

## Why only these two models

The original project used tree models, naïve baselines, recursive variants and an ensemble to explore the solution space. That work has already served its purpose: it led to the frozen incumbent. Repeating those models in this repository would dilute the only question that matters now.

The active benchmark therefore:

- trains only the frozen neural incumbent;
- runs only Chronos-2 as the challenger;
- uses one direct seven-day forecasting contract;
- excludes ensembles and post-hoc blending;
- exports only the two contenders to JSON and the dashboard;
- retains a short lineage explaining how **Best NN** was reached, without displaying obsolete leaderboard scores.

## Frozen incumbent: “Best NN”

The incumbent is the strongest confirmed direct NeuralNet specification from the original repository. Its essential design is frozen before Chronos is scored:

1. **Leakage-safe daily panel** — missing calendar rows, stockouts and valid zero demand remain distinct states.
2. **Direct multi-horizon formulation** — one stacked `(ForecastOrigin, Horizon, ProductId)` panel predicts all seven target days without recursive feedback.
3. **Availability-aware history** — unavailable quantities are censored from lag and rolling-demand features rather than treated as real zero demand.
4. **Residual target** — the network predicts a log-demand residual around a robust seasonal/rolling baseline.
5. **Structured categorical handling** — product, web campaign and app campaign identifiers use embeddings.
6. **Development-only selection** — all model decisions were made on walk-forward development origins; recent origins act only as confirmation.

Historical model implementations are excluded from the active repository. Their conclusions survive only as the compact incumbent lineage and the frozen configuration.

## Chronos-2 adapter

`ml/models/chronos2_model.py` translates the retail panel into the Chronos dataframe interface while preserving the original information boundary.

### Target handling

- Target: total daily demand, `QuantityApp + QuantityWeb`.
- Stockout rows are passed as missing historical targets, not zero demand.
- Synthetic calendar-gap rows are also missing targets.
- Products with no usable context receive the same availability-aware fallback used by the incumbent framework instead of aborting a fold.

### Known-future covariates

Only values genuinely known for the target dates are supplied as future covariates:

- calendar features;
- campaign subtype identifiers;
- discount values;
- sale/promotion flags;
- listed price, relative price and effective-price features.

`ProductAvailable` is deliberately **not** supplied for future dates because test-time availability is not known. Historical observation and availability indicators remain past-only covariates.

### Probabilistic output

Chronos produces `q10`, `q50` and `q90` forecasts. The point forecast is the median (`q50`); quantiles are exported in `final_forecasts.parquet`. Fallback rows receive internally consistent collapsed intervals.

## Fair comparison contract

Both contenders are evaluated with:

- the same development and recent-benchmark forecast origins;
- the same pre-origin information cutoff;
- the same `(ProductId, DateKey)` target keys;
- the same conditional-demand and realized-sales diagnostics;
- a common prediction population, so neither model improves its score by dropping hard rows;
- WAPE as the primary metric;
- development OOF only for winner selection;
- the recent benchmark as a one-time confirmation, never a tuning set.

The default winner protocol is `test-aligned`: development strata resembling the January test period receive the frozen weighting already defined by the original project. `--selection-protocol global` is available as a transparent alternative.

## Installation

Requires Python 3.14+ and `uv`.

The stable project environment and the Chronos overlay are kept separate so the original pipeline remains reproducible.

```bash
uv sync --group dev
```

Chronos-2 is loaded as an optional pinned overlay at execution time:

```bash
uv run --with "chronos-forecasting==2.3.1" python ml/pipeline.py --resume
```

The first run downloads `amazon/chronos-2` from Hugging Face. Later runs reuse the local model cache.

## Recommended challenge run

```bash
caffeinate -i uv run \
  --with "chronos-forecasting==2.3.1" \
  python ml/pipeline.py \
  --forecast-strategy direct \
  --submission-model auto \
  --selection-metric WAPE \
  --selection-protocol test-aligned \
  --chronos2 on \
  --chronos2-device auto \
  --chronos2-dtype float32 \
  --chronos2-batch-size 100 \
  --chronos2-cross-learning on \
  --chronos2-covariates on \
  --resume \
  2>&1 | tee pipeline_chronos_challenge.log
```

On Apple Silicon, `auto` resolves to MPS when supported. Use `--chronos2-device cpu` if the installed Chronos/Transformers combination exposes an unsupported MPS operation.

### Useful ablations

These are Chronos configuration checks, not extra competing models:

```bash
# Full challenger
--chronos2-cross-learning on  --chronos2-covariates on

# Test whether cross-product learning helps
--chronos2-cross-learning off --chronos2-covariates on

# Pure target-only zero-shot Chronos
--chronos2-cross-learning off --chronos2-covariates off
```

Changing a Chronos-specific option invalidates only the Chronos augmentation in compatible direct-fold checkpoints. Existing incumbent fold predictions remain reusable.

## Outputs

A complete run writes:

```text
outputs/
├── submission_best_nn.csv          # frozen incumbent forecast
├── submission_chronos2.csv          # Chronos point forecast
├── submission.csv                   # development-selected winner
├── challenge_comparison.csv         # two-model development + benchmark table
├── oof_predictions.parquet          # aligned row-level OOF predictions
├── final_forecasts.parquet          # point forecasts, fallbacks and quantiles
├── prediction_diagnostics.csv       # coverage/fallback diagnostics
├── prediction_diagnostics_by_origin.csv
├── per_product_summary.csv
├── top_decile_summary.csv
├── strategy_by_horizon.csv
└── results.json                     # strict two-model dashboard contract
```

Every model-bearing array in `results.json` is filtered at the exporter boundary to `NeuralNet` and `Chronos2`. This is not merely a frontend filter.

## Dashboard

Run the local web application:

```bash
uv run python webapp/server.py
```

Open:

```text
http://127.0.0.1:8998
```

The dashboard contains:

- the development-selected winner and recent-benchmark confirmation;
- WAPE/MAE/bias head-to-head cards;
- fold-by-fold and horizon-by-horizon deltas;
- high-volume-day performance;
- per-product win counts;
- a two-forecast product explorer;
- separate **Best NN** and **Chronos-2** method pages;
- a compact incumbent-development lineage;
- explicit pending state when Chronos has not yet been executed.

`docs/` contains the equivalent static GitHub Pages site.

## Repository map

```text
data/                         supplied train/test parquet files
ml/framework.py               shared panel, feature and evaluation contracts
ml/models/neural_net.py       frozen incumbent implementation
ml/models/chronos2_model.py   Chronos dataframe adapter and fallback logic
ml/pipeline.py                two-contender orchestration and artifact export
ml/export_results.py          rebuild dashboard JSON from persisted artifacts
webapp/                       local FastAPI server and two-model frontend
docs/                         static dashboard for GitHub Pages
tests/                        challenge, Chronos adapter and frontend contracts
```

## Acceptance checks

```bash
python -m compileall -q ml webapp
node tests/webapp_smoke_test.js
uv run pytest -q
```

The expensive model-weight inference is intentionally separate from unit tests. The local challenge run is the final runtime acceptance test for the installed hardware backend.
