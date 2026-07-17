let comparisonChart = null;
let horizonChart = null;
let productChart = null;
const visibleProductModels = new Set(CHALLENGE_MODELS);
let showProductHistory = true;

function modelColor(data, key) {
  return modelByKey(data, key)?.color || (key === "Chronos2" ? "#FF9900" : "#EE4C2C");
}

function modelAvailable(data, key) {
  const model = modelByKey(data, key);
  return Boolean(model?.available ?? forecastsFor(data)?.[key]);
}

function relativeDelta(challenger, incumbent) {
  const a = Number(challenger);
  const b = Number(incumbent);
  return Number.isFinite(a) && Number.isFinite(b) && b !== 0 ? a / b - 1 : null;
}

function renderStatus(data, regime) {
  const winner = canonicalModel(data);
  const complete = data.project?.status === "complete" || CHALLENGE_MODELS.every((key) => modelAvailable(data, key));
  document.getElementById("challenge-winner").textContent = complete
    ? modelLabel(data, winner)
    : "Awaiting Chronos-2 run";
  document.getElementById("challenge-status-copy").textContent = complete
    ? `Selected from development OOF only; the recent benchmark is confirmation, not another tuning set. Current lens: ${regime === "realized" ? "all observed days" : "available product-days"}.`
    : "The repository and dashboard are challenge-ready, but the checked-in artifact contains only the frozen incumbent.";
  document.getElementById("pending-banner").hidden = complete;

  const benchmarkRows = summaryRows(data, { source: "benchmark", regime });
  const benchmarkWinner = winnerFromRows(benchmarkRows, "WAPE");
  const badge = document.getElementById("benchmark-badge");
  badge.textContent = benchmarkWinner
    ? `Benchmark: ${modelLabel(data, benchmarkWinner.model)}`
    : "Benchmark pending";
}

function renderKpis(data, regime) {
  const dev = summaryRows(data, { source: "development", regime });
  const bench = summaryRows(data, { source: "benchmark", regime });
  const selectedModel = canonicalModel(data);
  const devWinner = dev.find((row) => row.model === selectedModel) || winnerFromRows(dev, "WAPE");
  const benchWinner = winnerFromRows(bench, "WAPE");
  const devMap = Object.fromEntries(dev.map((row) => [row.model, row]));
  const benchMap = Object.fromEntries(bench.map((row) => [row.model, row]));
  const devDelta = relativeDelta(devMap.Chronos2?.WAPE, devMap.NeuralNet?.WAPE);
  const benchDelta = relativeDelta(benchMap.Chronos2?.WAPE, benchMap.NeuralNet?.WAPE);
  const confirms = devWinner && benchWinner ? devWinner.model === benchWinner.model : null;

  const cards = [
    {
      label: "Development winner",
      value: devWinner ? modelLabel(data, devWinner.model) : "Pending",
      sub: devWinner ? `${data.config?.selection_protocol || "global"} development protocol` : "Chronos-2 score not available",
      color: devWinner ? modelColor(data, devWinner.model) : null,
    },
    {
      label: "Chronos Δ vs Best NN",
      value: signedPct(devDelta),
      sub: "Development WAPE; negative means Chronos is better",
      color: modelColor(data, "Chronos2"),
    },
    {
      label: "Recent benchmark winner",
      value: benchWinner ? modelLabel(data, benchWinner.model) : "Pending",
      sub: benchWinner ? `${ratePct(benchWinner.WAPE)} WAPE · Chronos Δ ${signedPct(benchDelta)}` : "Awaiting head-to-head run",
      color: benchWinner ? modelColor(data, benchWinner.model) : null,
    },
    {
      label: "Recent diagnostic agreement",
      value: confirms === null ? "Pending" : (confirms ? "Yes" : "No"),
      sub: "Previously inspected and never used for selection",
    },
  ];

  document.getElementById("kpi-grid").innerHTML = cards.map((card) => `
    <article class="kpi-card">
      <p class="kpi-label">${card.label}</p>
      <p class="kpi-value"${card.color ? ` style="color:${card.color}"` : ""}>${card.value}</p>
      <p class="kpi-sub">${card.sub}</p>
    </article>
  `).join("");
}

function renderChallengeColumns(data, regime) {
  const dev = Object.fromEntries(summaryRows(data, { source: "development", regime }).map((row) => [row.model, row]));
  const bench = Object.fromEntries(summaryRows(data, { source: "benchmark", regime }).map((row) => [row.model, row]));
  document.getElementById("challenge-columns").innerHTML = CHALLENGE_MODELS.map((key) => {
    const model = modelByKey(data, key) || { label: key, short: "", blurb: "", slug: key.toLowerCase(), kind: "contender" };
    const available = modelAvailable(data, key);
    const development = dev[key] || {};
    const benchmark = bench[key] || {};
    return `
      <a class="challenge-column${available ? "" : " unavailable"}" style="--mc:${model.color}" href="${modelHref(model.slug)}">
        <div class="model-column-header">
          <span class="model-badge">${key === "NeuralNet" ? "Frozen incumbent" : "Foundation challenger"}</span>
          <h3>${model.label}</h3>
          <span class="source">${model.short}</span>
        </div>
        <p class="challenge-blurb">${model.blurb}</p>
        <div class="model-stats">
          <div class="model-stat-row"><span>Development WAPE</span><strong>${ratePct(development.WAPE)}</strong></div>
          <div class="model-stat-row"><span>Benchmark WAPE</span><strong>${ratePct(benchmark.WAPE)}</strong></div>
          <div class="model-stat-row"><span>Benchmark MAE</span><strong>${fmt(benchmark.MAE)}</strong></div>
          <div class="model-stat-row"><span>Benchmark bias</span><strong>${fmt(benchmark.Bias)}</strong></div>
        </div>
        <span class="model-column-cta">${available ? "Open model analysis →" : "Implementation ready · run pending →"}</span>
      </a>
    `;
  }).join("");
}

function renderComparisonChart(data, regime) {
  const rows = summaryRows(data, { source: "benchmark", regime });
  const byModel = Object.fromEntries(rows.map((row) => [row.model, row]));
  const labels = CHALLENGE_MODELS.map((key) => modelLabel(data, key));
  const colors = CHALLENGE_MODELS.map((key) => modelColor(data, key));
  if (comparisonChart) comparisonChart.destroy();
  comparisonChart = new Chart(document.getElementById("chart-comparison"), {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "WAPE",
          data: CHALLENGE_MODELS.map((key) => Number.isFinite(Number(byModel[key]?.WAPE)) ? Number(byModel[key].WAPE) * 100 : null),
          backgroundColor: colors,
          yAxisID: "y",
          borderRadius: 0,
        },
        {
          label: "MAE",
          data: CHALLENGE_MODELS.map((key) => byModel[key]?.MAE ?? null),
          backgroundColor: colors.map((color) => `${color}66`),
          borderColor: colors,
          borderWidth: 1,
          yAxisID: "y1",
          borderRadius: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: "top", labels: { boxWidth: 12 } } },
      scales: {
        x: { grid: { display: false } },
        y: { beginAtZero: true, title: { display: true, text: "WAPE (%)" }, grid: { color: CHART_GRID } },
        y1: { beginAtZero: true, position: "right", title: { display: true, text: "MAE" }, grid: { drawOnChartArea: false } },
      },
    },
  });
}

function renderFairness(data, regime) {
  const n = summaryRows(data, { source: "benchmark", regime })[0]?.n_scored;
  const items = [
    ["Same forecast origins", `${data.config?.n_dev_origins || 12} development origins plus ${data.config?.n_cv_folds || 4} recent benchmark origins.`],
    ["Same scoring rows", `${Number.isFinite(Number(n)) ? Number(n).toLocaleString(NUMBER_LOCALE) : "Common"} ${regime === "realized" ? "observed" : "available"} product-days; neither model gets a private population.`],
    ["Same primary metric", "Global WAPE: total absolute error divided by total actual demand."],
    ["No diagnostic tuning", "The winner is selected on development OOF. Recent and final-audit results are non-selection evidence."],
  ];
  document.getElementById("fairness-list").innerHTML = items.map(([title, body]) => `
    <div class="definition-item"><strong>${title}</strong><span>${body}</span></div>
  `).join("");
}

function renderLineage(data) {
  const lineage = data.challenge?.lineage || [];
  document.getElementById("lineage-grid").innerHTML = lineage.map((item, index) => `
    <article class="lineage-card">
      <span class="lineage-index">0${index + 1}</span>
      <h3>${item.step}</h3>
      <strong>${item.decision}</strong>
      <p>${item.reason}</p>
    </article>
  `).join("");
}

function renderFoldTable(data, regime) {
  const rows = cvRows(data, regime);
  const folds = new Map();
  rows.forEach((row) => {
    const key = String(row.fold);
    if (!folds.has(key)) folds.set(key, {});
    folds.get(key)[row.model] = row;
  });
  const tbody = document.querySelector("#fold-table tbody");
  const rendered = [...folds.entries()].sort((a, b) => Number(a[0]) - Number(b[0])).map(([fold, models]) => {
    const incumbent = models.NeuralNet?.WAPE;
    const chronos = models.Chronos2?.WAPE;
    const delta = relativeDelta(chronos, incumbent);
    let winner = "Pending";
    if (Number.isFinite(Number(incumbent)) && Number.isFinite(Number(chronos))) {
      winner = Number(chronos) < Number(incumbent) ? "Chronos-2" : "Best NN";
    }
    return `<tr><td>${fold}</td><td>${ratePct(incumbent)}</td><td>${ratePct(chronos)}</td><td>${winner}</td><td>${signedPct(delta)}</td></tr>`;
  });
  tbody.innerHTML = rendered.length ? rendered.join("") : `<tr><td colspan="5">Fold comparison becomes available after the Chronos-2 run.</td></tr>`;
}

function renderHorizonChart(data, regime, metric) {
  const rows = (data.strategy_by_horizon || []).filter((row) => (
    CHALLENGE_MODELS.includes(row.model)
    && row.origin_type === "recent_benchmark"
    && row.strategy === "direct"
    && row.evaluation_regime === regime
    && row.comparison_population === "common"
    && row.aggregation === "global"
  ));
  if (horizonChart) horizonChart.destroy();
  horizonChart = new Chart(document.getElementById("chart-horizon"), {
    type: "line",
    data: {
      labels: Array.from({ length: data.config?.horizon || 7 }, (_, index) => `D+${index + 1}`),
      datasets: CHALLENGE_MODELS.map((key) => ({
        label: modelLabel(data, key),
        data: Array.from({ length: data.config?.horizon || 7 }, (_, index) => {
          const row = rows.find((candidate) => candidate.model === key && Number(candidate.horizon) === index + 1);
          const value = row?.[metric];
          return metric === "WAPE" || metric === "BiasRatio" ? (Number.isFinite(Number(value)) ? Number(value) * 100 : null) : value ?? null;
        }),
        borderColor: modelColor(data, key),
        backgroundColor: modelColor(data, key),
        tension: 0.18,
        spanGaps: false,
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: "top", labels: { boxWidth: 12 } } },
      scales: {
        x: { grid: { display: false } },
        y: { grid: { color: CHART_GRID }, title: { display: true, text: metric === "MAE" ? "MAE" : `${metric} (%)` } },
      },
    },
  });
}

function renderTopDemand(data) {
  const rows = (data.top_decile_summary || []).filter((row) => row.origin_type === "recent_benchmark" && CHALLENGE_MODELS.includes(row.model));
  const tbody = document.querySelector("#top-demand-table tbody");
  tbody.innerHTML = CHALLENGE_MODELS.map((key) => {
    const row = rows.find((candidate) => candidate.model === key) || {};
    return `<tr><td>${modelLabel(data, key)}</td><td>${ratePct(row.WAPE)}</td><td>${fmt(row.MAE)}</td><td>${fmt(row.Bias)}</td><td>${row.n ?? "—"}</td></tr>`;
  }).join("");
}

function renderProductVerdict(data) {
  const rows = (data.per_product_summary || []).filter((row) => row.origin_type === "recent_benchmark" && CHALLENGE_MODELS.includes(row.model));
  const products = new Map();
  rows.forEach((row) => {
    const key = String(row.ProductId);
    if (!products.has(key)) products.set(key, {});
    products.get(key)[row.model] = Number(row.WAPE);
  });
  let incumbentWins = 0;
  let chronosWins = 0;
  let ties = 0;
  products.forEach((values) => {
    if (!Number.isFinite(values.NeuralNet) || !Number.isFinite(values.Chronos2)) return;
    if (Math.abs(values.NeuralNet - values.Chronos2) < 1e-12) ties += 1;
    else if (values.Chronos2 < values.NeuralNet) chronosWins += 1;
    else incumbentWins += 1;
  });
  const total = incumbentWins + chronosWins + ties;
  document.getElementById("product-verdict").innerHTML = total ? `
    <div class="verdict-score" style="--mc:${modelColor(data, "NeuralNet")}"><strong>${incumbentWins}</strong><span>Best NN wins</span></div>
    <div class="verdict-vs">vs</div>
    <div class="verdict-score" style="--mc:${modelColor(data, "Chronos2")}"><strong>${chronosWins}</strong><span>Chronos-2 wins</span></div>
    ${ties ? `<p>${ties} tied products</p>` : ""}
  ` : `<p class="empty-state">Per-product verdict becomes available after both models are scored.</p>`;
}

function populateProductSelector(data) {
  const select = document.getElementById("product-select");
  const products = Object.keys(data.history || {}).sort((a, b) => Number(a) - Number(b));
  select.innerHTML = products.map((product) => `<option value="${product}">Product ${product}</option>`).join("");
  return products[0];
}

function renderProductLegend(data, refresh) {
  document.getElementById("product-model-legend").innerHTML = CHALLENGE_MODELS.map((key) => {
    const active = visibleProductModels.has(key);
    return `<button type="button" class="product-model-chip${active ? " active" : ""}" data-model="${key}" style="--mc:${modelColor(data, key)}" ${modelAvailable(data, key) ? "" : "disabled"}>${modelLabel(data, key)}</button>`;
  }).join("");
  document.querySelectorAll(".product-model-chip").forEach((button) => {
    button.addEventListener("click", () => {
      const key = button.dataset.model;
      if (visibleProductModels.has(key)) visibleProductModels.delete(key);
      else visibleProductModels.add(key);
      renderProductLegend(data, refresh);
      refresh();
    });
  });
}

function renderProductChart(data, productId) {
  const history = data.history?.[productId] || { dates: [], quantity: [] };
  const forecasts = forecastsFor(data);
  const datasets = [];
  if (showProductHistory) {
    datasets.push({
      label: "History",
      data: history.dates.map((date, index) => ({ x: date, y: history.quantity[index] })),
      borderColor: "#111111",
      backgroundColor: "#111111",
      pointRadius: 0,
      borderWidth: 1.5,
    });
  }
  CHALLENGE_MODELS.forEach((key) => {
    if (!visibleProductModels.has(key)) return;
    const series = forecasts?.[key]?.[productId];
    if (!series) return;
    datasets.push({
      label: modelLabel(data, key),
      data: series.dates.map((date, index) => ({ x: date, y: series.quantity[index] })),
      borderColor: modelColor(data, key),
      backgroundColor: modelColor(data, key),
      borderWidth: 3,
      pointRadius: 3,
    });
    const interval = data.probabilistic_evaluation?.status === "evaluated"
      ? data.probabilistic_evaluation?.forecasts?.Chronos2?.[productId]
      : null;
    if (key === "Chronos2" && interval) {
      datasets.push({
        label: "Chronos q90",
        data: interval.dates.map((date, index) => ({ x: date, y: interval.q90[index] })),
        borderColor: "transparent",
        backgroundColor: "rgba(255, 153, 0, 0.16)",
        pointRadius: 0,
      });
      datasets.push({
        label: "Chronos 80% interval",
        data: interval.dates.map((date, index) => ({ x: date, y: interval.q10[index] })),
        borderColor: "transparent",
        backgroundColor: "rgba(255, 153, 0, 0.16)",
        pointRadius: 0,
        fill: "-1",
      });
    }
  });
  if (productChart) productChart.destroy();
  productChart = new Chart(document.getElementById("chart-product"), {
    type: "line",
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      parsing: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { type: "category", grid: { display: false } },
        y: { beginAtZero: true, grid: { color: CHART_GRID } },
      },
    },
  });
}

function renderEvidence(data) {
  const baselineRows = (data.sanity_baseline || []).filter((row) => row.origin_type === "recent_benchmark");
  const baselineBody = document.querySelector("#baseline-table tbody");
  baselineBody.innerHTML = baselineRows.length
    ? baselineRows.map((row) => `<tr><td>${row.estimator === "SeasonalWeekdayNaive" ? "Seasonal weekday naive" : modelLabel(data, row.estimator)}</td><td>${ratePct(row.WAPE)}</td><td>${fmt(row.MAE)}</td><td>${fmt(row.Bias)}</td><td>${row.n ?? "—"}</td></tr>`).join("")
    : '<tr><td colspan="5">Baseline evidence is not available in this artifact.</td></tr>';

  const probability = data.probabilistic_evaluation || { status: "not_evaluated" };
  const metric = (probability.metrics || []).find((row) => row.origin_type === "recent_benchmark");
  document.getElementById("probabilistic-evidence").innerHTML = probability.status === "evaluated" && metric
    ? [
      ["Status", "Evaluated on real out-of-fold q10/q50/q90 forecasts."],
      ["80% interval coverage", ratePct(metric.interval_coverage)],
      ["Mean interval width", fmt(metric.interval_mean_width)],
      ["Pinball q10 / q50 / q90", `${fmt(metric.pinball_q10, 2)} / ${fmt(metric.pinball_q50, 2)} / ${fmt(metric.pinball_q90, 2)}`],
    ].map(([title, body]) => `<div class="definition-item"><strong>${title}</strong><span>${body}</span></div>`).join("")
    : `<div class="definition-item"><strong>Not yet evaluated</strong><span>${probability.reason || "A reproducible quantile rerun is required."}</span></div>`;

  const sensitivity = data.weight_sensitivity || [];
  const schemes = [...new Set(sensitivity.map((row) => row.scheme))];
  document.querySelector("#weight-table tbody").innerHTML = schemes.map((scheme) => {
    const rows = sensitivity.filter((row) => row.scheme === scheme);
    const scores = Object.fromEntries(rows.map((row) => [row.model, row.test_aligned_score]));
    return `<tr><td>${scheme.replaceAll("_", " ")}</td><td>${ratePct(scores.NeuralNet)}</td><td>${ratePct(scores.Chronos2)}</td><td>${rows[0]?.winner ? modelLabel(data, rows[0].winner) : "—"}</td></tr>`;
  }).join("") || '<tr><td colspan="4">Sensitivity evidence is not available.</td></tr>';

  const provenance = data.provenance || {};
  const source = provenance.source || {};
  const runtime = provenance.runtime || {};
  document.getElementById("provenance-list").innerHTML = [
    ["Run ID", provenance.run_id || "Unknown"],
    ["Source revision", source.revision || "Unknown"],
    ["Chronos revision", provenance.chronos?.model_revision || data.config?.chronos2_model_revision || "Unknown"],
    ["Runtime", `${runtime.os || "Unknown"} / ${runtime.machine || "Unknown"} / ${runtime.torch?.device || "Unknown"}`],
    ["Generated", provenance.generated_at || data.generated_at || "Unknown"],
  ].map(([title, body]) => `<div class="definition-item"><strong>${title}</strong><span class="provenance-value">${body}</span></div>`).join("");
}

function renderForecastGrid(data, modelKey) {
  const seriesByProduct = forecastsFor(data)?.[modelKey] || {};
  const products = Object.keys(seriesByProduct).sort((a, b) => Number(a) - Number(b));
  const dates = products.length ? seriesByProduct[products[0]].dates : [];
  const table = document.createElement("table");
  table.className = "data-table submission-grid";
  table.innerHTML = `
    <thead><tr><th>Product</th>${dates.map((date) => `<th>${date}</th>`).join("")}</tr></thead>
    <tbody>${products.map((product) => `<tr><td>${product}</td>${seriesByProduct[product].quantity.map((value) => `<td>${fmt(value, 0)}</td>`).join("")}</tr>`).join("")}</tbody>
  `;
  const wrap = document.getElementById("submission-table-wrap");
  wrap.innerHTML = "";
  if (!products.length) wrap.innerHTML = `<p class="empty-state">No final forecast is available for ${modelLabel(data, modelKey)}.</p>`;
  else wrap.appendChild(table);
  document.getElementById("submission-caption").textContent = `${modelLabel(data, modelKey)} final seven-day forecast`;
}

async function main() {
  try {
    const data = await loadResults();
    renderNav(data, "");
    updateSharedCopy(data);

    const regimeSelect = document.getElementById("regime-select");
    const horizonMetricSelect = document.getElementById("horizon-metric-select");
    const productSelect = document.getElementById("product-select");
    const historyToggle = document.getElementById("product-history-toggle");
    const gridModelSelect = document.getElementById("forecast-grid-model-select");

    const firstProduct = populateProductSelector(data);
    gridModelSelect.innerHTML = CHALLENGE_MODELS.map((key) => `<option value="${key}" ${modelAvailable(data, key) ? "" : "disabled"}>${modelLabel(data, key)}</option>`).join("");
    gridModelSelect.value = modelAvailable(data, canonicalModel(data)) ? canonicalModel(data) : "NeuralNet";

    const refreshProduct = () => renderProductChart(data, productSelect.value || firstProduct);
    renderProductLegend(data, refreshProduct);

    const renderAll = () => {
      const regime = regimeSelect.value;
      renderStatus(data, regime);
      renderKpis(data, regime);
      renderChallengeColumns(data, regime);
      renderComparisonChart(data, regime);
      renderFairness(data, regime);
      renderLineage(data);
      renderFoldTable(data, regime);
      renderHorizonChart(data, regime, horizonMetricSelect.value);
      renderTopDemand(data);
      renderProductVerdict(data);
      renderEvidence(data);
      refreshProduct();
      renderForecastGrid(data, gridModelSelect.value);
    };

    regimeSelect.addEventListener("change", renderAll);
    horizonMetricSelect.addEventListener("change", renderAll);
    productSelect.addEventListener("change", refreshProduct);
    historyToggle.addEventListener("change", () => {
      showProductHistory = historyToggle.checked;
      refreshProduct();
    });
    gridModelSelect.addEventListener("change", () => renderForecastGrid(data, gridModelSelect.value));
    renderAll();
  } catch (error) {
    document.getElementById("app").innerHTML = `<section class="panel error-panel"><h2>Could not load challenge results</h2><p>${error.message}</p></section>`;
  }
}

main();
