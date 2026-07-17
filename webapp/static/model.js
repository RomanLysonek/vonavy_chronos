let foldChart = null;
let productChart = null;
let showHistory = true;

function currentSlug() {
  if (window.STATIC_DASHBOARD) return new URLSearchParams(window.location.search).get("model") || "our-best";
  const parts = window.location.pathname.split("/").filter(Boolean);
  return parts[parts.length - 1] || "our-best";
}

function methodology(data, model) {
  const cfg = data.config || {};
  if (model.key === "NeuralNet") {
    return {
      intro: "A frozen, project-specific estimator: the strongest specification reached before the foundation-model challenge began.",
      items: [
        ["Forecast contract", "One direct model predicts all seven horizons from the same observed origin; no generated prediction is fed back into later days."],
        ["Target", `The network predicts a ${String(cfg.nn_target_mode || "log-residual").replaceAll("_", " ")} correction around the ${cfg.baseline_variant || "4:3:2:1 same-weekday"} seasonal anchor.`],
        ["Representation", "Product and campaign identities use learned embeddings; calendar, discount, price, availability-history and demand-lag signals enter as numeric features."],
        ["Robustness", `${(cfg.seeds || []).length || 3} fixed seeds are averaged. Missing annual history is imputed with explicit missingness indicators, and residual guards constrain implausible extrapolation.`],
        ["Training semantics", "Unavailable product-days are excluded as supervised targets and censored from demand lags instead of being learned as genuine zero demand."],
      ],
    };
  }
  return {
    intro: "A zero-shot pretrained time-series foundation model, wrapped so that it obeys the same retail semantics and leakage boundary as the incumbent.",
    items: [
      ["Checkpoint", cfg.chronos2_model_id || "amazon/chronos-2"],
      ["Forecast contract", "Chronos-2 predicts the complete seven-day horizon directly. Its median forecast is the point estimate used for WAPE, MAE, bias and final comparison."],
      ["Cross-series context", cfg.chronos2_cross_learning ? "All 30 products may be processed jointly so the model can exploit cross-series structure." : "Cross-learning is disabled; each product is forecast independently."],
      ["Known-future information", cfg.chronos2_covariates ? "Calendar, campaign, discount, price-relative and effective-price features are supplied only where their target-date values are known." : "Covariates are disabled for a target-only zero-shot ablation."],
      ["Leakage and stockouts", "Unavailable quantities and synthetic calendar gaps are masked as missing history, never recoded as zero. Future availability and future quantities are not supplied."],
      ["Uncertainty and fallback", `Requested quantiles: ${(cfg.chronos2_quantile_levels || [0.1, 0.5, 0.9]).join(", ")}. Products without usable context fall back explicitly to the incumbent seasonal anchor and are counted in diagnostics.`],
    ],
  };
}

function renderHero(model) {
  document.getElementById("page-title").textContent = `${model.label} — VOŇAVÝ CHRONOS`;
  const hero = document.getElementById("model-hero");
  hero.style.setProperty("--mc", model.color);
  document.getElementById("hero-badge").textContent = model.key === "NeuralNet" ? "Frozen incumbent" : "Foundation challenger";
  document.getElementById("hero-title").textContent = model.label;
  document.getElementById("hero-blurb").textContent = model.blurb;
  const source = document.getElementById("hero-source");
  if (model.source_url) {
    source.href = model.source_url;
    source.hidden = false;
  }
}

function renderMethod(data, model) {
  const content = methodology(data, model);
  document.getElementById("method-intro").textContent = content.intro;
  document.getElementById("model-method-list").innerHTML = content.items.map(([title, body]) => `
    <div class="definition-item"><strong>${title}</strong><span>${body}</span></div>
  `).join("");
}

function renderLineage(data, model) {
  const panel = document.getElementById("lineage-panel");
  if (model.key !== "NeuralNet") {
    panel.hidden = true;
    return;
  }
  const lineage = data.challenge?.lineage || [];
  document.getElementById("model-lineage-grid").innerHTML = lineage.map((item, index) => `
    <article class="lineage-card"><span class="lineage-index">0${index + 1}</span><h3>${item.step}</h3><strong>${item.decision}</strong><p>${item.reason}</p></article>
  `).join("");
  panel.hidden = false;
}

function renderKpis(data, model) {
  const benchmark = summaryRows(data, { source: "benchmark" });
  const development = summaryRows(data, { source: "development" });
  const benchmarkRow = benchmark.find((row) => row.model === model.key) || {};
  const developmentRow = development.find((row) => row.model === model.key) || {};
  const otherKey = model.key === "NeuralNet" ? "Chronos2" : "NeuralNet";
  const other = benchmark.find((row) => row.model === otherKey) || {};
  const delta = Number.isFinite(Number(benchmarkRow.WAPE)) && Number.isFinite(Number(other.WAPE)) && Number(other.WAPE) !== 0
    ? Number(benchmarkRow.WAPE) / Number(other.WAPE) - 1
    : null;
  const cards = [
    ["Development WAPE", ratePct(developmentRow.WAPE), "selection evidence"],
    ["Benchmark WAPE", ratePct(benchmarkRow.WAPE), "recent confirmation"],
    ["Benchmark MAE", fmt(benchmarkRow.MAE), "common product-days"],
    [`vs ${modelLabel(data, otherKey)}`, signedPct(delta), "benchmark WAPE; negative is better"],
  ];
  document.getElementById("kpi-grid").innerHTML = cards.map(([label, value, sub]) => `
    <article class="kpi-card"><p class="kpi-label">${label}</p><p class="kpi-value" style="color:${model.color}">${value}</p><p class="kpi-sub">${sub}</p></article>
  `).join("");
}

function renderFoldChart(data, model) {
  const rows = cvRows(data).filter((row) => row.model === model.key).sort((a, b) => Number(a.fold) - Number(b.fold));
  if (foldChart) foldChart.destroy();
  foldChart = new Chart(document.getElementById("chart-folds"), {
    type: "bar",
    data: {
      labels: rows.map((row) => `Fold ${row.fold}`),
      datasets: [{
        label: "WAPE (%)",
        data: rows.map((row) => Number(row.WAPE) * 100),
        backgroundColor: model.color,
        borderRadius: 0,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { x: { grid: { display: false } }, y: { beginAtZero: true, grid: { color: CHART_GRID } } },
    },
  });
}

function renderHeadToHead(data, model) {
  const benchmark = summaryRows(data, { source: "benchmark" });
  const current = benchmark.find((row) => row.model === model.key) || {};
  const otherKey = model.key === "NeuralNet" ? "Chronos2" : "NeuralNet";
  const other = benchmark.find((row) => row.model === otherKey) || {};
  const verdict = Number.isFinite(Number(current.WAPE)) && Number.isFinite(Number(other.WAPE))
    ? (Number(current.WAPE) < Number(other.WAPE) ? "wins" : Number(current.WAPE) > Number(other.WAPE) ? "loses" : "ties")
    : "is pending";
  const items = [
    ["Recent-benchmark verdict", `${model.label} ${verdict} on global WAPE.`],
    ["Coverage", `${fmt(current.coverage, 3)} coverage across ${current.n_scored ?? "—"} scored rows.`],
    ["Bias direction", `${fmt(current.Bias)} average signed error; positive means over-forecasting.`],
    ["Selection role", model.key === canonicalModel(data) ? "Development-selected final forecast." : "Challenger retained as a separately exported forecast, not blended into the winner."],
  ];
  document.getElementById("head-to-head-list").innerHTML = items.map(([title, body]) => `<div class="definition-item"><strong>${title}</strong><span>${body}</span></div>`).join("");
}

function populateProducts(data) {
  const select = document.getElementById("product-select");
  const products = Object.keys(data.history || {}).sort((a, b) => Number(a) - Number(b));
  select.innerHTML = products.map((product) => `<option value="${product}">Product ${product}</option>`).join("");
  return products[0];
}

function renderProduct(data, model, productId) {
  const history = data.history?.[productId] || { dates: [], quantity: [] };
  const forecast = forecastsFor(data)?.[model.key]?.[productId];
  const datasets = [];
  if (showHistory) {
    datasets.push({
      label: "History",
      data: history.dates.map((date, index) => ({ x: date, y: history.quantity[index] })),
      borderColor: "#111111",
      backgroundColor: "#111111",
      pointRadius: 0,
      borderWidth: 1.5,
    });
  }
  if (forecast) {
    datasets.push({
      label: model.label,
      data: forecast.dates.map((date, index) => ({ x: date, y: forecast.quantity[index] })),
      borderColor: model.color,
      backgroundColor: model.color,
      pointRadius: 3,
      borderWidth: 3,
    });
  }
  if (productChart) productChart.destroy();
  productChart = new Chart(document.getElementById("chart-product"), {
    type: "line",
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      parsing: false,
      plugins: { legend: { position: "top", labels: { boxWidth: 12 } } },
      scales: { x: { type: "category", grid: { display: false } }, y: { beginAtZero: true, grid: { color: CHART_GRID } } },
    },
  });
}

async function main() {
  try {
    const data = await loadResults();
    const slug = currentSlug();
    const model = modelByKey(data, slug);
    if (!model || !CHALLENGE_MODELS.includes(model.key)) throw new Error(`Unknown challenge contender: ${slug}`);
    renderNav(data, model.slug);
    updateSharedCopy(data);
    renderHero(model);
    renderMethod(data, model);
    renderLineage(data, model);
    renderKpis(data, model);
    renderFoldChart(data, model);
    renderHeadToHead(data, model);
    document.getElementById("model-pending-banner").hidden = Boolean(model.available);

    const firstProduct = populateProducts(data);
    const select = document.getElementById("product-select");
    const toggle = document.getElementById("model-product-history-toggle");
    const refresh = () => renderProduct(data, model, select.value || firstProduct);
    select.addEventListener("change", refresh);
    toggle.addEventListener("change", () => { showHistory = toggle.checked; refresh(); });
    refresh();
  } catch (error) {
    document.getElementById("app").innerHTML = `<section class="panel error-panel"><h2>Could not render contender</h2><p>${error.message}</p></section>`;
  }
}

main();
