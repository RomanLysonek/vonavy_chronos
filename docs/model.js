let foldChart = null;
let productChart = null;
let showHistory = true;

function currentSlug() {
  if (window.STATIC_DASHBOARD) return new URLSearchParams(window.location.search).get("model") || "neuralnet";
  const parts = window.location.pathname.split("/").filter(Boolean);
  return parts[parts.length - 1] || "neuralnet";
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
      ["Uncertainty and fallback", data.probabilistic_evaluation?.status === "evaluated" ? "q10/q50/q90 have real OOF pinball and interval diagnostics; the explorer shows the 80% band. Products without usable context use the explicit seasonal fallback." : "q10/q50/q90 were requested but are not presented as evaluated until real OOF calibration metrics exist."],
    ],
  };
}

function renderHero(model) {
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
    ["Bias direction", `${fmt(current.Bias)} average signed error; positive is over-forecasting, negative is under-forecasting, and zero is neutral.`],
    ["Selection role", model.key === canonicalModel(data) ? "Development-selected final forecast." : "Challenger retained as a separately exported forecast, not blended into the winner."],
  ];
  document.getElementById("head-to-head-list").innerHTML = items.map(([title, body]) => `<div class="definition-item"><strong>${title}</strong><span>${body}</span></div>`).join("");
}

function renderInterpretation(data, model) {
  const isChronos = model.key === "Chronos2";
  document.getElementById("model-interpretation-title").textContent = isChronos ? "Why zero-shot likely lost" : "Why the incumbent held";
  document.getElementById("model-interpretation-subtitle").textContent = isChronos ? "Plausible mechanisms; no unrun ablation is presented as proof." : "A narrow interpretation of the head-to-head.";
  const interpretation = isChronos
    ? [
      ["Small domain panel", "Thirty related series and roughly five annual retail cycles provide limited support for local zero-shot transfer."],
      ["Regime-heavy demand", "Channel migration, campaigns, stock censoring and concentrated retail events are unusually local dynamics."],
      ["No adaptation", "The pinned checkpoint received no task-specific fine-tuning, calibration fit or blending."],
      ["Covariate / inductive-bias mismatch", "The frozen incumbent explicitly learns around local seasonal anchors and retail representations that may fit this panel better."],
    ]
    : [
      ["Frozen first", "Best NN's architecture, inputs, seeds and selection weights were fixed before Chronos-2 entered the experiment."],
      ["Project-specific learning", "Its guarded residual objective and retail feature representation were fitted on development origins from this domain."],
      ["Consistent evidence", "It wins development selection, the previously inspected recent diagnostic and the consumed final audit."],
      ["Limited conclusion", "The result establishes superiority for this experiment, not universal superiority over foundation models."],
    ];
  document.getElementById("model-interpretation-list").innerHTML = interpretation.map(([title, body]) => `<div class="definition-item"><strong>${title}</strong><span>${body}</span></div>`).join("");

  document.getElementById("model-next-evidence-title").textContent = isChronos ? "What would justify a retry" : "Scientific safeguards";
  document.getElementById("model-next-evidence-subtitle").textContent = isChronos ? "A new run should change the evidence, not merely repeat the consumed audit." : "Why the incumbent claim remains bounded.";
  const threshold = isChronos
    ? [
      ["Broader domain data", "More products, histories, event cycles or comparable retail panels."],
      ["Development-only adaptation", "A pre-specified fine-tuning or calibration protocol selected without a new audit."],
      ["Richer context", "Inventory, traffic, product metadata and planned-event inputs that directly address local regimes."],
      ["New final audit", "Newly reserved origins and a pre-registered success threshold; the current audit is already consumed."],
    ]
    : [
      ["Same rows and origins", "Both contenders share forecast keys, information cut-offs and common scoring populations."],
      ["No post-hoc incumbent changes", "The challenger result did not trigger incumbent retuning."],
      ["Baseline sanity", "A same-row seasonal naive remains supporting context, never a third contender."],
      ["Provenance caveat", `Checkpoint provenance remains explicitly ${data.provenance?.verification?.status || "unknown"}; publication provenance is separately authenticated.`],
    ];
  document.getElementById("model-next-evidence-list").innerHTML = threshold.map(([title, body]) => `<div class="definition-item"><strong>${title}</strong><span>${body}</span></div>`).join("");
}

function populateProducts(data) {
  const select = document.getElementById("product-select");
  const products = Object.keys(data.history || {}).sort((a, b) => Number(a) - Number(b));
  if (!products.length) throw new Error("Product explorer history is missing from results.json.");
  select.innerHTML = products.map((product) => `<option value="${product}">Product ${product}</option>`).join("");
  return products[0];
}

function renderProduct(data, model, productId) {
  try {
    const history = chartSeries(data.history?.[productId], `Product ${productId} history`);
    const forecast = chartSeries(
      forecastsFor(data)?.[model.key]?.[productId],
      `${model.label} forecast for product ${productId}`,
    );
    const labels = showHistory ? [...history.dates, ...forecast.dates] : [...forecast.dates];
    const datasets = [];
    if (showHistory) {
      datasets.push({
        label: "History",
        data: [...history.quantity, ...forecast.dates.map(() => null)],
        borderColor: "#111111",
        backgroundColor: "transparent",
        pointRadius: 0,
        borderWidth: 2,
      });
    }
    datasets.push({
      label: model.label,
      data: showHistory
        ? [...history.dates.map(() => null), ...forecast.quantity]
        : [...forecast.quantity],
      borderColor: model.color,
      backgroundColor: "transparent",
      pointRadius: 3,
      borderWidth: 3,
    });
    const interval = model.key === "Chronos2" && data.probabilistic_evaluation?.status === "evaluated"
      ? chartInterval(
        data.probabilistic_evaluation?.forecasts?.Chronos2?.[productId],
        `Chronos-2 interval for product ${productId}`,
      )
      : null;
    if (interval) {
      if (interval.dates.join() !== forecast.dates.join()) {
        throw new Error("Chronos-2 interval dates do not match the product explorer axis.");
      }
      const historyPadding = showHistory ? history.dates.map(() => null) : [];
      datasets.push({
        label: "q90",
        data: [...historyPadding, ...interval.q90],
        borderColor: "rgba(255, 153, 0, 0.6)",
        borderWidth: 1,
        borderDash: [4, 3],
        backgroundColor: "rgba(255, 153, 0, 0.16)",
        pointRadius: 0,
      });
      datasets.push({
        label: "80% interval",
        data: [...historyPadding, ...interval.q10],
        borderColor: "rgba(255, 153, 0, 0.6)",
        borderWidth: 1,
        borderDash: [4, 3],
        backgroundColor: "rgba(255, 153, 0, 0.16)",
        pointRadius: 0,
        fill: "-1",
      });
    }
    clearChartError("chart-product");
    if (productChart) productChart.destroy();
    productChart = new Chart(document.getElementById("chart-product"), {
      type: "line",
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: { legend: { position: "top", labels: { boxWidth: 12 } } },
        scales: {
          x: {
            display: true,
            grid: { display: false },
            ticks: { display: true, autoSkip: true, maxTicksLimit: showHistory ? 10 : 7 },
          },
          y: { display: true, beginAtZero: true, grid: { color: CHART_GRID } },
        },
      },
    });
  } catch (error) {
    if (productChart) productChart.destroy();
    productChart = null;
    showChartError("chart-product", `Product explorer unavailable: ${error.message}`);
  }
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
    renderKpis(data, model);
    renderFoldChart(data, model);
    renderHeadToHead(data, model);
    renderInterpretation(data, model);
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
