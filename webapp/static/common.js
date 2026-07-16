const CHALLENGE_MODELS = ["NeuralNet", "Chronos2"];

async function loadResults() {
  const candidates = window.STATIC_DASHBOARD
    ? ["./results.json"]
    : ["/api/results", "/static/results.json", "./results.json"];
  let lastError = null;
  for (const url of candidates) {
    try {
      const response = await fetch(url);
      if (response.ok) return response.json();
      const body = await response.json().catch(() => ({}));
      lastError = new Error(body.detail || `HTTP ${response.status} from ${url}`);
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError || new Error("Could not load challenge results");
}

function overviewHref() {
  return window.STATIC_DASHBOARD ? "index.html" : "/";
}

function modelHref(slug) {
  return window.STATIC_DASHBOARD
    ? `model.html?model=${encodeURIComponent(slug)}`
    : `/model/${slug}`;
}

function datasetHref() {
  return window.STATIC_DASHBOARD ? "dataset.html" : "/dataset";
}

function evaluationHref() {
  return window.STATIC_DASHBOARD ? "evaluation.html" : "/evaluation";
}

function wireSharedLinks() {
  if (typeof document === "undefined" || !document.querySelectorAll) return;
  document.querySelectorAll("[data-overview-link]").forEach((link) => {
    link.href = overviewHref();
  });
  document.querySelectorAll("[data-dataset-link]").forEach((link) => {
    link.href = datasetHref();
  });
  document.querySelectorAll("[data-evaluation-link]").forEach((link) => {
    link.href = evaluationHref();
  });
}

function fmt(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return Number(value).toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

function ratePct(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return `${fmt(Number(value) * 100, digits)}%`;
}

function signedPct(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  const points = Number(value) * 100;
  return `${points > 0 ? "+" : ""}${fmt(points, digits)}%`;
}

function modelByKey(data, key) {
  return (data.models || []).find((model) => model.key === key || model.slug === key);
}

function canonicalModel(data) {
  return data.selection?.canonical_model || data.config?.submission_model || "NeuralNet";
}

function modelLabel(data, key) {
  return modelByKey(data, key)?.label || key;
}

function summaryRows(
  data,
  {
    source = "benchmark",
    regime = "conditional",
    population = "common",
    aggregation = "global",
  } = {},
) {
  const candidates = source === "development"
    ? [data.dev_summary_all, data.dev_summary]
    : [data.benchmark_summary_all, data.benchmark_summary, data.cv_summary];
  const rows = candidates.find((candidate) => Array.isArray(candidate) && candidate.length) || [];
  return rows.filter((row) => {
    if (!CHALLENGE_MODELS.includes(row.model)) return false;
    if (row.strategy && row.strategy !== "direct") return false;
    if (row.evaluation_regime && row.evaluation_regime !== regime) return false;
    if (row.comparison_population && row.comparison_population !== population) return false;
    if (row.aggregation && row.aggregation !== aggregation) return false;
    return true;
  });
}

function cvRows(data, regime = "conditional") {
  const rows = Array.isArray(data.cv_results_all) && data.cv_results_all.length
    ? data.cv_results_all
    : (data.cv_results || []);
  return rows.filter((row) => (
    CHALLENGE_MODELS.includes(row.model)
    && (!row.strategy || row.strategy === "direct")
    && (!row.regime || row.regime === regime)
    && (!row.evaluation_regime || row.evaluation_regime === regime)
    && (!row.comparison_population || row.comparison_population === "common")
  ));
}

function primaryScore(data, source, model, metric = "WAPE", regime = "conditional") {
  return summaryRows(data, { source, regime }).find((row) => row.model === model)?.[metric];
}

function winnerFromRows(rows, metric = "WAPE") {
  const usable = rows.filter((row) => Number.isFinite(Number(row[metric])));
  if (!usable.length) return null;
  return usable.reduce((best, row) => Number(row[metric]) < Number(best[metric]) ? row : best);
}

function forecastsFor(data) {
  return data.forecasts_by_strategy?.direct || data.forecasts || {};
}

function updateSharedCopy(data) {
  const promoStrategy = document.getElementById("promo-strategy");
  if (promoStrategy) promoStrategy.textContent = `${data.config?.horizon || 7}-Day Direct Forecast`;
  const promoCount = document.getElementById("promo-model-count");
  if (promoCount) promoCount.textContent = "2 Contenders";
  const winner = canonicalModel(data);
  const footer = document.getElementById("footer-method-text");
  if (footer) {
    footer.textContent = `Winner selection uses development OOF only. Current selected forecast: ${modelLabel(data, winner)}.`;
  }
}

function renderNav(data, activeSlug = "") {
  const nav = document.getElementById("site-nav");
  if (!nav) return;
  const items = [
    { slug: "", label: "Challenge", color: "#ffffff", href: overviewHref() },
    { slug: "dataset", label: "Data story", color: "#a78bfa", href: datasetHref() },
    { slug: "evaluation", label: "Evaluation", color: "#9ca3af", href: evaluationHref() },
    ...(data.models || []).filter((model) => CHALLENGE_MODELS.includes(model.key)).map((model) => ({
      slug: model.slug,
      label: model.label,
      color: model.color,
      href: modelHref(model.slug),
    })),
  ];
  nav.innerHTML = items.map((item) => {
    const active = item.slug === activeSlug;
    return `<a class="nav-pill${active ? " active" : ""}" style="--pill-color:${item.color}" href="${item.href}">${item.label}</a>`;
  }).join("");
}

const CHART_GRID = "#e4e4e4";
const CHART_TEXT = "#6b6b6b";
if (window.Chart) {
  Chart.defaults.color = CHART_TEXT;
  Chart.defaults.font.family = "Roboto, -apple-system, sans-serif";
  Chart.defaults.borderColor = CHART_GRID;
}

wireSharedLinks();
