function originPills(values) {
  return (values || []).map((value) => `<span class="origin-pill">${String(value).slice(0, 10)}</span>`).join("") || `<span class="empty-state">Origins are written during the next pipeline run.</span>`;
}

function inferredOrigins(data, originType) {
  const rows = data.prediction_diagnostics_by_origin || [];
  return [...new Set(rows.filter((row) => row.origin_type === originType).map((row) => row.origin))].sort();
}

function registeredOrigins(data, key, fallbackType) {
  const registered = (data.evaluation_origins || []).find((entry) => entry.key === key);
  return registered?.origins?.length ? registered.origins : inferredOrigins(data, fallbackType);
}

function renderKpis(data) {
  const dev = data.challenge?.development || {};
  const benchmark = data.challenge?.recent_benchmark || {};
  const cards = [
    ["Development origins", data.config?.n_dev_origins ?? "—", "winner selection"],
    ["Recent diagnostic folds", data.config?.n_cv_folds ?? "—", "previously inspected"],
    ["Forecast horizon", `${data.config?.horizon || 7} days`, "direct multi-horizon"],
    ["Primary metric", data.challenge?.selection_metric || "WAPE", "conditional / common / global"],
  ];
  document.getElementById("evaluation-kpis").innerHTML = cards.map(([label, value, sub]) => `<article class="kpi-card"><p class="kpi-label">${label}</p><p class="kpi-value">${value}</p><p class="kpi-sub">${sub}</p></article>`).join("");

  document.getElementById("development-origins").innerHTML = originPills(registeredOrigins(data, "development", "development"));
  document.getElementById("benchmark-origins").innerHTML = originPills(registeredOrigins(data, "recent_diagnostic", "recent_benchmark"));
  document.getElementById("audit-origins").innerHTML = originPills(registeredOrigins(data, "final_audit", "final_audit"));

  const weights = data.config?.validation_stratum_weights || {};
  const strata = [
    ["January / February proxy", "winter_test_like", "Closest seasonal analogue to the supplied January forecast week."],
    ["Regular periods", "regular", "Ordinary trading windows outside winter and major retail events."],
    ["Holiday / event stress", "holiday_event", "Late-November and December demand shifts, including Black Friday and pre-Christmas."],
  ];
  document.getElementById("strata-list").innerHTML = strata.map(([label, key, text]) => `<div class="definition-item"><strong>${label}</strong><span>${text} Weight: ${ratePct(weights[key], 0)}.</span></div>`).join("");
}

async function main() {
  try {
    const data = await loadResults();
    renderNav(data, "evaluation");
    updateSharedCopy(data);
    renderKpis(data);
  } catch (error) {
    document.getElementById("app").innerHTML = `<section class="panel error-panel"><h2>Could not load evaluation data</h2><p>${error.message}</p></section>`;
  }
}

main();
