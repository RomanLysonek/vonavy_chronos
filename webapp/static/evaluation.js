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
  const audit = data.challenge?.final_audit || {};
  const auditRows = Object.fromEntries((audit.rows || []).map((row) => [row.model, row]));
  const auditDelta = Number.isFinite(Number(auditRows.Chronos2?.WAPE)) && Number.isFinite(Number(auditRows.NeuralNet?.WAPE))
    ? Number(auditRows.Chronos2.WAPE) / Number(auditRows.NeuralNet.WAPE) - 1
    : null;
  document.getElementById("audit-result-note").innerHTML = `<strong>Consumed result:</strong> ${audit.winner ? modelLabel(data, audit.winner) : "Unavailable"} won the final audit${auditDelta === null ? "" : `; Chronos-2 WAPE was ${signedPct(auditDelta)} relative to Best NN`}. This evidence is non-selection and cannot be made fresh by rerunning it.`;

  const probability = data.probabilistic_evaluation || {};
  const probabilityRows = Object.fromEntries((probability.metrics || []).map((row) => [row.origin_type, row]));
  const recent = probabilityRows.recent_benchmark;
  const finalAudit = probabilityRows.final_audit;
  const probabilityItems = probability.status === "evaluated" && recent
    ? [
      ["Recent diagnostic", `${ratePct(recent.interval_coverage)} coverage for the nominal 80% interval; q10/q50/q90 empirical rates ${ratePct(recent.empirical_q10)} / ${ratePct(recent.empirical_q50)} / ${ratePct(recent.empirical_q90)}.`],
      ["Recent interval width", `${fmt(recent.interval_mean_width)} mean width; ${fmt(recent.interval_normalized_width, 2)} normalised width.`],
      ["Final audit", finalAudit ? `${ratePct(finalAudit.interval_coverage)} interval coverage; q50 empirical rate ${ratePct(finalAudit.empirical_q50)}.` : "No final-audit interval evidence."],
      ["Interpretation", "Intervals quantify uncertainty but do not overturn the point-forecast decision: Chronos q50 loses on the pre-specified WAPE comparison."],
    ]
    : [["Not evaluated", probability.reason || "Authenticated quantile artifacts are unavailable."]];
  document.getElementById("evaluation-probability").innerHTML = probabilityItems.map(([title, body]) => `<div class="definition-item"><strong>${title}</strong><span>${body}</span></div>`).join("");

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
