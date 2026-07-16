function renderDatasetCurrentDecision(data) {
  const note = document.getElementById("dataset-current-note");
  const status = data.project?.status === "complete" ? "complete" : "awaiting the Chronos-2 runtime";
  note.innerHTML = `<strong>Current challenge:</strong> forecast QuantityApp + QuantityWeb for the same 30 × 7 target grid with a frozen direct NeuralNet and Amazon Chronos-2. The checked-in result is ${status}; no historical baseline is carried into the active leaderboard.`;
}

function renderDatasetDecisionTrail(data) {
  const cfg = data.config || {};
  const lineage = data.challenge?.lineage || [];
  const stages = [
    {
      badge: "Data contract",
      title: "Repair the panel before modeling",
      body: "Daily reindexing, explicit gap/unavailable/pre-launch states, availability-aware lags, common scoring rows and leakage guards define what one observation means.",
      detail: "Missing rows are never silently converted to sales zeros",
    },
    ...lineage.map((item, index) => ({
      badge: `Incumbent ${index + 1}`,
      title: item.step,
      body: item.reason,
      detail: item.decision,
    })),
    {
      badge: "New challenge",
      title: "Introduce Chronos-2 without moving the goalposts",
      body: `Chronos receives the same seven target dates, with cross-learning ${cfg.chronos2_cross_learning ? "enabled" : "disabled"} and known-future covariates ${cfg.chronos2_covariates ? "enabled" : "disabled"}. Stockouts and gaps remain missing history, not zeros.`,
      detail: "Same origins, same rows, same WAPE",
    },
  ];
  document.getElementById("dataset-decision-trail").innerHTML = stages.map((stage) => `
    <article class="dataset-decision-card">
      <span class="dataset-decision-badge">${stage.badge}</span>
      <h3>${stage.title}</h3>
      <p>${stage.body}</p>
      <strong>${stage.detail}</strong>
    </article>
  `).join("");
}

function renderDatasetResponses() {
  const rows = [
    ["Staggered launches and isolated gaps", "Reindex every product to a daily calendar, keep inserted gaps unknown, and retain separate first-seen and first-available lifecycle clocks."],
    ["Stock-constrained realized sales", "Exclude unavailable observations from supervised targets and primary scoring. Their quantities are censored from demand lags rather than learned as zero demand."],
    ["Complete weekly cycle", "Use target-date calendar features and same-weekday seasonal anchors. The seven-day horizon covers one full Monday-to-Sunday cycle."],
    ["Annual retail events", "Keep nullable annual lags and deterministic event-distance features instead of dropping young products that lack a full year of history."],
    ["Price and promotion semantics", "Campaign subtype remains categorical. Web/app discounts, effective prices, app advantage and event context are represented explicitly."],
    ["Related products", "The incumbent pools all products through embeddings; Chronos-2 may additionally use cross-series group attention. Product identity is preserved in both routes."],
    ["Channel migration", "Total demand stays canonical. The historical app-share auxiliary experiment is documented as part of the incumbent lineage but is not another contender in this repo."],
    ["Unseen future availability", "Future ProductAvailable is forbidden. Chronos receives only genuinely known target-date covariates; the incumbent uses the same information boundary."],
    ["Right-skewed quantities", "The incumbent learns a guarded log-residual around a seasonal anchor. Chronos contributes a zero-shot median plus probabilistic quantiles."],
    ["Fair model comparison", "Both contenders are evaluated on identical rolling origins and common finite predictions. Global conditional-demand WAPE is primary."],
    ["Model selection", "Development OOF chooses the winner. The recent benchmark checks stability and is not reused as a tuning set."],
    ["Final artifacts", "The repo exports separate incumbent and Chronos submissions, plus submission.csv for the development-selected winner."],
  ];
  document.getElementById("dataset-response-list").innerHTML = rows.map(([title, body]) => `
    <div class="definition-item dataset-response-item"><strong>${title}</strong><span>${body}</span></div>
  `).join("");
}

async function main() {
  try {
    const data = await loadResults();
    renderNav(data, "dataset");
    updateSharedCopy(data);
    renderDatasetCurrentDecision(data);
    renderDatasetDecisionTrail(data);
    renderDatasetResponses(data);
  } catch (error) {
    document.getElementById("app").innerHTML = `<section class="panel error-panel"><h2>Could not load data-story metadata</h2><p>${error.message}</p></section>`;
  }
}

main();
