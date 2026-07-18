function renderDatasetCurrentDecision(data) {
  const note = document.getElementById("dataset-current-note");
  const status = data.project?.status === "complete" ? "complete" : "awaiting the Chronos-2 runtime";
  note.innerHTML = `<strong>Current result:</strong> the ${status} challenge compares a frozen direct NeuralNet with zero-shot Amazon Chronos-2 on the same 30 × 7 target grid. Best NN wins; no other estimator is an active contender.`;
}

async function main() {
  try {
    const data = await loadResults();
    renderNav(data, "dataset");
    updateSharedCopy(data);
    renderDatasetCurrentDecision(data);
  } catch (error) {
    document.getElementById("app").innerHTML = `<section class="panel error-panel"><h2>Could not load data-story metadata</h2><p>${error.message}</p></section>`;
  }
}

main();
