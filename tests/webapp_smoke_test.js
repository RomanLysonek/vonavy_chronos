"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const { spawnSync } = require("child_process");

const root = path.resolve(__dirname, "..");
const staticDir = path.join(root, "webapp", "static");
const docsDir = path.join(root, "docs");

for (const name of ["common.js", "app.js", "model.js", "evaluation.js", "dataset.js"]) {
  const result = spawnSync(process.execPath, ["--check", path.join(staticDir, name)], { encoding: "utf8" });
  assert.strictEqual(result.status, 0, `${name} syntax check failed:\n${result.stderr}`);
}

const data = JSON.parse(fs.readFileSync(path.join(staticDir, "results.json"), "utf8"));
assert.strictEqual(data.schema_version, "vonavy-chronos-v2");
assert.deepStrictEqual(data.models.map((model) => model.key), ["NeuralNet", "Chronos2"]);
assert.strictEqual(data.project.status, "complete");
assert.deepStrictEqual(Object.keys(data.forecasts), ["NeuralNet", "Chronos2"]);
assert.strictEqual(data.probabilistic_evaluation.status, "evaluated");
assert.ok(data.provenance.run_id);

function assertNoLegacyModel(value) {
  if (Array.isArray(value)) return value.forEach(assertNoLegacyModel);
  if (!value || typeof value !== "object") return;
  if (Object.prototype.hasOwnProperty.call(value, "model")) {
    assert.ok(["NeuralNet", "Chronos2"].includes(value.model), `legacy model leaked: ${value.model}`);
  }
  Object.values(value).forEach(assertNoLegacyModel);
}
assertNoLegacyModel(data);

const context = { window: {}, console };
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(staticDir, "common.js"), "utf8"), context);
assert.strictEqual(vm.runInContext("CHALLENGE_MODELS.join(',')", context), "NeuralNet,Chronos2");
assert.strictEqual(vm.runInContext("canonicalModel({selection:{canonical_model:'Chronos2'}})", context), "Chronos2");
assert.strictEqual(vm.runInContext("summaryRows({benchmark_summary_all:[{model:'NeuralNet',strategy:'direct',evaluation_regime:'conditional',comparison_population:'common',aggregation:'global',WAPE:0.2},{model:'XGBoost',strategy:'direct',evaluation_regime:'conditional',comparison_population:'common',aggregation:'global',WAPE:0.1}]},{source:'benchmark'}).length", context), 1);

const index = fs.readFileSync(path.join(staticDir, "index.html"), "utf8");
assert.ok(index.includes("Best NN vs Chronos-2"));
assert.ok(!index.includes("One frozen incumbent"));
assert.ok(!index.includes("No irrelevant leaderboard"));
assert.ok(index.includes('<span class="brand-logo">NOTINO</span>'));
assert.ok(index.includes('<span class="brand-tagline">/ Interview Assignment</span>'));
assert.ok(!index.includes("VONAVY_CHRONOS"));
assert.ok(!index.includes("Foundation Model Challenge"));
assert.ok(index.includes("Sanity baseline"));
assert.ok(index.includes("Published provenance"));
for (const legacy of ["XGBoost", "LightGBM", "Dynamic Ridge", "Moving Average", "Seasonal Naive", "Ensemble"]) {
  assert.ok(!index.includes(legacy), `legacy contender visible in index.html: ${legacy}`);
}

const evaluation = fs.readFileSync(path.join(staticDir, "evaluation.html"), "utf8");
assert.ok(evaluation.includes('class="model-hero evaluation-hero"'));
assert.ok(evaluation.includes("One fixed contract for both contenders"));
assert.ok(evaluation.includes("Development OOF selects the winner"));
assert.ok(!evaluation.includes("Leakage-Safe Head-to-Head"));
assert.ok(!evaluation.includes("What is fixed, what selects the winner"));

const expectedPromoFacts = [
  "30 Product Time Series",
  "7-Day Direct Forecast",
  "2 Contenders",
  "Same Walk-Forward Test",
];
for (const htmlName of ["index.html", "dataset.html", "evaluation.html", "model.html"]) {
  const html = fs.readFileSync(path.join(staticDir, htmlName), "utf8");
  const promo = html.match(/<div class="promo-bar">([\s\S]*?)<\/div>/);
  assert.ok(promo, `missing promo bar in ${htmlName}`);
  for (const fact of expectedPromoFacts) {
    assert.ok(promo[1].includes(fact), `${htmlName} has inconsistent promo fact: ${fact}`);
  }
  assert.ok(html.includes('lang="en-GB"'));
}
assert.ok(fs.readFileSync(path.join(staticDir, "common.js"), "utf8").includes(
  'item.textContent = "Same Walk-Forward Test"',
));

const styles = fs.readFileSync(path.join(staticDir, "styles.css"), "utf8");
assert.ok(styles.includes("scrollbar-gutter: stable"));
assert.ok(styles.includes("grid-template-columns: repeat(4, minmax(0, 1fr))"));
assert.ok(styles.includes(".promo-bar > *"));
assert.ok(styles.includes("--content-max: 1280px"));
assert.ok(styles.includes(".suite-switcher"));
for (const htmlName of ["index.html", "dataset.html", "evaluation.html", "model.html"]) {
  const html = fs.readFileSync(path.join(staticDir, htmlName), "utf8");
  assert.ok(html.includes("styles.css?v=chronos-3"), `${htmlName} uses stale strip CSS`);
}

const docsIndex = fs.readFileSync(path.join(docsDir, "index.html"), "utf8");
assert.ok(docsIndex.includes("window.STATIC_DASHBOARD = true"));
assert.ok(!docsIndex.includes('href="/static/'));
assert.ok(!docsIndex.includes('src="/static/'));
assert.deepStrictEqual(
  fs.readFileSync(path.join(root, "outputs", "results.json")),
  fs.readFileSync(path.join(staticDir, "results.json")),
);

console.log("webapp challenge smoke checks passed");
