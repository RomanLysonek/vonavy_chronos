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
assert.strictEqual(data.provenance.verification.status, "incomplete");
assert.strictEqual(data.publication_provenance.status, "authenticated");
assert.ok(data.publication_provenance.publication_id);

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
assert.ok(index.includes("Final challenge"));
assert.ok(index.includes("controlled negative-result experiment"));
assert.ok(!index.includes("One frozen incumbent"));
assert.ok(!index.includes("No irrelevant leaderboard"));
assert.ok(index.includes('<span class="brand-logo">NOTINO</span>'));
assert.ok(index.includes('<span class="brand-tagline">CHRONOS</span>'));
assert.ok(index.includes("<h1>Quantity Forecast Dashboard</h1>"));
assert.ok(!index.includes("/ Interview Assignment"));
assert.ok(!index.includes("VONAVY_CHRONOS"));
assert.ok(!index.includes("Foundation Model Challenge"));
assert.ok(index.includes("Sanity baseline"));
assert.ok(index.includes("Evidence provenance"));
assert.ok(index.includes("Recent diagnostic; each fold uses the same rows for both contenders"));
assert.ok(!index.includes("Recent benchmark; each fold"));
assert.ok(index.includes("Why Chronos-2 likely lost"));
assert.ok(index.includes("What would justify another attempt"));
const presentationFiles = [
  path.join(root, "README.md"),
  ...fs.readdirSync(staticDir).filter((name) => /\.(?:html|css|js)$/.test(name)).map((name) => path.join(staticDir, name)),
  ...fs.readdirSync(docsDir).filter((name) => /\.(?:html|css|js|md)$/.test(name)).map((name) => path.join(docsDir, name)),
];
const presentation = presentationFiles.map((name) => fs.readFileSync(name, "utf8")).join("\n");
for (const fragment of [
  "SUITE" + "_APPS",
  "suite-" + "switcher",
  "Classical" + " Forecasting",
  "Anomaly" + " Research",
  "vonava_" + "predikce",
  "vonave_" + "anomalie",
]) {
  assert.ok(!presentation.includes(fragment), `standalone identity leak: ${fragment}`);
}
for (const legacy of ["XGBoost", "LightGBM", "Dynamic Ridge", "Moving Average", "Seasonal Naive", "Ensemble"]) {
  assert.ok(!index.includes(legacy), `legacy contender visible in index.html: ${legacy}`);
}

const evaluation = fs.readFileSync(path.join(staticDir, "evaluation.html"), "utf8");
assert.ok(evaluation.includes('class="description-strip model-hero evaluation-hero"'));
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
const pageNames = ["index.html", "dataset.html", "evaluation.html", "model.html"];
function expectedPromoMarkup(datasetHref, evaluationHref) {
  return [
    '<div class="promo-bar">',
    `    <a class="promo-dataset-link" data-dataset-link href="${datasetHref}">${expectedPromoFacts[0]}</a>`,
    `    <span id="promo-strategy">${expectedPromoFacts[1]}</span>`,
    `    <span id="promo-model-count">${expectedPromoFacts[2]}</span>`,
    `    <a class="promo-evaluation-link" data-evaluation-link href="${evaluationHref}">${expectedPromoFacts[3]}</a>`,
    "  </div>",
  ].join("\n");
}
for (const [directory, datasetHref, evaluationHref] of [
  [staticDir, "/dataset", "/evaluation"],
  [docsDir, "./dataset.html", "./evaluation.html"],
]) {
  for (const htmlName of pageNames) {
    const html = fs.readFileSync(path.join(directory, htmlName), "utf8");
    const promo = html.match(/<div class="promo-bar">[\s\S]*?<\/div>/);
    assert.ok(promo, `missing promo bar in ${path.basename(directory)}/${htmlName}`);
    assert.strictEqual(
      promo[0],
      expectedPromoMarkup(datasetHref, evaluationHref),
      `inconsistent promo markup in ${path.basename(directory)}/${htmlName}`,
    );
  }
}
for (const htmlName of pageNames) {
  const html = fs.readFileSync(path.join(staticDir, htmlName), "utf8");
  assert.ok(html.includes('lang="en-GB"'));
  assert.strictEqual((html.match(/<title/g) || []).length, 1);
  assert.ok(html.includes("<title>NOTINO - chronos</title>"));
  assert.strictEqual((html.match(/class="description-strip model-hero\b[^"]*"/g) || []).length, 1);
  assert.strictEqual((html.match(/class="[^"]*\bmodel-hero\b[^"]*"/g) || []).length, 1);
  assert.ok(
    /<header class="hero[^"]*">[\s\S]*?<\/header>\s*<header class="description-strip model-hero/.test(html),
    `${htmlName} description strip is not immediately below the shared hero`,
  );
}

const commonScript = fs.readFileSync(path.join(staticDir, "common.js"), "utf8");
for (const selector of [
  "promo-dataset-link",
  "promo-strategy",
  "promo-model-count",
  "promo-evaluation-link",
]) {
  assert.ok(!commonScript.includes(selector), `shared JS mutates the stable promo item: ${selector}`);
}

const stablePromoElements = new Map([
  ["promo-strategy", { textContent: expectedPromoFacts[1] }],
  ["promo-model-count", { textContent: expectedPromoFacts[2] }],
  ["footer-method-text", { textContent: "" }],
]);
context.document = {
  getElementById(id) { return stablePromoElements.get(id) || null; },
  querySelectorAll() {
    return [
      { textContent: expectedPromoFacts[0] },
      { textContent: expectedPromoFacts[3] },
    ];
  },
};
vm.runInContext(
  "updateSharedCopy({config:{num_products:99,horizon:14},selection:{canonical_model:'Chronos2'},models:[{key:'Chronos2',label:'Chronos-2'}],provenance:{}})",
  context,
);
assert.strictEqual(stablePromoElements.get("promo-strategy").textContent, expectedPromoFacts[1]);
assert.strictEqual(stablePromoElements.get("promo-model-count").textContent, expectedPromoFacts[2]);

const styles = fs.readFileSync(path.join(staticDir, "styles.css"), "utf8");
assert.ok(styles.includes("scrollbar-gutter: stable"));
for (const declaration of [
  "--page-padding-inline: 56px;",
  "--description-strip-padding-block: 40px;",
  "--description-strip-border-width: 6px;",
  "--description-strip-min-height: 300px;",
]) {
  assert.ok(styles.includes(declaration));
}
for (const declaration of [
  "box-sizing: border-box;",
  "width: 100%;",
  "max-width: none;",
  "min-height: var(--description-strip-min-height);",
  "margin: 0;",
  "padding: var(--description-strip-padding-block) var(--page-padding-inline);",
  "border-bottom: var(--description-strip-border-width) solid var(--mc);",
]) {
  assert.ok(styles.match(/\.description-strip\s*\{[^}]*\}/s)[0].includes(declaration));
}
assert.strictEqual((styles.match(/^\.description-strip\s*\{/gm) || []).length, 1);
assert.strictEqual((styles.match(/^\.model-hero\s*\{/gm) || []).length, 0);
assert.ok(
  /@media \(max-width: 900px\)\s*\{\s*:root\s*\{\s*--page-padding-inline: 24px;\s*\}/s.test(styles),
);
assert.ok(styles.includes("grid-template-columns: repeat(4, minmax(0, 1fr))"));
const promoRule = styles.match(/\.promo-bar\s*\{[^}]*\}/s)[0];
for (const declaration of [
  "box-sizing: border-box;",
  "width: 100%;",
  "min-height: 40px;",
  "margin: 0;",
  "padding: 8px var(--page-padding-inline);",
  "display: grid;",
  "grid-template-columns: repeat(4, minmax(0, 1fr));",
  "align-items: center;",
  "column-gap: 24px;",
  "border-bottom: 1px solid var(--hairline);",
  "font-size: 10px;",
  "line-height: 1.2;",
  "letter-spacing: 0.04em;",
]) {
  assert.ok(promoRule.includes(declaration), `promo bar differs from prediction header: ${declaration}`);
}
const promoChildRule = styles.match(/\.promo-bar > \*\s*\{[^}]*\}/s)[0];
for (const declaration of ["min-width: 0;", "white-space: nowrap;"]) {
  assert.ok(promoChildRule.includes(declaration));
}
assert.ok(/\.promo-bar > :first-child\s*\{[^}]*text-align: left;/s.test(styles));
assert.ok(
  /\.promo-bar > :nth-child\(2\),\s*\.promo-bar > :nth-child\(3\)\s*\{[^}]*text-align: center;/s.test(styles),
);
assert.ok(/\.promo-bar > :last-child\s*\{[^}]*text-align: right;/s.test(styles));
assert.ok(
  /@media \(max-width: 800px\)\s*\{[\s\S]*?\.promo-bar\s*\{[^}]*min-height: 57px;[^}]*padding: 8px 24px;[^}]*grid-template-columns: repeat\(2, minmax\(0, 1fr\)\);[^}]*column-gap: 24px;[^}]*row-gap: 8px;[^}]*\}[\s\S]*?\.promo-bar > :nth-child\(odd\)\s*\{[^}]*text-align: left;[^}]*\}[\s\S]*?\.promo-bar > :nth-child\(even\)\s*\{[^}]*text-align: right;/s.test(styles),
);
assert.ok(
  /@media \(max-width: 480px\)\s*\{[\s\S]*?\.promo-bar\s*\{[^}]*min-height: 89px;[^}]*grid-template-columns: minmax\(0, 1fr\);[^}]*row-gap: 8px;[^}]*\}[\s\S]*?\.promo-bar > :nth-child\(n\)\s*\{[^}]*text-align: left;/s.test(styles),
);

function matchingBrace(source, openIndex) {
  let depth = 0;
  for (let index = openIndex; index < source.length; index += 1) {
    if (source[index] === "{") depth += 1;
    if (source[index] === "}") {
      depth -= 1;
      if (depth === 0) return index;
    }
  }
  throw new Error("Unbalanced CSS block");
}

function promoSelectorMatches(selector, childIndex) {
  const childSelector = selector.trim().replace(/^\.promo-bar\s*>\s*/, "");
  if (childSelector === "*") return true;
  if (childSelector === ":first-child") return childIndex === 1;
  if (childSelector === ":last-child") return childIndex === expectedPromoFacts.length;
  const nth = childSelector.match(/^:nth-child\((n|odd|even|\d+)\)$/);
  if (!nth) return false;
  if (nth[1] === "n") return true;
  if (nth[1] === "odd") return childIndex % 2 === 1;
  if (nth[1] === "even") return childIndex % 2 === 0;
  return childIndex === Number(nth[1]);
}

function selectorSpecificity(selector) {
  return (selector.match(/[.#:]|:(?=[\w-])/g) || []).length;
}

function computedPromoTextAlign(css, childIndex, viewportWidth) {
  const mediaRanges = [];
  for (const match of css.matchAll(/@media \(max-width: (\d+)px\)\s*\{/g)) {
    mediaRanges.push({
      maxWidth: Number(match[1]),
      start: match.index,
      end: matchingBrace(css, match.index + match[0].lastIndexOf("{")),
    });
  }

  let computed = null;
  let order = 0;
  for (const match of css.matchAll(/(\.promo-bar\s*>\s*[^{,]+(?:,\s*\.promo-bar\s*>\s*[^{,]+)*)\s*\{([^}]*)\}/g)) {
    const media = mediaRanges.find(({ start, end }) => start < match.index && match.index < end);
    if (media && viewportWidth > media.maxWidth) continue;
    const alignment = match[2].match(/text-align:\s*(left|center|right)\s*;/);
    if (!alignment) continue;
    for (const selector of match[1].split(",")) {
      if (!promoSelectorMatches(selector, childIndex)) continue;
      const candidate = { value: alignment[1], specificity: selectorSpecificity(selector), order };
      if (
        !computed
        || candidate.specificity > computed.specificity
        || (candidate.specificity === computed.specificity && candidate.order > computed.order)
      ) {
        computed = candidate;
      }
    }
    order += 1;
  }
  return computed?.value;
}

assert.strictEqual(computedPromoTextAlign(styles, 2, 480), "left");
assert.strictEqual(computedPromoTextAlign(styles, 4, 480), "left");
const heroRule = styles.match(/header\.hero\s*\{[^}]*\}/s)[0];
assert.ok(heroRule.includes("padding: 28px var(--page-padding-inline) 0;"));
const heroTopRule = styles.match(/\.hero-top\s*\{[^}]*\}/s)[0];
assert.ok(!heroTopRule.includes("max-width"));
assert.ok(!heroTopRule.includes("margin:"));
const navRule = styles.match(/nav\.site-nav\s*\{[^}]*\}/s)[0];
assert.ok(!navRule.includes("max-width"));
assert.ok(!navRule.includes("margin:"));
assert.ok(styles.includes("--content-max: 1280px"));
assert.ok(!styles.includes("suite-" + "switcher"));
for (const htmlName of ["index.html", "dataset.html", "evaluation.html", "model.html"]) {
  const html = fs.readFileSync(path.join(staticDir, htmlName), "utf8");
  assert.ok(html.includes("styles.css?v=chronos-7"), `${htmlName} uses stale strip CSS`);
}

for (const directory of [staticDir, docsDir]) {
  const javascript = fs.readdirSync(directory)
    .filter((name) => name.endsWith(".js"))
    .map((name) => fs.readFileSync(path.join(directory, name), "utf8"))
    .join("\n");
  assert.ok(!javascript.includes("document.title"));
  assert.ok(!javascript.includes("page-title"));
}

for (const htmlName of ["index.html", "dataset.html", "evaluation.html", "model.html"]) {
  const docsHtml = fs.readFileSync(path.join(docsDir, htmlName), "utf8");
  assert.ok(docsHtml.includes("window.STATIC_DASHBOARD = true"));
  assert.ok(!docsHtml.includes('href="/static/'));
  assert.ok(!docsHtml.includes('src="/static/'));
  assert.ok(!docsHtml.includes('href="/'));
  assert.ok(docsHtml.includes("<title>NOTINO - chronos</title>"));
  assert.strictEqual((docsHtml.match(/class="description-strip model-hero\b[^"]*"/g) || []).length, 1);
  assert.strictEqual((docsHtml.match(/class="[^"]*\bmodel-hero\b[^"]*"/g) || []).length, 1);
}
assert.deepStrictEqual(
  fs.readFileSync(path.join(root, "outputs", "results.json")),
  fs.readFileSync(path.join(staticDir, "results.json")),
);

function runProductChart(scriptName, renderExpression, payload = data) {
  const elements = new Map();
  const parent = { appendChild(element) { elements.set(element.id, element); } };
  elements.set("chart-product", { id: "chart-product", hidden: false, parentElement: parent });
  const document = {
    getElementById(id) { return elements.get(id) || null; },
    createElement() { return { hidden: false, textContent: "", className: "", id: "" }; },
    querySelectorAll() { return []; },
  };
  const charts = [];
  function Chart(element, config) {
    this.element = element;
    this.config = config;
    this.destroy = () => {};
    charts.push(this);
  }
  Chart.defaults = { font: {} };
  const chartContext = {
    window: { Chart, STATIC_DASHBOARD: false, location: { pathname: "/", search: "" } },
    document,
    Chart,
    console,
    URLSearchParams,
    fetch: async () => { throw new Error("fetch should not run in chart unit tests"); },
  };
  vm.createContext(chartContext);
  vm.runInContext(fs.readFileSync(path.join(staticDir, "common.js"), "utf8"), chartContext);
  const source = fs.readFileSync(path.join(staticDir, scriptName), "utf8").replace(/\nmain\(\);\s*$/, "\n");
  vm.runInContext(source, chartContext);
  chartContext.payload = payload;
  vm.runInContext(renderExpression, chartContext);
  return { charts, elements };
}

function productChartConfig(scriptName, renderExpression) {
  const { charts } = runProductChart(scriptName, renderExpression);
  assert.strictEqual(charts.length, 1, `${scriptName} did not create exactly one product chart`);
  return charts[0].config;
}

const firstProduct = Object.keys(data.history).sort((a, b) => Number(a) - Number(b))[0];
const historyCount = data.history[firstProduct].dates.length;
const forecastCount = data.forecasts.Chronos2[firstProduct].dates.length;
const challengeChart = productChartConfig(
  "app.js",
  `renderProductChart(payload, ${JSON.stringify(firstProduct)})`,
);
assert.strictEqual(challengeChart.data.labels.length, historyCount + forecastCount);
assert.strictEqual(challengeChart.data.datasets.find((set) => set.label === "History").data.filter(Number.isFinite).length, historyCount);
assert.strictEqual(challengeChart.data.datasets.find((set) => set.label === "Best NN").data.filter(Number.isFinite).length, forecastCount);
assert.strictEqual(challengeChart.data.datasets.find((set) => set.label === "Chronos-2").data.filter(Number.isFinite).length, forecastCount);
assert.strictEqual(challengeChart.data.datasets.find((set) => set.label === "Best NN").borderColor, data.models.find((model) => model.key === "NeuralNet").color);
assert.strictEqual(challengeChart.data.datasets.find((set) => set.label === "Chronos-2").borderColor, data.models.find((model) => model.key === "Chronos2").color);
assert.strictEqual(challengeChart.data.datasets.length, 5);
assert.ok(challengeChart.data.labels.every((label) => /^\d{4}-\d{2}-\d{2}$/.test(label)));
assert.strictEqual(challengeChart.options.scales.x.display, true);
assert.strictEqual(challengeChart.options.scales.x.ticks.display, true);

const challengeForecastOnlyChart = productChartConfig(
  "app.js",
  `showProductHistory = false; renderProductChart(payload, ${JSON.stringify(firstProduct)})`,
);
assert.strictEqual(challengeForecastOnlyChart.data.labels.length, forecastCount);
assert.ok(!challengeForecastOnlyChart.data.datasets.some((set) => set.label === "History"));

const incumbentOnly = JSON.parse(JSON.stringify(data));
delete incumbentOnly.forecasts.Chronos2;
delete incumbentOnly.forecasts_by_strategy.direct.Chronos2;
incumbentOnly.models.find((model) => model.key === "Chronos2").available = false;
const incumbentOnlyState = runProductChart(
  "app.js",
  `renderProductChart(payload, ${JSON.stringify(firstProduct)})`,
  incumbentOnly,
);
assert.strictEqual(incumbentOnlyState.charts.length, 1);
assert.strictEqual(
  incumbentOnlyState.charts[0].config.data.datasets.map((set) => set.label).join(","),
  "History,Best NN",
);

const modelChart = productChartConfig(
  "model.js",
  `renderProduct(payload, modelByKey(payload, "chronos2"), ${JSON.stringify(firstProduct)})`,
);
assert.strictEqual(modelChart.data.labels.length, historyCount + forecastCount);
assert.strictEqual(modelChart.data.datasets.find((set) => set.label === "History").data.filter(Number.isFinite).length, historyCount);
assert.strictEqual(modelChart.data.datasets.find((set) => set.label === "Chronos-2").data.filter(Number.isFinite).length, forecastCount);
assert.strictEqual(modelChart.data.datasets.find((set) => set.label === "Chronos-2").borderColor, data.models.find((model) => model.key === "Chronos2").color);
assert.strictEqual(modelChart.data.datasets.length, 4);
assert.ok(modelChart.data.labels.every((label) => /^\d{4}-\d{2}-\d{2}$/.test(label)));
assert.strictEqual(modelChart.options.scales.x.display, true);
assert.strictEqual(modelChart.options.scales.x.ticks.display, true);

const malformed = JSON.parse(JSON.stringify(data));
delete malformed.history[firstProduct];
const malformedState = runProductChart(
  "app.js",
  `renderProductChart(payload, ${JSON.stringify(firstProduct)})`,
  malformed,
);
assert.strictEqual(malformedState.charts.length, 0);
assert.ok(malformedState.elements.get("chart-product-error").textContent.includes("Product explorer unavailable"));
assert.strictEqual(malformedState.elements.get("chart-product").hidden, true);

console.log("webapp challenge smoke checks passed");
