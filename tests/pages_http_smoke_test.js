"use strict";

const assert = require("assert");
const net = require("net");
const path = require("path");
const { once } = require("events");
const { spawn } = require("child_process");

const root = path.resolve(__dirname, "..");

async function availablePort() {
  const server = net.createServer();
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  const { port } = server.address();
  server.close();
  await once(server, "close");
  return port;
}

async function waitFor(url, child) {
  let lastError;
  for (let attempt = 0; attempt < 50; attempt += 1) {
    if (child.exitCode !== null) throw new Error(`HTTP server exited with ${child.exitCode}`);
    try {
      const response = await fetch(url);
      if (response.ok) return;
      lastError = new Error(`HTTP ${response.status}`);
    } catch (error) {
      lastError = error;
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw lastError || new Error(`Timed out waiting for ${url}`);
}

async function main() {
  const port = await availablePort();
  const origin = `http://127.0.0.1:${port}`;
  const child = spawn(
    "python3",
    ["-m", "http.server", String(port), "--bind", "127.0.0.1", "--directory", "docs"],
    { cwd: root, stdio: ["ignore", "ignore", "pipe"] },
  );

  try {
    await waitFor(`${origin}/`, child);
    for (const route of ["/", "/dataset.html", "/evaluation.html", "/model.html?model=chronos2"]) {
      const response = await fetch(`${origin}${route}`);
      assert.strictEqual(response.status, 200, `${route} returned ${response.status}`);
      const html = await response.text();
      assert.ok(html.includes("<title>NOTINO - chronos</title>"), `${route} has the wrong title`);
      assert.ok(html.includes("Quantity Forecast Dashboard"), `${route} has the wrong shared heading`);
      assert.ok(html.includes('<span class="brand-tagline">CHRONOS</span>'), `${route} has the wrong brand`);
    }

    const resultsResponse = await fetch(`${origin}/results.json`);
    assert.strictEqual(resultsResponse.status, 200);
    const results = await resultsResponse.json();
    assert.strictEqual(Object.keys(results.history).length, 30);
    assert.deepStrictEqual(Object.keys(results.forecasts), ["NeuralNet", "Chronos2"]);

    const appResponse = await fetch(`${origin}/app.js`);
    assert.strictEqual(appResponse.status, 200);
    const app = await appResponse.text();
    assert.ok(app.includes("data: { labels, datasets }"));
    assert.ok(app.includes("display: true"));
  } finally {
    if (child.exitCode === null) {
      child.kill("SIGTERM");
      await Promise.race([
        once(child, "exit"),
        new Promise((resolve) => setTimeout(resolve, 2000)),
      ]);
    }
  }

  console.log("Pages HTTP smoke checks passed");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
