import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { generatePricePaths, simulateSls, expectedGain, kStar } from "../src/model/index.ts";

const fixture = JSON.parse(
  readFileSync(new URL("./fixtures/numerical_parity.json", import.meta.url), "utf8"),
);
const snapshotSteps = fixture.snapshot_steps;
const returnSteps = fixture.return_steps;

function compareTree(actual, expected, label, absTolerance = 2e-12, relTolerance = 2e-12) {
  if (Array.isArray(expected)) {
    assert.ok(Array.isArray(actual), `${label}: expected array`);
    assert.equal(actual.length, expected.length, `${label}: array length`);
    expected.forEach((value, index) =>
      compareTree(actual[index], value, `${label}[${index}]`, absTolerance, relTolerance),
    );
    return;
  }
  assert.equal(typeof actual, "number", `${label}: expected number`);
  const tolerance = absTolerance + relTolerance * Math.abs(expected);
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `${label}: actual ${actual}, expected ${expected}, tolerance ${tolerance}`,
  );
}

function selectColumns(matrix, indexes) {
  return matrix.map((row) => indexes.map((index) => row[index]));
}

for (const [caseName, expectedCase] of Object.entries(fixture.cases)) {
  const prices = generatePricePaths(fixture.normals, expectedCase.parameters);
  assert.equal(prices.times.length, 51, `${caseName}: stored time count`);
  compareTree(prices.times, Array.from({ length: 51 }, (_, n) => n / 50), `${caseName}.times`, 2e-16, 2e-16);
  compareTree(
    selectColumns(prices.logPrices, snapshotSteps),
    expectedCase.price_snapshots.log_prices,
    `${caseName}.log_prices snapshots`,
  );
  compareTree(
    selectColumns(prices.prices, snapshotSteps),
    expectedCase.price_snapshots.prices,
    `${caseName}.prices snapshots`,
  );
  compareTree(
    selectColumns(prices.returnIncrements, returnSteps),
    expectedCase.price_snapshots.return_increments,
    `${caseName}.return increments`,
  );

  for (const K of [2, -2, 0, 1e-12]) {
    const expectedGains = expectedCase.controller[String(K)];
    const gains = simulateSls(prices.returnIncrements, 5000, K);
    for (const [field, actual] of [
      ["long_gain_snapshots", gains.longGain],
      ["short_gain_snapshots", gains.shortGain],
      ["total_gain_snapshots", gains.totalGain],
    ]) {
      compareTree(
        selectColumns(actual, snapshotSteps),
        expectedGains[field],
        `${caseName} K=${K} ${field}`,
        2e-8,
        2e-12,
      );
    }
    const terminal = gains.totalGain.map((row) => row[row.length - 1]);
    compareTree(terminal, expectedGains.terminal_total_gain, `${caseName} K=${K} terminal gains`, 2e-8, 2e-12);
    const terminalMean = terminal.reduce((sum, value) => sum + value, 0) / terminal.length;
    compareTree(terminalMean, expectedGains.terminal_mean_gain, `${caseName} K=${K} terminal mean`, 2e-8, 2e-12);

    const theoryTimes = expectedCase.theory_steps.map((index) => prices.times[index]);
    const theory = expectedGain(theoryTimes, {
      ...expectedCase.parameters,
      I0: 5000,
      K,
    });
    const theoryAbsTolerance = K === 1e-12 ? 2e-24 : 2e-9;
    compareTree(theory, expectedCase.theory[String(K)], `${caseName} K=${K} analytical expected gain`, theoryAbsTolerance, 2e-12);
  }

  const bound = kStar(expectedCase.parameters);
  if (expectedCase.k_star === null) {
    assert.equal(bound, null, `${caseName}: K-star applicability`);
  } else {
    compareTree(bound, expectedCase.k_star, `${caseName} K-star`, 2e-12, 2e-12);
  }
}

console.log("TypeScript/Python parity passed: 5 model regimes, 100 paths × 50 steps, K = 2, -2, 0, 1e-12.");
