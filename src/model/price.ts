import type { Matrix, PricePaths, ProcessParameters } from "./types.ts";

function finiteScalar(value: number, name: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError(`${name} must be a finite number`);
  }
  return value;
}

function validateMatrix(values: Matrix, name: string): number[][] {
  if (!Array.isArray(values) || values.length === 0) {
    throw new TypeError(`${name} must be a nonempty matrix`);
  }
  const width = values[0]?.length;
  if (!Number.isInteger(width) || width === 0) {
    throw new TypeError(`${name} must have at least one column`);
  }
  return values.map((row, i) => {
    if (!Array.isArray(row) || row.length !== width) {
      throw new TypeError(`${name} must be rectangular (row ${i} has the wrong length)`);
    }
    return row.map((value, j) => finiteScalar(value, `${name}[${i}][${j}]`));
  });
}

/**
 * Euler–Maruyama log-price simulation using caller-supplied standard normals.
 * The production baseline passes a [100, 50] innovation matrix. Returns used
 * by the controller are Delta X + sigma^2 dt / 2, not exp(Delta X) - 1.
 */
export function generatePricePaths(
  innovations: Matrix,
  parameters: ProcessParameters,
): PricePaths {
  const normals = validateMatrix(innovations, "innovations");
  const nu0 = finiteScalar(parameters.nu0, "nu0");
  const nu1 = finiteScalar(parameters.nu1, "nu1");
  const kappa = finiteScalar(parameters.kappa, "kappa");
  const sigma = finiteScalar(parameters.sigma, "sigma");
  const T = finiteScalar(parameters.T ?? 1, "T");
  if (sigma < 0) throw new RangeError("sigma must be nonnegative");
  if (T <= 0) throw new RangeError("T must be positive");

  const pathCount = normals.length;
  const stepCount = normals[0]!.length;
  const dt = T / stepCount;
  if (kappa > 0 && kappa * dt >= 2) {
    throw new RangeError(
      "Euler log-price recursion is not mean-square stable for this kappa and step size",
    );
  }

  const times = Array.from({ length: stepCount + 1 }, (_, n) => (n * T) / stepCount);
  const logPrices = Array.from({ length: pathCount }, () => Array(stepCount + 1).fill(0));
  const returnIncrements = Array.from({ length: pathCount }, () => Array(stepCount).fill(0));
  const noiseScale = sigma * Math.sqrt(dt);

  for (let n = 0; n < stepCount; n += 1) {
    for (let i = 0; i < pathCount; i += 1) {
      const oldX = logPrices[i]![n]!;
      const drift = nu0 + nu1 * times[n]! - kappa * oldX;
      const dx = drift * dt + noiseScale * normals[i]![n]!;
      const nextX = oldX + dx;
      const dr = dx + 0.5 * sigma * sigma * dt;
      if (!Number.isFinite(nextX)) throw new RangeError(`nonfinite log-price at path ${i}, step ${n + 1}`);
      if (!Number.isFinite(dr)) throw new RangeError(`nonfinite return increment at path ${i}, step ${n}`);
      logPrices[i]![n + 1] = nextX;
      returnIncrements[i]![n] = dr;
    }
  }

  const prices = logPrices.map((row, i) =>
    row.map((value, n) => {
      const price = Math.exp(value);
      if (!Number.isFinite(price)) throw new RangeError(`nonfinite price at path ${i}, time index ${n}`);
      return price;
    }),
  );
  return { times, logPrices, prices, returnIncrements };
}
