import type { Matrix, SlsGains } from "./types.ts";

function finiteScalar(value: number, name: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError(`${name} must be a finite number`);
  }
  return value;
}

function validateIncrements(values: Matrix): number[][] {
  if (!Array.isArray(values) || values.length === 0 || !Array.isArray(values[0]) || values[0].length === 0) {
    throw new TypeError("returnIncrements must be a nonempty matrix");
  }
  const width = values[0].length;
  return values.map((row, i) => {
    if (!Array.isArray(row) || row.length !== width) {
      throw new TypeError("returnIncrements must be rectangular");
    }
    return row.map((value, j) => finiteScalar(value, `returnIncrements[${i}][${j}]`));
  });
}

/** Apply both signed-K SLS legs using each leg's old gain on every increment. */
export function simulateSls(returnIncrements: Matrix, I0: number, K: number): SlsGains {
  const drs = validateIncrements(returnIncrements);
  I0 = finiteScalar(I0, "I0");
  K = finiteScalar(K, "K");
  if (I0 <= 0) throw new RangeError("I0 must be positive");

  const pathCount = drs.length;
  const stepCount = drs[0]!.length;
  const longGain = Array.from({ length: pathCount }, () => Array(stepCount + 1).fill(0));
  const shortGain = Array.from({ length: pathCount }, () => Array(stepCount + 1).fill(0));

  for (let n = 0; n < stepCount; n += 1) {
    for (let i = 0; i < pathCount; i += 1) {
      const oldLong = longGain[i]![n]!;
      const oldShort = shortGain[i]![n]!;
      const dr = drs[i]![n]!;
      const nextLong = oldLong + (I0 + K * oldLong) * dr;
      const nextShort = oldShort + (-I0 - K * oldShort) * dr;
      if (!Number.isFinite(nextLong)) throw new RangeError(`nonfinite long gain at path ${i}, step ${n + 1}`);
      if (!Number.isFinite(nextShort)) throw new RangeError(`nonfinite short gain at path ${i}, step ${n + 1}`);
      longGain[i]![n + 1] = nextLong;
      shortGain[i]![n + 1] = nextShort;
    }
  }

  const totalGain = longGain.map((row, i) =>
    row.map((value, n) => {
      const total = value + shortGain[i]![n]!;
      if (!Number.isFinite(total)) throw new RangeError(`nonfinite total gain at path ${i}, time index ${n}`);
      return total;
    }),
  );
  return { longGain, shortGain, totalGain };
}
