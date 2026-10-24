import type { Moments, TheoryParameters } from "./types.ts";

const SMALL_ARGUMENT = 1e-4;
const SMALL_GAIN_ARGUMENT = 1e-3;

function finiteScalar(value: number, name: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError(`${name} must be a finite number`);
  }
  return value;
}

function validateTimes(times: readonly number[]): number[] {
  if (!Array.isArray(times) || !times.every((t) => Number.isFinite(t) && t >= 0)) {
    throw new TypeError("times must be a one-dimensional array of finite nonnegative values");
  }
  return [...times];
}

function checkFinite(values: number[], name: string): void {
  if (!values.every(Number.isFinite)) throw new RangeError(`nonfinite ${name} produced`);
}

function ouIntegrals(times: number[], kappa: number): [number[], number[], number[]] {
  const A: number[] = [];
  const B: number[] = [];
  const C: number[] = [];
  for (const t of times) {
    const q = kappa * t;
    let a: number;
    let b: number;
    let c: number;
    if (Math.abs(q) < SMALL_ARGUMENT) {
      a = t * (1 - q / 2 + q**2 / 6 - q**3 / 24 + q**4 / 120 - q**5 / 720 + q**6 / 5040);
      b = t**2 * (0.5 - q / 6 + q**2 / 24 - q**3 / 120 + q**4 / 720 - q**5 / 5040 + q**6 / 40320);
      c = t * (1 - q + (2 / 3) * q**2 - q**3 / 3 + (2 / 15) * q**4 - (2 / 45) * q**5 + (4 / 315) * q**6);
    } else {
      a = -Math.expm1(-q) / kappa;
      b = (t - a) / kappa;
      c = -Math.expm1(-2 * q) / (2 * kappa);
    }
    A.push(a);
    B.push(b);
    C.push(c);
  }
  checkFinite([...A, ...B, ...C], "OU moment integrals");
  return [A, B, C];
}

function meanVariance(
  times: number[],
  nu0: number,
  nu1: number,
  kappa: number,
  sigma: number,
): Moments {
  const [A, B, C] = ouIntegrals(times, kappa);
  const mean = times.map((_, i) => nu0 * A[i]! + nu1 * B[i]!);
  const variance = times.map((_, i) => sigma * sigma * C[i]!);
  checkFinite(mean, "OU mean");
  checkFinite(variance, "OU variance");
  return { mean, variance };
}

function zetaAt(times: number[], kappa: number, sigma: number): number[] {
  return times.map((t) => {
    const q = kappa * t;
    let difference: number;
    if (Math.abs(q) < SMALL_ARGUMENT) {
      difference = t * (-q + (2 / 3) * q**2 - q**3 / 3 + (2 / 15) * q**4 - (2 / 45) * q**5 + (4 / 315) * q**6);
    } else {
      const [, , C] = ouIntegrals([t], kappa);
      difference = C[0]! - t;
    }
    return 0.5 * sigma * sigma * difference;
  });
}

/** Analytical mean and variance of the trending OU log-price process. */
export function moments(
  timesInput: readonly number[],
  parameters: Omit<TheoryParameters, "I0" | "K" | "T">,
): Moments {
  const times = validateTimes(timesInput);
  const nu0 = finiteScalar(parameters.nu0, "nu0");
  const nu1 = finiteScalar(parameters.nu1, "nu1");
  const kappa = finiteScalar(parameters.kappa, "kappa");
  const sigma = finiteScalar(parameters.sigma, "sigma");
  if (sigma < 0) throw new RangeError("sigma must be nonnegative");
  return meanVariance(times, nu0, nu1, kappa, sigma);
}

function logCosh(value: number): number {
  const absolute = Math.abs(value);
  return absolute + Math.log1p(Math.exp(-2 * absolute)) - Math.LN2;
}

/** Continuous-time expected total dollar gain from the paper's expression. */
export function expectedGain(
  timesInput: readonly number[],
  parameters: TheoryParameters,
): number[] {
  const times = validateTimes(timesInput);
  const nu0 = finiteScalar(parameters.nu0, "nu0");
  const nu1 = finiteScalar(parameters.nu1, "nu1");
  const kappa = finiteScalar(parameters.kappa, "kappa");
  const sigma = finiteScalar(parameters.sigma, "sigma");
  const I0 = finiteScalar(parameters.I0, "I0");
  const K = finiteScalar(parameters.K, "K");
  if (sigma < 0) throw new RangeError("sigma must be nonnegative");
  if (I0 <= 0) throw new RangeError("I0 must be positive");
  if (K === 0) return times.map(() => 0);

  const { mean } = meanVariance(times, nu0, nu1, kappa, sigma);
  const zeta = zetaAt(times, kappa, sigma);
  const phi = times.map((t, i) => mean[i]! + 0.5 * sigma * sigma * t);
  const result = times.map((t, i) => {
    const scaledPhi = K * phi[i]!;
    const scaledZeta = K * (K * zeta[i]!);
    if (!Number.isFinite(scaledPhi) || !Number.isFinite(scaledZeta)) {
      throw new RangeError("nonfinite scaled expected-gain argument");
    }
    let value: number;
    if (Math.max(Math.abs(scaledPhi), Math.sqrt(Math.abs(scaledZeta))) < SMALL_GAIN_ARGUMENT) {
      const a = scaledZeta;
      const b = scaledPhi;
      const b2 = b * b;
      value = 2 * (I0 / K) * (
        a + b2 / 2 + a**2 / 2 + a * b2 / 2 + b2**2 / 24 + a**3 / 6 + a**2 * b2 / 4 + a * b2**2 / 24 + b2**3 / 720
      );
    } else {
      value = (2 * I0 / K) * Math.expm1(scaledZeta + logCosh(scaledPhi));
    }
    if (!Number.isFinite(value)) throw new RangeError("nonfinite analytical expected gain produced");
    return t === 0 ? 0 : value;
  });
  return result;
}

/** Sufficient positive-expectation threshold; null outside Theorem 3's domain. */
export function kStar(
  parameters: Omit<TheoryParameters, "I0" | "K" | "T"> & { T: number },
): number | null {
  const nu0 = finiteScalar(parameters.nu0, "nu0");
  const nu1 = finiteScalar(parameters.nu1, "nu1");
  const kappa = finiteScalar(parameters.kappa, "kappa");
  const sigma = finiteScalar(parameters.sigma, "sigma");
  const T = finiteScalar(parameters.T, "T");
  if (sigma < 0) throw new RangeError("sigma must be nonnegative");
  if (T <= 0) throw new RangeError("T must be positive");
  if (kappa <= 0 || sigma <= 0) return null;
  const { mean } = meanVariance([T], nu0, nu1, kappa, sigma);
  const zeta = zetaAt([T], kappa, sigma)[0]!;
  const phi = mean[0]! + 0.5 * sigma * sigma * T;
  if (!(zeta < 0) || !Number.isFinite(zeta)) throw new RangeError("K-star denominator is numerically invalid");
  const bound = Math.abs(phi) / zeta;
  if (!Number.isFinite(bound)) throw new RangeError("K-star is nonfinite for these parameters");
  return bound;
}
