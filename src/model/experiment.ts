import { expectedGain, kStar as calculateKStar } from "./theory.ts";
import { generatePricePaths } from "./price.ts";
import { simulateSls } from "./sls.ts";
import type { Matrix, PricePaths, ProcessParameters, SlsGains } from "./types.ts";

export interface SweepConfig {
  minimum: number;
  maximum: number;
  count: number;
}

export interface ExperimentConfig extends ProcessParameters {
  I0: number;
  selectedK: number;
  sweep: SweepConfig;
}

export interface SweepResult {
  kValues: number[];
  terminalGains: number[][];
  meanGains: number[];
  expectedGains: number[];
}

export interface ExperimentResult {
  pricePaths: PricePaths;
  selectedGains: SlsGains;
  selectedExpectedGain: number[];
  sweep: SweepResult;
  kStar: number | null;
}

export function makeKGrid(config: SweepConfig): number[] {
  const minimum = config.minimum;
  const maximum = config.maximum;
  const count = config.count;
  if (!Number.isFinite(minimum) || !Number.isFinite(maximum) || minimum >= maximum) {
    throw new RangeError("K sweep bounds must be finite and minimum must be below maximum");
  }
  if (!Number.isInteger(count) || count < 2 || count > 1001) {
    throw new RangeError("K sweep resolution must be an integer from 2 to 1001");
  }
  return Array.from({ length: count }, (_, i) => minimum + ((maximum - minimum) * i) / (count - 1));
}

function calculateSweep(
  returnIncrements: Matrix,
  parameters: ExperimentConfig,
): SweepResult {
  const kValues = makeKGrid(parameters.sweep);
  const terminalGains: number[][] = [];
  const meanGains: number[] = [];
  const expectedGains: number[] = [];
  for (const K of kValues) {
    const gains = simulateSls(returnIncrements, parameters.I0, K).totalGain;
    const terminal = gains.map((path) => path[path.length - 1]!);
    terminalGains.push(terminal);
    meanGains.push(terminal.reduce((sum, value) => sum + value, 0) / terminal.length);
    expectedGains.push(
      expectedGain([parameters.T ?? 1], { ...parameters, K })[0]!,
    );
  }
  return { kValues, terminalGains, meanGains, expectedGains };
}

export function runExperiment(
  normals: Matrix,
  parameters: ExperimentConfig,
): ExperimentResult {
  const pricePaths = generatePricePaths(normals, parameters);
  const selectedGains = simulateSls(
    pricePaths.returnIncrements,
    parameters.I0,
    parameters.selectedK,
  );
  const selectedExpectedGain = expectedGain(pricePaths.times, {
    ...parameters,
    K: parameters.selectedK,
  });
  const sweep = calculateSweep(pricePaths.returnIncrements, parameters);
  const { I0: _I0, selectedK: _selectedK, sweep: _sweep, ...process } = parameters;
  const bound =
    process.kappa > 0 && process.sigma > 0
      ? calculateKStar({ ...process, T: process.T ?? 1 })
      : null;
  return { pricePaths, selectedGains, selectedExpectedGain, sweep, kStar: bound };
}
