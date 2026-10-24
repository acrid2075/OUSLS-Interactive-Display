export type Matrix = readonly (readonly number[])[];

export interface ProcessParameters {
  nu0: number;
  nu1: number;
  kappa: number;
  sigma: number;
  T?: number;
}

export interface PricePaths {
  times: number[];
  logPrices: number[][];
  prices: number[][];
  returnIncrements: number[][];
}

export interface SlsGains {
  longGain: number[][];
  shortGain: number[][];
  totalGain: number[][];
}

export interface Moments {
  mean: number[];
  variance: number[];
}

export interface TheoryParameters extends ProcessParameters {
  I0: number;
  K: number;
}
