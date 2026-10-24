import "./style.css";
import {
  bindPathHighlighting,
  renderGainPaths,
  renderKCurve,
  renderPricePaths,
} from "../charts/charts.ts";
import { RNG_ALGORITHM, seededNormals } from "../model/random.ts";
import type { ExperimentConfig, ExperimentResult } from "../model/experiment.ts";
import type { Matrix, SlsGains } from "../model/types.ts";

interface RunRequest {
  type: "run";
  requestId: number;
  normals: Matrix;
  config: ExperimentConfig;
}

interface SelectKRequest {
  type: "select-k";
  requestId: number;
  K: number;
}

type WorkerRequest = RunRequest | SelectKRequest;

type WorkerResponse =
  | { type: "result"; requestId: number; result: ExperimentResult }
  | { type: "selected-k"; requestId: number; K: number; gains: SlsGains; expectedGain: number[] }
  | { type: "error"; requestId: number; message: string };

const form = required<HTMLFormElement>("parameter-form");
const priceChart = required<HTMLDivElement>("price-chart");
const sweepChart = required<HTMLDivElement>("sweep-chart");
const gainChart = required<HTMLDivElement>("gain-chart");
const seedInput = required<HTMLInputElement>("seed");
const selectedKInput = required<HTMLInputElement>("selected-k");
const statusTitle = required<HTMLElement>("status-title");
const statusText = required<HTMLElement>("status-text");
const statusIndicator = required<HTMLElement>("status-indicator");
const boundNote = required<HTMLElement>("bound-note");
const selectedKLabel = required<HTMLElement>("selected-k-label");
const highlightLabel = required<HTMLElement>("highlight-label");

const worker = new Worker(new URL("../workers/simulation.worker.ts", import.meta.url), {
  type: "module",
});

let requestSequence = 0;
let latestRequest = 0;
let currentNormals: number[][] = [];
let innovationSeed: number | null = null;
let lastRunConfig: ExperimentConfig | null = null;
let currentResult: ExperimentResult | null = null;
let highlightedPath: number | null = null;
let pathEventsBound = false;

function required<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Missing required interface element: #${id}`);
  return element as T;
}

function numberValue(id: string): number {
  const input = required<HTMLInputElement>(id);
  if (!input.checkValidity() || input.value.trim() === "") {
    input.reportValidity();
    throw new Error(`Check the ${input.labels?.[0]?.textContent?.trim() ?? id} value.`);
  }
  const value = input.valueAsNumber;
  if (!Number.isFinite(value)) throw new Error(`${id} must be finite.`);
  return value;
}

function readConfig(): ExperimentConfig {
  return {
    nu0: numberValue("nu0"),
    nu1: numberValue("nu1"),
    kappa: numberValue("kappa"),
    sigma: numberValue("sigma"),
    T: 1,
    I0: 5000,
    selectedK: numberValue("selected-k"),
    sweep: {
      minimum: numberValue("k-min"),
      maximum: numberValue("k-max"),
      count: numberValue("k-count"),
    },
  };
}

function status(title: string, message: string, state: "ready" | "busy" | "error"): void {
  statusTitle.textContent = title;
  statusText.textContent = message;
  statusIndicator.dataset.state = state;
}

function showBound(result: ExperimentResult, config: ExperimentConfig): void {
  if (result.kStar === null) {
    const reason = config.kappa <= 0 ? "κ must be positive" : "σ must be positive";
    boundNote.innerHTML = `<span class="bound-symbol">K*</span><p><strong>Not applicable.</strong> Theorem 3 requires κ &gt; 0, σ &gt; 0, and T &gt; 0 (${reason} here).</p>`;
    return;
  }
  const offRange =
    result.kStar < config.sweep.minimum || result.kStar > config.sweep.maximum;
  boundNote.innerHTML = `<span class="bound-symbol">K*</span><p><strong>K* = ${result.kStar.toPrecision(5)}${offRange ? " · outside sweep" : ""}.</strong> K &lt; K* is sufficient for positive continuous-time expected gain; it does not guarantee positive Euler sample gain.</p>`;
}

function configExceptK(config: ExperimentConfig): Omit<ExperimentConfig, "selectedK"> {
  const { selectedK: _selectedK, ...rest } = config;
  return rest;
}

function nonKInputsMatchLastRun(config: ExperimentConfig): boolean {
  return (
    lastRunConfig !== null &&
    JSON.stringify(configExceptK(config)) === JSON.stringify(configExceptK(lastRunConfig))
  );
}

function dispatch(request: WorkerRequest): void {
  latestRequest = request.requestId;
  worker.postMessage(request);
}

function submitRun(): void {
  try {
    const config = readConfig();
    const seed = numberValue("seed");
    if (!Number.isInteger(seed) || seed < 0 || seed > 0xffff_ffff) {
      throw new Error("Seed must be an integer between 0 and 4,294,967,295.");
    }
    if (innovationSeed !== seed) {
      currentNormals = seededNormals(seed, 100, 50);
      innovationSeed = seed;
    }
    lastRunConfig = config;
    status("Running experiment", "Generating 100 paths and sweeping the shared innovations over K…", "busy");
    dispatch({
      type: "run",
      requestId: ++requestSequence,
      normals: currentNormals,
      config,
    });
  } catch (error) {
    status("Check the inputs", error instanceof Error ? error.message : String(error), "error");
  }
}

function requestSelectedK(K: number): void {
  selectedKInput.value = String(K);
  selectedKLabel.textContent = `SELECTED K = ${K.toPrecision(4)}`;
  if (!currentResult) return;
  try {
    const config = readConfig();
    if (!nonKInputsMatchLastRun(config)) {
      status("Parameter edits pending", "Press Run simulation to apply the other edited parameters.", "busy");
      return;
    }
    status("Updating selected K", "Reusing the current prices and normal innovations.", "busy");
    dispatch({ type: "select-k", requestId: ++requestSequence, K });
  } catch (error) {
    status("Check the inputs", error instanceof Error ? error.message : String(error), "error");
  }
}

function refreshGainChart(): void {
  if (!currentResult) return;
  void renderGainPaths(gainChart, currentResult, numberValue("selected-k"), highlightedPath);
}

async function displayResult(result: ExperimentResult, config: ExperimentConfig): Promise<void> {
  currentResult = result;
  selectedKLabel.textContent = `SELECTED K = ${config.selectedK.toPrecision(4)}`;
  showBound(result, config);
  await Promise.all([
    renderPricePaths(priceChart, result, highlightedPath),
    renderKCurve(sweepChart, result, config.selectedK, requestSelectedK),
    renderGainPaths(gainChart, result, config.selectedK, highlightedPath),
  ]);
  if (!pathEventsBound) {
    bindPathHighlighting(priceChart, gainChart, sweepChart, (path) => {
      highlightedPath = path;
      highlightLabel.textContent = path === null ? "Hover a path to link it across charts" : `Highlighting path ${path + 1} across all three charts`;
    });
    pathEventsBound = true;
  }
  const seed = numberValue("seed");
  status(
    "Experiment ready",
    `Seed ${seed} · ${RNG_ALGORITHM} · same paths used for every K in the sweep.`,
    "ready",
  );
}

worker.addEventListener("message", (event: MessageEvent<WorkerResponse>) => {
  const message = event.data;
  if (message.requestId !== latestRequest) return;
  if (message.type === "error") {
    status("Simulation error", message.message, "error");
    return;
  }
  if (message.type === "result") {
    if (!lastRunConfig) return;
    void displayResult(message.result, lastRunConfig).catch((error: unknown) => {
      status("Chart error", error instanceof Error ? error.message : String(error), "error");
    });
    return;
  }

  if (!currentResult) return;
  currentResult = {
    ...currentResult,
    selectedGains: message.gains,
    selectedExpectedGain: message.expectedGain,
  };
  void renderKCurve(sweepChart, currentResult, message.K, requestSelectedK);
  refreshGainChart();
  status("Experiment ready", `K = ${message.K} · current prices and sweep retained.`, "ready");
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  submitRun();
});

required<HTMLButtonElement>("new-paths-button").addEventListener("click", () => {
  try {
    const seed = numberValue("seed");
    const nextSeed = seed >= 0xffff_ffff ? 0 : seed + 1;
    seedInput.value = String(nextSeed);
    innovationSeed = null;
    submitRun();
  } catch (error) {
    status("Check the seed", error instanceof Error ? error.message : String(error), "error");
  }
});

selectedKInput.addEventListener("change", () => {
  const value = selectedKInput.valueAsNumber;
  if (Number.isFinite(value)) requestSelectedK(value);
});

currentNormals = seededNormals(7, 100, 50);
innovationSeed = 7;
submitRun();
