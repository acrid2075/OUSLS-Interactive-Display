import Plotly, {
  type PlotConfig,
  type PlotHoverEvent,
  type PlotLayout,
  type PlotTrace,
  type PlotlyHTMLElement,
} from "plotly.js-basic-dist-min";
import type { ExperimentResult } from "../model/experiment.ts";

const PATH_COUNT = 100;
const PATH_COLOR = "rgba(49, 89, 125, 0.15)";
const SELECTED_COLOR = "#df704e";
const MEAN_COLOR = "#187c70";
const THEORY_COLOR = "#674c9b";
const sweepClickCallbacks = new WeakMap<HTMLElement, (K: number) => void>();
const wiredSweepCharts = new WeakSet<HTMLElement>();
const COMMON_CONFIG: PlotConfig = {
  responsive: true,
  displaylogo: false,
  modeBarButtonsToRemove: ["lasso2d", "select2d"],
};
const COMMON_LAYOUT: PlotLayout = {
  autosize: true,
  height: 340,
  margin: { l: 68, r: 20, t: 28, b: 50 },
  paper_bgcolor: "#ffffff",
  plot_bgcolor: "#ffffff",
  font: { family: "Inter, ui-sans-serif, system-ui, sans-serif", color: "#26384a", size: 12 },
  hovermode: "closest",
  showlegend: false,
  xaxis: { title: "Time (years)", gridcolor: "#e8edf2", zeroline: false },
  yaxis: { gridcolor: "#e8edf2", zeroline: false },
};

export interface ChartElements {
  prices: HTMLElement;
  sweep: HTMLElement;
  gains: HTMLElement;
}

function lineTrace(
  x: number[],
  y: number[],
  name: string,
  color: string,
  width: number,
): PlotTrace {
  return {
    type: "scatter",
    mode: "lines",
    x,
    y,
    name,
    showlegend:
      name === "Mean simulated gain (100 paths; Euler, 50 steps)" ||
      name === "Analytical expected gain (continuous time)",
    line: { color, width },
    hovertemplate: `${name}<br>x=%{x:.3f}<br>y=%{y:.5g}<extra></extra>`,
  };
}

async function draw(
  element: HTMLElement,
  data: PlotTrace[],
  layout: PlotLayout,
): Promise<void> {
  await Plotly.react(element, data, layout, COMMON_CONFIG);
  const onSelectK = sweepClickCallbacks.get(element);
  if (onSelectK && !wiredSweepCharts.has(element)) {
    (element as PlotlyHTMLElement).on("plotly_click", (event) => {
      const clickedK = event.points[0]?.x;
      if (typeof clickedK === "number" && Number.isFinite(clickedK)) onSelectK(clickedK);
    });
    wiredSweepCharts.add(element);
  }
}

export function setSweepClickHandler(element: HTMLElement, onSelectK: (K: number) => void): void {
  sweepClickCallbacks.set(element, onSelectK);
}

export async function renderPricePaths(
  element: HTMLElement,
  result: ExperimentResult,
  highlightedPath: number | null,
): Promise<void> {
  const traces = result.pricePaths.prices.map((path, i) =>
    lineTrace(
      result.pricePaths.times,
      path,
      `Path ${i + 1}`,
      i === highlightedPath ? SELECTED_COLOR : PATH_COLOR,
      i === highlightedPath ? 2.8 : 0.9,
    ),
  );
  await draw(element, traces, {
    ...COMMON_LAYOUT,
    yaxis: { ...COMMON_LAYOUT.yaxis, title: "Price S(t)", tickprefix: "$" },
  });
}

export async function renderKCurve(
  element: HTMLElement,
  result: ExperimentResult,
  selectedK: number,
  onSelectK: (K: number) => void,
): Promise<void> {
  sweepClickCallbacks.set(element, onSelectK);
  const { kValues, terminalGains, meanGains, expectedGains } = result.sweep;
  const traces: PlotTrace[] = [];
  for (let path = 0; path < PATH_COUNT; path += 1) {
    traces.push(
      lineTrace(
        kValues,
        terminalGains.map((atK) => atK[path]!),
        `Path ${path + 1}`,
        PATH_COLOR,
        0.8,
      ),
    );
  }
  traces.push(lineTrace(kValues, meanGains, "Mean simulated gain", MEAN_COLOR, 3));
  traces.push(lineTrace(kValues, expectedGains, "Analytical expected gain", THEORY_COLOR, 2.6));

  // Keep the sweep's vertical scale tied to the two summary curves. The
  // individual path traces remain visible when they fall inside this range.
  const summaryValues = [...meanGains, ...expectedGains].filter(Number.isFinite);
  const summaryMin = summaryValues.length ? Math.min(...summaryValues) : -1;
  const summaryMax = summaryValues.length ? Math.max(...summaryValues) : 1;
  let yMin = 2 * summaryMin;
  let yMax = 2 * summaryMax;
  // Expand a collapsed range enough to show both the series and its doubled bound.
  if (yMin === yMax) {
    const summaryValue = summaryMin;
    const padding = Math.max(1, Math.abs(summaryValue) * 0.05);
    yMin = Math.min(yMin, summaryValue) - padding;
    yMax = Math.max(yMax, summaryValue) + padding;
  }

  const sweepMin = kValues[0]!;
  const sweepMax = kValues[kValues.length - 1]!;
  const shapes: Array<Record<string, unknown>> = [
    {
      type: "line",
      x0: sweepMin,
      x1: sweepMax,
      y0: 0,
      y1: 0,
      line: { color: "#9aa9b6", width: 1, dash: "dot" },
    },
  ];
  const selectedInRange = selectedK >= sweepMin && selectedK <= sweepMax;
  const annotations: Array<Record<string, unknown>> = [];
  if (selectedInRange) {
    shapes.push({
      type: "line",
      x0: selectedK,
      x1: selectedK,
      y0: 0,
      y1: 1,
      yref: "paper",
      line: { color: SELECTED_COLOR, width: 1.5, dash: "dash" },
    });
    annotations.push({
      x: selectedK,
      y: 1,
      yref: "paper",
      text: `Selected K = ${selectedK.toFixed(3)}`,
      showarrow: false,
      yshift: 12,
      font: { color: SELECTED_COLOR, size: 10 },
    });
  } else {
    annotations.push({
      x: selectedK < sweepMin ? 0 : 1,
      xref: "paper",
      y: 0.88,
      yref: "paper",
      text: `Selected K = ${selectedK.toFixed(3)} (off-range)`,
      showarrow: false,
      xanchor: selectedK < sweepMin ? "left" : "right",
      font: { color: SELECTED_COLOR, size: 10 },
    });
  }
  if (result.kStar !== null) {
    if (result.kStar >= sweepMin && result.kStar <= sweepMax) {
      shapes.push({
        type: "line",
        x0: result.kStar,
        x1: result.kStar,
        y0: 0,
        y1: 1,
        yref: "paper",
        line: { color: THEORY_COLOR, width: 2, dash: "dot" },
      });
      annotations.push({
        x: result.kStar,
        y: 0,
        yref: "paper",
        text: `K* = ${result.kStar.toPrecision(4)}`,
        showarrow: false,
        yshift: -14,
        font: { color: THEORY_COLOR, size: 10 },
      });
    } else {
      annotations.push({
        x: result.kStar < sweepMin ? 0 : 1,
        xref: "paper",
        y: 0.94,
        yref: "paper",
        text: `K* = ${result.kStar.toPrecision(4)} is off-range`,
        showarrow: false,
        xanchor: result.kStar < sweepMin ? "left" : "right",
        font: { color: THEORY_COLOR, size: 10 },
      });
    }
  }
  await draw(element, traces, {
    ...COMMON_LAYOUT,
    hovermode: "x",
    xaxis: { ...COMMON_LAYOUT.xaxis, title: "Feedback K", range: [sweepMin, sweepMax] },
    yaxis: { ...COMMON_LAYOUT.yaxis, title: "Terminal gain ($)", tickprefix: "$", range: [yMin, yMax] },
    shapes,
    annotations,
  });
}

export async function renderGainPaths(
  element: HTMLElement,
  result: ExperimentResult,
  selectedK: number,
  highlightedPath: number | null,
): Promise<void> {
  const times = result.pricePaths.times;
  const traces: PlotTrace[] = result.selectedGains.totalGain.map((path, i) =>
    lineTrace(
      times,
      path,
      `Path ${i + 1}`,
      i === highlightedPath ? SELECTED_COLOR : PATH_COLOR,
      i === highlightedPath ? 2.8 : 0.9,
    ),
  );
  traces.push(lineTrace(times, meanPath(result.selectedGains.totalGain), "Mean simulated gain (100 paths; Euler, 50 steps)", MEAN_COLOR, 3));
  traces.push(lineTrace(times, result.selectedExpectedGain, "Analytical expected gain (continuous time)", THEORY_COLOR, 2.6));
  await draw(element, traces, {
    ...COMMON_LAYOUT,
    xaxis: { ...COMMON_LAYOUT.xaxis, title: "Time (years)" },
    yaxis: { ...COMMON_LAYOUT.yaxis, title: "Cumulative gain ($)", tickprefix: "$" },
    legend: { orientation: "h", y: -0.2, x: 0 },
    showlegend: true,
    uirevision: `gain-${selectedK}`,
  });
}

function meanPath(paths: number[][]): number[] {
  if (paths.length === 0) return [];
  return paths[0]!.map((_, n) => paths.reduce((sum, path) => sum + path[n]!, 0) / paths.length);
}

export function bindPathHighlighting(
  priceElement: HTMLElement,
  gainElement: HTMLElement,
  sweepElement: HTMLElement,
  onHighlight: (path: number | null) => void,
): void {
  const pricePlot = priceElement as PlotlyHTMLElement;
  const gainPlot = gainElement as PlotlyHTMLElement;
  const sweepPlot = sweepElement as PlotlyHTMLElement;
  const highlight = (path: number | null): void => {
    onHighlight(path);
    const indexes = Array.from({ length: PATH_COUNT }, (_, i) => i);
    const widths = indexes.map((i) => (i === path ? 2.8 : 0.9));
    const colors = indexes.map((i) => (i === path ? SELECTED_COLOR : PATH_COLOR));
    void Plotly.restyle(priceElement, { "line.width": widths, "line.color": colors }, indexes);
    void Plotly.restyle(gainElement, { "line.width": widths, "line.color": colors }, indexes);
    void Plotly.restyle(sweepElement, { "line.width": widths, "line.color": colors }, indexes);
  };

  const hoverPath = (event: PlotHoverEvent): void => {
    const trace = event.points[0]?.curveNumber;
    if (typeof trace === "number" && trace >= 0 && trace < PATH_COUNT) highlight(trace);
  };
  pricePlot.on("plotly_hover", hoverPath);
  gainPlot.on("plotly_hover", hoverPath);
  sweepPlot.on("plotly_hover", hoverPath);
  pricePlot.on("plotly_unhover", () => highlight(null));
  gainPlot.on("plotly_unhover", () => highlight(null));
  sweepPlot.on("plotly_unhover", () => highlight(null));
}
