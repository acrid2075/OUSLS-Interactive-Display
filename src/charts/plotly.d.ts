declare module "plotly.js-basic-dist-min" {
  export interface PlotTrace {
    type?: "scatter";
    mode?: string;
    x: readonly number[];
    y: readonly number[];
    name?: string;
    showlegend?: boolean;
    line?: { color?: string; width?: number; dash?: string };
    marker?: { color?: string; size?: number; symbol?: string };
    hovertemplate?: string;
  }

  export interface PlotLayout {
    title?: string | { text: string };
    xaxis?: { title?: string; range?: number[]; [key: string]: unknown };
    yaxis?: { title?: string; tickprefix?: string; zeroline?: boolean; [key: string]: unknown };
    shapes?: Array<Record<string, unknown>>;
    annotations?: Array<Record<string, unknown>>;
    hovermode?: string;
    showlegend?: boolean;
    autosize?: boolean;
    height?: number;
    margin?: { l?: number; r?: number; t?: number; b?: number };
    legend?: Record<string, unknown>;
    font?: Record<string, unknown>;
    paper_bgcolor?: string;
    plot_bgcolor?: string;
    uirevision?: string;
    [key: string]: unknown;
  }

  export interface PlotConfig {
    responsive?: boolean;
    displaylogo?: boolean;
    modeBarButtonsToRemove?: string[];
  }

  export interface PlotHoverEvent {
    points: Array<{ curveNumber: number; pointNumber: number; x: number; y: number }>;
  }

  export interface PlotClickEvent extends PlotHoverEvent {}

  export interface PlotlyHTMLElement extends HTMLElement {
    on(event: "plotly_hover", callback: (event: PlotHoverEvent) => void): this;
    on(event: "plotly_unhover", callback: () => void): this;
    on(event: "plotly_click", callback: (event: PlotClickEvent) => void): this;
  }

  const Plotly: {
    newPlot(
      target: HTMLElement,
      data: PlotTrace[],
      layout: PlotLayout,
      config?: PlotConfig,
    ): Promise<PlotlyHTMLElement>;
    react(
      target: HTMLElement,
      data: PlotTrace[],
      layout: PlotLayout,
      config?: PlotConfig,
    ): Promise<PlotlyHTMLElement>;
    restyle(
      target: HTMLElement,
      update: Record<string, unknown>,
      traceIndices?: number[],
    ): Promise<PlotlyHTMLElement>;
  };

  export default Plotly;
}
