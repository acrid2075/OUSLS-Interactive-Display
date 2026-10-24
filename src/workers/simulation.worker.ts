import { runExperiment, type ExperimentConfig, type ExperimentResult } from "../model/experiment.ts";
import { expectedGain } from "../model/theory.ts";
import { simulateSls } from "../model/sls.ts";
import type { Matrix, SlsGains } from "../model/types.ts";

type WorkerRequest =
  | { type: "run"; requestId: number; normals: Matrix; config: ExperimentConfig }
  | { type: "select-k"; requestId: number; K: number };

type WorkerResponse =
  | { type: "result"; requestId: number; result: ExperimentResult }
  | { type: "selected-k"; requestId: number; K: number; gains: SlsGains; expectedGain: number[] }
  | { type: "error"; requestId: number; message: string };

interface WorkerScope {
  addEventListener(
    type: "message",
    listener: (event: MessageEvent<WorkerRequest>) => void,
  ): void;
  postMessage(message: WorkerResponse): void;
}

const scope = self as unknown as WorkerScope;
let cached: { result: ExperimentResult; config: ExperimentConfig } | undefined;

scope.addEventListener("message", (event) => {
  const request = event.data;
  try {
    if (request.type === "run") {
      const result = runExperiment(request.normals, request.config);
      cached = { result, config: request.config };
      scope.postMessage({ type: "result", requestId: request.requestId, result });
      return;
    }

    if (!cached) throw new Error("Run the simulation before selecting a K value");
    const K = request.K;
    const { config, result } = cached;
    // A K selection reuses the cached process innovations and leaves prices and
    // the terminal K sweep untouched.
    const gains = simulateSls(result.pricePaths.returnIncrements, config.I0, K);
    const expected = expectedGain(result.pricePaths.times, { ...config, K });
    scope.postMessage({
      type: "selected-k",
      requestId: request.requestId,
      K,
      gains,
      expectedGain: expected,
    });
  } catch (error) {
    scope.postMessage({
      type: "error",
      requestId: request.requestId,
      message: error instanceof Error ? error.message : String(error),
    });
  }
});
