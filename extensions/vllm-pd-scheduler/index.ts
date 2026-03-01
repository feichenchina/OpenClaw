/**
 * OpenClaw plugin entry point for the vLLM PD (Prefill-Decode) scheduler.
 *
 * Registers the scheduler as a set of tools that agents can invoke and
 * exposes the REST router through the gateway.
 */

import type { PDSchedulerConfig } from "./src/types.js";
import { PDRouter } from "./src/router.js";

export { PDScheduler } from "./src/scheduler.js";
export { PDRouter } from "./src/router.js";
export { WorkerPool } from "./src/worker-pool.js";
export { VllmClient } from "./src/vllm-client.js";
export { KVCacheTransferManager } from "./src/kv-cache.js";
export { HealthMonitor } from "./src/health.js";
export type {
  PDSchedulerConfig,
  InferenceRequest,
  InferenceResult,
  WorkerInfo,
  WorkerRole,
  SchedulerMetrics,
  SchedulerEvent,
} from "./src/types.js";

// ---------------------------------------------------------------------------
// Default configuration
// ---------------------------------------------------------------------------

const DEFAULT_CONFIG: PDSchedulerConfig = {
  enabled: false,
  strategy: "least-loaded",
  healthCheckIntervalMs: 10_000,
  workerTimeoutMs: 30_000,
  maxQueueSize: 1000,
  defaultRequestTimeoutMs: 60_000,
  workers: [],
  kvTransfer: {
    maxConcurrent: 4,
    timeoutMs: 15_000,
  },
};

// ---------------------------------------------------------------------------
// Plugin definition
// ---------------------------------------------------------------------------

let routerInstance: PDRouter | null = null;

/**
 * Resolve a complete config by merging user-supplied values over defaults.
 */
function resolveConfig(
  userConfig?: Partial<PDSchedulerConfig>,
): PDSchedulerConfig {
  return {
    ...DEFAULT_CONFIG,
    ...userConfig,
    kvTransfer: {
      ...DEFAULT_CONFIG.kvTransfer,
      ...userConfig?.kvTransfer,
    },
  };
}

/**
 * Initialise and return the singleton router instance.
 *
 * Can be called from an OpenClaw plugin `register` hook or directly by
 * application code.
 */
export function createRouter(
  userConfig?: Partial<PDSchedulerConfig>,
): PDRouter {
  if (routerInstance) return routerInstance;
  const config = resolveConfig(userConfig);
  routerInstance = new PDRouter(config);
  if (config.enabled) routerInstance.start();
  return routerInstance;
}

/**
 * Return the existing router instance (if any).
 */
export function getRouter(): PDRouter | null {
  return routerInstance;
}

const plugin = {
  id: "vllm-pd-scheduler",
  name: "vLLM PD Scheduler",
  description:
    "Dynamic Prefill-Decode task scheduler for disaggregated vLLM inference serving.",

  register(api: {
    registerTool?: (def: unknown) => void;
    runtime?: unknown;
  }) {
    // Read scheduler config from the OpenClaw config tree.
    const userConfig = (
      api as { config?: { pdScheduler?: Partial<PDSchedulerConfig> } }
    ).config?.pdScheduler;

    const router = createRouter(userConfig);

    // Register tools that agents can invoke.
    if (api.registerTool) {
      api.registerTool({
        name: "pd_scheduler_submit",
        description:
          "Submit an inference request to the vLLM PD scheduler. " +
          "Returns the generated text after prefill → KV transfer → decode.",
        parameters: {
          type: "object",
          properties: {
            model: { type: "string", description: "Model ID to use for inference" },
            prompt: { type: "string", description: "Input prompt text" },
            max_tokens: { type: "number", description: "Maximum tokens to generate" },
            temperature: { type: "number", description: "Sampling temperature" },
            priority: {
              type: "string",
              enum: ["high", "normal", "low"],
              description: "Scheduling priority",
            },
          },
          required: ["model", "prompt"],
        },
        handler: async (args: {
          model: string;
          prompt: string;
          max_tokens?: number;
          temperature?: number;
          priority?: "high" | "normal" | "low";
        }) => {
          const result = await router.handle("POST", "/v1/completions", {
            model: args.model,
            prompt: args.prompt,
            max_tokens: args.max_tokens,
            temperature: args.temperature,
            priority: args.priority,
          });
          return result.body;
        },
      });

      api.registerTool({
        name: "pd_scheduler_metrics",
        description:
          "Get current metrics from the vLLM PD scheduler including " +
          "queue depth, active workers, and latency statistics.",
        parameters: { type: "object", properties: {} },
        handler: async () => {
          const result = await router.handle("GET", "/v1/metrics");
          return result.body;
        },
      });

      api.registerTool({
        name: "pd_scheduler_workers",
        description: "List all registered vLLM workers in the PD scheduler pool.",
        parameters: { type: "object", properties: {} },
        handler: async () => {
          const result = await router.handle("GET", "/v1/workers");
          return result.body;
        },
      });

      api.registerTool({
        name: "pd_scheduler_add_worker",
        description: "Register a new vLLM worker with the PD scheduler.",
        parameters: {
          type: "object",
          properties: {
            id: { type: "string", description: "Unique worker ID" },
            endpoint: {
              type: "string",
              description: "Worker base URL (e.g. http://gpu-host:8000)",
            },
            role: {
              type: "string",
              enum: ["prefill", "decode"],
              description: "Worker role",
            },
            modelId: { type: "string", description: "Model loaded on the worker" },
            maxConcurrency: {
              type: "number",
              description: "Max concurrent requests",
            },
          },
          required: ["id", "endpoint", "role", "modelId"],
        },
        handler: async (args: {
          id: string;
          endpoint: string;
          role: "prefill" | "decode";
          modelId: string;
          maxConcurrency?: number;
        }) => {
          const result = await router.handle("POST", "/v1/workers", args);
          return result.body;
        },
      });
    }
  },
};

export default plugin;
