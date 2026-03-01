/**
 * HTTP request router that exposes the PD scheduler as a REST API.
 *
 * Endpoints:
 *   POST   /v1/completions   – submit an inference request
 *   GET    /v1/metrics        – scheduler metrics snapshot
 *   GET    /v1/events         – recent scheduler events
 *   GET    /v1/workers        – list registered workers
 *   POST   /v1/workers        – register a new worker
 *   DELETE /v1/workers/:id    – remove a worker
 *   GET    /health            – simple liveness probe
 */

import type { PDSchedulerConfig, WorkerSeedConfig } from "./types.js";
import { PDScheduler } from "./scheduler.js";

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

export class PDRouter {
  readonly scheduler: PDScheduler;

  constructor(config: PDSchedulerConfig) {
    this.scheduler = new PDScheduler(config);
  }

  /** Start the underlying scheduler. */
  start(): void {
    this.scheduler.start();
  }

  /** Stop the scheduler gracefully. */
  stop(): void {
    this.scheduler.stop();
  }

  /**
   * Route an incoming HTTP-like request.
   *
   * This is intentionally transport-agnostic so it can be mounted on any
   * HTTP framework (express, fastify, raw Node http, etc.) or invoked
   * directly from an OpenClaw tool handler.
   */
  async handle(
    method: string,
    path: string,
    body?: unknown,
  ): Promise<{ status: number; body: unknown }> {
    try {
      // --- Inference ---------------------------------------------------------
      if (method === "POST" && path === "/v1/completions") {
        return await this.handleCompletions(body as CompletionRequestBody | undefined);
      }

      // --- Observability -----------------------------------------------------
      if (method === "GET" && path === "/v1/metrics") {
        return { status: 200, body: this.scheduler.metrics() };
      }

      if (method === "GET" && path === "/v1/events") {
        return { status: 200, body: this.scheduler.events() };
      }

      // --- Worker management -------------------------------------------------
      if (method === "GET" && path === "/v1/workers") {
        return { status: 200, body: this.scheduler.workerPool.list() };
      }

      if (method === "POST" && path === "/v1/workers") {
        return this.handleRegisterWorker(body as WorkerSeedConfig | undefined);
      }

      if (method === "DELETE" && path.startsWith("/v1/workers/")) {
        const id = decodeURIComponent(path.slice("/v1/workers/".length));
        const removed = this.scheduler.workerPool.remove(id);
        return removed
          ? { status: 200, body: { ok: true } }
          : { status: 404, body: { error: "Worker not found" } };
      }

      // --- Health ------------------------------------------------------------
      if (method === "GET" && path === "/health") {
        return { status: 200, body: { status: "ok" } };
      }

      return { status: 404, body: { error: "Not found" } };
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      return { status: 500, body: { error: message } };
    }
  }

  // -----------------------------------------------------------------------
  // Private handlers
  // -----------------------------------------------------------------------

  private async handleCompletions(
    body: CompletionRequestBody | undefined,
  ): Promise<{ status: number; body: unknown }> {
    if (!body?.model || !body?.prompt) {
      return {
        status: 400,
        body: { error: "Missing required fields: model, prompt" },
      };
    }

    try {
      const result = await this.scheduler.submit({
        modelId: body.model,
        prompt: body.prompt,
        samplingParams: {
          maxTokens: body.max_tokens,
          temperature: body.temperature,
          topP: body.top_p,
          topK: body.top_k,
          repetitionPenalty: body.repetition_penalty,
          stop: body.stop,
          stream: body.stream,
        },
        priority: body.priority,
        timeoutMs: body.timeout_ms,
      });

      return { status: 200, body: result };
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      return { status: 503, body: { error: message } };
    }
  }

  private handleRegisterWorker(
    body: WorkerSeedConfig | undefined,
  ): { status: number; body: unknown } {
    if (!body?.id || !body?.endpoint || !body?.role || !body?.modelId) {
      return {
        status: 400,
        body: {
          error: "Missing required fields: id, endpoint, role, modelId",
        },
      };
    }

    const info = this.scheduler.workerPool.register(body);
    return { status: 201, body: info };
  }
}

// ---------------------------------------------------------------------------
// Request body type (OpenAI-ish)
// ---------------------------------------------------------------------------

interface CompletionRequestBody {
  model: string;
  prompt: string | number[];
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  repetition_penalty?: number;
  stop?: string[];
  stream?: boolean;
  priority?: "high" | "normal" | "low";
  timeout_ms?: number;
}
