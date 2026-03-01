/**
 * Unit tests for the PDRouter HTTP-like handler.
 */

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import type { PDSchedulerConfig } from "../types.js";
import { PDRouter } from "../router.js";

const baseConfig: PDSchedulerConfig = {
  enabled: false, // don't auto-start dispatch loop in tests
  strategy: "least-loaded",
  healthCheckIntervalMs: 60_000,
  workerTimeoutMs: 30_000,
  maxQueueSize: 100,
  defaultRequestTimeoutMs: 5_000,
  workers: [],
  kvTransfer: { maxConcurrent: 2, timeoutMs: 5_000 },
};

describe("PDRouter", () => {
  let router: PDRouter;

  beforeEach(() => {
    router = new PDRouter(baseConfig);
  });

  afterEach(() => {
    router.stop();
  });

  // -----------------------------------------------------------------------
  // Health
  // -----------------------------------------------------------------------

  it("GET /health returns ok", async () => {
    const res = await router.handle("GET", "/health");
    expect(res.status).toBe(200);
    expect(res.body).toEqual({ status: "ok" });
  });

  // -----------------------------------------------------------------------
  // Workers
  // -----------------------------------------------------------------------

  it("POST /v1/workers registers a worker", async () => {
    const res = await router.handle("POST", "/v1/workers", {
      id: "p1",
      endpoint: "http://p1:8000",
      role: "prefill",
      modelId: "test-model",
    });
    expect(res.status).toBe(201);
    expect((res.body as { id: string }).id).toBe("p1");
  });

  it("POST /v1/workers rejects incomplete body", async () => {
    const res = await router.handle("POST", "/v1/workers", { id: "p1" });
    expect(res.status).toBe(400);
  });

  it("GET /v1/workers lists workers", async () => {
    await router.handle("POST", "/v1/workers", {
      id: "p1",
      endpoint: "http://p1:8000",
      role: "prefill",
      modelId: "m1",
    });
    const res = await router.handle("GET", "/v1/workers");
    expect(res.status).toBe(200);
    expect(Array.isArray(res.body)).toBe(true);
    expect((res.body as unknown[]).length).toBe(1);
  });

  it("DELETE /v1/workers/:id removes a worker", async () => {
    await router.handle("POST", "/v1/workers", {
      id: "p1",
      endpoint: "http://p1:8000",
      role: "prefill",
      modelId: "m1",
    });

    const del = await router.handle("DELETE", "/v1/workers/p1");
    expect(del.status).toBe(200);

    const list = await router.handle("GET", "/v1/workers");
    expect((list.body as unknown[]).length).toBe(0);
  });

  it("DELETE /v1/workers/:id returns 404 for unknown worker", async () => {
    const res = await router.handle("DELETE", "/v1/workers/unknown");
    expect(res.status).toBe(404);
  });

  // -----------------------------------------------------------------------
  // Metrics & Events
  // -----------------------------------------------------------------------

  it("GET /v1/metrics returns metrics", async () => {
    const res = await router.handle("GET", "/v1/metrics");
    expect(res.status).toBe(200);
    expect((res.body as { queueDepth: number }).queueDepth).toBe(0);
  });

  it("GET /v1/events returns events array", async () => {
    const res = await router.handle("GET", "/v1/events");
    expect(res.status).toBe(200);
    expect(Array.isArray(res.body)).toBe(true);
  });

  // -----------------------------------------------------------------------
  // Completions
  // -----------------------------------------------------------------------

  it("POST /v1/completions rejects missing fields", async () => {
    const res = await router.handle("POST", "/v1/completions", {});
    expect(res.status).toBe(400);
  });

  // -----------------------------------------------------------------------
  // 404
  // -----------------------------------------------------------------------

  it("returns 404 for unknown routes", async () => {
    const res = await router.handle("GET", "/unknown");
    expect(res.status).toBe(404);
  });
});
