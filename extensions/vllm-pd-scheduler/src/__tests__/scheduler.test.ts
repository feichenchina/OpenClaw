/**
 * Unit tests for the PDScheduler core logic.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import type { PDSchedulerConfig } from "../types.js";
import { PDScheduler } from "../scheduler.js";

// ---------------------------------------------------------------------------
// Test config with no real workers (unit-level tests mock the network)
// ---------------------------------------------------------------------------

const baseConfig: PDSchedulerConfig = {
  enabled: true,
  strategy: "least-loaded",
  healthCheckIntervalMs: 60_000, // don't interfere during tests
  workerTimeoutMs: 30_000,
  maxQueueSize: 10,
  defaultRequestTimeoutMs: 5_000,
  workers: [],
  kvTransfer: { maxConcurrent: 2, timeoutMs: 5_000 },
};

describe("PDScheduler", () => {
  let scheduler: PDScheduler;

  beforeEach(() => {
    scheduler = new PDScheduler(baseConfig);
  });

  afterEach(() => {
    scheduler.stop();
  });

  // -----------------------------------------------------------------------
  // Queue management
  // -----------------------------------------------------------------------

  it("rejects submissions when queue is full", async () => {
    const config: PDSchedulerConfig = { ...baseConfig, maxQueueSize: 0 };
    const s = new PDScheduler(config);

    await expect(
      s.submit({ modelId: "m", prompt: "hello" }),
    ).rejects.toThrow("queue is full");
  });

  // -----------------------------------------------------------------------
  // Metrics
  // -----------------------------------------------------------------------

  it("returns initial metrics", () => {
    const m = scheduler.metrics();
    expect(m.queueDepth).toBe(0);
    expect(m.activePrefills).toBe(0);
    expect(m.activeDecodes).toBe(0);
    expect(m.totalCompleted).toBe(0);
    expect(m.totalFailed).toBe(0);
    expect(m.workers).toEqual([]);
  });

  it("registers seed workers from config", () => {
    const config: PDSchedulerConfig = {
      ...baseConfig,
      workers: [
        { id: "p1", endpoint: "http://p1:8000", role: "prefill", modelId: "m1" },
        { id: "d1", endpoint: "http://d1:8000", role: "decode", modelId: "m1" },
      ],
    };
    const s = new PDScheduler(config);

    const m = s.metrics();
    expect(m.workers).toHaveLength(2);
    expect(m.workers.map((w) => w.id).sort()).toEqual(["d1", "p1"]);
  });

  // -----------------------------------------------------------------------
  // Worker pool access
  // -----------------------------------------------------------------------

  it("exposes the worker pool for runtime management", () => {
    const pool = scheduler.workerPool;
    pool.register({
      id: "p-runtime",
      endpoint: "http://p:8000",
      role: "prefill",
      modelId: "m1",
    });

    expect(pool.list()).toHaveLength(1);
  });

  // -----------------------------------------------------------------------
  // Events
  // -----------------------------------------------------------------------

  it("queues request and emits request_queued event", () => {
    // Don't start the scheduler so dispatch doesn't fire.
    void scheduler.submit({ modelId: "m", prompt: "test" });

    const events = scheduler.events();
    expect(events.length).toBeGreaterThanOrEqual(1);
    expect(events[0].kind).toBe("request_queued");
  });
});
