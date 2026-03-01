/**
 * Unit tests for the WorkerPool.
 */

import { describe, it, expect, beforeEach } from "vitest";
import { WorkerPool } from "../worker-pool.js";

describe("WorkerPool", () => {
  let pool: WorkerPool;

  beforeEach(() => {
    pool = new WorkerPool();
  });

  // -----------------------------------------------------------------------
  // Registration
  // -----------------------------------------------------------------------

  it("registers a new worker", () => {
    const info = pool.register({
      id: "p1",
      endpoint: "http://p1:8000",
      role: "prefill",
      modelId: "test-model",
    });

    expect(info.id).toBe("p1");
    expect(info.role).toBe("prefill");
    expect(info.status).toBe("idle");
    expect(info.activeRequests).toBe(0);
  });

  it("updates an existing worker without resetting status", () => {
    pool.register({
      id: "p1",
      endpoint: "http://p1:8000",
      role: "prefill",
      modelId: "m1",
    });
    pool.updateMetrics("p1", { status: "busy", gpuUtilization: 0.8 });

    const updated = pool.register({
      id: "p1",
      endpoint: "http://p1:8000",
      role: "prefill",
      modelId: "m1",
    });

    // Status should be preserved from the existing worker.
    expect(updated.status).toBe("busy");
    expect(updated.gpuUtilization).toBe(0.8);
  });

  it("removes a worker", () => {
    pool.register({ id: "p1", endpoint: "http://p1:8000", role: "prefill", modelId: "m1" });
    expect(pool.remove("p1")).toBe(true);
    expect(pool.get("p1")).toBeUndefined();
    expect(pool.remove("nonexistent")).toBe(false);
  });

  // -----------------------------------------------------------------------
  // Queries
  // -----------------------------------------------------------------------

  it("lists workers filtered by role", () => {
    pool.register({ id: "p1", endpoint: "http://p1:8000", role: "prefill", modelId: "m1" });
    pool.register({ id: "d1", endpoint: "http://d1:8000", role: "decode", modelId: "m1" });
    pool.register({ id: "p2", endpoint: "http://p2:8000", role: "prefill", modelId: "m1" });

    expect(pool.list("prefill")).toHaveLength(2);
    expect(pool.list("decode")).toHaveLength(1);
    expect(pool.list()).toHaveLength(3);
  });

  it("returns available workers (not offline, not at capacity)", () => {
    pool.register({ id: "p1", endpoint: "http://p1:8000", role: "prefill", modelId: "m1", maxConcurrency: 2 });
    pool.register({ id: "p2", endpoint: "http://p2:8000", role: "prefill", modelId: "m1" });

    // Fill p1 to capacity.
    pool.incrementActive("p1");
    pool.incrementActive("p1");

    const available = pool.available("prefill");
    expect(available).toHaveLength(1);
    expect(available[0].id).toBe("p2");
  });

  // -----------------------------------------------------------------------
  // Selection strategies
  // -----------------------------------------------------------------------

  it("selects workers round-robin", () => {
    pool.register({ id: "p1", endpoint: "http://p1:8000", role: "prefill", modelId: "m1" });
    pool.register({ id: "p2", endpoint: "http://p2:8000", role: "prefill", modelId: "m1" });

    const first = pool.select("prefill", "round-robin");
    const second = pool.select("prefill", "round-robin");
    const third = pool.select("prefill", "round-robin");

    expect(first?.id).toBe("p1");
    expect(second?.id).toBe("p2");
    expect(third?.id).toBe("p1"); // wraps around
  });

  it("selects least-loaded worker", () => {
    pool.register({ id: "p1", endpoint: "http://p1:8000", role: "prefill", modelId: "m1" });
    pool.register({ id: "p2", endpoint: "http://p2:8000", role: "prefill", modelId: "m1" });
    pool.incrementActive("p1");
    pool.incrementActive("p1");
    pool.incrementActive("p2");

    const selected = pool.select("prefill", "least-loaded");
    expect(selected?.id).toBe("p2");
  });

  it("selects latency-aware (lowest GPU utilisation)", () => {
    pool.register({ id: "p1", endpoint: "http://p1:8000", role: "prefill", modelId: "m1" });
    pool.register({ id: "p2", endpoint: "http://p2:8000", role: "prefill", modelId: "m1" });
    pool.updateMetrics("p1", { gpuUtilization: 0.9 });
    pool.updateMetrics("p2", { gpuUtilization: 0.3 });

    const selected = pool.select("prefill", "latency-aware");
    expect(selected?.id).toBe("p2");
  });

  it("returns undefined when no workers are available", () => {
    expect(pool.select("prefill", "least-loaded")).toBeUndefined();
  });

  // -----------------------------------------------------------------------
  // Status management
  // -----------------------------------------------------------------------

  it("increments and decrements active requests", () => {
    pool.register({ id: "p1", endpoint: "http://p1:8000", role: "prefill", modelId: "m1", maxConcurrency: 2 });

    pool.incrementActive("p1");
    expect(pool.get("p1")?.activeRequests).toBe(1);

    pool.incrementActive("p1");
    expect(pool.get("p1")?.status).toBe("busy");

    pool.decrementActive("p1");
    expect(pool.get("p1")?.activeRequests).toBe(1);
    expect(pool.get("p1")?.status).toBe("idle");
  });

  it("does not decrement below zero", () => {
    pool.register({ id: "p1", endpoint: "http://p1:8000", role: "prefill", modelId: "m1" });
    pool.decrementActive("p1");
    expect(pool.get("p1")?.activeRequests).toBe(0);
  });

  it("marks stale workers offline", () => {
    pool.register({ id: "p1", endpoint: "http://p1:8000", role: "prefill", modelId: "m1" });

    // Manually backdate the health check.
    const w = pool.get("p1")!;
    w.lastHealthCheck = Date.now() - 60_000;

    const expired = pool.expireStaleWorkers(30_000);
    expect(expired).toContain("p1");
    expect(pool.get("p1")?.status).toBe("offline");
  });

  it("offline workers are not available", () => {
    pool.register({ id: "p1", endpoint: "http://p1:8000", role: "prefill", modelId: "m1" });
    pool.markOffline("p1");
    expect(pool.available("prefill")).toHaveLength(0);
  });
});
