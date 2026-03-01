/**
 * Health-check loop and metrics aggregation.
 *
 * Periodically pings every registered worker via the vLLM client and
 * updates the worker pool accordingly.  Also maintains a rolling window
 * of latency samples for the {@link SchedulerMetrics} snapshot.
 */

import type { SchedulerMetrics, SchedulerEvent, WorkerRole } from "./types.js";
import type { WorkerPool } from "./worker-pool.js";
import type { VllmClient } from "./vllm-client.js";

// ---------------------------------------------------------------------------
// Rolling latency tracker
// ---------------------------------------------------------------------------

const WINDOW_SIZE = 200; // keep the last N samples

class LatencyWindow {
  private readonly samples: number[] = [];

  push(value: number): void {
    this.samples.push(value);
    if (this.samples.length > WINDOW_SIZE) this.samples.shift();
  }

  average(): number {
    if (this.samples.length === 0) return 0;
    return this.samples.reduce((a, b) => a + b, 0) / this.samples.length;
  }
}

// ---------------------------------------------------------------------------
// HealthMonitor
// ---------------------------------------------------------------------------

export class HealthMonitor {
  private readonly pool: WorkerPool;
  private readonly client: VllmClient;
  private readonly intervalMs: number;
  private readonly workerTimeoutMs: number;
  private timer: ReturnType<typeof setInterval> | null = null;

  readonly latency = {
    total: new LatencyWindow(),
    prefill: new LatencyWindow(),
    decode: new LatencyWindow(),
  };

  private counters = { completed: 0, failed: 0 };

  private readonly eventLog: SchedulerEvent[] = [];
  private onEvent?: (event: SchedulerEvent) => void;

  constructor(opts: {
    pool: WorkerPool;
    client: VllmClient;
    intervalMs: number;
    workerTimeoutMs: number;
    onEvent?: (event: SchedulerEvent) => void;
  }) {
    this.pool = opts.pool;
    this.client = opts.client;
    this.intervalMs = opts.intervalMs;
    this.workerTimeoutMs = opts.workerTimeoutMs;
    this.onEvent = opts.onEvent;
  }

  // -----------------------------------------------------------------------
  // Lifecycle
  // -----------------------------------------------------------------------

  start(): void {
    if (this.timer) return;
    this.timer = setInterval(() => void this.runHealthChecks(), this.intervalMs);
  }

  stop(): void {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
  }

  // -----------------------------------------------------------------------
  // Health checks
  // -----------------------------------------------------------------------

  async runHealthChecks(): Promise<void> {
    const workers = this.pool.list();
    const checks = workers.map(async (w) => {
      const result = await this.client.health(w);
      if (result.healthy) {
        this.pool.updateMetrics(w.id, {
          gpuUtilization: result.gpuUtilization,
          activeRequests: result.activeRequests,
          status: result.activeRequests >= w.maxConcurrency ? "busy" : "idle",
        });
      } else {
        this.pool.markOffline(w.id);
        this.emit({ kind: "worker_offline", workerId: w.id, timestamp: Date.now() });
      }
    });

    await Promise.allSettled(checks);

    // Also expire workers that haven't been probed in a while.
    const expired = this.pool.expireStaleWorkers(this.workerTimeoutMs);
    for (const wid of expired) {
      this.emit({ kind: "worker_offline", workerId: wid, timestamp: Date.now() });
    }
  }

  // -----------------------------------------------------------------------
  // Metrics
  // -----------------------------------------------------------------------

  recordCompletion(totalMs: number, prefillMs: number, decodeMs: number): void {
    this.counters.completed++;
    this.latency.total.push(totalMs);
    this.latency.prefill.push(prefillMs);
    this.latency.decode.push(decodeMs);
  }

  recordFailure(): void {
    this.counters.failed++;
  }

  snapshot(extra: {
    queueDepth: number;
    activePrefills: number;
    activeTransfers: number;
    activeDecodes: number;
  }): SchedulerMetrics {
    return {
      ...extra,
      totalCompleted: this.counters.completed,
      totalFailed: this.counters.failed,
      avgLatencyMs: Math.round(this.latency.total.average()),
      avgPrefillLatencyMs: Math.round(this.latency.prefill.average()),
      avgDecodeLatencyMs: Math.round(this.latency.decode.average()),
      workers: this.pool.list().map((w) => ({
        id: w.id,
        role: w.role,
        status: w.status,
        gpuUtilization: w.gpuUtilization,
        activeRequests: w.activeRequests,
      })),
    };
  }

  // -----------------------------------------------------------------------
  // Events
  // -----------------------------------------------------------------------

  emit(event: SchedulerEvent): void {
    this.eventLog.push(event);
    if (this.eventLog.length > 1000) this.eventLog.shift();
    this.onEvent?.(event);
  }

  recentEvents(limit = 50): SchedulerEvent[] {
    return this.eventLog.slice(-limit);
  }
}
