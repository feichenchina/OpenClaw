/**
 * Worker pool manager – tracks vLLM worker instances and their health.
 *
 * Workers are categorised by role (prefill / decode) and selected by the
 * scheduler according to the configured strategy. The pool manager runs
 * periodic health checks and removes unresponsive workers.
 */

import type {
  WorkerInfo,
  WorkerRole,
  WorkerStatus,
  WorkerSeedConfig,
  SchedulingStrategy,
} from "./types.js";

export class WorkerPool {
  private readonly workers = new Map<string, WorkerInfo>();
  private roundRobinCounters = { prefill: 0, decode: 0 };

  // -------------------------------------------------------------------------
  // Registration
  // -------------------------------------------------------------------------

  /** Register a new worker (or update an existing one). */
  register(seed: WorkerSeedConfig): WorkerInfo {
    const existing = this.workers.get(seed.id);
    const info: WorkerInfo = {
      id: seed.id,
      endpoint: seed.endpoint,
      role: seed.role,
      status: existing?.status ?? "idle",
      gpuUtilization: existing?.gpuUtilization ?? 0,
      activeRequests: existing?.activeRequests ?? 0,
      maxConcurrency: seed.maxConcurrency ?? 32,
      lastHealthCheck: Date.now(),
      modelId: seed.modelId,
    };
    this.workers.set(seed.id, info);
    return info;
  }

  /** Remove a worker from the pool. */
  remove(workerId: string): boolean {
    return this.workers.delete(workerId);
  }

  // -------------------------------------------------------------------------
  // Queries
  // -------------------------------------------------------------------------

  get(workerId: string): WorkerInfo | undefined {
    return this.workers.get(workerId);
  }

  /** List all workers, optionally filtered by role. */
  list(role?: WorkerRole): WorkerInfo[] {
    const all = Array.from(this.workers.values());
    return role ? all.filter((w) => w.role === role) : all;
  }

  /** Return workers that can accept new work. */
  available(role: WorkerRole): WorkerInfo[] {
    return this.list(role).filter(
      (w) =>
        (w.status === "idle" || w.status === "busy") &&
        w.activeRequests < w.maxConcurrency,
    );
  }

  // -------------------------------------------------------------------------
  // Selection
  // -------------------------------------------------------------------------

  /** Select the best available worker for the given role and strategy. */
  select(
    role: WorkerRole,
    strategy: SchedulingStrategy,
  ): WorkerInfo | undefined {
    const candidates = this.available(role);
    if (candidates.length === 0) return undefined;

    switch (strategy) {
      case "round-robin":
        return this.selectRoundRobin(candidates, role);
      case "least-loaded":
        return this.selectLeastLoaded(candidates);
      case "latency-aware":
        // Latency-aware falls back to least GPU utilisation as a proxy.
        return this.selectLatencyAware(candidates);
      default:
        return candidates[0];
    }
  }

  // -------------------------------------------------------------------------
  // Status updates
  // -------------------------------------------------------------------------

  /** Update runtime metrics for a worker after a health probe. */
  updateMetrics(
    workerId: string,
    metrics: {
      gpuUtilization?: number;
      activeRequests?: number;
      status?: WorkerStatus;
    },
  ): void {
    const w = this.workers.get(workerId);
    if (!w) return;
    if (metrics.gpuUtilization !== undefined) w.gpuUtilization = metrics.gpuUtilization;
    if (metrics.activeRequests !== undefined) w.activeRequests = metrics.activeRequests;
    if (metrics.status !== undefined) w.status = metrics.status;
    w.lastHealthCheck = Date.now();
  }

  /** Mark a worker offline (e.g. after a failed health check). */
  markOffline(workerId: string): void {
    const w = this.workers.get(workerId);
    if (w) w.status = "offline";
  }

  /** Increment the active-request counter for a worker. */
  incrementActive(workerId: string): void {
    const w = this.workers.get(workerId);
    if (!w) return;
    w.activeRequests++;
    if (w.activeRequests >= w.maxConcurrency) w.status = "busy";
  }

  /** Decrement the active-request counter for a worker. */
  decrementActive(workerId: string): void {
    const w = this.workers.get(workerId);
    if (!w) return;
    w.activeRequests = Math.max(0, w.activeRequests - 1);
    if (w.status === "busy" && w.activeRequests < w.maxConcurrency) {
      w.status = "idle";
    }
  }

  /** Mark all workers whose last health check is older than `timeoutMs` as offline. */
  expireStaleWorkers(timeoutMs: number): string[] {
    const now = Date.now();
    const expired: string[] = [];
    for (const w of this.workers.values()) {
      if (w.status !== "offline" && now - w.lastHealthCheck > timeoutMs) {
        w.status = "offline";
        expired.push(w.id);
      }
    }
    return expired;
  }

  // -------------------------------------------------------------------------
  // Private selection helpers
  // -------------------------------------------------------------------------

  private selectRoundRobin(
    candidates: WorkerInfo[],
    role: WorkerRole,
  ): WorkerInfo {
    const idx = this.roundRobinCounters[role] % candidates.length;
    this.roundRobinCounters[role]++;
    return candidates[idx];
  }

  private selectLeastLoaded(candidates: WorkerInfo[]): WorkerInfo {
    return candidates.reduce((best, cur) =>
      cur.activeRequests < best.activeRequests ? cur : best,
    );
  }

  private selectLatencyAware(candidates: WorkerInfo[]): WorkerInfo {
    // Use GPU utilisation as a latency proxy – lower utilisation ≈ lower
    // queuing delay inside the vLLM engine.
    return candidates.reduce((best, cur) =>
      cur.gpuUtilization < best.gpuUtilization ? cur : best,
    );
  }
}
