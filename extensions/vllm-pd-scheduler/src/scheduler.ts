/**
 * PD dynamic task scheduler – the central orchestrator.
 *
 * Lifecycle of a request:
 *   1. **Queued** – request enters the scheduler queue.
 *   2. **Prefilling** – a prefill worker is selected and begins processing.
 *   3. **Transferring** – the KV cache is moved from the prefill worker to a
 *      decode worker.
 *   4. **Decoding** – the decode worker generates tokens.
 *   5. **Completed / Failed** – terminal states.
 *
 * The scheduler runs a continuous dispatch loop that:
 *   • Picks the next queued request (priority-aware).
 *   • Selects a prefill worker according to the configured strategy.
 *   • After prefill, coordinates a KV cache transfer.
 *   • Selects a decode worker and kicks off token generation.
 *   • Emits events at each phase transition for observability.
 */

import type {
  InferenceRequest,
  InferenceResult,
  PDSchedulerConfig,
  SchedulerMetrics,
  SchedulerEvent,
  RequestPriority,
  SamplingParams,
} from "./types.js";
import { WorkerPool } from "./worker-pool.js";
import { VllmClient } from "./vllm-client.js";
import { KVCacheTransferManager } from "./kv-cache.js";
import { HealthMonitor } from "./health.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

let idCounter = 0;

function generateRequestId(): string {
  return `req-${Date.now()}-${++idCounter}`;
}

const PRIORITY_ORDER: Record<RequestPriority, number> = {
  high: 0,
  normal: 1,
  low: 2,
};

// ---------------------------------------------------------------------------
// Scheduler
// ---------------------------------------------------------------------------

export class PDScheduler {
  private readonly config: PDSchedulerConfig;
  private readonly pool: WorkerPool;
  private readonly client: VllmClient;
  private readonly kvManager: KVCacheTransferManager;
  private readonly health: HealthMonitor;

  /** Requests waiting to be dispatched. */
  private readonly queue: InferenceRequest[] = [];
  /** In-flight request tracking. */
  private readonly inflight = new Map<
    string,
    {
      request: InferenceRequest;
      resolve: (result: InferenceResult) => void;
      reject: (error: Error) => void;
    }
  >();

  private activePrefills = 0;
  private activeDecodes = 0;
  private dispatchTimer: ReturnType<typeof setInterval> | null = null;
  private running = false;

  constructor(config: PDSchedulerConfig) {
    this.config = config;
    this.pool = new WorkerPool();
    this.client = new VllmClient({ timeoutMs: config.defaultRequestTimeoutMs });
    this.kvManager = new KVCacheTransferManager(config.kvTransfer);
    this.health = new HealthMonitor({
      pool: this.pool,
      client: this.client,
      intervalMs: config.healthCheckIntervalMs,
      workerTimeoutMs: config.workerTimeoutMs,
    });

    // Register seed workers.
    for (const seed of config.workers) {
      this.pool.register(seed);
    }
  }

  // -----------------------------------------------------------------------
  // Lifecycle
  // -----------------------------------------------------------------------

  /** Start the scheduler dispatch loop and health monitor. */
  start(): void {
    if (this.running) return;
    this.running = true;
    this.health.start();
    this.dispatchTimer = setInterval(() => void this.dispatch(), 50);
  }

  /** Gracefully stop the scheduler. In-flight requests will finish. */
  stop(): void {
    this.running = false;
    this.health.stop();
    if (this.dispatchTimer) {
      clearInterval(this.dispatchTimer);
      this.dispatchTimer = null;
    }
  }

  // -----------------------------------------------------------------------
  // Submit
  // -----------------------------------------------------------------------

  /**
   * Submit an inference request to the scheduler.
   *
   * Returns a promise that resolves with the generated text once the
   * full P→transfer→D pipeline completes.
   */
  submit(opts: {
    modelId: string;
    prompt: string | number[];
    samplingParams?: SamplingParams;
    priority?: RequestPriority;
    timeoutMs?: number;
  }): Promise<InferenceResult> {
    if (this.queue.length >= this.config.maxQueueSize) {
      return Promise.reject(new Error("Scheduler queue is full"));
    }

    const request: InferenceRequest = {
      requestId: generateRequestId(),
      modelId: opts.modelId,
      prompt: opts.prompt,
      samplingParams: opts.samplingParams ?? {},
      priority: opts.priority ?? "normal",
      phase: "queued",
      createdAt: Date.now(),
      timeoutMs: opts.timeoutMs ?? this.config.defaultRequestTimeoutMs,
    };

    return new Promise<InferenceResult>((resolve, reject) => {
      this.queue.push(request);
      this.inflight.set(request.requestId, { request, resolve, reject });
      this.health.emit({
        kind: "request_queued",
        requestId: request.requestId,
        timestamp: Date.now(),
      });
    });
  }

  // -----------------------------------------------------------------------
  // Dispatch loop
  // -----------------------------------------------------------------------

  /** Try to move the next queued request into the prefill phase. */
  private async dispatch(): Promise<void> {
    if (this.queue.length === 0) return;

    // Sort by priority then by creation time (FIFO within same priority).
    this.queue.sort((a, b) => {
      const pd = PRIORITY_ORDER[a.priority] - PRIORITY_ORDER[b.priority];
      return pd !== 0 ? pd : a.createdAt - b.createdAt;
    });

    // Check for timed-out requests.
    const now = Date.now();
    for (let i = this.queue.length - 1; i >= 0; i--) {
      const req = this.queue[i];
      if (req.timeoutMs && now - req.createdAt > req.timeoutMs) {
        this.queue.splice(i, 1);
        this.fail(req.requestId, new Error("Request timed out in queue"));
      }
    }

    // Pick the first request that can be served.
    const prefillWorker = this.pool.select("prefill", this.config.strategy);
    if (!prefillWorker) return; // no capacity right now

    const request = this.queue.shift();
    if (!request) return;

    // Run the full pipeline asynchronously.
    void this.runPipeline(request, prefillWorker.id);
  }

  // -----------------------------------------------------------------------
  // Pipeline
  // -----------------------------------------------------------------------

  private async runPipeline(
    request: InferenceRequest,
    prefillWorkerId: string,
  ): Promise<void> {
    const rid = request.requestId;
    const prefillStart = Date.now();

    try {
      // --- Prefill ---
      request.phase = "prefilling";
      request.prefillWorkerId = prefillWorkerId;
      this.activePrefills++;
      this.pool.incrementActive(prefillWorkerId);

      this.health.emit({
        kind: "prefill_started",
        requestId: rid,
        workerId: prefillWorkerId,
        timestamp: Date.now(),
      });

      const prefillWorker = this.pool.get(prefillWorkerId)!;
      const prefillResult = await this.client.prefill(
        prefillWorker,
        rid,
        request.prompt,
        request.modelId,
      );

      this.pool.decrementActive(prefillWorkerId);
      this.activePrefills--;

      this.health.emit({
        kind: "prefill_completed",
        requestId: rid,
        workerId: prefillWorkerId,
        latencyMs: prefillResult.latencyMs,
        timestamp: Date.now(),
      });

      request.kvCacheHandle = prefillResult.kvCacheHandle;

      // --- KV Cache transfer ---
      request.phase = "transferring";

      const decodeWorker = this.pool.select("decode", this.config.strategy);
      if (!decodeWorker) {
        throw new Error("No decode worker available");
      }
      request.decodeWorkerId = decodeWorker.id;

      this.health.emit({
        kind: "transfer_started",
        requestId: rid,
        from: prefillWorkerId,
        to: decodeWorker.id,
        timestamp: Date.now(),
      });

      const transferResult = await this.kvManager.transfer({
        requestId: rid,
        sourceWorkerId: prefillWorker.endpoint,
        targetWorkerId: decodeWorker.endpoint,
        cacheHandle: prefillResult.kvCacheHandle,
        cacheSizeBytes: 0, // unknown until runtime
      });

      if (!transferResult.success) {
        throw new Error(`KV cache transfer failed: ${transferResult.error}`);
      }

      this.health.emit({
        kind: "transfer_completed",
        requestId: rid,
        durationMs: transferResult.transferDurationMs,
        timestamp: Date.now(),
      });

      // --- Decode ---
      request.phase = "decoding";
      this.activeDecodes++;
      this.pool.incrementActive(decodeWorker.id);

      this.health.emit({
        kind: "decode_started",
        requestId: rid,
        workerId: decodeWorker.id,
        timestamp: Date.now(),
      });

      const decodeResult = await this.client.decode(
        decodeWorker,
        rid,
        transferResult.targetCacheHandle ?? prefillResult.kvCacheHandle,
        request.modelId,
        request.samplingParams,
      );

      this.pool.decrementActive(decodeWorker.id);
      this.activeDecodes--;

      const totalLatencyMs = Date.now() - prefillStart;

      this.health.emit({
        kind: "decode_completed",
        requestId: rid,
        workerId: decodeWorker.id,
        latencyMs: decodeResult.latencyMs,
        timestamp: Date.now(),
      });

      this.health.emit({
        kind: "request_completed",
        requestId: rid,
        totalLatencyMs,
        timestamp: Date.now(),
      });

      // Record metrics.
      this.health.recordCompletion(
        totalLatencyMs,
        prefillResult.latencyMs,
        decodeResult.latencyMs,
      );

      // Resolve the caller's promise.
      request.phase = "completed";
      const entry = this.inflight.get(rid);
      if (entry) {
        this.inflight.delete(rid);
        entry.resolve({
          requestId: rid,
          text: decodeResult.text,
          tokenCount: decodeResult.completionTokens,
          prefillLatencyMs: prefillResult.latencyMs,
          decodeLatencyMs: decodeResult.latencyMs,
          totalLatencyMs,
          prefillWorkerId,
          decodeWorkerId: decodeWorker.id,
        });
      }
    } catch (err: unknown) {
      this.fail(
        rid,
        err instanceof Error ? err : new Error(String(err)),
      );
    }
  }

  // -----------------------------------------------------------------------
  // Failure handling
  // -----------------------------------------------------------------------

  private fail(requestId: string, error: Error): void {
    this.health.recordFailure();
    this.health.emit({
      kind: "request_failed",
      requestId,
      error: error.message,
      timestamp: Date.now(),
    });

    const entry = this.inflight.get(requestId);
    if (entry) {
      entry.request.phase = "failed";
      this.inflight.delete(requestId);
      entry.reject(error);
    }
  }

  // -----------------------------------------------------------------------
  // Observability
  // -----------------------------------------------------------------------

  /** Return a snapshot of current scheduler metrics. */
  metrics(): SchedulerMetrics {
    return this.health.snapshot({
      queueDepth: this.queue.length,
      activePrefills: this.activePrefills,
      activeTransfers: this.kvManager.active,
      activeDecodes: this.activeDecodes,
    });
  }

  /** Return recent scheduler events. */
  events(limit?: number): SchedulerEvent[] {
    return this.health.recentEvents(limit);
  }

  /** Expose the worker pool for external management (e.g. adding workers at runtime). */
  get workerPool(): WorkerPool {
    return this.pool;
  }
}
