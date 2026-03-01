/**
 * Core type definitions for vLLM PD (Prefill-Decode) dynamic task scheduler.
 *
 * In disaggregated serving, the prefill phase (processing all input tokens to
 * build the KV cache) and the decode phase (auto-regressive token generation)
 * run on separate worker pools. This module defines the shared types used
 * across the scheduler, worker pool, KV cache transfer, and routing layers.
 */

// ---------------------------------------------------------------------------
// Worker & cluster topology
// ---------------------------------------------------------------------------

/** Role a vLLM worker can assume. */
export type WorkerRole = "prefill" | "decode";

/** Operational state reported by a worker. */
export type WorkerStatus = "idle" | "busy" | "draining" | "offline";

/** Descriptor for a single vLLM worker instance. */
export interface WorkerInfo {
  /** Unique worker identifier (e.g. hostname:port). */
  id: string;
  /** Base URL of the vLLM OpenAI-compatible API. */
  endpoint: string;
  /** Current role assignment. */
  role: WorkerRole;
  /** Operational status. */
  status: WorkerStatus;
  /** GPU memory utilisation in [0, 1]. */
  gpuUtilization: number;
  /** Number of requests currently being processed. */
  activeRequests: number;
  /** Maximum concurrent requests the worker can handle. */
  maxConcurrency: number;
  /** Epoch-ms of last successful health check. */
  lastHealthCheck: number;
  /** Model ID loaded on this worker (e.g. "Qwen/Qwen2-72B"). */
  modelId: string;
  /** Arbitrary metadata attached by the operator. */
  metadata?: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Inference requests & responses
// ---------------------------------------------------------------------------

/** Scheduling priority – lower numeric value = higher priority. */
export type RequestPriority = "high" | "normal" | "low";

/** Lifecycle state of an inference request inside the scheduler. */
export type RequestPhase =
  | "queued"
  | "prefilling"
  | "transferring"
  | "decoding"
  | "completed"
  | "failed";

/** An inference request flowing through the PD pipeline. */
export interface InferenceRequest {
  /** Globally unique request ID. */
  requestId: string;
  /** Model to run inference on. */
  modelId: string;
  /** Tokenised (or raw text) prompt – the prefill input. */
  prompt: string | number[];
  /** Sampling / generation parameters forwarded to vLLM. */
  samplingParams: SamplingParams;
  /** Scheduling priority. */
  priority: RequestPriority;
  /** Current lifecycle phase. */
  phase: RequestPhase;
  /** Epoch-ms when the request entered the scheduler. */
  createdAt: number;
  /** ID of the worker handling prefill (set after assignment). */
  prefillWorkerId?: string;
  /** ID of the worker handling decode (set after assignment). */
  decodeWorkerId?: string;
  /** Opaque handle for the KV cache produced by prefill. */
  kvCacheHandle?: string;
  /** Maximum time (ms) the request may stay in the queue. */
  timeoutMs?: number;
}

/** Generation parameters forwarded to vLLM. */
export interface SamplingParams {
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  repetitionPenalty?: number;
  stop?: string[];
  stream?: boolean;
}

/** Result returned when the decode phase completes. */
export interface InferenceResult {
  requestId: string;
  text: string;
  tokenCount: number;
  prefillLatencyMs: number;
  decodeLatencyMs: number;
  totalLatencyMs: number;
  prefillWorkerId: string;
  decodeWorkerId: string;
}

// ---------------------------------------------------------------------------
// KV cache transfer
// ---------------------------------------------------------------------------

/** Describes a KV cache block that must be transferred from P → D worker. */
export interface KVCacheTransferRequest {
  requestId: string;
  /** Base URL of the source (prefill) worker's API. */
  sourceEndpoint: string;
  /** Base URL of the target (decode) worker's API. */
  targetEndpoint: string;
  /** Opaque handle identifying the cache on the source worker. */
  cacheHandle: string;
  /** Size of the cache payload in bytes (for future bandwidth estimation). */
  cacheSizeBytes: number;
}

export interface KVCacheTransferResult {
  requestId: string;
  success: boolean;
  transferDurationMs: number;
  /** Handle on the *target* worker after successful transfer. */
  targetCacheHandle?: string;
  error?: string;
}

// ---------------------------------------------------------------------------
// Scheduler configuration
// ---------------------------------------------------------------------------

/** Scheduling algorithm to use when picking workers. */
export type SchedulingStrategy =
  | "round-robin"
  | "least-loaded"
  | "latency-aware";

/** Top-level configuration for the PD scheduler plugin. */
export interface PDSchedulerConfig {
  /** Whether the plugin is active. */
  enabled: boolean;
  /** Scheduling strategy for assigning requests to workers. */
  strategy: SchedulingStrategy;
  /** Interval (ms) between health-check probes. */
  healthCheckIntervalMs: number;
  /** How long (ms) before an unresponsive worker is marked offline. */
  workerTimeoutMs: number;
  /** Max requests waiting in the scheduler queue. */
  maxQueueSize: number;
  /** Default per-request timeout (ms). */
  defaultRequestTimeoutMs: number;
  /** Initial set of workers to register at startup. */
  workers: WorkerSeedConfig[];
  /** KV cache transfer settings. */
  kvTransfer: {
    /** Maximum concurrent transfers. */
    maxConcurrent: number;
    /** Transfer timeout (ms). */
    timeoutMs: number;
  };
}

/** Seed configuration for a worker supplied via config file. */
export interface WorkerSeedConfig {
  id: string;
  endpoint: string;
  role: WorkerRole;
  modelId: string;
  maxConcurrency?: number;
}

// ---------------------------------------------------------------------------
// Metrics & events
// ---------------------------------------------------------------------------

/** Aggregate metrics snapshot exposed by the scheduler. */
export interface SchedulerMetrics {
  /** Number of requests currently queued. */
  queueDepth: number;
  /** Number of requests in the prefill phase. */
  activePrefills: number;
  /** Number of KV cache transfers in flight. */
  activeTransfers: number;
  /** Number of requests in the decode phase. */
  activeDecodes: number;
  /** Total requests completed since startup. */
  totalCompleted: number;
  /** Total requests failed since startup. */
  totalFailed: number;
  /** Average end-to-end latency (ms) over a recent window. */
  avgLatencyMs: number;
  /** Average prefill latency (ms) over a recent window. */
  avgPrefillLatencyMs: number;
  /** Average decode latency (ms) over a recent window. */
  avgDecodeLatencyMs: number;
  /** Per-worker status summary. */
  workers: Pick<WorkerInfo, "id" | "role" | "status" | "gpuUtilization" | "activeRequests">[];
}

/** Events emitted by the scheduler for observability. */
export type SchedulerEvent =
  | { kind: "request_queued"; requestId: string; timestamp: number }
  | { kind: "prefill_started"; requestId: string; workerId: string; timestamp: number }
  | { kind: "prefill_completed"; requestId: string; workerId: string; latencyMs: number; timestamp: number }
  | { kind: "transfer_started"; requestId: string; from: string; to: string; timestamp: number }
  | { kind: "transfer_completed"; requestId: string; durationMs: number; timestamp: number }
  | { kind: "decode_started"; requestId: string; workerId: string; timestamp: number }
  | { kind: "decode_completed"; requestId: string; workerId: string; latencyMs: number; timestamp: number }
  | { kind: "request_completed"; requestId: string; totalLatencyMs: number; timestamp: number }
  | { kind: "request_failed"; requestId: string; error: string; timestamp: number }
  | { kind: "worker_online"; workerId: string; role: WorkerRole; timestamp: number }
  | { kind: "worker_offline"; workerId: string; timestamp: number };
