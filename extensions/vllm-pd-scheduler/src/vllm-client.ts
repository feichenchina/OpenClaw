/**
 * Lightweight vLLM API client.
 *
 * Talks to the OpenAI-compatible API exposed by vLLM workers for:
 *   1. Running prefill-only passes (returns a KV cache handle).
 *   2. Running decode passes (given a KV cache handle).
 *   3. Health / readiness probes.
 *
 * NOTE: The "prefill-only" and "KV-cache-aware decode" endpoints rely on
 * vLLM's disaggregated-serving extensions.  When those are unavailable the
 * client falls back to a standard `/v1/completions` call for both phases.
 */

import type { SamplingParams, WorkerInfo } from "./types.js";

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export interface PrefillResult {
  kvCacheHandle: string;
  promptTokens: number;
  latencyMs: number;
}

export interface DecodeResult {
  text: string;
  completionTokens: number;
  latencyMs: number;
}

export interface HealthResult {
  healthy: boolean;
  gpuUtilization: number;
  activeRequests: number;
  error?: string;
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

export class VllmClient {
  private readonly timeoutMs: number;

  constructor(opts?: { timeoutMs?: number }) {
    this.timeoutMs = opts?.timeoutMs ?? 30_000;
  }

  // -----------------------------------------------------------------------
  // Prefill
  // -----------------------------------------------------------------------

  /**
   * Submit a prefill-only request to a vLLM worker.
   *
   * Tries the disaggregated `/v1/prefill` endpoint first.  If the worker
   * does not expose it (404), falls back to `/v1/completions` with
   * `max_tokens=1` so we still obtain a KV cache handle.
   */
  async prefill(
    worker: WorkerInfo,
    requestId: string,
    prompt: string | number[],
    modelId: string,
  ): Promise<PrefillResult> {
    const start = Date.now();

    // Try the disaggregated endpoint first.
    try {
      const res = await this.post(`${worker.endpoint}/v1/prefill`, {
        model: modelId,
        prompt,
        request_id: requestId,
      });

      if (res.ok) {
        const body = (await res.json()) as {
          kv_cache_handle: string;
          prompt_tokens: number;
        };
        return {
          kvCacheHandle: body.kv_cache_handle,
          promptTokens: body.prompt_tokens,
          latencyMs: Date.now() - start,
        };
      }

      // If 404 the endpoint is not available – fall through to the fallback.
      if (res.status !== 404) {
        throw new Error(`prefill endpoint returned ${res.status}: ${await res.text()}`);
      }
    } catch (err: unknown) {
      // Network errors fall through to the fallback path as well.
      if (err instanceof TypeError) {
        /* fetch network error – fall through */
      } else if (err instanceof Error && err.message.startsWith("prefill endpoint")) {
        throw err;
      }
    }

    // Fallback: use a standard completion with max_tokens=1.
    const fallbackRes = await this.post(`${worker.endpoint}/v1/completions`, {
      model: modelId,
      prompt,
      max_tokens: 1,
      request_id: requestId,
    });

    if (!fallbackRes.ok) {
      throw new Error(
        `prefill fallback returned ${fallbackRes.status}: ${await fallbackRes.text()}`,
      );
    }

    const fallbackBody = (await fallbackRes.json()) as {
      id: string;
      usage?: { prompt_tokens?: number };
    };

    return {
      kvCacheHandle: fallbackBody.id,
      promptTokens: fallbackBody.usage?.prompt_tokens ?? 0,
      latencyMs: Date.now() - start,
    };
  }

  // -----------------------------------------------------------------------
  // Decode
  // -----------------------------------------------------------------------

  /**
   * Submit a decode request that resumes from a transferred KV cache.
   *
   * Like {@link prefill}, falls back to a standard completions call when
   * the disaggregated endpoint is not available.
   */
  async decode(
    worker: WorkerInfo,
    requestId: string,
    kvCacheHandle: string,
    modelId: string,
    params: SamplingParams,
  ): Promise<DecodeResult> {
    const start = Date.now();

    try {
      const res = await this.post(`${worker.endpoint}/v1/decode`, {
        model: modelId,
        kv_cache_handle: kvCacheHandle,
        request_id: requestId,
        max_tokens: params.maxTokens ?? 512,
        temperature: params.temperature,
        top_p: params.topP,
        top_k: params.topK,
        repetition_penalty: params.repetitionPenalty,
        stop: params.stop,
      });

      if (res.ok) {
        const body = (await res.json()) as {
          text: string;
          completion_tokens: number;
        };
        return {
          text: body.text,
          completionTokens: body.completion_tokens,
          latencyMs: Date.now() - start,
        };
      }

      if (res.status !== 404) {
        throw new Error(`decode endpoint returned ${res.status}: ${await res.text()}`);
      }
    } catch (err: unknown) {
      if (err instanceof TypeError) {
        /* network error – fall through */
      } else if (err instanceof Error && err.message.startsWith("decode endpoint")) {
        throw err;
      }
    }

    // Fallback: standard completion. The `<kv_cache:…>` prefix is a
    // convention used by some vLLM forks to hint that the prompt should
    // be resolved from an existing KV cache rather than re-tokenised.
    // When the backend does not support this, it is treated as a literal
    // text prompt and the decode quality may degrade.
    const fallbackRes = await this.post(`${worker.endpoint}/v1/completions`, {
      model: modelId,
      prompt: `<kv_cache:${kvCacheHandle}>`,
      max_tokens: params.maxTokens ?? 512,
      temperature: params.temperature,
      top_p: params.topP,
      top_k: params.topK,
      repetition_penalty: params.repetitionPenalty,
      stop: params.stop,
      request_id: requestId,
    });

    if (!fallbackRes.ok) {
      throw new Error(
        `decode fallback returned ${fallbackRes.status}: ${await fallbackRes.text()}`,
      );
    }

    const body = (await fallbackRes.json()) as {
      choices?: { text?: string }[];
      usage?: { completion_tokens?: number };
    };

    return {
      text: body.choices?.[0]?.text ?? "",
      completionTokens: body.usage?.completion_tokens ?? 0,
      latencyMs: Date.now() - start,
    };
  }

  // -----------------------------------------------------------------------
  // Health
  // -----------------------------------------------------------------------

  /** Probe a worker for health and runtime metrics. */
  async health(worker: WorkerInfo): Promise<HealthResult> {
    try {
      const res = await this.get(`${worker.endpoint}/health`);

      if (!res.ok) {
        return {
          healthy: false,
          gpuUtilization: 0,
          activeRequests: 0,
          error: `HTTP ${res.status}`,
        };
      }

      const body = (await res.json()) as {
        status?: string;
        gpu_utilization?: number;
        active_requests?: number;
      };

      return {
        healthy: body.status === "ok" || res.ok,
        gpuUtilization: body.gpu_utilization ?? 0,
        activeRequests: body.active_requests ?? 0,
      };
    } catch (err: unknown) {
      return {
        healthy: false,
        gpuUtilization: 0,
        activeRequests: 0,
        error: err instanceof Error ? err.message : String(err),
      };
    }
  }

  // -----------------------------------------------------------------------
  // Helpers
  // -----------------------------------------------------------------------

  private post(url: string, body: unknown): Promise<Response> {
    return fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(this.timeoutMs),
    });
  }

  private get(url: string): Promise<Response> {
    return fetch(url, {
      method: "GET",
      signal: AbortSignal.timeout(this.timeoutMs),
    });
  }
}
