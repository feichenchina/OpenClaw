/**
 * KV cache transfer coordinator.
 *
 * After a prefill worker completes the prefill phase it holds a KV cache in
 * GPU memory.  Before the decode worker can continue token generation the
 * cache must be transferred.  This module coordinates those transfers with
 * concurrency control and timeout handling.
 */

import type {
  KVCacheTransferRequest,
  KVCacheTransferResult,
  PDSchedulerConfig,
} from "./types.js";

export class KVCacheTransferManager {
  private readonly maxConcurrent: number;
  private readonly timeoutMs: number;
  private activeTransfers = 0;
  private readonly pending: Array<{
    req: KVCacheTransferRequest;
    resolve: (r: KVCacheTransferResult) => void;
    reject: (e: Error) => void;
  }> = [];

  constructor(config: PDSchedulerConfig["kvTransfer"]) {
    this.maxConcurrent = config.maxConcurrent;
    this.timeoutMs = config.timeoutMs;
  }

  /** Number of currently in-flight transfers. */
  get active(): number {
    return this.activeTransfers;
  }

  /** Number of transfers waiting in the queue. */
  get pendingCount(): number {
    return this.pending.length;
  }

  /**
   * Initiate a KV cache transfer from the source (prefill) worker to the
   * target (decode) worker.  If the concurrency limit is reached the call
   * will wait until a slot opens.
   */
  async transfer(req: KVCacheTransferRequest): Promise<KVCacheTransferResult> {
    if (this.activeTransfers < this.maxConcurrent) {
      return this.executeTransfer(req);
    }

    // Queue the request until a slot is available.
    return new Promise<KVCacheTransferResult>((resolve, reject) => {
      this.pending.push({ req, resolve, reject });
    });
  }

  // -----------------------------------------------------------------------
  // Internal
  // -----------------------------------------------------------------------

  private async executeTransfer(
    req: KVCacheTransferRequest,
  ): Promise<KVCacheTransferResult> {
    this.activeTransfers++;
    const start = Date.now();

    try {
      const result = await this.doTransfer(req);
      return {
        requestId: req.requestId,
        success: true,
        transferDurationMs: Date.now() - start,
        targetCacheHandle: result.targetCacheHandle,
      };
    } catch (err: unknown) {
      return {
        requestId: req.requestId,
        success: false,
        transferDurationMs: Date.now() - start,
        error: err instanceof Error ? err.message : String(err),
      };
    } finally {
      this.activeTransfers--;
      this.drainPending();
    }
  }

  /**
   * Perform the actual transfer by calling the source worker's export
   * endpoint and the target worker's import endpoint.
   *
   * In a production system this would use RDMA / NCCL / TCP depending on
   * the cluster topology.  Here we model it as two HTTP calls to the
   * vLLM disaggregated-serving control plane.
   */
  private async doTransfer(
    req: KVCacheTransferRequest,
  ): Promise<{ targetCacheHandle: string }> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      // Step 1: Export cache from source worker.
      const exportRes = await fetch(
        `${req.sourceWorkerId}/v1/kv_cache/export`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ cache_handle: req.cacheHandle }),
          signal: controller.signal,
        },
      );

      if (!exportRes.ok) {
        throw new Error(
          `KV cache export failed (${exportRes.status}): ${await exportRes.text()}`,
        );
      }

      const exportBody = (await exportRes.json()) as {
        transfer_token: string;
      };

      // Step 2: Import cache into target worker.
      const importRes = await fetch(
        `${req.targetWorkerId}/v1/kv_cache/import`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            transfer_token: exportBody.transfer_token,
            source_worker: req.sourceWorkerId,
          }),
          signal: controller.signal,
        },
      );

      if (!importRes.ok) {
        throw new Error(
          `KV cache import failed (${importRes.status}): ${await importRes.text()}`,
        );
      }

      const importBody = (await importRes.json()) as {
        cache_handle: string;
      };

      return { targetCacheHandle: importBody.cache_handle };
    } finally {
      clearTimeout(timeout);
    }
  }

  /** Process waiting transfers when a concurrency slot opens. */
  private drainPending(): void {
    while (this.pending.length > 0 && this.activeTransfers < this.maxConcurrent) {
      const next = this.pending.shift()!;
      this.executeTransfer(next.req).then(next.resolve, next.reject);
    }
  }
}
