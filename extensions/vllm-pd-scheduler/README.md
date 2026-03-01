# vLLM PD (Prefill-Decode) Dynamic Task Scheduler

OpenClaw plugin that implements dynamic scheduling for disaggregated vLLM inference serving. It separates the **Prefill** (P) and **Decode** (D) phases onto dedicated worker pools and coordinates KV cache transfers between them.

## Architecture

```
                  ┌─────────────┐
   Request ──▶   │  Scheduler   │
                  │  (priority   │
                  │   queue)     │
                  └──────┬──────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
      ┌──────────────┐     ┌──────────────┐
      │  Prefill Pool │     │  Decode Pool  │
      │  ┌────┐┌────┐│     │  ┌────┐┌────┐│
      │  │ P1 ││ P2 ││     │  │ D1 ││ D2 ││
      │  └────┘└────┘│     │  └────┘└────┘│
      └──────┬───────┘     └──────▲───────┘
             │                     │
             │  KV Cache Transfer  │
             └─────────────────────┘
```

### Request Lifecycle

1. **Queued** – request enters the priority-aware scheduler queue
2. **Prefilling** – a prefill worker processes all input tokens and builds the KV cache
3. **Transferring** – the KV cache is moved from the prefill worker to a decode worker
4. **Decoding** – the decode worker generates output tokens auto-regressively
5. **Completed / Failed** – terminal states

## Scheduling Strategies

| Strategy | Description |
|----------|-------------|
| `round-robin` | Cycles through available workers evenly |
| `least-loaded` | Picks the worker with the fewest active requests |
| `latency-aware` | Selects based on GPU utilisation as a latency proxy |

## Configuration

Add to your `openclaw.json` under `plugins.entries`:

```json
{
  "vllm-pd-scheduler": {
    "enabled": true
  }
}
```

Full configuration (with defaults):

```json
{
  "pdScheduler": {
    "enabled": true,
    "strategy": "least-loaded",
    "healthCheckIntervalMs": 10000,
    "workerTimeoutMs": 30000,
    "maxQueueSize": 1000,
    "defaultRequestTimeoutMs": 60000,
    "workers": [
      {
        "id": "prefill-0",
        "endpoint": "http://gpu-host-1:8000",
        "role": "prefill",
        "modelId": "Qwen/Qwen2-72B",
        "maxConcurrency": 32
      },
      {
        "id": "decode-0",
        "endpoint": "http://gpu-host-2:8000",
        "role": "decode",
        "modelId": "Qwen/Qwen2-72B",
        "maxConcurrency": 64
      }
    ],
    "kvTransfer": {
      "maxConcurrent": 4,
      "timeoutMs": 15000
    }
  }
}
```

## REST API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/completions` | Submit an inference request |
| `GET` | `/v1/metrics` | Scheduler metrics snapshot |
| `GET` | `/v1/events` | Recent scheduler events |
| `GET` | `/v1/workers` | List registered workers |
| `POST` | `/v1/workers` | Register a new worker |
| `DELETE` | `/v1/workers/:id` | Remove a worker |
| `GET` | `/health` | Liveness probe |

### Example: Submit a request

```bash
curl -X POST http://localhost:18789/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen2-72B",
    "prompt": "Hello, world!",
    "max_tokens": 256,
    "temperature": 0.7,
    "priority": "normal"
  }'
```

## Agent Tools

When registered as an OpenClaw plugin, the following tools are available to agents:

- **`pd_scheduler_submit`** – Submit an inference request through the PD pipeline
- **`pd_scheduler_metrics`** – Get current scheduler metrics
- **`pd_scheduler_workers`** – List all registered workers
- **`pd_scheduler_add_worker`** – Register a new worker at runtime

## Development

```bash
cd extensions/vllm-pd-scheduler
npm install
npm test          # run unit tests
npm run build     # type-check with tsc
```

## License

MIT
