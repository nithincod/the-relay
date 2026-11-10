# The Relay stand-in service

This is a local, deterministic HTTP service for exercising The Relay's model-client path. It is **not** a substitute for Groq and its quota/latency values are simulated.

## Run

From the repository root, with the project virtual environment active:

```bash
uvicorn simulator.app:app --host 127.0.0.1 --port 8090
```

Health and counters:

```bash
curl http://127.0.0.1:8090/health
curl http://127.0.0.1:8090/metrics
```

## Configuration

All settings are environment variables read when the service starts:

| Variable | Default | Meaning |
|---|---:|---|
| `STANDIN_MODEL` | `relay-standin-v1` | Displayed model name |
| `STANDIN_BASE_LATENCY_MS` | `900` | Delay before first streamed chunk |
| `STANDIN_LOAD_LATENCY_MS` | `250` | Additional delay per already-active request |
| `STANDIN_CHUNK_DELAY_MS` | `10` | Delay between subsequent chunks |
| `STANDIN_CHUNK_CHARS` | `32` | Characters per stream chunk |
| `STANDIN_MAX_CONCURRENT` | `3` | Simultaneous service capacity |
| `STANDIN_MAX_QUEUE` | `20` | Maximum requests waiting for a slot |
| `STANDIN_QUEUE_TIMEOUT_MS` | `30000` | Maximum slot wait |
| `STANDIN_RPM_LIMIT` | `60` | Simulated requests per process-relative minute |
| `STANDIN_TPM_LIMIT` | `30000` | Simulated tokens per process-relative minute |
| `STANDIN_DAILY_TOKEN_LIMIT` | `1000000` | Simulated tokens per process-relative day |
| `STANDIN_FAILURE_RATE` | `0` | Random server-failure probability, 0 to 1 |
| `STANDIN_RANDOM_SEED` | `17` | Seed for repeatable failure injection |
| `STANDIN_SCENARIO` | `normal` | Default forced scenario |
| `STANDIN_LOG_PATH` | `results/standin_service.jsonl` | Per-model-call JSONL log; empty disables it |

Token counts are rough estimates using four characters per token; they are not tokenizer-equivalent to Groq. RPM/TPM windows start when the service process starts and reset after 60 seconds. The daily window is a process-relative 24-hour window. Restarting the service resets all in-memory quotas and counters. `/admin/reset` resets in-memory counters for repeatable local experiments.

## Forced failure scenarios

Pass `X-Sim-Scenario` on a request to force one response:

- `rate_limit`: HTTP 429, simulated per-minute limit.
- `daily_exhausted`: HTTP 429, simulated daily budget exhaustion.
- `timeout`: HTTP 408.
- `server_error`: HTTP 500.

The `/admin/reset` and `/metrics` endpoints are intentionally unauthenticated and should only be bound to localhost during this assignment.

## Calibration caveat

The initial 900 ms default is a provisional starting point based on the observed successful-run non-verifier Groq calls (approximately 454 ms, 984 ms and 1,101 ms). One verifier call took approximately 22.9 seconds and is treated as an observed outlier pending more samples. Do not describe the default as a statistically calibrated distribution. Measure more real calls, then update the profile and document the sample size and limitations.

## Purpose-aware timing profile

Set `STANDIN_PROFILE_MODE=groq_observed` to enable workflow-stage timing.
The HTTP client sends `X-Sim-Purpose` with one of:

- `understand_question`
- `generate_answer`
- `regenerate_answer`
- `verify_answer`

Each stage has separate TTFT and total-response targets. The simulator
distributes the interval between first token and completion across its
streamed chunks. Load-dependent delay is added to both targets.

The default `generic` profile is unchanged and is used for direct load tests
that do not provide a recognized workflow purpose.

### Provisional calibration source

The initial `groq_observed` profile uses one successful real Groq workflow
observation with `openai/gpt-oss-20b`. It is a four-point, single-run
calibration, not a statistically representative latency distribution.
The unusually slow verifier observation is retained as a measured case,
not claimed as typical behavior.

Token counts remain character-based estimates and are not equivalent to
Groq tokenizer usage. The simulated service does not reproduce Groq's
internal scheduling, network path, hardware, or quota enforcement.

## Updated calibration: November 10, 2026

The purpose-specific defaults use provisional medians from observed Groq
workflow calls recorded in `results/groq_provider_observations.jsonl`.

| Purpose | Samples | Median TTFT (ms) | Median total (ms) |
|---|---:|---:|---:|
| Understand | 3 | 452.712 | 453.653 |
| Generate | 3 | 410.007 | 607.787 |
| Regenerate | 2 | 435.448 | 877.948 |
| Verify | 3 | 798.022 | 853.471 |

Sample sizes are small. One verification observation took approximately
22.9 seconds. It is retained as a slow-tail observation but is not used
as the default verification latency.

These are provisional calibration points, not production latency
distributions or guarantees. More repeated measurements are required.
