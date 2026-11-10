# Stand-in load testing runbook

The load runner calls the local stand-in HTTP service directly. It does not route these requests through LangGraph. This isolates service queueing, rate limits, streaming TTFT and service latency from RAG/retrieval costs.

Each request is appended to `results/results.csv` using the assignment's required 12-column schema. Direct requests use `retrieval_setting=direct_standin_loadtest`, `answer_grade` remains blank, and retries are currently zero. A timestamped per-run JSON file beside the CSV contains status counts, p50/p95/p99 latency, offered rate and the service's `/metrics` snapshot.

Run the service in one terminal:

```bash
uvicorn simulator.app:app --host 127.0.0.1 --port 8090
```

Run the load tool from another terminal with `.venv` active:

```bash
python -m experiments.load_test --scenario steady
python -m experiments.load_test --scenario burst
python -m experiments.load_test --scenario overload
python -m experiments.load_test --scenario daily_exhausted
python -m experiments.load_test --scenario rate_limit
python -m experiments.load_test --scenario timeout
python -m experiments.load_test --scenario server_error
```

## Half-RPM experiment

The half-RPM profile requires a fresh simulator process configured at 30 RPM. Stop the normal simulator with Ctrl+C, then restart:

```bash
STANDIN_RPM_LIMIT=30 uvicorn simulator.app:app --host 127.0.0.1 --port 8090
```

In the other terminal:

```bash
python -m experiments.load_test --scenario half_rpm
```

The runner checks that `/metrics` reports exactly 30 RPM and submits 32 requests at 2 requests/second. It does not wait for retry-after; these are quota-observation requests, not a retry-policy test.

## Defaults

- steady: 12 requests at 2 RPS
- burst: 20 requests launched together
- overload: 40 requests at 5 RPS
- half_rpm: 32 requests at 2 RPS (requires service RPM=30)
- daily_exhausted: one normal request followed by forced daily-budget exhaustion
- rate_limit, timeout, server_error: five forced responses

The stand-in token estimator is approximate (roughly four characters per token). Its simulated quotas are not Groq quotas. The load runner does not retry by default; this makes each offered attempt visible in the CSV and avoids waiting a full simulated minute or day. The harness captures direct model-service metrics and must not be described as an end-to-end LangGraph latency measurement.

For each profile, preserve the timestamped JSON output and the corresponding CSV rows. For percentiles in the report, group the raw CSV rows by `scenario` and `source`. The run JSON contains the same run-specific summary for convenience.
