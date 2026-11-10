from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

CSV_COLUMNS = [
    "request_id", "scenario", "source", "start_time", "first_token_ms", "total_ms",
    "status", "retries", "tokens_in", "tokens_out", "retrieval_setting", "answer_grade",
]

DEFAULTS = {
    "steady": (12, 2.0),
    "burst": (20, None),
    "overload": (40, 5.0),
    "half_rpm": (32, 2.0),
    "daily_exhausted": (5, None),
    "rate_limit": (5, None),
    "timeout": (5, None),
    "server_error": (5, None),
}


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile; returns None when there are no observations."""
    if not values:
        return None
    if not 0 < p <= 100:
        raise ValueError("p must be in (0, 100]")
    ordered = sorted(values)
    index = max(0, math.ceil((p / 100) * len(ordered)) - 1)
    return ordered[index]


def _status_for_http(status_code: int, error_code: str = "") -> str:
    if 200 <= status_code < 300:
        return "ok"
    if error_code == "queue_full":
        return "error"
    if status_code == 429:
        return "rate_limited"
    if status_code == 408 or status_code == 504:
        return "timeout"
    return "error"


def _messages(index: int) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are a deterministic model endpoint used only for service load testing.",
        },
        {
            "role": "user",
            "content": (
                f"Load-test request {index:05d}. Briefly summarize attention mechanisms. "
                "[CHUNK_ID:loadtest-p001-c01]"
            ),
        },
    ]


async def _one_request(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    scenario: str,
    index: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    request_id = str(uuid4())
    first_token_ms: float | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    status_code: int | None = None
    response_request_id = ""
    queue_wait_ms: float | None = None
    error = ""
    error_code = ""

    # The daily-budget case allows one normal request, then forces a simulated
    # exhausted-budget response. Other forced cases apply to every request.
    header_scenario = scenario
    if scenario == "daily_exhausted" and index == 0:
        header_scenario = "normal"
    elif scenario in {"steady", "burst", "overload", "half_rpm"}:
        header_scenario = "normal"
    elif scenario == "rate_limit":
        header_scenario = "rate_limit"
    elif scenario == "timeout":
        header_scenario = "timeout"
    elif scenario == "server_error":
        header_scenario = "server_error"

    try:
        async with client.stream(
            "POST",
            f"{base_url.rstrip('/')}/v1/chat/completions",
            json={"model": "relay-standin-v1", "messages": _messages(index), "stream": True},
            headers={"X-Sim-Scenario": header_scenario},
            timeout=timeout_seconds,
        ) as response:
            status_code = response.status_code
            response_request_id = response.headers.get("x-request-id", "")
            queue_header = response.headers.get("x-sim-queue-wait-ms")
            if queue_header:
                try:
                    queue_wait_ms = float(queue_header)
                except ValueError:
                    pass

            if status_code >= 400:
                body = await response.aread()
                try:
                    payload = json.loads(body.decode("utf-8"))
                    detail = payload.get("detail", payload)
                    if isinstance(detail, dict):
                        error = detail.get("message", str(detail))
                        error_code = str(detail.get("error", ""))
                    else:
                        error = str(detail)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    error = body.decode("utf-8", errors="replace")
            else:
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    usage = event.get("usage") or {}
                    if usage:
                        tokens_in = usage.get("prompt_tokens", tokens_in)
                        tokens_out = usage.get("completion_tokens", tokens_out)
                    for choice in event.get("choices") or []:
                        content = (choice.get("delta") or {}).get("content")
                        if content and first_token_ms is None:
                            first_token_ms = (time.perf_counter() - started) * 1000

    except httpx.TimeoutException as exc:
        status_code = 408
        error = f"client_timeout: {type(exc).__name__}"
    except httpx.RequestError as exc:
        status_code = 0
        error = f"connection_error: {type(exc).__name__}: {exc}"

    total_ms = (time.perf_counter() - started) * 1000
    status = _status_for_http(status_code or 0, error_code)
    return {
        "request_id": response_request_id or request_id,
        "scenario": scenario,
        "source": "stand-in",
        "start_time": started_at,
        "first_token_ms": first_token_ms,
        "total_ms": total_ms,
        "status": status,
        "retries": 0,
        "tokens_in": tokens_in or 0,
        "tokens_out": tokens_out or 0,
        "retrieval_setting": "direct_standin_loadtest",
        "answer_grade": "",
        "status_code": status_code,
        "queue_wait_ms": queue_wait_ms,
        "error_code": error_code,
        "error": error,
    }


def append_assignment_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append direct-service observations using the assignment's exact CSV schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
        if header != CSV_COLUMNS:
            raise ValueError(
                f"Unexpected results CSV header in {path}; expected {CSV_COLUMNS}, got {header}"
            )
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_COLUMNS})


def summarize(rows: list[dict[str, Any]], elapsed_seconds: float, offered_rps: float | None) -> dict[str, Any]:
    count = len(rows)
    successes = sum(row["status"] == "ok" for row in rows)
    rate_limited = sum(row["status"] == "rate_limited" for row in rows)
    timeouts = sum(row["status"] == "timeout" for row in rows)
    errors = sum(row["status"] == "error" for row in rows)
    total_values = [float(row["total_ms"]) for row in rows]
    ttft_values = [float(row["first_token_ms"]) for row in rows if row["first_token_ms"] is not None]
    return {
        "requests": count,
        "duration_seconds": round(elapsed_seconds, 3),
        "offered_rate_rps": offered_rps,
        "completed_rate_rps": round(count / elapsed_seconds, 3) if elapsed_seconds > 0 else 0.0,
        "success_percent": round(100 * successes / count, 2) if count else 0.0,
        "rate_limited_percent": round(100 * rate_limited / count, 2) if count else 0.0,
        "timeout_percent": round(100 * timeouts / count, 2) if count else 0.0,
        "error_percent": round(100 * errors / count, 2) if count else 0.0,
        "ttft_ms": {f"p{p}": percentile(ttft_values, p) for p in (50, 95, 99)},
        "total_ms": {f"p{p}": percentile(total_values, p) for p in (50, 95, 99)},
        "tokens_in": sum(int(row["tokens_in"] or 0) for row in rows),
        "tokens_out": sum(int(row["tokens_out"] or 0) for row in rows),
        "retries_per_request": 0.0,
        "status_counts": {
            "ok": successes,
            "rate_limited": rate_limited,
            "timeout": timeouts,
            "error": errors,
        },
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    base_url = args.base_url.rstrip("/")
    async with httpx.AsyncClient() as client:
        health = await client.get(f"{base_url}/health", timeout=5.0)
        health.raise_for_status()
        if args.scenario == "half_rpm":
            metrics_response = await client.get(f"{base_url}/metrics", timeout=5.0)
            metrics_response.raise_for_status()
            configured_rpm = int(metrics_response.json().get("limits", {}).get("rpm", -1))
            if configured_rpm != 30:
                raise SystemExit(
                    "half_rpm requires the simulator to be restarted with "
                    "STANDIN_RPM_LIMIT=30; current configured RPM is "
                    f"{configured_rpm}."
                )
            if args.requests < 31:
                raise SystemExit("half_rpm requires at least 31 requests to cross the 30 RPM limit.")
        if not args.keep_counters:
            reset = await client.post(f"{base_url}/admin/reset", timeout=5.0)
            reset.raise_for_status()

    connector = httpx.Limits(
        max_connections=max(args.requests, 1),
        max_keepalive_connections=max(args.requests, 1),
    )
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    async with httpx.AsyncClient(limits=connector) as client:
        tasks: list[asyncio.Task[dict[str, Any]]] = []
        interval = (1.0 / args.rps) if args.rps and args.rps > 0 else None

        start_index = 0
        if args.scenario == "daily_exhausted" and args.requests > 0:
            # Complete one normal request before forcing daily exhaustion.
            first_row = await _one_request(
                client,
                base_url=base_url,
                scenario=args.scenario,
                index=0,
                timeout_seconds=args.timeout_seconds,
            )
            rows.append(first_row)
            start_index = 1

        for index in range(start_index, args.requests):
            tasks.append(asyncio.create_task(
                _one_request(
                    client,
                    base_url=base_url,
                    scenario=args.scenario,
                    index=index,
                    timeout_seconds=args.timeout_seconds,
                )
            ))
            if interval is not None and index + 1 < args.requests:
                await asyncio.sleep(interval)

        if tasks:
            rows.extend(await asyncio.gather(*tasks))
    elapsed = time.perf_counter() - started

    offered_rps = args.rps
    if args.scenario in {"burst", "daily_exhausted", "rate_limit", "timeout", "server_error"}:
        offered_rps = None
    summary = summarize(rows, elapsed, offered_rps)
    append_assignment_csv(Path(args.output), rows)

    async with httpx.AsyncClient() as client:
        metrics_response = await client.get(f"{base_url}/metrics", timeout=5.0)
        metrics_response.raise_for_status()
        service_metrics = metrics_response.json()

    result = {
        "scenario": args.scenario,
        "service": health.json(),
        "summary": summary,
        "service_metrics_after_run": service_metrics,
        "requests_detail": rows,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_path = Path(args.output).parent / f"loadtest_{args.scenario}_{stamp}.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "scenario": args.scenario,
        "summary": summary,
        "service_counters": {
            key: service_metrics.get(key)
            for key in ("requests_total", "accepted_total", "rejected_total", "peak_active", "peak_queue")
        },
        "raw_csv": str(Path(args.output)),
        "run_json": str(summary_path),
    }, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load-test The Relay local stand-in service.")
    parser.add_argument(
        "--scenario",
        choices=tuple(DEFAULTS),
        required=True,
        help="steady, burst, overload, half_rpm, daily_exhausted, rate_limit, timeout, server_error",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8090")
    parser.add_argument("--requests", type=int)
    parser.add_argument("--rps", type=float)
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    parser.add_argument("--output", default="results/results.csv")
    parser.add_argument("--keep-counters", action="store_true", help="Do not reset simulator counters before this run")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    default_count, default_rps = DEFAULTS[args.scenario]
    if args.requests is None:
        args.requests = default_count
    if args.rps is None:
        args.rps = default_rps
    if args.requests < 1:
        parser.error("--requests must be >= 1")
    if args.rps is not None and args.rps <= 0:
        parser.error("--rps must be > 0")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
