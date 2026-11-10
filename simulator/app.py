from __future__ import annotations

import asyncio
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


@dataclass(frozen=True, slots=True)
class SimConfig:
    """Configuration for one stand-in service process.

    Token counts are estimates based on roughly four UTF-8 characters per token.
    The service's windows are process-relative fixed windows, not provider quotas.
    """

    model_name: str = "relay-standin-v1"
    base_latency_ms: float = 900.0
    load_latency_ms: float = 250.0
    chunk_delay_ms: float = 10.0
    chunk_chars: int = 32
    max_concurrent: int = 3
    max_queue: int = 20
    queue_timeout_ms: float = 30_000.0
    rpm_limit: int = 60
    tpm_limit: int = 30_000
    daily_token_limit: int = 1_000_000
    failure_rate: float = 0.0
    random_seed: int = 17
    default_scenario: str = "normal"
    log_path: str = "results/standin_service.jsonl"

    # Provisional single-observation Groq timing profiles.
    profile_mode: str = "generic"
    understand_ttft_ms: float = 452.712
    understand_total_ms: float = 453.653
    generate_ttft_ms: float = 410.007
    generate_total_ms: float = 607.787
    regenerate_ttft_ms: float = 435.448
    regenerate_total_ms: float = 877.948
    verify_ttft_ms: float = 798.022
    verify_total_ms: float = 853.471

    @classmethod
    def from_env(cls) -> "SimConfig":
        def integer(name: str, default: int) -> int:
            return int(os.getenv(name, str(default)))

        def number(name: str, default: float) -> float:
            return float(os.getenv(name, str(default)))

        return cls(
            model_name=os.getenv("STANDIN_MODEL", "relay-standin-v1"),
            base_latency_ms=number("STANDIN_BASE_LATENCY_MS", 900.0),
            load_latency_ms=number("STANDIN_LOAD_LATENCY_MS", 250.0),
            chunk_delay_ms=number("STANDIN_CHUNK_DELAY_MS", 10.0),
            chunk_chars=integer("STANDIN_CHUNK_CHARS", 32),
            max_concurrent=integer("STANDIN_MAX_CONCURRENT", 3),
            max_queue=integer("STANDIN_MAX_QUEUE", 20),
            queue_timeout_ms=number("STANDIN_QUEUE_TIMEOUT_MS", 30_000.0),
            rpm_limit=integer("STANDIN_RPM_LIMIT", 60),
            tpm_limit=integer("STANDIN_TPM_LIMIT", 30_000),
            daily_token_limit=integer("STANDIN_DAILY_TOKEN_LIMIT", 1_000_000),
            failure_rate=number("STANDIN_FAILURE_RATE", 0.0),
            random_seed=integer("STANDIN_RANDOM_SEED", 17),
            default_scenario=os.getenv("STANDIN_SCENARIO", "normal"),
            log_path=os.getenv("STANDIN_LOG_PATH", "results/standin_service.jsonl"),
            profile_mode=os.getenv("STANDIN_PROFILE_MODE", "generic").strip().lower(),
            understand_ttft_ms=number("STANDIN_UNDERSTAND_TTFT_MS", 452.712),
            understand_total_ms=number("STANDIN_UNDERSTAND_TOTAL_MS", 453.653),
            generate_ttft_ms=number("STANDIN_GENERATE_TTFT_MS", 410.007),
            generate_total_ms=number("STANDIN_GENERATE_TOTAL_MS", 607.787),
            regenerate_ttft_ms=number("STANDIN_REGENERATE_TTFT_MS", 435.448),
            regenerate_total_ms=number("STANDIN_REGENERATE_TOTAL_MS", 877.948),
            verify_ttft_ms=number("STANDIN_VERIFY_TTFT_MS", 798.022),
            verify_total_ms=number("STANDIN_VERIFY_TOTAL_MS", 853.471),
        )

    def validate(self) -> None:
        if self.base_latency_ms < 0 or self.load_latency_ms < 0 or self.chunk_delay_ms < 0:
            raise ValueError("Latency settings must be non-negative")
        if self.chunk_chars < 1:
            raise ValueError("chunk_chars must be at least 1")
        if self.max_concurrent < 1 or self.max_queue < 0:
            raise ValueError("max_concurrent must be >= 1 and max_queue must be >= 0")
        if self.queue_timeout_ms < 0:
            raise ValueError("queue_timeout_ms must be non-negative")
        if min(self.rpm_limit, self.tpm_limit, self.daily_token_limit) < 1:
            raise ValueError("RPM, TPM and daily token limits must be positive")
        if not 0 <= self.failure_rate <= 1:
            raise ValueError("failure_rate must be between 0 and 1")
        if self.profile_mode not in {"generic", "groq_observed"}:
            raise ValueError("profile_mode must be generic or groq_observed")
        for stage in ("understand", "generate", "regenerate", "verify"):
            ttft = getattr(self, f"{stage}_ttft_ms")
            total = getattr(self, f"{stage}_total_ms")
            if ttft < 0 or total < ttft:
                raise ValueError(f"Invalid {stage} timing profile: total must be >= TTFT >= 0")


class CompletionRequest(BaseModel):
    model: str = "relay-standin-v1"
    messages: list[dict[str, str]] = Field(default_factory=list)
    stream: bool = True


def _estimate_tokens(text: str) -> int:
    """Cheap, deterministic estimate; not a provider tokenizer."""
    return max(1, math.ceil(len(text) / 4)) if text else 0


def _latency_targets(
    config: SimConfig, purpose: str, active_requests: int
) -> tuple[float, float | None, str]:
    """Return (TTFT target, total target, profile label) for one model call."""
    load_extra_ms = max(0, active_requests - 1) * config.load_latency_ms

    profiles = {
        "understand_question": (config.understand_ttft_ms, config.understand_total_ms),
        "generate_answer": (config.generate_ttft_ms, config.generate_total_ms),
        "regenerate_answer": (config.regenerate_ttft_ms, config.regenerate_total_ms),
        "verify_answer": (config.verify_ttft_ms, config.verify_total_ms),
    }

    if config.profile_mode == "groq_observed" and purpose in profiles:
        ttft_ms, total_ms = profiles[purpose]
        return ttft_ms + load_extra_ms, total_ms + load_extra_ms, "groq_observed"

    return config.base_latency_ms + load_extra_ms, None, "generic"


def _make_completion(messages: list[dict[str, str]]) -> str:
    system = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
    user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
    lowered = (system + " " + user).lower()

    if "retrieval_query" in lowered or "classify the question" in lowered:
        question = user.split("Question:", 1)[-1].strip()
        return json.dumps({
            "intent": "research_question",
            "complexity": "normal",
            "retrieval_query": question,
        })

    if "verify whether" in lowered or ("verify" in lowered and "return only pass" in lowered):
        return "PASS - deterministic stand-in verification fixture; this is not a semantic quality assessment."

    match = re.search(r"\[CHUNK_ID:([A-Za-z0-9_.-]+)\]|\[\[([A-Za-z0-9_.-]+)\]\]", user)
    chunk_id = next((part for part in match.groups() if part), None) if match else None
    if chunk_id:
        return (
            "The stand-in located a supplied passage for this question. "
            f"[[{chunk_id}]]"
        )
    return "The stand-in could not identify a passage to cite, so it will not guess."


class SimulatorEngine:
    def __init__(self, config: SimConfig) -> None:
        config.validate()
        self.config = config
        self.lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(config.max_concurrent)
        self.rng = random.Random(config.random_seed)
        self.minute_started = time.monotonic()
        self.day_started = self.minute_started
        self.requests_minute = 0
        self.tokens_minute = 0
        self.tokens_day = 0
        self.requests_total = 0
        self.accepted_total = 0
        self.rejected_total = 0
        self.active_requests = 0
        self.queued_requests = 0
        self.peak_queue = 0
        self.peak_active = 0
        self.last_requests: list[dict[str, Any]] = []

    def _refresh_windows(self, now: float) -> None:
        if now - self.minute_started >= 60:
            self.minute_started = now
            self.requests_minute = 0
            self.tokens_minute = 0
        if now - self.day_started >= 86_400:
            self.day_started = now
            self.tokens_day = 0

    async def _reject(self, status: int, code: str, message: str, retry_after: float = 1.0) -> None:
        async with self.lock:
            self.rejected_total += 1
        raise HTTPException(
            status_code=status,
            detail={"error": code, "message": message},
            headers={"retry-after": str(max(1, math.ceil(retry_after)))},
        )

    async def admit(self, input_tokens: int, output_tokens: int, scenario: str) -> None:
        now = time.monotonic()
        async with self.lock:
            self.requests_total += 1
            self._refresh_windows(now)

        if scenario in {"rate_limit", "force_rate_limit", "per_minute_exhausted"}:
            await self._reject(429, "rate_limit_exceeded", "Forced simulated per-minute limit", 60)
        if scenario in {"daily_exhausted", "daily_budget_exhausted"}:
            async with self.lock:
                self.tokens_day = self.config.daily_token_limit
            await self._reject(429, "daily_budget_exhausted", "Forced simulated daily token budget exhaustion", 86_400)
        if scenario in {"timeout", "server_timeout"}:
            await self._reject(408, "simulated_timeout", "Forced simulated service timeout", 1)
        if scenario in {"server_error", "failure"}:
            await self._reject(500, "simulated_server_error", "Forced simulated server failure", 1)

        async with self.lock:
            now = time.monotonic()
            self._refresh_windows(now)
            predicted_tokens = input_tokens + output_tokens
            minute_remaining = self.config.tpm_limit - self.tokens_minute
            day_remaining = self.config.daily_token_limit - self.tokens_day
            if self.requests_minute + 1 > self.config.rpm_limit:
                self.rejected_total += 1
                retry = 60 - (now - self.minute_started)
                raise HTTPException(
                    status_code=429,
                    detail={"error": "rate_limit_exceeded", "message": "Simulated RPM limit reached"},
                    headers={"retry-after": str(max(1, math.ceil(retry))), "x-ratelimit-limit-requests": str(self.config.rpm_limit), "x-ratelimit-remaining-requests": "0"},
                )
            if predicted_tokens > minute_remaining:
                self.rejected_total += 1
                retry = 60 - (now - self.minute_started)
                raise HTTPException(
                    status_code=429,
                    detail={"error": "rate_limit_exceeded", "message": "Simulated TPM limit reached"},
                    headers={"retry-after": str(max(1, math.ceil(retry))), "x-ratelimit-limit-tokens": str(self.config.tpm_limit), "x-ratelimit-remaining-tokens": str(max(0, minute_remaining))},
                )
            if predicted_tokens > day_remaining:
                self.rejected_total += 1
                retry = 86_400 - (now - self.day_started)
                raise HTTPException(
                    status_code=429,
                    detail={"error": "daily_budget_exhausted", "message": "Simulated daily token budget exhausted"},
                    headers={"retry-after": str(max(1, math.ceil(retry))), "x-ratelimit-limit-daily-tokens": str(self.config.daily_token_limit), "x-ratelimit-remaining-daily-tokens": "0"},
                )
            self.requests_minute += 1
            self.tokens_minute += predicted_tokens
            self.tokens_day += predicted_tokens
            self.accepted_total += 1

    async def acquire_slot(self) -> float:
        queued = False
        queued_at = time.perf_counter()
        async with self.lock:
            if self.active_requests >= self.config.max_concurrent:
                if self.queued_requests >= self.config.max_queue:
                    self.rejected_total += 1
                    raise HTTPException(
                        status_code=429,
                        detail={"error": "queue_full", "message": "Stand-in request queue is full"},
                        headers={"retry-after": "1"},
                    )
                self.queued_requests += 1
                queued = True
                self.peak_queue = max(self.peak_queue, self.queued_requests)
        try:
            await asyncio.wait_for(
                self.semaphore.acquire(),
                timeout=self.config.queue_timeout_ms / 1000,
            )
        except TimeoutError as exc:
            async with self.lock:
                if queued:
                    self.queued_requests -= 1
                self.rejected_total += 1
            raise HTTPException(
                status_code=408,
                detail={"error": "queue_timeout", "message": "Timed out waiting for a simulator slot"},
            ) from exc
        async with self.lock:
            if queued:
                self.queued_requests -= 1
            self.active_requests += 1
            self.peak_active = max(self.peak_active, self.active_requests)
        return (time.perf_counter() - queued_at) * 1000

    async def release_slot(self) -> None:
        async with self.lock:
            self.active_requests = max(0, self.active_requests - 1)
        self.semaphore.release()

    async def snapshot(self) -> dict[str, Any]:
        async with self.lock:
            self._refresh_windows(time.monotonic())
            return {
                "model": self.config.model_name,
                "requests_total": self.requests_total,
                "accepted_total": self.accepted_total,
                "rejected_total": self.rejected_total,
                "requests_in_current_minute": self.requests_minute,
                "tokens_in_current_minute": self.tokens_minute,
                "tokens_in_current_day": self.tokens_day,
                "active_requests": self.active_requests,
                "queued_requests": self.queued_requests,
                "peak_active": self.peak_active,
                "peak_queue": self.peak_queue,
                "limits": {
                    "rpm": self.config.rpm_limit,
                    "tpm": self.config.tpm_limit,
                    "daily_tokens": self.config.daily_token_limit,
                    "max_concurrent": self.config.max_concurrent,
                },
            }

    async def reset(self) -> None:
        async with self.lock:
            now = time.monotonic()
            self.minute_started = now
            self.day_started = now
            self.requests_minute = self.tokens_minute = self.tokens_day = 0
            self.requests_total = self.accepted_total = self.rejected_total = 0
            self.peak_queue = self.peak_active = 0
            self.last_requests.clear()

    async def record(self, event: dict[str, Any]) -> None:
        async with self.lock:
            self.last_requests.append(event)
            if len(self.last_requests) > 100:
                self.last_requests.pop(0)
        if self.config.log_path:
            path = Path(self.config.log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def create_app(config: SimConfig | None = None) -> FastAPI:
    cfg = config or SimConfig.from_env()
    engine = SimulatorEngine(cfg)
    app = FastAPI(title="The Relay Stand-in Model Service", version="0.1.0")
    app.state.engine = engine

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "model": cfg.model_name}

    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        result = await engine.snapshot()
        result["recent_requests"] = list(engine.last_requests)
        return result

    @app.post("/admin/reset")
    async def admin_reset() -> dict[str, str]:
        await engine.reset()
        return {"status": "reset"}

    @app.post("/v1/chat/completions")
    async def completions(
        body: CompletionRequest,
        x_sim_scenario: str | None = Header(default=None),
        x_sim_purpose: str | None = Header(default=None),
    ) -> StreamingResponse:
        request_id = str(uuid4())
        scenario = (x_sim_scenario or cfg.default_scenario or "normal").strip().lower()
        purpose = (x_sim_purpose or "completion").strip().lower()
        messages_text = "\n".join(m.get("content", "") for m in body.messages)
        output_text = _make_completion(body.messages)
        tokens_in = _estimate_tokens(messages_text)
        tokens_out = _estimate_tokens(output_text)
        await engine.admit(tokens_in, tokens_out, scenario)
        queue_wait_ms = await engine.acquire_slot()

        if cfg.failure_rate and engine.rng.random() < cfg.failure_rate:
            await engine.release_slot()
            raise HTTPException(
                status_code=500,
                detail={"error": "injected_failure", "message": "Random stand-in failure"},
            )

        active_at_start = engine.active_requests
        service_delay_ms, target_total_ms, profile_label = _latency_targets(
            cfg, purpose, active_at_start
        )
        chunks = [
            output_text[i:i + cfg.chunk_chars]
            for i in range(0, len(output_text), cfg.chunk_chars)
        ] or [""]
        gap_count = max(0, len(chunks) - 1)
        profile_gap_delay_ms = (
            max(0.0, target_total_ms - service_delay_ms) / gap_count
            if target_total_ms is not None and gap_count
            else None
        )
        started = time.perf_counter()

        async def stream_events() -> AsyncIterator[str]:
            first_token_ms: float | None = None
            try:
                # Base delay models service TTFT; additional active requests increase latency.
                if service_delay_ms:
                    await asyncio.sleep(service_delay_ms / 1000)
                for index, part in enumerate(chunks):
                    if index:
                        gap_delay_ms = (
                            profile_gap_delay_ms
                            if profile_gap_delay_ms is not None
                            else cfg.chunk_delay_ms
                        )
                        if gap_delay_ms:
                            await asyncio.sleep(gap_delay_ms / 1000)
                    if first_token_ms is None:
                        first_token_ms = (time.perf_counter() - started) * 1000
                    payload = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": cfg.model_name,
                        "choices": [{"index": 0, "delta": {"content": part}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                if target_total_ms is not None:
                    elapsed_before_usage = (time.perf_counter() - started) * 1000
                    remaining_ms = target_total_ms - elapsed_before_usage
                    if remaining_ms > 0:
                        await asyncio.sleep(remaining_ms / 1000)

                final_payload = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": cfg.model_name,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": tokens_in,
                        "completion_tokens": tokens_out,
                        "total_tokens": tokens_in + tokens_out,
                    },
                }
                yield f"data: {json.dumps(final_payload)}\n\n"
                yield "data: [DONE]\n\n"
                total_ms = (time.perf_counter() - started) * 1000
                await engine.record({
                    "request_id": request_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "scenario": scenario,
                    "purpose": purpose,
                    "latency_profile": profile_label,
                    "target_total_ms": target_total_ms,
                    "status": "ok",
                    "tokens_in_estimated": tokens_in,
                    "tokens_out_estimated": tokens_out,
                    "queue_wait_ms": round(queue_wait_ms, 3),
                    "simulated_service_delay_ms": round(service_delay_ms, 3),
                    "first_token_ms": round(first_token_ms or 0.0, 3),
                    "total_ms": round(total_ms, 3),
                })
            finally:
                await engine.release_slot()

        headers = {
            "x-request-id": request_id,
            "x-ratelimit-limit-requests": str(cfg.rpm_limit),
            "x-ratelimit-limit-tokens": str(cfg.tpm_limit),
            "x-sim-queue-wait-ms": f"{queue_wait_ms:.3f}",
            "x-sim-purpose": purpose,
            "x-sim-latency-profile": profile_label,
        }
        return StreamingResponse(stream_events(), media_type="text/event-stream", headers=headers)

    return app


app = create_app()
