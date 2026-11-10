from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from groq import AsyncGroq
from groq import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from relay.model_clients.base import ModelCallError, ModelResponse


logger = logging.getLogger(__name__)

EXTRA_PROVIDER_HEADERS = {
    "retry-after",
    "x-request-id",
    "request-id",
}


def filter_provider_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Keep rate-limit and request-identification headers, not arbitrary headers."""
    result: dict[str, str] = {}
    for key, value in headers.items():
        normalized = str(key).lower()
        if normalized.startswith("x-ratelimit-") or normalized in EXTRA_PROVIDER_HEADERS:
            result[normalized] = str(value)
    return result


def parse_retry_after(headers: Mapping[str, str]) -> float | None:
    value = next(
        (v for k, v in headers.items() if str(k).lower() == "retry-after"),
        None,
    )
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


class GroqModelClient:
    provider_name = "groq"

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_seconds: float = 30.0,
        observation_path: Path | str = "results/groq_provider_observations.jsonl",
    ) -> None:
        if not api_key:
            raise ValueError("GROQ_API_KEY is required when MODEL_BACKEND=groq")

        self.model_name = model
        self.observation_path = Path(observation_path)

        # Disable implicit SDK retries so benchmark retry counts remain observable.
        self._client = AsyncGroq(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=0,
        )

    def _record_observation(self, event: dict[str, Any]) -> None:
        """Append one metadata-only observation; never persist prompts or API keys."""
        self.observation_path.parent.mkdir(parents=True, exist_ok=True)
        with self.observation_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _error_observation(
        self,
        exc: Exception,
        status_code: int | None,
        purpose: str,
        started: float,
    ) -> tuple[dict[str, str], float | None]:
        response = getattr(exc, "response", None)
        raw_headers = dict(response.headers) if response is not None else {}
        headers = filter_provider_headers(raw_headers)
        retry_after = parse_retry_after(headers)

        self._record_observation({
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "provider": self.provider_name,
            "model": self.model_name,
            "purpose": purpose,
            "outcome": "http_error" if status_code else "connection_error",
            "status_code": status_code,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "sdk_retries": 0,
            "retry_after_seconds": retry_after,
            "response_headers": headers,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        })

        return headers, retry_after

    def _raise_normalized(
        self,
        exc: Exception,
        status_code: int | None,
        purpose: str,
        started: float,
    ) -> None:
        headers, retry_after = self._error_observation(
            exc, status_code, purpose, started
        )
        raise ModelCallError(
            str(exc),
            status_code=status_code,
            retry_after=retry_after,
            response_headers=headers,
        ) from exc

    async def complete(
        self,
        messages: list[dict[str, str]],
        purpose: str = "completion",
    ) -> ModelResponse:
        started = time.perf_counter()
        first_token_ms: float | None = None
        parts: list[str] = []
        tokens_in: int | None = None
        tokens_out: int | None = None
        response_headers: dict[str, str] = {}
        sdk_retries = 0

        try:
            # The SDK's streaming response exposes HTTP headers before consuming SSE.
            async with self._client.chat.completions.with_streaming_response.create(
                model=self.model_name,
                messages=messages,
                temperature=0,
                reasoning_effort="low",
                max_completion_tokens=1024,
                include_reasoning=False,
                stream=True,
            ) as response:
                response_headers = filter_provider_headers(dict(response.headers))
                sdk_retries = int(getattr(response, "retries_taken", 0))

                async for line in response.iter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue

                    data = line[5:].strip()
                    if data == "[DONE]":
                        break

                    try:
                        event: dict[str, Any] = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    usage = event.get("usage")
                    if usage is None:
                        x_groq = event.get("x_groq") or {}
                        usage = x_groq.get("usage")

                    if usage:
                        tokens_in = usage.get("prompt_tokens", tokens_in)
                        tokens_out = usage.get("completion_tokens", tokens_out)

                    for choice in event.get("choices") or []:
                        delta = choice.get("delta") or {}
                        content = delta.get("content")
                        if content:
                            if first_token_ms is None:
                                first_token_ms = (
                                    time.perf_counter() - started
                                ) * 1000
                            parts.append(content)

            elapsed_ms = (time.perf_counter() - started) * 1000
            answer = "".join(parts)

            self._record_observation({
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "provider": self.provider_name,
                "model": self.model_name,
                "purpose": purpose,
                "outcome": "success",
                "status_code": 200,
                "elapsed_ms": round(elapsed_ms, 3),
                "first_token_ms": (
                    round(first_token_ms, 3)
                    if first_token_ms is not None else None
                ),
                "prompt_tokens": tokens_in,
                "completion_tokens": tokens_out,
                "sdk_retries": sdk_retries,
                "retry_after_seconds": None,
                "response_headers": response_headers,
            })

            return ModelResponse(
                text=answer,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                first_token_ms=first_token_ms,
                elapsed_ms=elapsed_ms,
                response_headers=response_headers,
                provider=self.provider_name,
                model=self.model_name,
                sdk_retries=sdk_retries,
            )

        except RateLimitError as exc:
            self._raise_normalized(exc, 429, purpose, started)
        except APITimeoutError as exc:
            self._raise_normalized(exc, 408, purpose, started)
        except APIStatusError as exc:
            self._raise_normalized(
                exc, getattr(exc, "status_code", None), purpose, started
            )
        except APIConnectionError as exc:
            self._error_observation(exc, None, purpose, started)
            raise ModelCallError(f"Groq connection error: {exc}") from exc

    async def aclose(self) -> None:
        await self._client.close()
