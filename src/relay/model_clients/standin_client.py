from __future__ import annotations

import json
import time
from typing import Any

import httpx

from relay.model_clients.base import ModelCallError, ModelResponse


class StandInModelClient:
    """HTTP adapter for the local simulated model service."""

    provider_name = "stand-in"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8090",
        model: str = "relay-standin-v1",
        timeout_seconds: float = 35.0,
        scenario: str = "normal",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model_name = model
        self.scenario = scenario
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds),
            transport=transport,
        )

    async def complete(
        self, messages: list[dict[str, str]], purpose: str = "completion"
    ) -> ModelResponse:
        started = time.perf_counter()
        first_token_ms: float | None = None
        parts: list[str] = []
        tokens_in: int | None = None
        tokens_out: int | None = None
        response_headers: dict[str, str] = {}
        response_model = self.model_name

        headers = {
            "X-Sim-Scenario": self.scenario,
            "X-Sim-Purpose": purpose,
        }

        try:
            async with self._client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": True,
                },
                headers=headers,
            ) as response:
                response_headers = dict(response.headers)

                if response.status_code >= 400:
                    body = await response.aread()
                    try:
                        payload = json.loads(body.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        payload = {"detail": body.decode("utf-8", errors="replace")}

                    detail = payload.get("detail", payload)
                    if isinstance(detail, dict):
                        message = (
                            detail.get("message")
                            or detail.get("error")
                            or str(detail)
                        )
                    else:
                        message = str(detail)

                    retry_after = None
                    try:
                        value = response.headers.get("retry-after")
                        retry_after = float(value) if value else None
                    except (TypeError, ValueError):
                        pass

                    raise ModelCallError(
                        f"Stand-in HTTP {response.status_code}: {message}",
                        status_code=response.status_code,
                        retry_after=retry_after,
                        response_headers=response_headers,
                    )

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue

                    data = line[5:].strip()
                    if data == "[DONE]":
                        break

                    try:
                        event: dict[str, Any] = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise ModelCallError(
                            "Stand-in returned malformed SSE JSON"
                        ) from exc

                    response_model = event.get("model", response_model)

                    usage = event.get("usage")
                    if usage:
                        tokens_in = usage.get("prompt_tokens", tokens_in)
                        tokens_out = usage.get("completion_tokens", tokens_out)

                    choices = event.get("choices") or []
                    for choice in choices:
                        delta = choice.get("delta") or {}
                        content = delta.get("content")
                        if content:
                            if first_token_ms is None:
                                first_token_ms = (
                                    time.perf_counter() - started
                                ) * 1000
                            parts.append(content)

            elapsed_ms = (time.perf_counter() - started) * 1000

            return ModelResponse(
                text="".join(parts),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                first_token_ms=first_token_ms,
                elapsed_ms=elapsed_ms,
                response_headers=response_headers,
                provider=self.provider_name,
                model=response_model,
            )

        except ModelCallError:
            raise
        except httpx.TimeoutException as exc:
            raise ModelCallError(
                f"Stand-in request timed out: {exc}",
                status_code=408,
            ) from exc
        except httpx.RequestError as exc:
            raise ModelCallError(
                f"Stand-in connection error: {exc}"
            ) from exc

    async def aclose(self) -> None:
        await self._client.aclose()
