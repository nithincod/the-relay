from __future__ import annotations

import time
from typing import Any

from groq import AsyncGroq
from groq import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from relay.model_clients.base import ModelCallError, ModelResponse


class GroqModelClient:
    provider_name = "groq"

    def __init__(self, api_key: str, model: str, timeout_seconds: float = 30.0) -> None:
        if not api_key:
            raise ValueError("GROQ_API_KEY is required when MODEL_BACKEND=groq")
        self.model_name = model
        self._client = AsyncGroq(api_key=api_key, timeout=timeout_seconds)

    async def complete(self, messages: list[dict[str, str]]) -> ModelResponse:
        started = time.perf_counter()
        first_token_ms: float | None = None
        parts: list[str] = []
        tokens_in: int | None = None
        tokens_out: int | None = None
        response_headers: dict[str, str] = {}

        try:
            # Stream responses to measure model time-to-first-token.
            # Groq exposes usage on the final chunk through x_groq.usage.
            stream = await self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0,
                reasoning_effort="low",
                max_completion_tokens=1024,
                include_reasoning=False,
                stream=True,
            )
            async for chunk in stream:
                x_groq = getattr(chunk, "x_groq", None)
                usage = getattr(x_groq, "usage", None) if x_groq else None
                if usage is None:
                    usage = getattr(chunk, "usage", None)
                if usage is not None:
                    tokens_in = getattr(usage, "prompt_tokens", None)
                    tokens_out = getattr(usage, "completion_tokens", None)

                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    if first_token_ms is None:
                        first_token_ms = (time.perf_counter() - started) * 1000
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
                model=self.model_name,
            )
        except RateLimitError as exc:
            self._raise_normalized(exc, status_code=429)
        except APITimeoutError as exc:
            self._raise_normalized(exc, status_code=408)
        except APIStatusError as exc:
            self._raise_normalized(exc, status_code=getattr(exc, "status_code", None))
        except APIConnectionError as exc:
            raise ModelCallError(f"Groq connection error: {exc}") from exc

    @staticmethod
    def _raise_normalized(exc: Exception, status_code: int | None) -> None:
        response = getattr(exc, "response", None)
        headers_obj: Any = getattr(response, "headers", None)
        headers = {str(k).lower(): str(v) for k, v in headers_obj.items()} if headers_obj else {}
        retry_after = None
        try:
            retry_after = float(headers["retry-after"]) if "retry-after" in headers else None
        except (TypeError, ValueError):
            retry_after = None
        raise ModelCallError(
            str(exc),
            status_code=status_code,
            retry_after=retry_after,
            response_headers=headers,
        ) from exc
