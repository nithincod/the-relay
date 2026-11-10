from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(slots=True)
class ModelResponse:
    text: str
    tokens_in: int | None = None
    tokens_out: int | None = None
    first_token_ms: float | None = None
    elapsed_ms: float = 0.0
    response_headers: dict[str, str] = field(default_factory=dict)
    provider: str = "unknown"
    model: str = "unknown"
    sdk_retries: int = 0


class ModelCallError(RuntimeError):
    """A provider error with normalized HTTP/rate-limit details."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
        response_headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.response_headers = response_headers or {}


class ModelClient(Protocol):
    provider_name: str
    model_name: str

    async def complete(
        self, messages: list[dict[str, str]], purpose: str = "completion"
    ) -> ModelResponse:
        """Generate one completion and report timing/usage metadata."""
