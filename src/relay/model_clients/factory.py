from __future__ import annotations

from relay.config import Settings
from relay.model_clients.base import ModelClient
from relay.model_clients.groq_client import GroqModelClient
from relay.model_clients.stub_client import StubModelClient


def build_model_client(settings: Settings) -> ModelClient:
    backend = settings.model_backend.strip().lower()
    if backend == "stub":
        return StubModelClient()
    if backend == "groq":
        return GroqModelClient(
            api_key=settings.groq_api_key or "",
            model=settings.groq_model,
            timeout_seconds=settings.groq_timeout_seconds,
        )
    raise ValueError(f"Unsupported MODEL_BACKEND={settings.model_backend!r}; use 'stub' or 'groq'.")
