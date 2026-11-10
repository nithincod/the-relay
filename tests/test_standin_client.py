from __future__ import annotations

import json

import httpx
import pytest

from relay.model_clients.base import ModelCallError
from relay.model_clients.standin_client import StandInModelClient


@pytest.mark.asyncio
async def test_standin_client_parses_stream_and_usage():
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is True
        assert body["model"] == "relay-standin-v1"

        events = [
            {
                "model": "relay-standin-v1",
                "choices": [{"delta": {"content": "Hello "}}],
            },
            {
                "model": "relay-standin-v1",
                "choices": [{"delta": {"content": "Relay"}}],
            },
            {
                "model": "relay-standin-v1",
                "choices": [],
                "usage": {
                    "prompt_tokens": 42,
                    "completion_tokens": 12,
                    "total_tokens": 54,
                },
            },
        ]

        content = "".join(
            f"data: {json.dumps(event)}\n\n" for event in events
        ) + "data: [DONE]\n\n"

        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=content,
        )

    client = StandInModelClient(
        transport=httpx.MockTransport(handler)
    )

    try:
        result = await client.complete([
            {"role": "user", "content": "Hello"}
        ])

        assert result.text == "Hello Relay"
        assert result.tokens_in == 42
        assert result.tokens_out == 12
        assert result.first_token_ms is not None
        assert result.elapsed_ms >= result.first_token_ms
        assert result.provider == "stand-in"
        assert result.model == "relay-standin-v1"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_standin_client_normalizes_rate_limit():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"retry-after": "17"},
            json={
                "detail": {
                    "error": "rate_limit_exceeded",
                    "message": "Simulated RPM limit reached",
                }
            },
        )

    client = StandInModelClient(
        transport=httpx.MockTransport(handler)
    )

    try:
        with pytest.raises(ModelCallError) as captured:
            await client.complete([
                {"role": "user", "content": "Test"}
            ])

        error = captured.value
        assert error.status_code == 429
        assert error.retry_after == 17
        assert "RPM limit" in str(error)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_standin_client_forwards_workflow_purpose():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-sim-purpose"] == "verify_answer"
        event = {
            "model": "relay-standin-v1",
            "choices": [{"delta": {"content": "PASS"}}],
        }
        usage = {
            "model": "relay-standin-v1",
            "choices": [],
            "usage": {"prompt_tokens": 4, "completion_tokens": 1},
        }
        content = (
            f"data: {json.dumps(event)}\n\n"
            f"data: {json.dumps(usage)}\n\n"
            "data: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=content,
        )

    client = StandInModelClient(transport=httpx.MockTransport(handler))
    try:
        result = await client.complete(
            [{"role": "user", "content": "Verify this"}],
            purpose="verify_answer",
        )
        assert result.text == "PASS"
        assert result.provider == "stand-in"
    finally:
        await client.aclose()
