from __future__ import annotations

import pytest

from relay.model_clients.groq_client import (
    GroqModelClient,
    filter_provider_headers,
    parse_retry_after,
)


def test_filter_provider_headers_keeps_rate_limits_and_request_id():
    result = filter_provider_headers({
        "X-RateLimit-Limit-Requests": "1000",
        "x-ratelimit-limit-tokens": "8000",
        "x-ratelimit-remaining-tokens": "7920",
        "x-ratelimit-reset-tokens": "3.2s",
        "Retry-After": "7",
        "x-request-id": "req_example",
        "content-type": "text/event-stream",
        "server": "cloudflare",
    })

    assert result == {
        "x-ratelimit-limit-requests": "1000",
        "x-ratelimit-limit-tokens": "8000",
        "x-ratelimit-remaining-tokens": "7920",
        "x-ratelimit-reset-tokens": "3.2s",
        "retry-after": "7",
        "x-request-id": "req_example",
    }


def test_parse_retry_after():
    assert parse_retry_after({"Retry-After": "12"}) == 12.0
    assert parse_retry_after({"retry-after": "not-a-number"}) is None
    assert parse_retry_after({}) is None


@pytest.mark.asyncio
async def test_groq_sdk_automatic_retries_are_disabled(tmp_path):
    client = GroqModelClient(
        api_key="test-key-not-used",
        model="openai/gpt-oss-20b",
        observation_path=tmp_path / "provider.jsonl",
    )
    try:
        assert client._client.max_retries == 0
        assert client.observation_path == tmp_path / "provider.jsonl"
    finally:
        await client.aclose()
