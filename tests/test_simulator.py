from __future__ import annotations

import json

from fastapi.testclient import TestClient

from simulator.app import SimConfig, create_app


def _payload() -> dict:
    return {
        "model": "relay-standin-v1",
        "stream": True,
        "messages": [
            {"role": "system", "content": "Answer only from the supplied passages."},
            {"role": "user", "content": "[CHUNK_ID:1706.03762-v7-p003-c01] Explain the passage."},
        ],
    }


def _events(response_text: str) -> list[dict]:
    result = []
    for line in response_text.splitlines():
        if line.startswith("data: ") and line[6:] != "[DONE]":
            result.append(json.loads(line[6:]))
    return result


def test_stream_returns_cited_text_and_usage():
    app = create_app(SimConfig(base_latency_ms=0, chunk_delay_ms=0, log_path=""))
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=_payload())
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = _events(response.text)
        text = "".join(
            event["choices"][0]["delta"].get("content", "")
            for event in events if event.get("choices")
        )
        assert "[[1706.03762-v7-p003-c01]]" in text
        assert events[-1]["usage"]["prompt_tokens"] > 0
        assert "[DONE]" in response.text


def test_rpm_limit_returns_429_and_retry_after():
    app = create_app(SimConfig(
        base_latency_ms=0, chunk_delay_ms=0, rpm_limit=1, log_path=""
    ))
    with TestClient(app) as client:
        first = client.post("/v1/chat/completions", json=_payload())
        second = client.post("/v1/chat/completions", json=_payload())
        assert first.status_code == 200
        assert second.status_code == 429
        assert int(second.headers["retry-after"]) >= 1
        assert second.json()["detail"]["error"] == "rate_limit_exceeded"


def test_forced_daily_exhaustion_is_distinguishable():
    app = create_app(SimConfig(base_latency_ms=0, log_path=""))
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json=_payload(),
            headers={"X-Sim-Scenario": "daily_exhausted"},
        )
        assert response.status_code == 429
        assert response.json()["detail"]["error"] == "daily_budget_exhausted"


def test_forced_timeout_and_server_error_scenarios():
    app = create_app(SimConfig(base_latency_ms=0, log_path=""))
    with TestClient(app) as client:
        timeout = client.post(
            "/v1/chat/completions", json=_payload(), headers={"X-Sim-Scenario": "timeout"}
        )
        failure = client.post(
            "/v1/chat/completions", json=_payload(), headers={"X-Sim-Scenario": "server_error"}
        )
        assert timeout.status_code == 408
        assert failure.status_code == 500


def test_metrics_endpoint_exposes_limits_and_counters():
    app = create_app(SimConfig(base_latency_ms=0, chunk_delay_ms=0, log_path=""))
    with TestClient(app) as client:
        client.post("/v1/chat/completions", json=_payload())
        result = client.get("/metrics").json()
        assert result["accepted_total"] == 1
        assert result["limits"]["rpm"] == 60
        assert result["peak_active"] == 1


def test_generic_latency_profile_remains_unchanged():
    from simulator.app import SimConfig, _latency_targets

    config = SimConfig(
        profile_mode="generic",
        base_latency_ms=900,
        load_latency_ms=250,
    )

    assert _latency_targets(config, "verify_answer", 1) == (
        900,
        None,
        "generic",
    )
    assert _latency_targets(config, "verify_answer", 3) == (
        1400,
        None,
        "generic",
    )


def test_groq_observed_profile_uses_stage_specific_timings():
    from simulator.app import SimConfig, _latency_targets

    config = SimConfig(profile_mode="groq_observed")

    assert _latency_targets(config, "understand_question", 1) == (
        452.712,
        453.653,
        "groq_observed",
    )
    assert _latency_targets(config, "generate_answer", 1) == (
        410.007,
        607.787,
        "groq_observed",
    )
    assert _latency_targets(config, "regenerate_answer", 1) == (
        435.448,
        877.948,
        "groq_observed",
    )
    assert _latency_targets(config, "verify_answer", 1) == (
        798.022,
        853.471,
        "groq_observed",
    )
    actual = _latency_targets(config, "generate_answer", 3)
    assert abs(actual[0] - 910.007) < 1e-6
    assert abs(actual[1] - 1107.787) < 1e-6
    assert actual[2] == "groq_observed"
