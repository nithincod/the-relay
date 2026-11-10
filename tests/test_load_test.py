from __future__ import annotations

import csv

import pytest

from experiments.load_test import CSV_COLUMNS, append_assignment_csv, percentile, summarize


def test_percentile_uses_nearest_rank():
    assert percentile([10, 20, 30, 40], 50) == 20
    assert percentile([10, 20, 30, 40], 95) == 40
    assert percentile([], 50) is None


def test_summary_counts_statuses_and_latency_percentiles():
    rows = [
        {"status": "ok", "total_ms": 100, "first_token_ms": 50, "tokens_in": 10, "tokens_out": 5},
        {"status": "rate_limited", "total_ms": 20, "first_token_ms": None, "tokens_in": 0, "tokens_out": 0},
        {"status": "timeout", "total_ms": 200, "first_token_ms": None, "tokens_in": 0, "tokens_out": 0},
    ]
    result = summarize(rows, 2.0, 1.5)
    assert result["requests"] == 3
    assert result["success_percent"] == pytest.approx(33.33)
    assert result["rate_limited_percent"] == pytest.approx(33.33)
    assert result["timeout_percent"] == pytest.approx(33.33)
    assert result["ttft_ms"]["p50"] == 50
    assert result["total_ms"]["p95"] == 200
    assert result["tokens_in"] == 10


def test_append_assignment_csv_preserves_required_schema(tmp_path):
    path = tmp_path / "results.csv"
    row = {
        "request_id": "abc",
        "scenario": "steady",
        "source": "stand-in",
        "start_time": "2026-11-10T00:00:00+00:00",
        "first_token_ms": 10.0,
        "total_ms": 20.0,
        "status": "ok",
        "retries": 0,
        "tokens_in": 10,
        "tokens_out": 5,
        "retrieval_setting": "direct_standin_loadtest",
        "answer_grade": "",
    }
    append_assignment_csv(path, [row])
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == CSV_COLUMNS
        stored = next(reader)
    assert stored["request_id"] == "abc"
    assert stored["scenario"] == "steady"


def test_append_assignment_csv_refuses_wrong_existing_header(tmp_path):
    path = tmp_path / "results.csv"
    path.write_text("wrong,header\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected results CSV header"):
        append_assignment_csv(path, [])


def test_queue_full_is_not_classified_as_quota_rate_limit():
    from experiments.load_test import _status_for_http

    assert _status_for_http(429, "queue_full") == "error"


def test_quota_429_remains_rate_limited():
    from experiments.load_test import _status_for_http

    assert _status_for_http(429, "rate_limit_exceeded") == "rate_limited"
    assert _status_for_http(429, "daily_budget_exhausted") == "rate_limited"
