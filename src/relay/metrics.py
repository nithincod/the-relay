from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


CSV_COLUMNS = [
    "request_id", "scenario", "source", "start_time", "first_token_ms", "total_ms",
    "status", "retries", "tokens_in", "tokens_out", "retrieval_setting", "answer_grade",
]


def append_result(path: Path, state: dict[str, Any], retrieval_setting: str = "default") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    row = {
        "request_id": state.get("request_id", ""),
        "scenario": state.get("scenario", "manual"),
        "source": state.get("source", "unknown"),
        "start_time": state.get("started_at", ""),
        "first_token_ms": _format_num(state.get("first_token_ms")),
        "total_ms": _format_num(state.get("total_ms")),
        "status": state.get("status", "error"),
        "retries": (
            max(0, int(state.get("retrieval_attempts", 0)) - 1)
            + max(0, int(state.get("generation_attempts", 0)) - 1)
            + sum(int(call.get("sdk_retries", 0)) for call in state.get("model_calls", []))
        ),
        "tokens_in": state.get("tokens_in", 0),
        "tokens_out": state.get("tokens_out", 0),
        "retrieval_setting": retrieval_setting,
        # Filled by the manual/assisted answer-quality evaluation pass.
        "answer_grade": "",
    }
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, lineterminator="\n")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _format_num(value: Any) -> str:
    return "" if value is None else f"{float(value):.3f}"
