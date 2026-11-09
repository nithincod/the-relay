from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class RelayState(TypedDict, total=False):
    request_id: str
    question: str
    scenario: str
    source: str
    started_at: str
    started_perf: float

    intent: str
    complexity: str
    retrieval_query: str
    retrieval_strategy: str
    retrieval_setting: str
    retrieved_chunks: list[dict[str, Any]]
    evidence_sufficient: bool
    evidence_reason: str
    retrieval_attempts: int
    generation_attempts: int

    draft_answer: str
    verification_result: str
    verification_feedback: str
    citations: list[str]
    final_answer: str
    status: str
    error_message: str
    provider_error_headers: dict[str, str]
    provider_retry_after: float | None

    tokens_in: int
    tokens_out: int
    model_first_token_ms: float | None
    first_token_ms: float | None
    total_ms: float | None
    node_timings_ms: dict[str, float]
    model_calls: Annotated[list[dict[str, Any]], operator.add]
