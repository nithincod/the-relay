from __future__ import annotations

import pytest

from relay.model_clients.stub_client import StubModelClient
from relay.retrieval import PaperRetriever
from relay.workflow import RelayWorkflow


class EmptyRetriever(PaperRetriever):
    def __init__(self):
        self.chunks = []
        self._tokens = []
        self._bm25 = None

    @property
    def document_count(self):
        return 0

    def search(self, query: str, top_k: int = 4, **kwargs):
        return []


class OneChunkRetriever(EmptyRetriever):
    def search(self, query: str, top_k: int = 4, **kwargs):
        return [{
            "chunk_id": "1706.03762-v7-p001-c01",
            "paper_id": "1706.03762",
            "version": "v7",
            "title": "Attention Is All You Need",
            "page": 1,
            "text": "The Transformer uses attention mechanisms and removes recurrence.",
            "score": 2.5,
        }]


@pytest.mark.asyncio
async def test_empty_library_refuses_without_guessing():
    app = RelayWorkflow(StubModelClient(), EmptyRetriever())
    result = await app.run("What does the paper say about an unrelated topic?")
    assert result["status"] == "ok"
    assert "do not provide enough evidence" in result["final_answer"].lower()
    assert result["retrieval_attempts"] == 2


@pytest.mark.asyncio
async def test_empty_question_is_rejected():
    app = RelayWorkflow(StubModelClient(), EmptyRetriever())
    result = await app.run("   ")
    assert result["status"] == "error"
    assert "non-empty" in result["final_answer"].lower()


@pytest.mark.asyncio
async def test_graph_generates_and_formats_a_citation():
    app = RelayWorkflow(StubModelClient(), OneChunkRetriever(), min_score=0.01)
    result = await app.run("What does the paper say about the Transformer?")
    assert result["status"] == "ok"
    assert "arXiv:1706.03762v7" in result["final_answer"]
    assert result["generation_attempts"] == 1
    assert len(result["model_calls"]) == 3

    # The initial draft and successful semantic verification are preserved.
    assert len(result["draft_history"]) == 1
    assert result["draft_history"][0]["kind"] == "initial"
    assert result["draft_history"][0]["answer"] == result["draft_answer"]

    assert len(result["verification_history"]) == 1
    assert result["verification_history"][0]["check"] == "llm_semantic_verification"
    assert result["verification_history"][0]["result"] == "pass"


@pytest.mark.asyncio
async def test_missing_citation_is_recorded_without_llm_verification():
    retriever = OneChunkRetriever()
    app = RelayWorkflow(StubModelClient(), retriever)

    result = await app.verify_answer({
        "question": "Explain the Transformer.",
        "retrieved_chunks": retriever.search("Transformer"),
        "draft_answer": "The Transformer uses attention mechanisms.",
    })

    assert result["verification_result"] == "fail"
    assert len(result["verification_history"]) == 1

    event = result["verification_history"][0]
    assert event["check"] == "local_citation_precheck"
    assert event["result"] == "fail"
    assert event["used_ids"] == []
    assert event["invalid_ids"] == []
