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
