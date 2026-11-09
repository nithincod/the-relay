from __future__ import annotations

from relay.retrieval import PaperChunk, PaperRetriever, expand_query, tokenize


class FixedScores:
    def __init__(self, scores: list[float]):
        self.scores = scores

    def get_scores(self, _query_tokens: list[str]):
        return self.scores


def make_retriever() -> PaperRetriever:
    retriever = PaperRetriever.__new__(PaperRetriever)
    retriever.chunks = [
        PaperChunk(
            chunk_id="2507.07955-v2-p037-c01",
            paper_id="2507.07955",
            version="v2",
            title="Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
            page=37,
            text=(
                "Transformers in the encoder network do not appear helpful for text. "
                "This passage compares model architecture ablations in H-Net."
            ),
        ),
        PaperChunk(
            chunk_id="2410.00037-v1-p007-c01",
            paper_id="2410.00037",
            version="v1",
            title="Moshi: a speech-text foundation model for real-time dialogue",
            page=7,
            text=(
                "Moshi is built on top of Helium, a text language model, and models "
                "text and audio tokens for real-time dialogue."
            ),
        ),
        PaperChunk(
            chunk_id="1706.03762-v7-p003-c01",
            paper_id="1706.03762",
            version="v7",
            title="Attention Is All You Need",
            page=3,
            text=(
                "The Transformer is based solely on attention mechanisms, dispensing "
                "with recurrence and convolutions. It uses multi-head self-attention, "
                "positional encodings, encoder and decoder stacks."
            ),
        ),
    ]
    retriever._tokens = [tokenize(chunk.text) for chunk in retriever.chunks]
    retriever._token_sets = [set(tokens) for tokens in retriever._tokens]
    # Simulate a naive BM25 result where the unrelated mention of Transformers
    # is the strongest lexical match. Topic-aware retrieval should recover the
    # canonical Transformer paper instead of blindly taking this first result.
    retriever._bm25 = FixedScores([8.48, 4.73, 2.10])
    return retriever


def test_query_expansion_adds_transformer_concepts_and_preserves_original_terms():
    query = "Explain the Transformer architecture"
    expanded = tokenize(expand_query(query))
    assert "transformer" in expanded
    assert "architecture" in expanded
    assert "attention" in expanded
    assert "positional" in expanded
    assert "encoder" in expanded


def test_paper_affinity_identifies_primary_transformer_paper():
    query = "Explain the Transformer architecture"
    assert PaperRetriever.paper_affinity(query, "1706.03762") == 1.0
    assert PaperRetriever.paper_affinity(query, "2410.00037") == 0.0
    assert PaperRetriever.paper_affinity(query, "2507.07955") == 0.0


def test_multi_paper_query_can_activate_more_than_one_paper_hint():
    query = "Compare Moshi speech modeling with H-Net hierarchical chunking"
    assert PaperRetriever.paper_affinity(query, "2410.00037") > 0.5
    assert PaperRetriever.paper_affinity(query, "2507.07955") > 0.5


def test_bm25_baseline_preserves_naive_global_ranking_for_comparison():
    retriever = make_retriever()
    results = retriever.search("Explain the Transformer architecture", top_k=2, strategy="bm25")
    assert results[0]["paper_id"] == "2507.07955"


def test_expanded_retrieval_surfaces_attention_paper_over_contextual_mention():
    retriever = make_retriever()
    results = retriever.search("Explain the Transformer architecture", top_k=2, strategy="expanded")
    assert results[0]["paper_id"] == "1706.03762"
    assert results[0]["chunk_id"] == "1706.03762-v7-p003-c01"
    assert results[0]["paper_affinity"] == 1.0


def test_reranked_results_include_explainable_diagnostics():
    retriever = make_retriever()
    results = retriever.search("Explain the Transformer architecture", top_k=2, strategy="reranked")
    assert results[0]["paper_id"] == "1706.03762"
    assert "rerank_score" in results[0]
    assert "query_overlap" in results[0]
    assert "expanded_query_overlap" in results[0]
