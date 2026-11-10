from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import pymupdf
from rank_bm25 import BM25Okapi


PAPER_CATALOG: dict[str, dict[str, str]] = {
    "attention_is_all_you_need_v7.pdf": {
        "paper_id": "1706.03762",
        "version": "v7",
        "title": "Attention Is All You Need",
    },
    "moshi_v1.pdf": {
        "paper_id": "2410.00037",
        "version": "v1",
        "title": "Moshi: a speech-text foundation model for real-time dialogue",
    },
    "dynamic_chunking_v2.pdf": {
        "paper_id": "2507.07955",
        "version": "v2",
        "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
    },
}

RetrievalStrategy = Literal["bm25", "expanded", "reranked"]

# Deliberately small, transparent domain expansions. These are not model-generated.
# Keep the original query unchanged in the state; expansions are only used for search.
QUERY_EXPANSIONS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("transformer",), (
        "self attention", "multi head attention", "scaled dot product attention",
        "encoder decoder", "positional encoding", "feed forward network",
    )),
    (("transformers",), (
        "self attention", "multi head attention", "scaled dot product attention",
        "encoder decoder", "positional encoding", "feed forward network",
    )),
    (("attention",), (
        "self attention", "multi head attention", "scaled dot product attention",
        "query key value", "attention weights",
    )),
    (("moshi",), (
        "speech text model", "real time dialogue", "audio tokens", "speech generation",
        "Mimi codec", "Helium language model",
    )),
    (("speech",), (
        "audio tokens", "speech text", "speech generation", "real time dialogue",
        "audio codec", "streaming audio",
    )),
    (("audio",), (
        "speech text", "audio tokens", "audio codec", "streaming speech",
    )),
    (("chunking",), (
        "hierarchical sequence modeling", "H Net", "subword tokenization",
        "dynamic segmentation", "sequence compression", "chunk boundaries",
        "boundary probabilities", "routing module", "downsampler",
        "smoothing module",
    )),
    (("hierarchical",), (
        "dynamic chunking", "H Net", "subword tokenization", "sequence abstraction",
    )),
    (("hnet",), (
        "H Net", "hierarchical chunking", "sequence modeling", "dynamic segmentation",
    )),
    (("h", "net"), (
        "H Net", "hierarchical chunking", "sequence modeling", "dynamic segmentation",
    )),
)

# Weighted paper hints are used only as a modest prior. Multiple-paper questions can
# activate more than one paper; a paper is never excluded solely by this mapping.
PAPER_TOPIC_HINTS: dict[str, tuple[tuple[str, float], ...]] = {
    "1706.03762": (
        ("transformer", 4.0),
        ("transformers", 4.0),
        ("self attention", 2.5),
        ("multi head attention", 2.8),
        ("scaled dot product", 2.5),
        ("positional encoding", 2.5),
        ("attention", 1.5),
        ("encoder decoder", 1.5),
    ),
    "2410.00037": (
        ("moshi", 4.0),
        ("speech text", 2.8),
        ("speech", 1.8),
        ("audio", 1.5),
        ("dialogue", 1.8),
        ("mimi", 2.5),
        ("helium", 2.5),
        ("real time", 1.0),
    ),
    "2507.07955": (
        ("dynamic chunking", 3.5),
        ("chunking", 2.5),
        ("h net", 2.8),
        ("hnet", 2.8),
        ("hierarchical", 1.8),
        ("subword", 1.8),
        ("tokenization", 1.5),
        ("sequence modeling", 1.8),
    ),
}


def tokenize(text: str) -> list[str]:
    """Tokenize text while making common hyphenated forms searchable.

    For example, both 'sub-layers' and 'sublayers' produce the token
    'sublayers'. The component tokens are retained for multiword matching.
    """
    result: list[str] = []
    for term in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower()):
        parts = term.split("-")
        result.extend(parts)
        if len(parts) > 1:
            result.append("".join(parts))
    return result


def expand_query(query: str) -> str:
    """Return a deterministic expanded query while retaining original terms."""
    original_tokens = tokenize(query)
    if not original_tokens:
        return query

    token_set = set(original_tokens)
    extra: list[str] = []
    for triggers, expansions in QUERY_EXPANSIONS:
        if all(trigger in token_set for trigger in triggers):
            for phrase in expansions:
                extra.extend(tokenize(phrase))

    return " ".join(dict.fromkeys(original_tokens + extra))


def _normalise_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    low, high = min(scores), max(scores)
    if high - low < 1e-12:
        return [0.5 for _ in scores]
    return [(score - low) / (high - low) for score in scores]


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


@dataclass(slots=True)
class PaperChunk:
    chunk_id: str
    paper_id: str
    version: str
    title: str
    page: int
    text: str
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PaperRetriever:
    """BM25 over exact-version PDFs with optional expansion and transparent reranking.

    Pages are 1-based PDF page numbers. `bm25` is the unmodified lexical baseline;
    `expanded` adds query expansion and a soft paper-topic prior; `reranked` adds
    passage-overlap scoring and a small redundancy penalty.
    """

    def __init__(
        self,
        papers_dir: Path,
        chunk_words: int = 500,
        overlap_words: int = 60,
    ) -> None:
        self.papers_dir = papers_dir
        self.chunk_words = max(80, chunk_words)
        self.overlap_words = max(0, min(overlap_words, self.chunk_words // 2))
        self.chunks = self._load_chunks()
        self._tokens = [tokenize(chunk.text) for chunk in self.chunks]
        self._bm25 = BM25Okapi(self._tokens) if self._tokens else None
        self._token_sets = [set(tokens) for tokens in self._tokens]

    def _load_chunks(self) -> list[PaperChunk]:
        chunks: list[PaperChunk] = []
        for filename, meta in PAPER_CATALOG.items():
            pdf_path = self.papers_dir / filename
            if not pdf_path.exists():
                continue
            try:
                doc = pymupdf.open(pdf_path)
            except (RuntimeError, ValueError) as exc:
                raise RuntimeError(f"Could not read {pdf_path}: {exc}") from exc

            try:
                for page_index, page in enumerate(doc):
                    text = page.get_text("text").strip()
                    if not text:
                        continue
                    words = text.split()
                    stride = max(1, self.chunk_words - self.overlap_words)
                    for offset in range(0, len(words), stride):
                        block = words[offset : offset + self.chunk_words]
                        if not block:
                            continue
                        part = offset // stride + 1
                        chunk_id = (
                            f"{meta['paper_id']}-{meta['version']}-"
                            f"p{page_index + 1:03d}-c{part:02d}"
                        )
                        chunks.append(PaperChunk(
                            chunk_id=chunk_id,
                            paper_id=meta["paper_id"],
                            version=meta["version"],
                            title=meta["title"],
                            page=page_index + 1,
                            text=" ".join(block),
                        ))
                        if offset + self.chunk_words >= len(words):
                            break
            finally:
                doc.close()
        return chunks

    @property
    def document_count(self) -> int:
        return len({chunk.paper_id for chunk in self.chunks})

    def weighted_query_coverage(self, query: str, passage: str) -> float:
        """IDF-weighted fraction of meaningful query terms present in a passage.

        Rare query terms carry more weight than generic terms. Terms absent
        from the indexed corpus remain in the denominator, making unsupported
        topics harder to accept as answerable evidence.
        """
        stop_words = {
            "a", "an", "and", "are", "as", "at", "be", "by", "can", "could",
            "did", "do", "does", "for", "from", "how", "i", "in", "is", "it",
            "of", "on", "or", "that", "the", "their", "them", "there",
            "these", "this", "to", "was", "what", "when", "where", "which",
            "who", "why", "with", "would", "according", "have", "has", "had",
            "paper", "papers", "about", "say",
        }
        terms = set(tokenize(query)) - stop_words
        if not terms:
            return 0.0

        passage_terms = set(tokenize(passage))
        corpus = getattr(self, "_token_sets", [])
        document_count = max(1, len(corpus))

        total_weight = 0.0
        matched_weight = 0.0

        for term in terms:
            df = sum(1 for document in corpus if term in document)
            idf = math.log1p(
                (document_count - df + 0.5) / (df + 0.5)
            )
            total_weight += idf
            if term in passage_terms:
                matched_weight += idf

        return matched_weight / total_weight if total_weight else 0.0

    @staticmethod
    def paper_affinity(query: str, paper_id: str) -> float:
        """Return a bounded 0..1 prior based on explicit paper-topic phrases."""
        query_tokens = tokenize(query)
        query_set = set(query_tokens)
        if not query_set:
            return 0.0

        normalised_query = " ".join(query_tokens)
        score = 0.0
        for phrase, weight in PAPER_TOPIC_HINTS.get(paper_id, ()):
            phrase_tokens = tokenize(phrase)
            if not phrase_tokens:
                continue
            if phrase in normalised_query:
                score += weight
            elif len(phrase_tokens) == 1 and phrase_tokens[0] in query_set:
                score += weight
            elif len(phrase_tokens) > 1 and all(token in query_set for token in phrase_tokens):
                score += weight * 0.65

        # A numeric arXiv ID or paper title in the query is an explicit source request.
        if paper_id in query:
            score += 10.0
        return min(1.0, score / 4.0)

    def _candidate_indices(
        self,
        original_scores: list[float],
        expanded_scores: list[float],
        candidate_k: int,
    ) -> set[int]:
        count = len(self.chunks)
        overall_k = min(count, max(candidate_k, 1))
        candidates = {
            index
            for scores in (original_scores, expanded_scores)
            for index in sorted(range(count), key=lambda i: scores[i], reverse=True)[:overall_k]
        }

        # Keep a few candidates from every paper so a strong source-specific topic
        # hint cannot be defeated simply because another PDF has more lexical hits.
        per_paper_k = min(3, max(1, candidate_k // max(1, len(PAPER_CATALOG))))
        paper_ids = {chunk.paper_id for chunk in self.chunks}
        for paper_id in paper_ids:
            in_paper = [i for i, chunk in enumerate(self.chunks) if chunk.paper_id == paper_id]
            candidates.update(
                sorted(in_paper, key=lambda i: expanded_scores[i], reverse=True)[:per_paper_k]
            )
        return candidates

    def search(
        self,
        query: str,
        top_k: int = 4,
        strategy: RetrievalStrategy | str = "bm25",
        candidate_k: int = 12,
    ) -> list[dict[str, Any]]:
        if not self._bm25 or not self.chunks:
            return []
        if strategy not in {"bm25", "expanded", "reranked"}:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")

        original_tokens = tokenize(query)
        if not original_tokens:
            return []
        original_set = set(original_tokens)
        original_scores = [float(score) for score in self._bm25.get_scores(original_tokens)]
        original_norm = _normalise_scores(original_scores)

        if strategy == "bm25":
            ranked = sorted(
                range(len(self.chunks)),
                key=lambda index: original_scores[index],
                reverse=True,
            )[: max(1, top_k)]
            return [self._make_result(index, original_scores[index], strategy) for index in ranked]

        expanded_query = expand_query(query)
        expanded_tokens = tokenize(expanded_query)
        expanded_set = set(expanded_tokens)
        expanded_scores = [float(score) for score in self._bm25.get_scores(expanded_tokens)]
        expanded_norm = _normalise_scores(expanded_scores)
        candidates = self._candidate_indices(
            original_scores,
            expanded_scores,
            candidate_k=max(candidate_k, top_k),
        )

        base_scores: dict[int, float] = {}
        affinities: dict[int, float] = {}
        overlaps: dict[int, float] = {}
        expanded_overlaps: dict[int, float] = {}

        for index in candidates:
            paper_id = self.chunks[index].paper_id
            affinity = self.paper_affinity(query, paper_id)
            token_set = self._token_sets[index]
            overlap = len(original_set & token_set) / max(1, len(original_set))
            expanded_overlap = len(expanded_set & token_set) / max(1, len(expanded_set))
            affinities[index] = affinity
            overlaps[index] = overlap
            expanded_overlaps[index] = expanded_overlap
            # Stronger topic prior applies only when the user explicitly names a
            # paper-specific concept; unanchored questions remain BM25-led.
            base_scores[index] = (
                0.25 * original_norm[index]
                + 0.15 * expanded_norm[index]
                + 0.60 * affinity
            )

        if strategy == "expanded":
            ranked = sorted(candidates, key=lambda i: base_scores[i], reverse=True)[: max(1, top_k)]
        else:
            # Transparent lexical reranker: it rewards direct query coverage and
            # topic affinity, then applies MMR-style redundancy reduction.
            rerank_scores = {
                index: (
                    0.65 * base_scores[index]
                    + 0.25 * overlaps[index]
                    + 0.10 * expanded_overlaps[index]
                )
                for index in candidates
            }
            remaining = set(candidates)
            selected: list[int] = []
            target_count = min(max(1, top_k), len(remaining))
            while remaining and len(selected) < target_count:
                best_index = max(
                    remaining,
                    key=lambda index: rerank_scores[index] - 0.12 * max(
                        (_jaccard(self._token_sets[index], self._token_sets[chosen]) for chosen in selected),
                        default=0.0,
                    ),
                )
                selected.append(best_index)
                remaining.remove(best_index)
            ranked = selected

            # Reserve a slot for the strongest direct BM25 candidate that
            # has meaningful query coverage, if reranking pushed it out.
            if ranked:
                for direct_index in sorted(
                    candidates,
                    key=lambda i: original_scores[i],
                    reverse=True,
                ):
                    direct_coverage = self.weighted_query_coverage(
                        query, self.chunks[direct_index].text
                    )
                    if direct_coverage < 0.35:
                        continue
                    if direct_index not in ranked:
                        ranked[-1] = direct_index
                    break

        results: list[dict[str, Any]] = []
        for index in ranked:
            result = self._make_result(index, original_scores[index], strategy)
            result.update({
                "expanded_score": expanded_scores[index],
                "retrieval_score": float(base_scores[index]),
                "paper_affinity": float(affinities[index]),
                "query_overlap": float(overlaps[index]),
                "expanded_query_overlap": float(expanded_overlaps[index]),
            })
            if strategy == "reranked":
                result["rerank_score"] = float(rerank_scores[index])
            results.append(result)
        return results

    def _make_result(self, index: int, score: float, strategy: str) -> dict[str, Any]:
        result = self.chunks[index].to_dict()
        result["score"] = float(score)  # Original BM25 score retained for diagnostics.
        result["strategy"] = strategy
        return result
