from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from relay.model_clients.base import ModelCallError, ModelClient, ModelResponse
from relay.retrieval import PaperRetriever, tokenize
from relay.state import RelayState


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "did", "do",
    "does", "for", "from", "how", "i", "in", "is", "it", "of", "on", "or", "that",
    "the", "their", "them", "there", "these", "this", "to", "was", "what", "when",
    "where", "which", "who", "why", "with", "would",
}


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I)
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}


def _chunk_ids(chunks: list[dict[str, Any]]) -> set[str]:
    return {str(c.get("chunk_id", "")) for c in chunks if c.get("chunk_id")}


def _citation_markers(text: str) -> list[str]:
    return re.findall(r"\[\[([A-Za-z0-9_.-]+)\]\]", text)


class RelayWorkflow:
    def __init__(
        self,
        model: ModelClient,
        retriever: PaperRetriever,
        top_k: int = 4,
        min_score: float = 0.01,
        retrieval_strategy: str = "reranked",
        retrieval_candidates: int = 12,
    ):
        if retrieval_strategy not in {"bm25", "expanded", "reranked"}:
            raise ValueError(f"Unknown retrieval strategy: {retrieval_strategy}")
        self.model = model
        self.retriever = retriever
        self.top_k = max(1, top_k)
        # Kept for compatibility with the initial constructor. Evidence sufficiency
        # now uses query coverage because raw BM25 scores vary with query length.
        self.min_score = min_score
        self.retrieval_strategy = retrieval_strategy
        self.retrieval_candidates = max(self.top_k, retrieval_candidates)
        self.graph = self._build_graph()

    async def _call_model(
        self, purpose: str, messages: list[dict[str, str]]
    ) -> tuple[ModelResponse, dict[str, Any]]:
        response = await self.model.complete(messages)
        metric = {
            "purpose": purpose,
            "provider": response.provider,
            "model": response.model,
            "elapsed_ms": response.elapsed_ms,
            "first_token_ms": response.first_token_ms,
            "tokens_in": response.tokens_in,
            "tokens_out": response.tokens_out,
        }
        return response, metric

    async def validate_request(self, state: RelayState) -> dict[str, Any]:
        question = str(state.get("question", "")).strip()
        if not question:
            return {"status": "error", "final_answer": "Please provide a non-empty research question."}
        return {"question": question, "status": "running"}

    async def understand_question(self, state: RelayState) -> dict[str, Any]:
        prompt = [
            {
                "role": "system",
                "content": (
                    "Classify a research question for retrieval from a fixed library of three papers. "
                    "Return one JSON object with keys intent, complexity, retrieval_query. "
                    "Do not answer the research question. Keep retrieval_query faithful to the question."
                ),
            },
            {"role": "user", "content": f"Question:\n{state['question']}"},
        ]
        response, metric = await self._call_model("understand_question", prompt)
        parsed = _parse_json_object(response.text)
        query = str(parsed.get("retrieval_query") or state["question"]).strip()
        return {
            "intent": str(parsed.get("intent") or "research_question"),
            "complexity": str(parsed.get("complexity") or "normal"),
            "retrieval_query": query or state["question"],
            "tokens_in": int(state.get("tokens_in", 0)) + int(response.tokens_in or 0),
            "tokens_out": int(state.get("tokens_out", 0)) + int(response.tokens_out or 0),
            "model_calls": [metric],
        }

    async def retrieve_papers(self, state: RelayState) -> dict[str, Any]:
        attempt = int(state.get("retrieval_attempts", 0)) + 1
        # Keep top_k fixed across retries so one retrieval experiment has a stable
        # setting. The retry changes the query, not the experimental configuration.
        query = state.get("retrieval_query", state["question"])
        chunks = self.retriever.search(
            query,
            top_k=self.top_k,
            strategy=self.retrieval_strategy,
            candidate_k=self.retrieval_candidates,
        )
        return {
            "retrieved_chunks": chunks,
            "retrieval_attempts": attempt,
            "retrieval_strategy": self.retrieval_strategy,
        }

    async def evaluate_evidence(self, state: RelayState) -> dict[str, Any]:
        chunks = state.get("retrieved_chunks", [])
        best_score = max((float(c.get("score", 0.0)) for c in chunks), default=0.0)
        query_text = state.get("retrieval_query", state.get("question", ""))
        query_terms = set(tokenize(query_text)) - STOP_WORDS
        best_overlap = max(
            (
                len(query_terms.intersection(tokenize(str(c.get("text", "")))))
                / max(1, len(query_terms))
                for c in chunks
            ),
            default=0.0,
        )
        best_expanded_overlap = max(
            (float(c.get("expanded_query_overlap", 0.0)) for c in chunks),
            default=0.0,
        )
        best_affinity = max((float(c.get("paper_affinity", 0.0)) for c in chunks), default=0.0)

        # A positive BM25 score alone is not proof that evidence answers the question.
        # Require direct lexical coverage, or expanded coverage plus a relevant paper hint.
        sufficient = bool(chunks) and (
            best_overlap >= 0.20
            or (best_expanded_overlap >= 0.16 and best_affinity >= 0.50)
        )
        reason = (
            f"Retrieved {len(chunks)} passage(s); best BM25 score={best_score:.4f}; "
            f"direct query overlap={best_overlap:.1%}; "
            f"expanded overlap={best_expanded_overlap:.1%}; "
            f"paper affinity={best_affinity:.2f}."
            if chunks else "No passages were retrieved from the indexed library."
        )
        return {"evidence_sufficient": sufficient, "evidence_reason": reason}

    async def prepare_retrieval_retry(self, state: RelayState) -> dict[str, Any]:
        terms = [t for t in re.findall(r"[A-Za-z0-9]+", state["question"].lower()) if t not in STOP_WORDS]
        # Keep the user's own technical terms; this is a lexical broadening, not an answer rewrite.
        broadened = " ".join(dict.fromkeys(terms)) or state["question"]
        return {"retrieval_query": broadened}

    async def generate_answer(self, state: RelayState) -> dict[str, Any]:
        context = "\n\n".join(
            f"[CHUNK_ID:{c['chunk_id']}] [PAPER:{c['title']}; arXiv:{c['paper_id']}{c['version']}; PAGE:{c['page']}]\n{c['text']}"
            for c in state.get("retrieved_chunks", [])
        )
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are The Relay, a research-paper assistant. Answer ONLY from the supplied passages. "
                    "Do not use outside knowledge. Every substantive factual claim must cite one or more supplied "
                    "chunk IDs using exactly [[chunk_id]]. If the passages do not establish an answer, say that "
                    "the supplied papers do not provide enough information. Never invent page numbers or citations."
                ),
            },
            {
                "role": "user",
                "content": f"Question:\n{state['question']}\n\nPassages:\n{context}",
            },
        ]
        response, metric = await self._call_model("generate_answer", prompt)
        draft = response.text.strip()
        attempt = int(state.get("generation_attempts", 0)) + 1
        return {
            "draft_answer": draft,
            "draft_history": [{
                "attempt": attempt,
                "kind": "initial",
                "answer": draft,
            }],
            "generation_attempts": attempt,
            "tokens_in": int(state.get("tokens_in", 0)) + int(response.tokens_in or 0),
            "tokens_out": int(state.get("tokens_out", 0)) + int(response.tokens_out or 0),
            "model_first_token_ms": response.first_token_ms,
            "model_calls": [metric],
        }

    async def verify_answer(self, state: RelayState) -> dict[str, Any]:
        allowed_ids = _chunk_ids(state.get("retrieved_chunks", []))
        used_ids = _citation_markers(state.get("draft_answer", ""))
        invalid_ids = [cid for cid in used_ids if cid not in allowed_ids]
        if not used_ids or invalid_ids:
            feedback = (
                "The draft is missing citations or contains citation IDs "
                "not present in retrieved evidence."
            )
            event = {
                "check": "local_citation_precheck",
                "result": "fail",
                "feedback": feedback,
                "used_ids": used_ids,
                "invalid_ids": invalid_ids,
            }
            return {
                "verification_result": "fail",
                "verification_feedback": feedback,
                "verification_history": [event],
            }

        evidence = "\n\n".join(
            f"[[{c['chunk_id']}]] {c['text']}" for c in state.get("retrieved_chunks", [])
        )
        prompt = [
            {
                "role": "system",
                "content": (
                    "Verify whether the draft is fully supported by the supplied paper passages and whether its "
                    "claims are appropriately cited. Do not reward plausible outside knowledge. Return only PASS "
                    "or FAIL, followed by a short reason."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {state['question']}\n\nEvidence:\n{evidence}\n\nDraft:\n{state['draft_answer']}",
            },
        ]
        response, metric = await self._call_model("verify_answer", prompt)
        passed = response.text.strip().upper().startswith("PASS")
        feedback = response.text.strip()
        event = {
            "check": "llm_semantic_verification",
            "result": "pass" if passed else "fail",
            "feedback": feedback,
            "used_ids": used_ids,
        }
        return {
            "verification_result": "pass" if passed else "fail",
            "verification_feedback": feedback,
            "verification_history": [event],
            "tokens_in": int(state.get("tokens_in", 0)) + int(response.tokens_in or 0),
            "tokens_out": int(state.get("tokens_out", 0)) + int(response.tokens_out or 0),
            "model_calls": [metric],
        }

    async def regenerate_answer(self, state: RelayState) -> dict[str, Any]:
        context = "\n\n".join(
            f"[CHUNK_ID:{c['chunk_id']}] [PAPER:{c['title']}; arXiv:{c['paper_id']}{c['version']}; PAGE:{c['page']}]\n{c['text']}"
            for c in state.get("retrieved_chunks", [])
        )
        prompt = [
            {
                "role": "system",
                "content": (
                    "Revise the answer using ONLY supplied passages. Fix the verifier's concern. "
                    "Every factual claim must cite supplied chunk IDs as [[chunk_id]]. If evidence is insufficient, say so."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {state['question']}\n\nEvidence:\n{context}\n\nPrevious draft:\n{state['draft_answer']}"
                    f"\n\nVerifier feedback:\n{state.get('verification_feedback', '')}"
                ),
            },
        ]
        response, metric = await self._call_model("regenerate_answer", prompt)
        draft = response.text.strip()
        attempt = int(state.get("generation_attempts", 0)) + 1
        return {
            "draft_answer": draft,
            "draft_history": [{
                "attempt": attempt,
                "kind": "regeneration",
                "answer": draft,
            }],
            "generation_attempts": attempt,
            "tokens_in": int(state.get("tokens_in", 0)) + int(response.tokens_in or 0),
            "tokens_out": int(state.get("tokens_out", 0)) + int(response.tokens_out or 0),
            "model_first_token_ms": response.first_token_ms,
            "model_calls": [metric],
        }

    async def refuse_answer(self, state: RelayState) -> dict[str, Any]:
        reason = state.get("evidence_reason", "The available evidence did not pass verification.")
        if state.get("verification_result") == "fail":
            reason = state.get("verification_feedback") or reason
            answer = "I can't provide a reliable answer from the retrieved passages because the draft did not pass evidence and citation verification."
        else:
            answer = "The specified papers do not provide enough evidence to answer this question reliably. I won't guess."
        return {"final_answer": f"{answer}\n\nReason: {reason}", "status": "ok"}

    async def finalize_response(self, state: RelayState) -> dict[str, Any]:
        by_id = {str(c["chunk_id"]): c for c in state.get("retrieved_chunks", [])}
        citations: list[str] = []

        def replace(match: re.Match[str]) -> str:
            chunk_id = match.group(1)
            chunk = by_id.get(chunk_id)
            if not chunk:
                return "[invalid citation]"
            citation = f"{chunk['title']}, arXiv:{chunk['paper_id']}{chunk['version']}, p. {chunk['page']}"
            if citation not in citations:
                citations.append(citation)
            return f"[{citation}]"

        answer = re.sub(r"\[\[([A-Za-z0-9_.-]+)\]\]", replace, state.get("draft_answer", ""))
        return {"final_answer": answer, "citations": citations, "status": "ok"}

    async def handle_error(self, state: RelayState, exc: Exception) -> dict[str, Any]:
        if isinstance(exc, ModelCallError):
            status = "rate_limited" if exc.status_code == 429 else "timeout" if exc.status_code == 408 else "error"
            detail = f"{exc}; status_code={exc.status_code}; retry_after={exc.retry_after}"
            return {
                "status": status,
                "error_message": detail,
                "provider_error_headers": exc.response_headers,
                "provider_retry_after": exc.retry_after,
                "final_answer": "The assistant could not complete this request safely. Please try again later.",
            }
        detail = str(exc)
        return {
            "status": "error",
            "error_message": detail,
            "final_answer": "The assistant could not complete this request safely. Please try again later.",
        }

    def _build_graph(self):
        builder = StateGraph(RelayState)
        builder.add_node("validate_request", self.validate_request)
        builder.add_node("understand_question", self.understand_question)
        builder.add_node("retrieve_papers", self.retrieve_papers)
        builder.add_node("evaluate_evidence", self.evaluate_evidence)
        builder.add_node("prepare_retrieval_retry", self.prepare_retrieval_retry)
        builder.add_node("generate_answer", self.generate_answer)
        builder.add_node("verify_answer", self.verify_answer)
        builder.add_node("regenerate_answer", self.regenerate_answer)
        builder.add_node("refuse_answer", self.refuse_answer)
        builder.add_node("finalize_response", self.finalize_response)

        builder.add_edge(START, "validate_request")
        builder.add_conditional_edges(
            "validate_request",
            lambda s: "continue" if s.get("status") == "running" else "stop",
            {"continue": "understand_question", "stop": END},
        )
        builder.add_edge("understand_question", "retrieve_papers")
        builder.add_edge("retrieve_papers", "evaluate_evidence")
        builder.add_conditional_edges(
            "evaluate_evidence",
            self._route_evidence,
            {"generate": "generate_answer", "retry": "prepare_retrieval_retry", "refuse": "refuse_answer"},
        )
        builder.add_edge("prepare_retrieval_retry", "retrieve_papers")
        builder.add_edge("generate_answer", "verify_answer")
        builder.add_conditional_edges(
            "verify_answer",
            self._route_verification,
            {"finalize": "finalize_response", "regenerate": "regenerate_answer", "refuse": "refuse_answer"},
        )
        builder.add_edge("regenerate_answer", "verify_answer")
        builder.add_edge("finalize_response", END)
        builder.add_edge("refuse_answer", END)
        return builder.compile()

    @staticmethod
    def _route_evidence(state: RelayState) -> Literal["generate", "retry", "refuse"]:
        if state.get("evidence_sufficient"):
            return "generate"
        if int(state.get("retrieval_attempts", 0)) < 2:
            return "retry"
        return "refuse"

    @staticmethod
    def _route_verification(state: RelayState) -> Literal["finalize", "regenerate", "refuse"]:
        if state.get("verification_result") == "pass":
            return "finalize"
        if int(state.get("generation_attempts", 0)) < 2:
            return "regenerate"
        return "refuse"

    async def run(self, question: str, request_id: str | None = None, scenario: str = "manual") -> RelayState:
        now = datetime.now(timezone.utc).isoformat()
        start_perf = time.perf_counter()
        initial: RelayState = {
            "request_id": request_id or str(uuid4()),
            "question": question,
            "scenario": scenario,
            "retrieval_strategy": self.retrieval_strategy,
            "source": (
                "real" if self.model.provider_name == "groq"
                else "stand-in" if self.model.provider_name == "stand-in"
                else "stub"
            ),
            "started_at": now,
            "started_perf": start_perf,
            "retrieval_attempts": 0,
            "generation_attempts": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "model_calls": [],
            "draft_history": [],
            "verification_history": [],
            "node_timings_ms": {},
            "status": "running",
        }
        try:
            result = await self.graph.ainvoke(initial)
        except Exception as exc:  # Normalize unhandled provider/retrieval failures at request boundary.
            result = initial | await self.handle_error(initial, exc)
        total_ms = (time.perf_counter() - start_perf) * 1000
        result["total_ms"] = total_ms
        # For the current safe-release policy, response tokens are exposed only after verification.
        # This makes first_token_ms a user-visible measure, not the model's internal draft TTFT.
        result["first_token_ms"] = total_ms if result.get("final_answer") else None
        return result
