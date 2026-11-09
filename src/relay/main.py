from __future__ import annotations

import argparse
import asyncio

from dotenv import load_dotenv

from relay.config import get_settings
from relay.metrics import append_result
from relay.model_clients.factory import build_model_client
from relay.retrieval import PaperRetriever
from relay.workflow import RelayWorkflow


async def _run(question: str, debug_retrieval: bool = False, debug_workflow: bool = False) -> None:
    load_dotenv()
    settings = get_settings()
    model = build_model_client(settings)
    retriever = PaperRetriever(
        papers_dir=settings.papers_dir,
        chunk_words=settings.retrieval_chunk_words,
        overlap_words=settings.retrieval_overlap_words,
    )
    workflow = RelayWorkflow(
        model=model,
        retriever=retriever,
        top_k=settings.retrieval_top_k,
        min_score=settings.retrieval_min_score,
        retrieval_strategy=settings.retrieval_strategy,
        retrieval_candidates=settings.retrieval_candidates,
    )
    result = await workflow.run(question, scenario=settings.scenario)
    if result.get("source") in {"real", "stand-in"}:
        rerank_label = "yes" if settings.retrieval_strategy == "reranked" else "no"
        setting_label = (
            f"{settings.retrieval_strategy}_{settings.retrieval_chunk_words}w_"
            f"top{settings.retrieval_top_k}_rerank_{rerank_label}"
        )
        append_result(settings.results_path, result, retrieval_setting=setting_label)
    else:
        print("[development stub run: excluded from benchmark results.csv]")

    print("\n=== THE RELAY ===")
    print(result.get("final_answer", "No response was produced."))
    print("\n--- Run metrics ---")
    print(f"request_id: {result.get('request_id')}")
    print(f"status: {result.get('status')}")
    print(f"source: {result.get('source')}")
    print(f"papers indexed: {retriever.document_count}")
    print(f"retrieval strategy: {settings.retrieval_strategy}")
    print(f"retrieved chunks: {len(result.get('retrieved_chunks', []))}")
    if debug_workflow:
        print("\n=== WORKFLOW DEBUG ===")
        print("verification_result:", result.get("verification_result"))
        print("verification_feedback:", result.get("verification_feedback"))
        print("generation_attempts:", result.get("generation_attempts"))
        print("retrieval_attempts:", result.get("retrieval_attempts"))
        print("draft_answer:")
        print(result.get("draft_answer", "<no draft recorded>"))
        print("model_calls:")
        for call in result.get("model_calls", []):
            print(call)

    if debug_retrieval:
        print("\n=== RETRIEVAL DEBUG ===")
        for rank, chunk in enumerate(result.get("retrieved_chunks", []), start=1):
            print(f"\nRank: {rank}")
            print(f"Paper: {chunk.get('title')}")
            print(f"Paper ID: {chunk.get('paper_id')} / {chunk.get('version')}")
            print(f"Page: {chunk.get('page')}")
            print(f"BM25 score: {float(chunk.get('score', 0.0)):.4f}")
            if "retrieval_score" in chunk:
                print(f"Combined retrieval score: {float(chunk['retrieval_score']):.4f}")
            if "rerank_score" in chunk:
                print(f"Rerank score: {float(chunk['rerank_score']):.4f}")
            print(f"Paper affinity: {float(chunk.get('paper_affinity', 0.0)):.2f}")
            excerpt = " ".join(str(chunk.get("text", "")).split())[:500]
            print(f"Excerpt: {excerpt}")
    print(f"total_ms: {result.get('total_ms'):.2f}")
    print(f"tokens_in/out: {result.get('tokens_in')}/{result.get('tokens_out')}")
    if result.get("error_message"):
        print(f"error: {result['error_message']}")
    if result.get("source") in {"real", "stand-in"}:
        print(f"raw row appended to: {settings.results_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask The Relay a research-paper question.")
    parser.add_argument("--debug-retrieval", action="store_true", help="Print retrieved passage diagnostics")
    parser.add_argument("--debug-workflow", action="store_true", help="Print draft and verification diagnostics")
    parser.add_argument("question", nargs="+", help="Question to ask about the three-paper library")
    args = parser.parse_args()
    asyncio.run(_run(
        " ".join(args.question),
        debug_retrieval=args.debug_retrieval,
        debug_workflow=args.debug_workflow,
    ))


if __name__ == "__main__":
    main()
