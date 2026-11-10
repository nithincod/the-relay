from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from relay.config import ROOT_DIR, get_settings
from relay.metrics import append_result
from relay.model_clients.factory import build_model_client
from relay.retrieval import PaperRetriever
from relay.workflow import RelayWorkflow


STRATEGIES = ("bm25", "expanded", "reranked")

# Keep candidate_k=40 identical to the local evaluation just completed.
CANDIDATE_K = 40
CHUNK_WORDS = 500
CHUNK_OVERLAP = 60
TOP_K = 4

# Conservative limits for this experiment.
DEFAULT_DELAY_SECONDS = 75
MAX_TOKENS_PER_RUN = 7500
MAX_EXPERIMENT_TOKENS = 100_000


QUESTIONS: list[dict[str, Any]] = [
    {
        "id": "Q1",
        "question": "What are the two sublayers in a Transformer encoder layer?",
        "expected_refs": ["1706.03762-v7-p003-c01"],
        "answer_guide": (
            "Multi-head self-attention and a position-wise "
            "fully connected feed-forward network."
        ),
    },
    {
        "id": "Q2",
        "question": "What is the purpose of positional encodings?",
        "expected_refs": ["1706.03762-v7-p006-c01"],
        "answer_guide": (
            "They inject relative or absolute token-position information "
            "because the Transformer has no recurrence or convolution."
        ),
    },
    {
        "id": "Q3",
        "question": (
            "What role does the Mimi codec play in Moshi's "
            "speech-generation pipeline?"
        ),
        "expected_refs": ["2410.00037-v1-p006-c01"],
        "answer_guide": (
            "Mimi is a neural audio codec using residual vector quantization "
            "and semantic information; it converts audio to discrete tokens "
            "and back."
        ),
    },
    {
        "id": "Q4",
        "question": "How is the Helium language model used in Moshi?",
        "expected_refs": [
            "2410.00037-v1-p007-c01",
            "2410.00037-v1-p003-c01",
        ],
        "answer_guide": (
            "Helium is Moshi's text LLM backbone and provides text knowledge "
            "and reasoning abilities."
        ),
    },
    {
        "id": "Q5",
        "question": (
            "What is H-Net, and how does it organize sequence "
            "representations hierarchically?"
        ),
        "expected_refs": ["2507.07955-v2-p004-c01"],
        "answer_guide": (
            "H-Net recursively groups low-level sequence elements into "
            "higher-level abstractions such as words, phrases and sentences."
        ),
    },
    {
        "id": "Q6",
        "question": (
            "How does Dynamic Chunking determine where to create "
            "chunk boundaries?"
        ),
        "expected_refs": [
            "2507.07955-v2-p003-c01",
            "2507.07955-v2-p006-c02",
        ],
        "answer_guide": (
            "The routing module predicts boundary probabilities; boundary "
            "indicators determine which vectors the downsampler retains."
        ),
    },
    {
        "id": "Q7",
        "question": (
            "According to these papers, what effect does quantum computing "
            "have on agricultural yields in Andhra Pradesh?"
        ),
        "expected_refs": [],
        "answer_guide": (
            "Correctly refuse: the three papers do not provide evidence "
            "about this topic."
        ),
    },
]


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
    return records


def read_csv_request_ids(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()

    with path.open("r", newline="", encoding="utf-8") as f:
        return {
            row.get("request_id", "")
            for row in csv.DictReader(f)
            if row.get("request_id")
        }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        f.flush()


def retrieval_label(strategy: str) -> str:
    rerank = "yes" if strategy == "reranked" else "no"
    return (
        f"{strategy}_{CHUNK_WORDS}w_top{TOP_K}_cand{CANDIDATE_K}_"
        f"rerank_{rerank}"
    )


def build_record(
    question: dict[str, Any],
    strategy: str,
    request_id: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    chunks = []
    for chunk in result.get("retrieved_chunks", []):
        fields = (
            "chunk_id", "paper_id", "version", "title", "page", "text",
            "score", "expanded_score", "retrieval_score", "rerank_score",
            "paper_affinity", "query_overlap", "expanded_query_overlap",
        )
        chunks.append({key: chunk.get(key) for key in fields if key in chunk})

    # Preserve exactly the state fields needed to reconstruct a missing CSV row.
    csv_state = {
        "request_id": request_id,
        "scenario": "retrieval_eval",
        "source": result.get("source", "unknown"),
        "started_at": result.get("started_at", ""),
        "first_token_ms": result.get("first_token_ms"),
        "total_ms": result.get("total_ms"),
        "status": result.get("status", "error"),
        "retrieval_attempts": result.get("retrieval_attempts", 0),
        "generation_attempts": result.get("generation_attempts", 0),
        "tokens_in": result.get("tokens_in", 0),
        "tokens_out": result.get("tokens_out", 0),
        "model_calls": result.get("model_calls", []),
    }

    return {
        "experiment": "retrieval_quality_v1",
        "question_id": question["id"],
        "question": question["question"],
        "strategy": strategy,
        "retrieval_setting": retrieval_label(strategy),
        "expected_reference_chunk_ids": question["expected_refs"],
        "answer_guide": question["answer_guide"],
        "request_id": request_id,
        "source": result.get("source"),
        "status": result.get("status"),
        "final_answer": result.get("final_answer", ""),
        "draft_answer": result.get("draft_answer", ""),
        "verification_result": result.get("verification_result"),
        "verification_feedback": result.get("verification_feedback"),
        "first_token_ms": result.get("first_token_ms"),
        "total_ms": result.get("total_ms"),
        "tokens_in": result.get("tokens_in", 0),
        "tokens_out": result.get("tokens_out", 0),
        "retrieval_attempts": result.get("retrieval_attempts", 0),
        "generation_attempts": result.get("generation_attempts", 0),
        "model_calls": result.get("model_calls", []),
        "verification_history": result.get("verification_history", []),
        "retrieved_chunks": chunks,
        "answer_grade": None,
        "csv_state": csv_state,
    }


async def run_experiment(args: argparse.Namespace) -> None:
    load_dotenv()
    settings = get_settings()

    output_path = resolve_path(args.output)
    results_path = settings.results_path

    selected_ids = set(args.question_ids or [q["id"] for q in QUESTIONS])
    selected_questions = [
        q for q in QUESTIONS if q["id"] in selected_ids
    ]

    plan = [
        (question, strategy)
        for question in selected_questions
        for strategy in STRATEGIES
    ]
    if args.limit is not None:
        plan = plan[:args.limit]

    if args.dry_run:
        print(f"Backend configured: {settings.model_backend}")
        print(f"Questions: {len(QUESTIONS)}")
        print(f"Strategies: {', '.join(STRATEGIES)}")
        print(f"Planned runs: {len(plan)}")
        print(
            f"Settings: chunks={CHUNK_WORDS}, overlap={CHUNK_OVERLAP}, "
            f"top_k={TOP_K}, candidate_k={CANDIDATE_K}"
        )
        print(f"Start-to-start delay: {args.delay_seconds}s")
        print(
            "Approximate schedule duration: "
            f"{max(0, len(plan) - 1) * args.delay_seconds / 60:.1f} minutes"
        )
        for question, strategy in plan:
            print(f"{question['id']} / {strategy}")
        print(f"Detailed output: {output_path}")
        print(f"CSV output: {results_path}")
        return

    if settings.model_backend.lower() != "groq":
        raise SystemExit(
            "Refusing to run: set MODEL_BACKEND=groq. "
            "No API key or request was printed."
        )
    if not settings.groq_api_key:
        raise SystemExit("GROQ_API_KEY is missing; no API request was made.")
    if args.delay_seconds < 65:
        raise SystemExit(
            "For quota safety, delay must be at least 65 seconds."
        )

    existing = read_jsonl(output_path)
    if existing and not args.resume:
        raise SystemExit(
            f"{output_path} already contains records. Use --resume to "
            "continue without repeating completed runs."
        )

    completed: set[tuple[str, str]] = set()
    jsonl_request_ids: set[str] = set()
    prior_tokens = 0

    for record in existing:
        key = (record["question_id"], record["strategy"])
        if key in completed:
            raise SystemExit(f"Duplicate experiment record found: {key}")
        completed.add(key)
        jsonl_request_ids.add(record["request_id"])
        prior_tokens += int(record.get("tokens_in", 0))
        prior_tokens += int(record.get("tokens_out", 0))

    csv_request_ids = read_csv_request_ids(results_path)
    evaluation_csv_ids = {
        rid for rid in csv_request_ids if rid.startswith("retrieval-eval-")
    }
    orphaned = evaluation_csv_ids - jsonl_request_ids
    if orphaned:
        raise SystemExit(
            "Found retrieval-evaluation CSV rows without matching JSONL "
            f"records. Reconcile before continuing: {sorted(orphaned)}"
        )

    # Repair a crash that wrote the JSONL record but not its CSV row.
    for record in existing:
        rid = record["request_id"]
        if rid not in csv_request_ids:
            append_result(
                results_path,
                record["csv_state"],
                retrieval_setting=record["retrieval_setting"],
            )
            csv_request_ids.add(rid)

    remaining = [
        (question, strategy)
        for question, strategy in plan
        if (question["id"], strategy) not in completed
    ]

    if not remaining:
        print("All planned question/strategy pairs are already recorded.")
        print(f"Detailed output: {output_path}")
        return

    print(f"Backend: Groq ({settings.groq_model})")
    print(f"Previously completed: {len(completed)}")
    print(f"Runs remaining: {len(remaining)}")
    print(f"Starting token budget for this experiment: {prior_tokens:,}")
    print(f"Maximum additional experiment tokens: {args.max_tokens:,}")
    print(f"Start-to-start delay: {args.delay_seconds}s")
    print("Answers are withheld until the workflow verification step completes.")

    model = build_model_client(settings)
    if model.provider_name != "groq":
        raise SystemExit("Model client is not Groq; refusing to run benchmark.")

    retriever = PaperRetriever(
        papers_dir=settings.papers_dir,
        chunk_words=CHUNK_WORDS,
        overlap_words=CHUNK_OVERLAP,
    )

    workflows = {
        strategy: RelayWorkflow(
            model=model,
            retriever=retriever,
            top_k=TOP_K,
            min_score=settings.retrieval_min_score,
            retrieval_strategy=strategy,
            retrieval_candidates=CANDIDATE_K,
        )
        for strategy in STRATEGIES
    }

    experiment_tokens = prior_tokens
    next_start = time.monotonic()

    for index, (question, strategy) in enumerate(remaining, start=1):
        if experiment_tokens >= args.max_tokens:
            print("Stopping before next run: cumulative token budget reached.")
            break

        wait = next_start - time.monotonic()
        if wait > 0:
            print(f"\nQuota pacing: sleeping {wait:.1f}s before next workflow.")
            await asyncio.sleep(wait)

        started = time.monotonic()
        request_id = (
            f"retrieval-eval-{question['id'].lower()}-{strategy}"
        )

        print(
            f"\n[{index}/{len(remaining)}] "
            f"{question['id']} / {strategy}"
        )
        result = await workflows[strategy].run(
            question["question"],
            request_id=request_id,
            scenario="retrieval_eval",
        )

        if result.get("source") != "real":
            raise RuntimeError(
                f"Unexpected result source for {request_id}: "
                f"{result.get('source')}"
            )

        record = build_record(question, strategy, request_id, result)

        # Append the required CSV row and then the richer JSONL record.
        append_result(
            results_path,
            result,
            retrieval_setting=retrieval_label(strategy),
        )
        append_jsonl(output_path, record)

        used = int(result.get("tokens_in", 0)) + int(
            result.get("tokens_out", 0)
        )
        experiment_tokens += used

        print(f"request_id: {request_id}")
        print(f"status: {result.get('status')}")
        print(f"workflow release latency: {result.get('first_token_ms')} ms")
        print(f"workflow total latency: {result.get('total_ms')} ms")
        print(f"tokens in/out: {result.get('tokens_in')}/{result.get('tokens_out')}")
        print(f"tokens used this run: {used:,}")
        print(f"cumulative experiment tokens: {experiment_tokens:,}")
        print(f"answer:\n{result.get('final_answer', '')[:1600]}")

        if result.get("status") != "ok":
            print(
                "Stopping after a failed run. Its metrics and diagnostics "
                "have been preserved; inspect the provider response before "
                "resuming."
            )
            break

        if used > MAX_TOKENS_PER_RUN:
            print(
                f"Stopping: this workflow used more than "
                f"{MAX_TOKENS_PER_RUN:,} tokens."
            )
            break

        if experiment_tokens >= args.max_tokens:
            print("Stopping: the experiment token budget has been reached.")
            break

        next_start = started + args.delay_seconds

    print("\nExperiment runner finished.")
    print(f"Detailed JSONL: {output_path}")
    print(f"Assignment CSV: {results_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run paced, fixed-question Groq retrieval evaluation."
    )
    parser.add_argument(
        "--question-ids",
        nargs="+",
        choices=[q["id"] for q in QUESTIONS],
        default=None,
        help="Optional subset of fixed evaluation questions.",
    )
    parser.add_argument(
        "--output",
        default="results/retrieval_eval_runs.jsonl",
        help="Detailed JSONL output path, relative to project root by default.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=int,
        default=DEFAULT_DELAY_SECONDS,
        help="Minimum start-to-start delay between workflow runs (>=65).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_EXPERIMENT_TOKENS,
        help="Maximum cumulative input+output tokens for this experiment.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of pending runs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing JSONL records without repeating completed pairs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without creating a model client or making API calls.",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1.")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive.")

    asyncio.run(run_experiment(args))


if __name__ == "__main__":
    main()
