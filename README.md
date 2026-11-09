# The Relay

A citation-grounded research assistant built around LangGraph. The same workflow can call either Groq or a local deterministic stub while the retrieval and experiment infrastructure is being developed.

## Current implementation status

- [x] LangGraph StateGraph with explicit conditional routing
- [x] Bounded retrieval retry (maximum two retrieval attempts)
- [x] Answer verification and one regeneration attempt
- [x] Groq async streaming adapter with first-token timing and provider usage capture
- [x] Offline deterministic stub for workflow smoke tests
- [x] Version-pinned PDF loading and BM25 retrieval
- [x] Three selectable retrieval settings: BM25 baseline, expanded paper-aware retrieval, deterministic reranking
- [x] Retrieval diagnostics: original/expanded overlap, paper affinity and ranking score
- [x] Request-level CSV logging with the assignment's required columns
- [ ] Download and validate the three exact arXiv PDFs
- [ ] Dedicated FastAPI stand-in service with configurable latency, failures, quotas and chunked streaming
- [ ] Admission control, bounded queue and deadline management
- [ ] Load scenarios, 15+ question evaluation and report generation

## Architecture

- **LangGraph:** orchestration, state transitions, conditional edges and bounded loops.
- **Python services:** retrieval, model-provider adapters, metrics, rate-limit handling and benchmarking.
- **Groq adapter:** async streaming chat completions. `GROQ_API_KEY` is read from the environment.
- **Stub adapter:** deterministic local responses for development. It is not the calibrated load simulator.
- **Retriever:** PDF page extraction and overlapping word-window chunks. Select `RETRIEVAL_STRATEGY=bm25`, `expanded`, or `reranked`. The latter two use deterministic, auditable query expansion and paper-topic hints; `reranked` also applies lexical overlap and a small redundancy penalty. This is not embedding-based semantic search. Every chunk retains paper ID, exact version and 1-based PDF page number.

The graph intentionally does not stream an unverified draft to the user. Accordingly, `first_token_ms` in the current CSV is the user-visible release time after verification; `model_first_token_ms` is retained inside the graph result for the model's internal generation TTFT. We will keep these metrics distinct during benchmarking.

## Setup

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -e .[dev]
cp .env.example .env
```

Download the exact paper versions:

```bash
python scripts/download_papers.py
```

The download script uses the versioned arXiv PDF endpoints specified by the assignment. Check the printed file sizes and keep the three version suffixes unchanged.

## Retrieval experiments

Set these values in `.env` before each experiment. Keep the question set and all other settings fixed within a comparison.

| Setting | `RETRIEVAL_STRATEGY` | `RETRIEVAL_CHUNK_WORDS` | `RETRIEVAL_TOP_K` | Meaning |
|---|---|---:|---:|---|
| A | `bm25` | 300 | 2 | Original global BM25 baseline; no expansion or reranking |
| B | `expanded` | 500 | 4 | Deterministic query expansion plus paper-topic prior |
| C | `reranked` | 500 | 6 | Setting B plus lexical overlap and redundancy-aware reranking |

For each setting, preserve the resulting CSV rows and label them separately. The query expansion and paper-topic hints are explicit heuristics, not semantic embeddings; report their measured quality and failure cases honestly.

Example for setting B:

```env
RETRIEVAL_STRATEGY=expanded
RETRIEVAL_CHUNK_WORDS=500
RETRIEVAL_OVERLAP_WORDS=60
RETRIEVAL_TOP_K=4
RETRIEVAL_CANDIDATES=12
```

## Run an offline workflow smoke test

The stub backend requires no API key:

```bash
python -m relay.main "What is the Transformer architecture?"
```

The first run will refuse if the PDFs have not been downloaded. Once the PDFs are in `papers/`, it will retrieve passages and exercise the graph. The stub is intentionally not treated as a real answer-quality model.

## Run against Groq

1. Create your own Groq account and API key.
2. Put the key in `.env` as `GROQ_API_KEY=...`.
3. Set `MODEL_BACKEND=groq`.
4. Confirm the model is currently available to your account and set `GROQ_MODEL` accordingly.

```bash
python -m relay.main "What is the Transformer architecture?"
```

Never commit `.env` or include the API key in the submission.

## Required CSV

Each completed manual request appends one row to `results/results.csv`. The columns match the assignment exactly. `answer_grade` is left blank until the human quality-evaluation pass assigns one of:

- `correct`
- `wrong`
- `refused_correctly`
- `refused_wrongly`

Do not treat blank grades as evaluated answers.

## Notes on measurements

Groq returns rate-limit information in HTTP headers; the adapter preserves response headers on normalized errors where exposed by the SDK. We will add explicit request/response header capture and the controlled simulator in the next phase. Groq's account-level limits vary by model and plan, so use the account dashboard and actual response headers rather than copying a generic limit into the report.
