# The Relay retrieval update

This is an overlay patch for the existing `the-relay` project. It contains only updated source, tests, docs, and the example environment file.

## Apply

1. From the existing project root, make a quick backup of the five source files:

   ```bash
   mkdir -p backup/pre-retrieval-update
   cp src/relay/{retrieval.py,workflow.py,state.py,config.py,main.py} backup/pre-retrieval-update/
   cp tests/test_workflow.py backup/pre-retrieval-update/ 2>/dev/null || true
   ```

2. Extract this archive into the project root, preserving its folder structure:

   ```bash
   unzip -o the-relay-retrieval-update.zip -d .
   ```

3. Keep your existing `.env` and API key. This archive does **not** contain `.env`, paper PDFs, or `results/results.csv`.

4. Add these settings to your existing `.env` if they are absent:

   ```env
   RETRIEVAL_STRATEGY=reranked
   RETRIEVAL_CANDIDATES=12
   ```

5. Run:

   ```bash
   python -m pytest -q
   python -m relay.main "Explain the Transformer architecture."
   ```

## Retrieval experiment settings

Use the same question set for all three settings and record the run labels from `results.csv`.

| Setting | `RETRIEVAL_STRATEGY` | `RETRIEVAL_CHUNK_WORDS` | `RETRIEVAL_TOP_K` | Re-ranking |
|---|---|---:|---:|---|
| A | `bm25` | 300 | 2 | No |
| B | `expanded` | 500 | 4 | No |
| C | `reranked` | 500 | 6 | Yes |

`expanded` uses a small, deterministic query-expansion map and a soft paper-topic prior. `reranked` adds lexical query-overlap scoring and a redundancy penalty. Neither is a semantic embedding model or an LLM reranker. The three papers remain searchable in every setting.

## Validation note

The source files passed Python bytecode compilation. The six retrieval unit tests passed in the build environment using a small BM25 test double because `rank-bm25` and LangGraph were not installed in that environment. Run the full project test suite in your local `.venv` after applying the patch; do not treat the local tests as passed until that command succeeds.
