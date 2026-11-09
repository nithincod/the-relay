# Decision 001: Use LangGraph for agent orchestration

**Status:** Accepted

**Decision:** Use LangGraph `StateGraph` for the research assistant's workflow. Keep retrieval, provider clients, rate limiting, queues, deadlines, metrics, and benchmark runners as ordinary Python services/functions.

**Why:** The workflow has explicit conditional routes and two bounded cycles: a retrieval retry and a verification/regeneration retry. A shared state object is useful for passing the question, evidence, citations, retry counters, token usage, and status across nodes. LangGraph makes those routes inspectable and easier to modify as experiments introduce budget-aware retrieval or provider fallbacks.

**Costs:** More concepts and dependencies than a sequential Python function chain; graph state and node boundaries must be designed deliberately. LangGraph does not itself enforce Groq RPM/TPM/RPD/TPD limits or provide production admission control.

**Trade-off:** Accept modest orchestration complexity in exchange for visible, testable control flow. Do not use LangChain abstractions for every component; direct Groq SDK and plain Python services keep the implementation small.
