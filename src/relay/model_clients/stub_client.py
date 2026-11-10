from __future__ import annotations

import json
import re
import time

from relay.model_clients.base import ModelResponse


class StubModelClient:
    """Deterministic offline adapter for graph development, not the load simulator."""

    provider_name = "stub"
    model_name = "local-deterministic-stub"

    async def complete(
        self, messages: list[dict[str, str]], purpose: str = "completion"
    ) -> ModelResponse:
        started = time.perf_counter()
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        low = (system + " " + user).lower()

        if "retrieval_query" in low or "classify the question" in low:
            question = user.split("Question:", 1)[-1].strip()
            text = json.dumps({
                "intent": "research_question",
                "complexity": "normal",
                "retrieval_query": question,
            })
        elif "verify" in low or "grounded" in low and "return only" in low:
            text = "PASS"
        else:
            # Return a transparent response with a real chunk marker if supplied.
            match = re.search(r"\[CHUNK_ID:([A-Za-z0-9_.-]+)\]", user)
            if match:
                text = (
                    "The local stub can confirm that a passage was retrieved, but it does not "
                    "generate a substantive research answer. [[" + match.group(1) + "]]"
                )
            else:
                text = "I cannot answer this from the indexed paper passages."

        elapsed_ms = (time.perf_counter() - started) * 1000
        return ModelResponse(
            text=text,
            tokens_in=max(1, len((system + user).split())),
            tokens_out=max(1, len(text.split())),
            first_token_ms=elapsed_ms,
            elapsed_ms=elapsed_ms,
            provider=self.provider_name,
            model=self.model_name,
        )
