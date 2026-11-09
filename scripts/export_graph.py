from __future__ import annotations

from relay.config import get_settings
from relay.model_clients.factory import build_model_client
from relay.retrieval import PaperRetriever
from relay.workflow import RelayWorkflow


def main() -> None:
    settings = get_settings()
    model = build_model_client(settings)
    retriever = PaperRetriever(settings.papers_dir, settings.retrieval_chunk_words, settings.retrieval_overlap_words)
    workflow = RelayWorkflow(model, retriever, settings.retrieval_top_k, settings.retrieval_min_score)
    output = workflow.graph.get_graph().draw_mermaid()
    path = settings.results_path.parent.parent / "docs" / "workflow.mmd"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(output, encoding="utf-8")
    print(f"Saved Mermaid graph to {path}")


if __name__ == "__main__":
    main()
