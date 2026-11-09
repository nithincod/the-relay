from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_backend: str = "stub"
    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-20b"
    groq_timeout_seconds: float = 30.0

    retrieval_chunk_words: int = 500
    retrieval_overlap_words: int = 60
    retrieval_top_k: int = 4
    retrieval_min_score: float = 0.01  # Retained for compatibility / diagnostics.
    retrieval_strategy: str = "reranked"  # bm25 | expanded | reranked
    retrieval_candidates: int = 12

    results_csv: str = "results/results.csv"
    scenario: str = "manual"

    @property
    def papers_dir(self) -> Path:
        return ROOT_DIR / "papers"

    @property
    def results_path(self) -> Path:
        p = Path(self.results_csv)
        return p if p.is_absolute() else ROOT_DIR / p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
