"""Application configuration loaded from environment variables."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["ollama", "openai", "azure", "anthropic"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM ---
    # Default to a fully local, open-source stack served by Ollama.
    # See https://ollama.com for installation & supported models.
    llm_provider: LLMProvider = "ollama"
    llm_model: str = "qwen2.5-coder:7b"
    llm_vision_model: str = "minicpm-v"
    embedding_model: str = "nomic-embed-text"

    # OpenAI-compatible providers (also used for Azure / OpenAI-compatible gateways).
    openai_api_key: str = Field(default="", repr=False)
    openai_base_url: str = ""

    # Ollama exposes an OpenAI-compatible REST API at /v1.
    # When `llm_provider == "ollama"` this base URL is used and `openai_api_key`
    # is ignored (Ollama does not require auth for local installs).
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_request_timeout: float = 600.0  # seconds; CPU inference can be slow

    # --- Vector store ---
    chroma_persist_dir: str = "./.chroma"
    collection_name: str = "analyst_corpus"

    # --- Guardrails ---
    grounding_min_score: float = 0.55
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    enable_nli: bool = True

    # --- Service ---
    max_file_mb: int = 50
    log_level: str = "INFO"
    app_env: str = "dev"

    # ------------------------------------------------------------------
    @property
    def effective_base_url(self) -> str:
        """Resolve the OpenAI-compatible base URL for the active provider."""
        if self.llm_provider == "ollama":
            return self.ollama_base_url
        return self.openai_base_url

    @property
    def effective_api_key(self) -> str:
        """Ollama ignores the API key but the OpenAI SDK requires *some* value."""
        if self.llm_provider == "ollama":
            return "ollama"
        return self.openai_api_key or "sk-not-set"


@lru_cache
def get_settings() -> Settings:
    return Settings()
