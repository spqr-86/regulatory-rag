from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    TEMPERATURE: float = 0.0

    # Embeddings
    EMBEDDING_PROVIDER: str = "openai"  # варианты: openai, hf_api, local
    EMBEDDING_MODEL_NAME: str = "text-embedding-3-small"

    # Reranker (flashrank | crossencoder)
    RERANKER_BACKEND: str = "crossencoder"
    RERANKING_MODEL: str = "ms-marco-MiniLM-L-12-v2"
    CROSSENCODER_MODEL: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    FLASHRANK_CACHE_DIR: str = ".flashrank_cache"

    # Indexing
    MAX_FILE_SIZE: int = 50 * 1024 * 1024
    MAX_TOTAL_SIZE: int = 200 * 1024 * 1024
    ALLOWED_TYPES: list[str] = [".txt", ".pdf", ".docx", ".md"]
    SOURCE_DOCS_PATH: str = "./source_docs"
    # Department Q&A snapshot metadata (issue #36); empty keeps legacy indexing
    CORPUS_MANIFEST_PATH: str = ""
    # Department Q&A object-profile mode (issue #44). One switch picks the whole bundle:
    # Chroma path + collection, profiles, prompt version, output schema. Separate paths,
    # because index.py drops the whole CHROMA_DB_PATH before writing.
    DEPARTMENT_QA_MODE: Literal["v1", "v2"] = "v2"
    DEPARTMENT_V1_CHROMA_DB_PATH: str = "./chroma_db_dept"
    DEPARTMENT_V1_COLLECTION: str = "department_demo"
    DEPARTMENT_V2_CHROMA_DB_PATH: str = "./chroma_db_dept_v2"
    DEPARTMENT_V2_COLLECTION: str = "department_demo_v2"
    DEPARTMENT_RETRIEVAL_TIMEOUT_S: float = 120.0
    DEPARTMENT_RETRIEVAL_WORKERS: int = 4
    DEPARTMENT_RETRIEVAL_PENDING: int = 8
    CHUNK_SIZE: int = 1200
    CACHE_DIR: str = "document_cache"
    CACHE_EXPIRE_DAYS: int = 7
    # UI reindex button rebuilds the whole index; off for public deployments (issue #32)
    ENABLE_UI_REINDEX: bool = False

    # Vector store
    CHROMA_DB_PATH: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "documents"
    # Generic normative search keeps its own store even when the root UI is
    # launched with CHROMA_DB_PATH pointed at the Department V2 collection.
    GENERIC_CHROMA_DB_PATH: str = "./chroma_db"
    GENERIC_CHROMA_COLLECTION: str = "documents"
    GENERIC_QUERY_WORKERS: int = Field(default=2, gt=0)
    GENERIC_QUERY_PENDING: int = Field(default=2, ge=0)
    VECTOR_STORE: str = "chroma"

    # HTTP
    REQUEST_TIMEOUT: float = 120.0

    # Per-path model config — change independently without touching other paths
    # Showcase default (17.09.2026): cheapest model that passed all trap-set questions,
    # see eval/runs/object_profile_traps_2026-09-17/summary.md. Override in .env freely.
    SIMPLE_LLM_PROVIDER: str = "openrouter"
    SIMPLE_MODEL_NAME: str = "deepseek/deepseek-v4.1-flash"
    COMPLEX_LLM_PROVIDER: str = "openrouter"
    COMPLEX_MODEL_NAME: str = "deepseek/deepseek-v4.1-flash"
    # OpenRouter request controls. None = field not sent, provider default applies
    # (DeepSeek V4.1 Flash: reasoning on, effort "high"). thinking_budget is
    # Gemini-only; for OpenRouter these are the knobs that actually reach the model.
    # Effort default "low" (23.09.2026): retrieval unaffected (expand runs with
    # reasoning off), answers faster and cheaper; see
    # eval/runs/openrouter_reasoning_ab_2026-09-23/. Set "high" for provider behaviour.
    OPENROUTER_REASONING_EFFORT: (
        Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None
    ) = "low"
    OPENROUTER_MAX_TOKENS: int | None = Field(default=None, gt=0)
    OPENROUTER_PROVIDER_SORT: Literal["latency", "throughput", "price"] | None = None
    # Eval judge — independent from pipeline provider
    JUDGE_LLM_PROVIDER: str = "openai"
    JUDGE_MODEL_NAME: str = "gpt-4o"

    # V7 node limits
    MAX_SEARCH_CALLS: int = 2
    MAX_VISUAL_PROOF_CALLS: int = 1
    MAX_VISUAL_PROOFS: int = 1  # was 3; each attempt costs up to 3s timeout on VPS


settings = Settings()
