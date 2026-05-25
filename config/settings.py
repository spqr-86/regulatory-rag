from pydantic_settings import BaseSettings, SettingsConfigDict

from .constants import ALLOWED_TYPES, MAX_FILE_SIZE, MAX_TOTAL_SIZE


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM (fallback provider when path-specific overrides not set)
    LLM_PROVIDER: str = "gemini"  # supported: gemini | openai
    MODEL_NAME: str = "gpt-4o-mini"  # used when LLM_PROVIDER=openai
    TEMPERATURE: float = 0.0

    # Embeddings
    EMBEDDING_PROVIDER: str = "openai"  # варианты: openai, hf_api, local
    EMBEDDING_MODEL_NAME: str = "text-embedding-3-small"

    # Параметры для FlashRank
    RERANKING_MODEL: str = "ms-marco-MiniLM-L-12-v2"
    FLASHRANK_CACHE_DIR: str = ".flashrank_cache"

    # Параметры для индексации и обработки документов
    MAX_FILE_SIZE: int = MAX_FILE_SIZE
    MAX_TOTAL_SIZE: int = MAX_TOTAL_SIZE
    ALLOWED_TYPES: list[str] = ALLOWED_TYPES

    CHROMA_DB_PATH: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "documents"

    # Vector store backend — currently supported: chroma
    # Future: qdrant, pgvector (see docs/plans/ for roadmap)
    VECTOR_STORE: str = "chroma"

    VECTOR_SEARCH_K: int = 10
    HYBRID_RETRIEVER_WEIGHTS: list[float] = [0.6, 0.4]

    LOG_LEVEL: str = "INFO"

    REQUEST_TIMEOUT: float = 120.0

    CACHE_DIR: str = "document_cache"
    CACHE_EXPIRE_DAYS: int = 7

    CHUNK_SIZE: int = 1500
    CHUNK_OVERLAP: int = 400

    SOURCE_DOCS_PATH: str = "./source_docs"

    # Google Gemini
    GEMINI_API_KEY: str = ""
    GEMINI_FAST_MODEL: str = "gemini-3-flash"  # fallback if path-specific model not set

    # Per-path provider/model — override to switch models independently.
    # Falls back to LLM_PROVIDER + GEMINI_FAST_MODEL when empty.
    SIMPLE_LLM_PROVIDER: str = "gemini"
    SIMPLE_MODEL_NAME: str = "gemini-2.5-flash"
    COMPLEX_LLM_PROVIDER: str = "gemini"
    COMPLEX_MODEL_NAME: str = "gemini-3-flash-preview"

    # Agent workflow settings
    THINKING_BUDGET: int = 8192
    THINKING_VERIFIER: int = 1024
    MAX_REVISIONS: int = 1
    MAX_AGENT_STEPS: int = 16
    MAX_SEARCH_CALLS: int = 2
    MAX_VISUAL_PROOF_CALLS: int = 1
    MAX_VISUAL_PROOFS: int = 3  # How many chunks to process for visual proof

    # RAG node specific settings
    MIN_CHUNK_LENGTH_FOR_FILTERING: int = (
        50  # Minimum length for a chunk to be considered relevant after filtering
    )
    SIMILARITY_THRESHOLD_ACCEPTANCE: float = (
        0.10  # Minimum similarity to consider results found (max across all results)
    )
    MAX_CHUNKS_FOR_LLM: int = 10  # Maximum chunks passed to LLM context
    SIMILARITY_THRESHOLD_FOR_VERIFIER_SKIP: float = (
        0.85  # If similarity score is above this, skip verifier in simple RAG
    )

    # V7 graph toggle
    USE_V7_GRAPH: bool = False

    # динамически подставим суффикс по провайдеру,
    # если путь не задан через .env
    def model_post_init(self, __context) -> None:
        if self.CHROMA_DB_PATH == "./chroma_db":
            object.__setattr__(
                self, "CHROMA_DB_PATH", f"chroma_db_{self.LLM_PROVIDER.lower()}"
            )


settings = Settings()
