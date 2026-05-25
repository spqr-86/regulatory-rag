from pydantic_settings import BaseSettings, SettingsConfigDict

from .constants import ALLOWED_TYPES, MAX_FILE_SIZE, MAX_TOTAL_SIZE


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

    # FlashRank reranker
    RERANKING_MODEL: str = "ms-marco-MiniLM-L-12-v2"
    FLASHRANK_CACHE_DIR: str = ".flashrank_cache"

    # Indexing
    MAX_FILE_SIZE: int = MAX_FILE_SIZE
    MAX_TOTAL_SIZE: int = MAX_TOTAL_SIZE
    ALLOWED_TYPES: list[str] = ALLOWED_TYPES
    SOURCE_DOCS_PATH: str = "./source_docs"
    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 150
    CACHE_DIR: str = "document_cache"
    CACHE_EXPIRE_DAYS: int = 7

    # Vector store
    CHROMA_DB_PATH: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "documents"
    VECTOR_STORE: str = "chroma"

    # HTTP
    REQUEST_TIMEOUT: float = 120.0

    # Per-path model config — change independently without touching other paths
    SIMPLE_LLM_PROVIDER: str = "gemini"
    SIMPLE_MODEL_NAME: str = "gemini-2.5-flash"
    COMPLEX_LLM_PROVIDER: str = "gemini"
    COMPLEX_MODEL_NAME: str = "gemini-3-flash-preview"

    # V7 node limits
    MAX_SEARCH_CALLS: int = 2
    MAX_VISUAL_PROOF_CALLS: int = 1
    MAX_VISUAL_PROOFS: int = 3


settings = Settings()
