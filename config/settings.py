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

    # FlashRank reranker
    RERANKING_MODEL: str = "ms-marco-MiniLM-L-12-v2"
    FLASHRANK_CACHE_DIR: str = ".flashrank_cache"

    # Indexing
    MAX_FILE_SIZE: int = 50 * 1024 * 1024
    MAX_TOTAL_SIZE: int = 200 * 1024 * 1024
    ALLOWED_TYPES: list[str] = [".txt", ".pdf", ".docx", ".md"]
    SOURCE_DOCS_PATH: str = "./source_docs"
    CHUNK_SIZE: int = 1200
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
    # Eval judge — independent from pipeline provider
    JUDGE_LLM_PROVIDER: str = "openai"
    JUDGE_MODEL_NAME: str = "gpt-4o-mini"

    # V7 node limits
    MAX_SEARCH_CALLS: int = 2
    MAX_VISUAL_PROOF_CALLS: int = 1
    MAX_VISUAL_PROOFS: int = 3


settings = Settings()
