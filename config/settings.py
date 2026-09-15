from typing import Literal

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
    DEPARTMENT_QA_MODE: Literal["v1", "v2"] = "v1"
    DEPARTMENT_V1_CHROMA_DB_PATH: str = "./chroma_db_dept"
    DEPARTMENT_V1_COLLECTION: str = "department_demo"
    DEPARTMENT_V2_CHROMA_DB_PATH: str = "./chroma_db_dept_v2"
    DEPARTMENT_V2_COLLECTION: str = "department_demo_v2"
    CHUNK_SIZE: int = 1200
    CACHE_DIR: str = "document_cache"
    CACHE_EXPIRE_DAYS: int = 7
    # UI reindex button rebuilds the whole index; off for public deployments (issue #32)
    ENABLE_UI_REINDEX: bool = False

    # Vector store
    CHROMA_DB_PATH: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "documents"
    VECTOR_STORE: str = "chroma"

    # HTTP
    REQUEST_TIMEOUT: float = 120.0

    # Per-path model config — change independently without touching other paths
    SIMPLE_LLM_PROVIDER: str = "openai"
    SIMPLE_MODEL_NAME: str = "gpt-4o-mini"
    COMPLEX_LLM_PROVIDER: str = "openai"
    COMPLEX_MODEL_NAME: str = "gpt-4o"
    # Eval judge — independent from pipeline provider
    JUDGE_LLM_PROVIDER: str = "openai"
    JUDGE_MODEL_NAME: str = "gpt-4o"

    # V7 node limits
    MAX_SEARCH_CALLS: int = 2
    MAX_VISUAL_PROOF_CALLS: int = 1
    MAX_VISUAL_PROOFS: int = 1  # was 3; each attempt costs up to 3s timeout on VPS


settings = Settings()
