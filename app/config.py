''' Application configuration settings using Pydantic '''

from pydantic_settings import BaseSettings,SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    ''' Application configuration settings '''


    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False, 
        extra = "ignore"     
    )
    # openai api key
    openai_api_key: str

    # qdrant configuration
    qdrant_url: str
    qdrant_api_key: str

    #collection name
    qdrant_collection_name: str = "rag_qa_collection"

    # Document chunk size
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # RETRIEVER SETTINGS
    retrieval_k: int = 4

    # Model Configuration
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0

    # Logging
    log_level: str = "INFO"

    # RAGAS Evaluation Settings
    enable_ragas_evaluation: bool = True
    ragas_timeout_seconds: float = 30.0
    ragas_log_results: bool = True
    ragas_llm_model: str | None = None  # Defaults to llm_model if not set
    ragas_llm_temperature: float | None = None  # Defaults to llm_temperature if not set
    ragas_embedding_model: str | None = None  # Defaults to embedding_model if not set

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Application Info
    app_name: str = "RAG Q&A System"
    app_version: str = "0.1.0"

@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()