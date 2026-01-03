""" Embedding utilities and classes. """

from functools import lru_cache
from langchain_openai import OpenAIEmbeddings
from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

@lru_cache()
def get_embeddings() -> OpenAIEmbeddings:
    """Get cached OpenAI embeddings instance."""
    settings = get_settings()
    logger.info("Initializing OpenAI Embeddings with model: %s", settings.embedding_model)
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )
    logger.info("OpenAI Embeddings initialized successfully.")
    return embeddings

class EmbeddingService:
    """Service class for handling embeddings."""

    def __init__(self):
        """Initialize the EmbeddingService."""
        logger.info("Initializing EmbeddingService")
        settings = get_settings()
        self.embeddings = get_embeddings()
        self.model_name = settings.embedding_model

    def embed_query(self, text: str) -> list[float]:
        """Generate embeddings for the given text."""
        logger.debug("Generating embeddings for text of length %d", len(text))
        embedding_query = self.embeddings.embed_query(text)
        logger.debug("Generated embedding of length %d", len(embedding))
        return embedding_query
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of documents."""
        logger.debug("Generating embeddings for %d documents", len(texts))
        embeddings_document = self.embeddings.embed_documents(texts)
        logger.debug("Generated embeddings for %d documents", len(embeddings))
        return embeddings_document