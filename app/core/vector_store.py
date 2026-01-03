from functools import lru_cache
from typing import Any
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse
from uuid import uuid4

from app.config import get_settings
from app.core.embeddings import get_embeddings
from app.utils.logger import get_logger
logger = get_logger(__name__)
settings = get_settings()
@lru_cache()
def get_qdrant_client() -> QdrantClient:
    """Get cached Qdrant client instance."""
    logger.info("Initializing Qdrant client") 
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )
    logger.info("Qdrant client initialized successfully.")
    return client
class VectorStoreService:
    """Service class for handling vector stores."""

    def __init__(self,collection_name: str|None = None):
        """Initialize the VectorStoreService."""
        logger.info("Initializing VectorStoreService")
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.client = get_qdrant_client()
        self.embeddings = get_embeddings()

        self._ensure_collection()
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )
        logger.info("VectorStoreService initialized successfully.") 

    def _ensure_collection(self) -> None:
        """Ensure the collection exists in Qdrant."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            logger.info("Collection '%s' already exists.", self.collection_name)
        except UnexpectedResponse:
            logger.info("Collection '%s' does not exist. Creating...", self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=1536,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Collection '%s' created successfully.", self.collection_name)
    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to the vector store."""
        if not documents:
            logger.warning("No documents to add to the vector store.")
            return []
        logger.info("Adding %d documents to the vector store.", len(documents))

        # Generate unique IDs for each document
        ids = [str(uuid4()) for _ in documents]

        self.vector_store.add_documents(documents, ids=ids)
        logger.info("Added %d documents to the vector store.", len(documents))
        return ids
    def search(
        self,
        query: str,
        k: int | None = None,
    ) -> list[Document]:
        """Search the vector store for similar documents."""
        logger.info("Searching the vector store for similar documents.")
        k = k or settings.retrieval_k
        logger.debug("Using k=%d for search.", k)
        results = self.vector_store.similarity_search(
            query,
            k=k,
        )
        logger.info("Found %d similar documents.", len(results))
        return results
    
    def search_with_scores(
        self,
        query: str,
        k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """Search the vector store for similar documents with scores."""
        logger.info("Searching the vector store for similar documents with scores.")
        k = k or settings.retrieval_k
        logger.debug("Using k=%d for search.", k)
        results = self.vector_store.similarity_search_with_score(
            query,
            k=k,
        )
        logger.info("Found %d similar documents with scores.", len(results))
        return results
    def get_retriever(self, k: int | None = None) -> Any:
        """Get a retriever for the vector store."""
        logger.info("Getting retriever for the vector store.")
        k = k or settings.retrieval_k
        logger.debug("Using k=%d for retriever.", k)
        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
        logger.info("Retriever obtained successfully.")
        return retriever
    def delete_collection(self) -> None:
        """Delete the collection from Qdrant."""
        logger.warning("Deleting collection '%s' from Qdrant.", self.collection_name)
        self.client.delete_collection(self.collection_name)
        logger.info("Collection '%s' deleted successfully.", self.collection_name)
    def get_collection_info(self) -> dict[str, Any]:
        """Get information about the collection in Qdrant."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            logger.info("Collection '%s' information obtained successfully.", self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": collection_info.points_count,
                "status": collection_info.status
            }
        except UnexpectedResponse as e:
            logger.error("Failed to get collection info for '%s': %s", self.collection_name, str(e))
            return {
                "error": str(e),
                "points_count": 0,
                "indexed_vectors_count": 0,
                "created_at": None,
                "status": "Collection does not exist"
            }
    def health_check(self) -> bool:
        """Perform a health check on the Qdrant client."""
        logger.info("Performing health check on Qdrant client.")
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error("Qdrant client health check exception: %s", str(e))
            return False
                   