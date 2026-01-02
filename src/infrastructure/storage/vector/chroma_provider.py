"""
ChromaDB vector store provider implementation.
"""

from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from src.shared.constant.enums import VectorStoreType
from src.shared.utils import get_logger
from .base_provider import BaseVectorStoreProvider

logger = get_logger(__name__)


class ChromaStoreProvider(BaseVectorStoreProvider):
    """
    Provider for ChromaDB Store.
    
    Creates and configures ChromaDB-based vector stores for memory persistence.
    """

    @property
    def store_type(self) -> str:
        """Get the ChromaDB store type identifier."""
        return VectorStoreType.CHROMA.value

    def create_store(
        self,
        embeddings: Embeddings,
        **kwargs: Any
    ) -> VectorStore:
        """
        Create a ChromaDB vector store instance.
        
        Args:
            embeddings: Embedding function to use
            **kwargs: Additional configuration:
                - persist_directory: Directory to persist data
                  (default: "./chroma_db")
                - collection_name: Collection name (default: "agent_memory")
        
        Returns:
            Configured Chroma instance
        """
        from langchain_community.vectorstores import Chroma

        persist_directory = kwargs.get("persist_directory", "./chroma_db")
        collection_name = kwargs.get("collection_name", "agent_memory")

        logger.debug(
            f"Initializing Chroma store: dir={persist_directory}, "
            f"collection={collection_name}"
        )
        return Chroma(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_function=embeddings
        )

