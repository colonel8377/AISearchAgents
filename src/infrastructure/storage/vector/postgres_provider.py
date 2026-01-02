"""
PostgreSQL (PGVector) vector store provider implementation.
"""

from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from src.shared.constant.enums import VectorStoreType
from src.shared.utils import get_logger
from .base_provider import BaseVectorStoreProvider

logger = get_logger(__name__)


class PostgresStoreProvider(BaseVectorStoreProvider):
    """
    Provider for PostgreSQL (PGVector) Store.
    
    Creates and configures PostgreSQL-based vector stores using PGVector extension.
    """

    @property
    def store_type(self) -> str:
        """Get the PostgreSQL store type identifier."""
        return VectorStoreType.POSTGRES.value

    def create_store(
        self,
        embeddings: Embeddings,
        **kwargs: Any
    ) -> VectorStore:
        """
        Create a PostgreSQL vector store instance.
        
        Args:
            embeddings: Embedding function to use
            **kwargs: Additional configuration:
                - connection_string: PostgreSQL persistence string
                  (default: "postgresql://postgres:@localhost:5432/vectordb")
                - collection_name: Collection name (default: "agent_memory")
        
        Returns:
            Configured PGVector instance
        """
        from langchain_community.vectorstores.pgvector import PGVector

        connection_string = kwargs.get(
            "connection_string",
            "postgresql://postgres:@localhost:5432/vectordb"
        )
        collection_name = kwargs.get("collection_name", "agent_memory")

        logger.debug(
            f"Initializing Postgres store: collection={collection_name}"
        )
        return PGVector(
            connection_string=connection_string,
            collection_name=collection_name,
            embedding_function=embeddings
        )

