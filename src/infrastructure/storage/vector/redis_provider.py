"""
Redis vector store provider implementation.
"""

from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from src.shared.constant.enums import VectorStoreType
from src.shared.utils import get_logger
from .base_provider import BaseVectorStoreProvider

logger = get_logger(__name__)


class RedisStoreProvider(BaseVectorStoreProvider):
    """
    Provider for Redis Vector Store.
    
    Creates and configures Redis-based vector stores for memory persistence.
    """

    @property
    def store_type(self) -> str:
        """Get the Redis store type identifier."""
        return VectorStoreType.REDIS.value

    def create_store(
        self,
        embeddings: Embeddings,
        **kwargs: Any
    ) -> VectorStore:
        """
        Create a Redis vector store instance.
        
        Args:
            embeddings: Embedding function to use
            **kwargs: Additional configuration:
                - redis_url: Redis persistence URL (default: "redis://localhost:6379")
                - index_name: Index name (default: "agent_memory")
        
        Returns:
            Configured Redis VectorStore instance
        """
        from langchain_community.vectorstores import Redis

        redis_url = kwargs.get("redis_url", "redis://localhost:6379")
        index_name = kwargs.get("index_name", "agent_memory")

        logger.debug(
            f"Initializing Redis store: url={redis_url}, index={index_name}"
        )
        return Redis(
            redis_url=redis_url,
            index_name=index_name,
            embedding=embeddings
        )

