"""
Factory for creating Embedding models and Vector Stores.

Uses the Provider/Registry pattern to allow easy extension without modifying core logic.
"""

from __future__ import annotations

from typing import Optional, Any

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from ...shared.llm.embedding_config import EmbeddingConfig
from ...shared.utils.logger import get_logger
from .registry import ProviderRegistry

logger = get_logger(__name__)


class VectorStoreFacade:
    """
    Facade for creating and managing Embeddings and Vector Stores.

    Provides a unified interface for creating embeddings and vector stores,
    delegating implementation details to registered providers.
    """

    @classmethod
    def create_embeddings(
        cls,
        provider_key: Optional[str] = None
    ) -> Embeddings:
        """
        Create an embedding model instance.

        Args:
            provider_key: Optional override for provider. Defaults to settings.

        Returns:
            Configured Embeddings instance
            
        Raises:
            ValueError: If provider configuration is invalid
            Exception: If embedding creation fails
        """
        config = EmbeddingConfig.from_settings()
        if provider_key:
            config.provider = provider_key.lower()

        provider = ProviderRegistry.get_embedding_provider(config.provider)

        logger.info(
            f"Creating embeddings: {provider.name} (model={config.model})"
        )
        provider.validate_config(config)

        try:
            return provider.create_embeddings(config)
        except Exception as e:
            logger.error(
                f"Failed to create embeddings for {provider.name}: {e}"
            )
            raise

    @classmethod
    def create_vector_store(
        cls,
        store_type: str,
        embeddings: Optional[Embeddings] = None,
        **kwargs: Any
    ) -> VectorStore:
        """
        Create a vector store instance.

        Args:
            store_type: Identifier for the store (e.g., 'redis', 'chroma').
            embeddings: Optional existing embedding instance.
                       If None, creates a new one using settings.
            **kwargs: Configuration specific to the vector store.

        Returns:
            Configured VectorStore instance
            
        Raises:
            ValueError: If store type is not registered
            Exception: If vector store creation fails
        """
        if embeddings is None:
            embeddings = cls.create_embeddings()

        try:
            provider = ProviderRegistry.get_vector_store_provider(store_type)
            logger.info(f"Creating vector store: {store_type}")

            return provider.create_store(embeddings, **kwargs)
        except Exception as e:
            logger.error(
                f"Failed to create vector store '{store_type}': {e}",
                exc_info=True
            )
            raise

