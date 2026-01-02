"""
Provider registry for managing embedding and vector store providers.
"""

from typing import Dict

from src.infrastructure.storage.vector import BaseVectorStoreProvider
from src.shared.llm import BaseEmbeddingProvider
from src.shared.utils import get_logger

logger = get_logger(__name__)


class ProviderRegistry:
    """
    Unified registry for both Embedding and Vector Store providers.
    
    This registry maintains mappings of provider keys to provider instances,
    allowing dynamic lookup and registration of providers at runtime.
    """

    _embedding_providers: Dict[str, BaseEmbeddingProvider] = {}
    _vector_store_providers: Dict[str, BaseVectorStoreProvider] = {}

    # --- Embedding Provider Management ---
    @classmethod
    def register_embedding(
        cls,
        provider: BaseEmbeddingProvider
    ) -> None:
        """
        Register an embedding provider.
        
        Args:
            provider: Embedding provider instance to register
        """
        cls._embedding_providers[provider.provider_key] = provider
        logger.debug(f"Registered embedding provider: {provider.name}")

    @classmethod
    def get_embedding_provider(
        cls,
        key: str
    ) -> BaseEmbeddingProvider:
        """
        Get an embedding provider by key.
        
        Args:
            key: Provider key identifier
            
        Returns:
            Registered embedding provider instance
            
        Raises:
            ValueError: If the provider key is not registered
        """
        if key not in cls._embedding_providers:
            available = list(cls._embedding_providers.keys())
            raise ValueError(
                f"Unknown embedding provider '{key}'. "
                f"Available: {available}"
            )
        return cls._embedding_providers[key]

    # --- Vector Store Provider Management ---
    @classmethod
    def register_vector_store(
        cls,
        provider: BaseVectorStoreProvider
    ) -> None:
        """
        Register a vector store provider.
        
        Args:
            provider: Vector store provider instance to register
        """
        cls._vector_store_providers[provider.store_type] = provider
        logger.debug(f"Registered vector store provider: {provider.store_type}")

    @classmethod
    def get_vector_store_provider(
        cls,
        key: str
    ) -> BaseVectorStoreProvider:
        """
        Get a vector store provider by key.
        
        Args:
            key: Store type identifier
            
        Returns:
            Registered vector store provider instance
            
        Raises:
            ValueError: If the store type is not registered
        """
        if key not in cls._vector_store_providers:
            available = list(cls._vector_store_providers.keys())
            raise ValueError(
                f"Unknown vector store type '{key}'. "
                f"Available: {available}"
            )
        return cls._vector_store_providers[key]
