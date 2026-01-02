"""
Provider registration module for embedding and vector store providers.

This module handles the registration of all available providers at module import time.
"""
from ..storage.vector import RedisStoreProvider, PostgresStoreProvider, ChromaStoreProvider
from ...shared.llm.embedding_providers import (
    OpenAICompatibleProvider,
    GoogleGeminiProvider
)
from ...shared.constant.enums import EmbeddingProviderType
from ...shared.utils.logger import get_logger
from .registry import ProviderRegistry

logger = get_logger(__name__)


def register_embedding_providers():
    """
    Register all embedding providers with the registry.
    
    This function registers:
    - OpenAI
    - Qwen
    - DeepSeek
    - Google Gemini
    """
    ProviderRegistry.register_embedding(
        OpenAICompatibleProvider(
            "OpenAI",
            EmbeddingProviderType.OPENAI.value,
            "https://api.openai.com/v1"
        )
    )
    ProviderRegistry.register_embedding(
        OpenAICompatibleProvider(
            "Qwen",
            EmbeddingProviderType.QWEN.value
        )
    )
    ProviderRegistry.register_embedding(
        OpenAICompatibleProvider(
            "DeepSeek",
            EmbeddingProviderType.DEEPSEEK.value
        )
    )
    ProviderRegistry.register_embedding(
        GoogleGeminiProvider(
            "Google Gemini",
            EmbeddingProviderType.GEMINI.value
        )
    )
    logger.debug("All embedding providers registered")


def register_vector_store_providers():
    """
    Register all vector store providers with the registry.
    
    This function registers:
    - Redis
    - PostgreSQL (PGVector)
    - ChromaDB
    """
    ProviderRegistry.register_vector_store(RedisStoreProvider())
    ProviderRegistry.register_vector_store(PostgresStoreProvider())
    ProviderRegistry.register_vector_store(ChromaStoreProvider())
    logger.debug("All vector store providers registered")


def register_all_providers():
    """
    Register all embedding and vector store providers.
    
    This is the main entry point for provider registration.
    """
    register_embedding_providers()
    register_vector_store_providers()
    logger.info("All providers registered successfully")

