"""
Concrete implementations of embedding providers.
"""

from typing import Optional, TYPE_CHECKING

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from . import EmbeddingConfig
from .base_embedding_provider import BaseEmbeddingProvider


if TYPE_CHECKING:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings


class OpenAICompatibleProvider(BaseEmbeddingProvider):
    """
    Provider for OpenAI, DeepSeek, Qwen, and other OpenAI-compatible APIs.
    """

    def __init__(
        self,
        name: str,
        provider_key: str,
        default_base_url: Optional[str] = None
    ):
        """
        Initialize OpenAI-compatible provider.
        
        Args:
            name: Human-readable name (e.g., "OpenAI", "Qwen")
            provider_key: Unique key identifier
            default_base_url: Default API base URL for this provider
        """
        super().__init__(name, provider_key)
        self.default_base_url = default_base_url

    def validate_config(self, config: EmbeddingConfig) -> None:
        """
        Validate configuration for OpenAI-compatible providers.
        
        Args:
            config: Embedding configuration to validate
            
        Raises:
            ValueError: If API key is missing
        """
        if not config.api_key:
            raise ValueError(f"API key is required for {self.name}")

    def create_embeddings(self, config: EmbeddingConfig) -> Embeddings:
        """
        Create OpenAI-compatible embeddings instance.
        
        Args:
            config: Embedding configuration
            
        Returns:
            Configured OpenAIEmbeddings instance
        """
        self._check_dependencies("langchain_openai")

        kwargs = {
            "model": config.model,
            "api_key": config.api_key,
            "base_url": config.api_base or self.default_base_url,
            **self._filter_embedding_params(config.custom_params)
        }
        return OpenAIEmbeddings(**kwargs)


class GoogleGeminiProvider(BaseEmbeddingProvider):
    """Provider for Google Gemini embeddings."""

    def validate_config(self, config: EmbeddingConfig) -> None:
        """
        Validate configuration for Google Gemini provider.
        
        Args:
            config: Embedding configuration to validate
            
        Raises:
            ValueError: If API key is missing
        """
        if not config.api_key:
            raise ValueError(f"API key is required for {self.name}")

    def create_embeddings(self, config: EmbeddingConfig) -> Embeddings:
        """
        Create Google Gemini embeddings instance.
        
        Args:
            config: Embedding configuration
            
        Returns:
            Configured GoogleGenerativeAIEmbeddings instance
        """
        self._check_dependencies("langchain_google_genai")
        # Lazy import to avoid hard dependency at module level
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=config.model,
            google_api_key=config.api_key,
            **self._filter_embedding_params(config.custom_params)
        )
