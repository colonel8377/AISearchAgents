"""
Abstract base class for embedding providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

from langchain_core.embeddings import Embeddings

from .embedding_config import EmbeddingConfig


class BaseEmbeddingProvider(ABC):
    """Abstract base class for all embedding providers."""

    def __init__(self, name: str, provider_key: str):
        """
        Initialize the embedding provider.
        
        Args:
            name: Human-readable name of the provider
            provider_key: Unique key identifier for the provider
        """
        self._name = name
        self._provider_key = provider_key

    @property
    def name(self) -> str:
        """Get the human-readable name of the provider."""
        return self._name

    @property
    def provider_key(self) -> str:
        """Get the unique key identifier of the provider."""
        return self._provider_key

    @abstractmethod
    def validate_config(self, config: EmbeddingConfig) -> None:
        """
        Validate the configuration specific to this provider.
        
        Args:
            config: Embedding configuration to validate
            
        Raises:
            ValueError: If the configuration is invalid
        """
        raise NotImplementedError

    @abstractmethod
    def create_embeddings(self, config: EmbeddingConfig) -> Embeddings:
        """
        Instantiate the LangChain Embeddings object.
        
        Args:
            config: Embedding configuration
            
        Returns:
            Configured Embeddings instance
        """
        raise NotImplementedError

    def _check_dependencies(self, *module_names: str) -> None:
        """
        Ensure required Python packages are installed.
        
        Args:
            *module_names: Names of required modules
            
        Raises:
            ImportError: If any required module is missing
        """
        for module_name in module_names:
            try:
                __import__(module_name)
            except ImportError as e:
                raise ImportError(
                    f"Missing dependency '{module_name}' for {self.name}: {e}"
                )

    @staticmethod
    def _filter_embedding_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize parameters to remove those known to cause conflicts.
        
        Args:
            params: Parameters dictionary to filter
            
        Returns:
            Filtered parameters dictionary
        """
        problematic_keys = {
            "http_client_kwargs",
            "http_kwargs",
            "https_kwargs",
            "httpx_kwargs",
            "aiohttp_kwargs",
            "requests_kwargs",
            "proxy_kwargs",
            "connection_kwargs",
            "session_kwargs",
            "client_kwargs",
            "async_client_kwargs",
            "model_kwargs"
        }
        return {k: v for k, v in params.items() if k not in problematic_keys}
