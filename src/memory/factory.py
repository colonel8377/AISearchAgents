"""Vector store factory for creating different types of vector stores."""

import os
from typing import Optional, Any, Dict, Callable, Type, TYPE_CHECKING
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from ..config.settings import settings
from ..utils.logger import get_logger

# Optional imports for type checking
if TYPE_CHECKING:
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
    except ImportError:
        GoogleGenerativeAIEmbeddings = Any  # type: ignore

logger = get_logger(__name__)


def _filter_embedding_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Filter out problematic parameters that should not be passed to embeddings."""
    # Parameters that cause issues when passed to embeddings constructors or methods
    problematic_params = {
        # HTTP client related parameters
        "http_client_kwargs",
        "http_kwargs",
        "https_kwargs",
        "httpx_kwargs",
        "aiohttp_kwargs",
        "requests_kwargs",

        # Proxy related parameters
        "proxy_kwargs",
        "connection_kwargs",
        "session_kwargs",

        # Client related parameters
        "client_kwargs",
        "async_client_kwargs",

        # Other potentially problematic parameters
        "model_kwargs",  # Sometimes causes issues if nested incorrectly
    }

    return {k: v for k, v in params.items() if k not in problematic_params}


@dataclass
class EmbeddingConfig:
    """Configuration for embedding providers."""
    provider: str = ""
    model: str = ""
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    custom_params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_settings(cls) -> 'EmbeddingConfig':
        """Create config from global settings."""
        return cls(
            provider=settings.embedding_provider.lower(),
            model=settings.embedding_model,
            api_key=settings.embedding_api_key or settings.openai_api_key,
            api_base=settings.embedding_api_base or settings.openai_api_base
        )


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider display name."""
        pass

    @property
    @abstractmethod
    def provider_key(self) -> str:
        """Provider identifier key."""
        pass

    @abstractmethod
    def validate_config(self, config: EmbeddingConfig) -> None:
        """Validate configuration for this provider."""
        pass

    @abstractmethod
    def create_embeddings(self, config: EmbeddingConfig) -> Embeddings:
        """Create embeddings instance."""
        pass


class BaseEmbeddingProvider(EmbeddingProvider):
    """Base class for embedding providers."""

    def __init__(self, name: str, provider_key: str):
        self._name = name
        self._provider_key = provider_key

    @property
    def name(self) -> str:
        return self._name

    @property
    def provider_key(self) -> str:
        return self._provider_key

    @abstractmethod
    def validate_config(self, config: EmbeddingConfig) -> None:
        """Validate configuration. Subclasses must implement."""
        raise NotImplementedError

    @abstractmethod
    def create_embeddings(self, config: EmbeddingConfig) -> Embeddings:
        """Create embeddings instance. Subclasses must implement."""
        raise NotImplementedError

    def _check_dependencies(self, *module_names: str) -> None:
        """Check if required dependencies are available."""
        for module_name in module_names:
            try:
                __import__(module_name)
            except ImportError as e:
                raise ValueError(f"Missing dependency '{module_name}' for {self.name} provider: {e}")


class OpenAICompatibleProvider(BaseEmbeddingProvider):
    """Provider for OpenAI-compatible APIs (OpenAI, Qwen, DeepSeek, etc.)."""

    def __init__(self, name: str, provider_key: str, default_base_url: Optional[str] = None):
        super().__init__(name, provider_key)
        self.default_base_url = default_base_url

    def validate_config(self, config: EmbeddingConfig) -> None:
        """Validate OpenAI-compatible configuration."""
        if not config.api_key:
            raise ValueError(f"{self.name} API key is required")

    def create_embeddings(self, config: EmbeddingConfig) -> Embeddings:
        """Create OpenAIEmbeddings instance."""
        self._check_dependencies("langchain_openai")

        # Prepare constructor arguments
        kwargs = {
            "model": config.model,
            "api_key": config.api_key,
            "base_url": config.api_base or self.default_base_url,
        }

        # Add filtered custom parameters
        kwargs.update(_filter_embedding_params(config.custom_params))

        return OpenAIEmbeddings(**kwargs)


class GoogleGeminiProvider(BaseEmbeddingProvider):
    """Provider for Google Gemini embeddings."""

    def validate_config(self, config: EmbeddingConfig) -> None:
        """Validate Gemini configuration."""
        if not config.api_key:
            raise ValueError(f"{self.name} API key is required")

    def create_embeddings(self, config: EmbeddingConfig) -> Embeddings:
        """Create GoogleGenerativeAIEmbeddings instance."""
        self._check_dependencies("langchain_google_genai")

        # Import at runtime to handle optional dependencies
        from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore

        # Filter out problematic parameters
        filtered_params = _filter_embedding_params(config.custom_params)

        return GoogleGenerativeAIEmbeddings(
            model=config.model,
            google_api_key=config.api_key,
            **filtered_params
        )


class EmbeddingProviderRegistry:
    """Registry for managing embedding providers."""

    _providers: Dict[str, EmbeddingProvider] = {}

    @classmethod
    def register(cls, provider: EmbeddingProvider) -> None:
        """Register a provider."""
        cls._providers[provider.provider_key] = provider
        logger.debug(f"Registered embedding provider: {provider.provider_key} ({provider.name})")

    @classmethod
    def get_provider(cls, provider_key: str) -> EmbeddingProvider:
        """Get provider by key."""
        if provider_key not in cls._providers:
            available = list(cls._providers.keys())
            raise ValueError(f"Unknown embedding provider '{provider_key}'. Available: {available}")
        return cls._providers[provider_key]

    @classmethod
    def list_providers(cls) -> Dict[str, str]:
        """List all registered providers."""
        return {key: provider.name for key, provider in cls._providers.items()}

    @classmethod
    def has_provider(cls, provider_key: str) -> bool:
        """Check if provider is registered."""
        return provider_key in cls._providers


# Register built-in providers
EmbeddingProviderRegistry.register(
    OpenAICompatibleProvider("OpenAI", "openai", "https://api.openai.com/v1")
)
EmbeddingProviderRegistry.register(
    OpenAICompatibleProvider("Qwen", "qwen")
)
EmbeddingProviderRegistry.register(
    OpenAICompatibleProvider("DeepSeek", "deepseek")
)
EmbeddingProviderRegistry.register(
    GoogleGeminiProvider("Google Gemini", "gemini")
)

class EmbeddingFactory:
    """Factory for creating embeddings using registered providers."""

    @classmethod
    def create_embeddings_from_config(cls, config: EmbeddingConfig) -> Embeddings:
        """Create embeddings from configuration."""
        provider = EmbeddingProviderRegistry.get_provider(config.provider)
        logger.info(f"Creating embeddings: provider={config.provider} ({provider.name}), model={config.model}")

        # Validate configuration
        provider.validate_config(config)

        try:
            return provider.create_embeddings(config)
        except Exception as e:
            logger.error(f"Failed to create {provider.name} embeddings: {e}")
            raise ValueError(f"Failed to create {provider.name} embeddings: {e}") from e

    @classmethod
    def create_embeddings(cls, provider_key: Optional[str] = None) -> Embeddings:
        """Create embeddings using global settings."""
        if provider_key is None:
            provider_key = settings.embedding_provider.lower()

        config = EmbeddingConfig.from_settings()
        config.provider = provider_key

        return cls.create_embeddings_from_config(config)

    @classmethod
    def list_available_providers(cls) -> Dict[str, str]:
        """List all available providers."""
        return EmbeddingProviderRegistry.list_providers()

    @classmethod
    def register_provider(cls, provider: EmbeddingProvider) -> None:
        """Register a new provider."""
        EmbeddingProviderRegistry.register(provider)

    @classmethod
    def is_provider_available(cls, provider_key: str) -> bool:
        """Check if a provider is available."""
        return EmbeddingProviderRegistry.has_provider(provider_key)


class VectorStoreFactory:
    """Factory for creating vector store instances based on configuration."""

    @staticmethod
    def create_embeddings() -> Embeddings:
        """
        Create an embeddings instance based on the configured provider.

        Returns:
            Embeddings instance

        Raises:
            ValueError: If the embedding provider is not supported
        """
        provider = settings.embedding_provider.lower()
        logger.info(f"Creating embeddings: provider={provider}, model={settings.embedding_model}")

        return EmbeddingFactory.create_embeddings(provider)

    @staticmethod
    def create_vector_store(
        store_type: str,
        embeddings: Optional[Embeddings] = None,
        **kwargs: Any
    ):
        """
        Create a vector store instance based on the specified type.

        Args:
            store_type: Type of vector store ('redis', 'postgres', or 'chroma')
            embeddings: Embeddings instance to use (defaults to configured embeddings)
            **kwargs: Additional keyword arguments specific to the vector store type

        Returns:
            VectorStore instance

        Raises:
            ValueError: If store_type is not supported
        """
        logger.info(f"Creating vector store: type={store_type}")

        if embeddings is None:
            logger.debug("No embeddings provided, using configured embeddings")
            embeddings = VectorStoreFactory.create_embeddings()
        
        try:
            if store_type == "redis":
                store = VectorStoreFactory._create_redis_store(embeddings, **kwargs)
            elif store_type == "postgres":
                store = VectorStoreFactory._create_postgres_store(embeddings, **kwargs)
            elif store_type == "chroma":
                store = VectorStoreFactory._create_chroma_store(embeddings, **kwargs)
            else:
                logger.error(f"Unsupported vector store type requested: {store_type}")
                raise ValueError(f"Unsupported vector store type: {store_type}")
            
            logger.info(f"Vector store created successfully: type={store_type}")
            return store
        except Exception as e:
            logger.error(f"Failed to create vector store: type={store_type}, error={e}", exc_info=True)
            raise
    
    @staticmethod
    def _create_redis_store(embeddings: Embeddings, **kwargs: Any):
        """Create a Redis vector store."""
        from langchain_community.vectorstores import Redis
        
        redis_url = kwargs.get("redis_url", "redis://localhost:6379")
        index_name = kwargs.get("index_name", "agent_memory")
        
        logger.debug(f"Creating Redis store: url={redis_url}, index={index_name}")
        
        return Redis(
            redis_url=redis_url,
            index_name=index_name,
            embedding=embeddings
        )
    
    @staticmethod
    def _create_postgres_store(embeddings: Embeddings, **kwargs: Any):
        """Create a PostgreSQL with pgvector store."""
        from langchain_community.vectorstores.pgvector import PGVector
        
        connection_string = kwargs.get(
            "connection_string",
            "postgresql://postgres:@localhost:5432/vectordb"
        )
        collection_name = kwargs.get("collection_name", "agent_memory")
        
        logger.debug(f"Creating Postgres store: collection={collection_name}")
        
        return PGVector(
            connection_string=connection_string,
            collection_name=collection_name,
            embedding_function=embeddings
        )
    
    @staticmethod
    def _create_chroma_store(embeddings: Embeddings, **kwargs: Any):
        """Create a Chroma vector store."""
        from langchain_community.vectorstores import Chroma
        
        persist_directory = kwargs.get("persist_directory", "./chroma_db")
        collection_name = kwargs.get("collection_name", "agent_memory")
        
        logger.debug(f"Creating Chroma store: persist_dir={persist_directory}, collection={collection_name}")
        
        return Chroma(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_function=embeddings
        )
