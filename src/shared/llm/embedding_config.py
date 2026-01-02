"""
Configuration data classes for embedding providers.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from src.shared.config.settings import settings


@dataclass
class EmbeddingConfig:
    """Configuration data object for embedding providers."""
    provider: str
    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    custom_params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_settings(cls) -> "EmbeddingConfig":
        """Create configuration from global application settings."""
        return cls(
            provider=settings.embedding_provider.lower(),
            model=settings.embedding_model,
            api_key=settings.embedding_api_key or settings.openai_api_key,
            api_base=settings.embedding_api_base or settings.openai_api_base
        )
