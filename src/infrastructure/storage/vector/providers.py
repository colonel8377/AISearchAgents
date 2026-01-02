"""
Legacy providers module - redirects to individual provider modules.

This file is kept for backward compatibility. New code should import from
individual provider modules directly:
- redis_provider
- postgres_provider
- chroma_provider
"""

from .redis_provider import RedisStoreProvider
from .postgres_provider import PostgresStoreProvider
from .chroma_provider import ChromaStoreProvider

__all__ = [
    "RedisStoreProvider",
    "PostgresStoreProvider",
    "ChromaStoreProvider"
]
