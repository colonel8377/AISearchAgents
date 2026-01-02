"""
Agent cache module with decorators and backends.
"""
from .decorator import cached
from .factory import (
    get_agent_cache,
    create_cache_backend,
    reset_agent_cache,
    reset_all_caches,
)
from .backends import (
    BaseCacheBackend,
    LocalCacheBackend,
    RedisCacheBackend,
    LocalCacheBackend as LocalCache,
    RedisCacheBackend as RedisCache,
)


__all__ = [
    "cached",
    "get_agent_cache",
    "reset_agent_cache",
    "reset_all_caches",
    "create_cache_backend",
    "LocalCache",
    "RedisCache",
    "BaseCacheBackend",
    "LocalCacheBackend",
    "RedisCacheBackend",
]
