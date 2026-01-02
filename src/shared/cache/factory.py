"""
Cache factory and management functions.
"""

import threading
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

from ...shared.config.settings import settings, BASE_DIR
from ...shared.utils.logger import get_logger
from .backends import BaseCacheBackend, LocalCacheBackend, RedisCacheBackend

logger = get_logger(__name__)


# Global cache instance (singleton)
_cache_instance: Optional[BaseCacheBackend] = None
_cache_lock = threading.Lock()


def get_agent_cache() -> BaseCacheBackend:
    """
    Get or create the global cache instance (singleton).
    Thread-safe implementation.

    Returns:
        BaseCacheBackend instance
    """
    global _cache_instance
    if _cache_instance is None:
        with _cache_lock:
            if _cache_instance is None:
                _cache_instance = create_cache_backend()
    return _cache_instance


def create_cache_backend() -> BaseCacheBackend:
    """
    Create cache backend based on settings.
    
    Returns:
        BaseCacheBackend instance (LocalCache or RedisCache)
    """
    cache_backend = getattr(settings, 'cache_backend', 'local')
    
    if cache_backend == "redis":
        try:
            return RedisCacheBackend(
                redis_host=settings.redis_host,
                redis_port=settings.redis_port,
                redis_user=settings.redis_user,
                redis_password=settings.redis_password,
                redis_db=getattr(settings, 'cache_redis_db', 1)
            )
        except Exception as e:
            logger.warning(
                f"Failed to initialize Redis cache, "
                f"falling back to local: {e}"
            )
            # Fallthrough to local cache

    # Default to local cache
    return LocalCacheBackend(
        db_path=getattr(settings, 'cache_db_path', None)
    )


def reset_agent_cache(
    function_name: Optional[str] = None,
    class_name: Optional[str] = None
) -> None:
    """
    Reset the agent cache by clearing all or specific cached entries.
    
    Args:
        function_name: Optional function name to clear specific function cache
        class_name: Optional class name to clear specific class cache
    """
    try:
        cache = get_agent_cache()
        cache.clear(function_name=function_name, class_name=class_name)
        logger.info(
            f"Agent cache reset: function={function_name}, "
            f"class={class_name}"
        )
    except Exception as e:
        logger.error(f"Failed to reset agent cache: {e}", exc_info=True)
        raise


def reset_all_caches() -> Dict[str, Any]:
    """
    Reset all caches in the system.
    
    Clears:
    - Agent cache (Local SQLite or Redis)
    - Web opinion cache (Local SQLite and Redis)
    - Vector stores (Chroma, PostgreSQL, Redis)
    
    Returns:
        Dict with success status and details.
    """
    result = {
        "success": True,
        "errors": [],
        "cleared_items": []
    }
    
    # 1. Clear agent cache
    _clear_agent_cache(result)

    # 2. Clear web opinion cache (local SQLite)
    _clear_web_opinion_cache_local(result)

    # 3. Clear Redis caches (web opinion cache and vector store)
    _clear_redis_caches(result)

    # 4. Clear vector stores (Chroma and PostgreSQL)
    _clear_vector_stores(result)

    return result


def _clear_agent_cache(result: Dict[str, Any]) -> None:
    try:
        reset_agent_cache()
        result["cleared_items"].append("agent_cache")
    except Exception as e:
        msg = f"Failed to clear agent cache: {e}"
        logger.error(msg, exc_info=True)
        result["errors"].append(msg)
        result["success"] = False


def _clear_web_opinion_cache_local(result: Dict[str, Any]) -> None:
    """Clear web opinion cache local SQLite database."""
    try:
        db_path = BASE_DIR / "data" / "web_opinion_cache.db"
        
        if db_path.exists():
            db_path.unlink()
            result["cleared_items"].append("web_opinion_cache_local")
            logger.info(f"Web opinion cache local SQLite deleted: {db_path}")
        else:
            logger.debug("Web opinion cache local SQLite file not found, skipping")
    except Exception as e:
        msg = f"Failed to clear web opinion cache local: {e}"
        logger.error(msg, exc_info=True)
        result["errors"].append(msg)
        result["success"] = False


def _clear_redis_caches(result: Dict[str, Any]) -> None:
    try:
        import redis

        # Helper to get redis client
        def get_redis_client(db: int) -> redis.Redis:
            return redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                username=settings.redis_user or None,
                password=settings.redis_password or None,
                db=db,
                decode_responses=True
            )

        # Clear web opinion cache in Redis (db=2)
        # Note: This is independent of cache_backend setting
        try:
            client = get_redis_client(db=2)
            _clear_redis_keys(client, "web_opinion_cache:*", "redis_web_opinion_cache", result)
        except Exception as e:
            logger.warning(f"Failed to clear Redis web opinion cache: {e}")
            # Don't fail hard, just log warning

        # Clear Redis vector store (only if vector store type is Redis)
        vector_store_type = getattr(settings, 'vector_store_type', 'chroma')
        if vector_store_type == "redis":
            try:
                redis_db = getattr(settings, 'redis_db', 0)
                client = get_redis_client(db=redis_db)
                _clear_redis_keys(client, "agent_memory_*", "redis_vector_store", result)
            except Exception as e:
                result["errors"].append(f"Failed to clear Redis vector store: {e}")
                result["success"] = False
        else:
            logger.debug(f"Vector store type is {vector_store_type}, skipping Redis vector store clearing")

    except ImportError:
        logger.warning("Redis package not installed")
    except Exception as e:
        msg = f"Failed to clear Redis caches: {e}"
        logger.error(msg, exc_info=True)
        result["errors"].append(msg)
        result["success"] = False


def _clear_redis_keys(client: Any, pattern: str, item_name: str, result: Dict[str, Any]) -> None:
    """Helper to clear keys matching a pattern in Redis."""
    try:
        keys = client.keys(pattern)
        if keys:
            client.delete(*keys)
            result["cleared_items"].append(item_name)
            logger.info(f"{item_name} cleared ({len(keys)} keys)")
        else:
            logger.debug(f"No keys found for pattern {pattern}")
    except Exception as e:
        logger.warning(f"Failed to clear Redis keys with pattern {pattern}: {e}")


def _clear_vector_stores(result: Dict[str, Any]) -> None:
    """Clear vector stores (Chroma and PostgreSQL)."""
    vector_store_type = getattr(settings, 'vector_store_type', 'chroma')
    
    if vector_store_type == "chroma":
        _clear_chroma_vector_store(result)
    elif vector_store_type == "postgres":
        _clear_postgres_vector_store(result)
    elif vector_store_type == "redis":
        # Redis vector store is cleared in _clear_redis_caches
        logger.debug("Redis vector store clearing handled in _clear_redis_caches")
    else:
        logger.debug(f"Unknown vector store type: {vector_store_type}, skipping")


def _clear_chroma_vector_store(result: Dict[str, Any]) -> None:
    """Clear Chroma vector store (local file system)."""
    try:
        chroma_dir = Path(getattr(settings, 'chroma_persist_directory', './chroma_db'))
        
        if chroma_dir.exists():
            # Delete all contents of the directory (files and subdirectories)
            for item in chroma_dir.iterdir():
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                    logger.debug(f"Deleted {item}")
                except Exception as e:
                    logger.warning(f"Failed to delete {item}: {e}")
                    # Continue with other items
            
            result["cleared_items"].append("chroma_vector_store")
            logger.info(f"Chroma vector store cleared: {chroma_dir}")
        else:
            logger.debug(f"Chroma directory does not exist: {chroma_dir}")
            
    except Exception as e:
        msg = f"Failed to clear Chroma vector store: {e}"
        logger.error(msg, exc_info=True)
        result["errors"].append(msg)
        result["success"] = False


def _clear_postgres_vector_store(result: Dict[str, Any]) -> None:
    """Clear PostgreSQL vector store (PGVector)."""
    try:
        import psycopg2
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
        
        conn = psycopg2.connect(
            host=getattr(settings, 'postgres_host', 'localhost'),
            port=getattr(settings, 'postgres_port', 5432),
            user=getattr(settings, 'postgres_user', 'postgres'),
            password=getattr(settings, 'postgres_password', ''),
            database=getattr(settings, 'postgres_db', 'vectordb')
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        # Truncate tables instead of DELETE for performance and resetting identity
        # Assuming standard LangChain/PGVector tables
        tables_to_clear = ["langchain_pg_embedding", "langchain_pg_collection"]

        for table in tables_to_clear:
            try:
                cursor.execute(f"TRUNCATE TABLE {table} CASCADE")
                logger.debug(f"Truncated table {table}")
            except psycopg2.errors.UndefinedTable:
                logger.debug(f"Table {table} does not exist, skipping")
            except Exception as e:
                logger.warning(f"Failed to truncate table {table}: {e}")
                # Don't fail hard if table doesn't exist or other minor issue

        cursor.close()
        conn.close()
        result["cleared_items"].append("postgres_vector_store")
        logger.info("PostgreSQL vector store cleared")
        
    except ImportError:
        msg = "psycopg2 not available, skipping PostgreSQL vector store clearing"
        logger.warning(msg)
        result["errors"].append(msg)
        result["success"] = False
    except Exception as e:
        msg = f"Failed to clear PostgreSQL vector store: {e}"
        logger.error(msg, exc_info=True)
        result["errors"].append(msg)
        result["success"] = False
