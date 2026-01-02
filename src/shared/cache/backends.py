"""
Cache backend implementations (Local SQLite and Redis).
"""

import json
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from datetime import datetime, timedelta

from ..utils.logger import get_logger
from .utils import generate_cache_key, get_parameters_dict

logger = get_logger(__name__)


class BaseCacheBackend(ABC):
    """
    Abstract base class for cache backends.

    Implements the Template Method pattern for get/set/clear operations,
    handling key generation, error catching, and logging centrally.
    """

    def get(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        class_name: Optional[str] = None
    ) -> Optional[Any]:
        """Get cached result if it exists."""
        try:
            key = generate_cache_key(func, args, kwargs, class_name)
            result = self._get_impl(key)

            if result is not None:
                logger.debug(
                    f"{self.__class__.__name__} hit for {func.__name__} "
                    f"(key: {key[:16]}...)"
                )
                return result

            logger.debug(
                f"{self.__class__.__name__} miss for {func.__name__} "
                f"(key: {key[:16]}...)"
            )
            return None
        except Exception as e:
            logger.error(
                f"Error reading from {self.__class__.__name__}: {e}",
                exc_info=True
            )
            return None

    def set(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        result: Any,
        class_name: Optional[str] = None,
        ttl: int = 86400
    ):
        """Store result in cache with TTL."""
        try:
            key = generate_cache_key(func, args, kwargs, class_name)
            params = get_parameters_dict(func, args, kwargs, class_name)

            self._set_impl(
                key=key,
                func_name=func.__name__,
                class_name=class_name,
                params=params,
                result=result,
                ttl=ttl
            )

            logger.debug(
                f"{self.__class__.__name__} stored result for {func.__name__} "
                f"(key: {key[:16]}...)"
            )
        except Exception as e:
            logger.error(
                f"Error writing to {self.__class__.__name__}: {e}",
                exc_info=True
            )

    def clear(
        self,
        function_name: Optional[str] = None,
        class_name: Optional[str] = None
    ):
        """Clear cached results."""
        try:
            count = self._clear_impl(function_name, class_name)
            logger.info(
                f"{self.__class__.__name__} cleared: {count if count is not None else 'all'} entries "
                f"(function={function_name}, class={class_name})"
            )
        except Exception as e:
            logger.error(
                f"Error clearing {self.__class__.__name__}: {e}",
                exc_info=True
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            return self._get_stats_impl()
        except Exception as e:
            logger.error(
                f"Error getting stats for {self.__class__.__name__}: {e}",
                exc_info=True
            )
            return {"error": str(e)}

    # --- Abstract Methods to be implemented by subclasses ---

    @abstractmethod
    def _get_impl(self, key: str) -> Optional[Any]:
        """Retrieve value from storage and update access time."""
        pass

    @abstractmethod
    def _set_impl(
        self,
        key: str,
        func_name: str,
        class_name: Optional[str],
        params: dict,
        result: Any,
        ttl: int
    ):
        """Save value and metadata to storage."""
        pass

    @abstractmethod
    def _clear_impl(
        self,
        function_name: Optional[str],
        class_name: Optional[str]
    ) -> Optional[int]:
        """Clear data from storage. Returns count of deleted items if available."""
        pass

    @abstractmethod
    def _get_stats_impl(self) -> Dict[str, Any]:
        """Retrieve statistics from storage."""
        pass


class LocalCacheBackend(BaseCacheBackend):
    """
    Local SQLite cache backend.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            base_dir = Path(__file__).resolve().parents[3]
            db_path = str(base_dir / "data" / "agent_cache.db")

        self.db_path = db_path
        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"LocalCache initialized: db_path={db_path}")

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Enable WAL mode for better concurrency
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_cache (
                    cache_key TEXT PRIMARY KEY,
                    function_name TEXT NOT NULL,
                    class_name TEXT,
                    parameters_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_function_name ON agent_cache(function_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_class_function ON agent_cache(class_name, function_name)")
            conn.commit()

    def _get_impl(self, key: str) -> Optional[Any]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT result_json 
                FROM agent_cache 
                WHERE cache_key = ? 
                  AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """, (key,))

            row = cursor.fetchone()
            if row:
                # Update access time
                cursor.execute("""
                    UPDATE agent_cache 
                    SET last_accessed = CURRENT_TIMESTAMP 
                    WHERE cache_key = ?
                """, (key,))
                conn.commit()
                return json.loads(row[0])
            return None

    def _set_impl(self, key: str, func_name: str, class_name: Optional[str], params: dict, result: Any, ttl: int):
        params_json = json.dumps(params, sort_keys=True, default=str)
        result_json = json.dumps(result, default=str)
        expires_at = datetime.now() + timedelta(seconds=ttl)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO agent_cache
                (cache_key, function_name, class_name, parameters_json,
                 result_json, created_at, last_accessed, expires_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
            """, (
                key, func_name, class_name, params_json, result_json, expires_at.isoformat()
            ))
            conn.commit()

    def _clear_impl(self, function_name: Optional[str], class_name: Optional[str]) -> Optional[int]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            conditions = []
            params = []

            if function_name:
                conditions.append("function_name = ?")
                params.append(function_name)

            if class_name:
                conditions.append("class_name = ?")
                params.append(class_name)

            if conditions:
                query = f"DELETE FROM agent_cache WHERE {' AND '.join(conditions)}"
                cursor.execute(query, tuple(params))
            else:
                cursor.execute("DELETE FROM agent_cache")

            conn.commit()
            return cursor.rowcount

    def _get_stats_impl(self) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM agent_cache")
            count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT function_name) FROM agent_cache")
            unique_functions = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT class_name) FROM agent_cache WHERE class_name IS NOT NULL")
            unique_classes = cursor.fetchone()[0]

            cursor.execute("SELECT MIN(created_at), MAX(created_at) FROM agent_cache")
            oldest, newest = cursor.fetchone()

            return {
                "backend": "local",
                "db_path": self.db_path,
                "total_entries": count,
                "unique_functions": unique_functions,
                "unique_classes": unique_classes,
                "oldest_entry": oldest,
                "newest_entry": newest
            }


class RedisCacheBackend(BaseCacheBackend):
    """
    Redis cache backend.
    """

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_user: str = "default",
        redis_password: str = "",
        redis_db: int = 1
    ):
        try:
            import redis
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                username=redis_user if redis_user else None,
                password=redis_password if redis_password else None,
                db=redis_db,
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info(f"RedisCache initialized: host={redis_host}, port={redis_port}, db={redis_db}")
        except ImportError:
            raise ImportError("redis package is required. Install with: pip install redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}", exc_info=True)
            raise

    def _get_redis_key(self, key: str) -> str:
        return f"agent_cache:{key}"

    def _get_impl(self, key: str) -> Optional[Any]:
        redis_key = self._get_redis_key(key)
        result_json = self.redis_client.get(redis_key)

        if result_json:
            # Update metadata last_accessed
            metadata_key = f"{redis_key}:metadata"
            self.redis_client.hset(metadata_key, "last_accessed", datetime.now().isoformat())
            return json.loads(result_json)
        return None

    def _set_impl(self, key: str, func_name: str, class_name: Optional[str], params: dict, result: Any, ttl: int):
        redis_key = self._get_redis_key(key)
        result_json = json.dumps(result, default=str)

        # Store result
        self.redis_client.setex(redis_key, ttl, result_json)

        # Store metadata
        metadata_key = f"{redis_key}:metadata"
        metadata = {
            "function_name": func_name,
            "class_name": class_name or "",
            "parameters_json": json.dumps(params, sort_keys=True, default=str),
            "created_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat()
        }
        self.redis_client.hset(metadata_key, mapping=metadata)
        # Metadata should expire when the key expires, but Redis doesn't link them automatically.
        # We set a slightly longer TTL for metadata or same.
        self.redis_client.expire(metadata_key, ttl)

    def _clear_impl(self, function_name: Optional[str], class_name: Optional[str]) -> Optional[int]:
        pattern = "agent_cache:*"
        # Use scan_iter instead of keys to avoid blocking Redis
        keys_iterator = self.redis_client.scan_iter(pattern)
        deleted = 0
        keys_to_delete = []

        # If clearing everything
        if not function_name and not class_name:
            # Batch delete for performance
            batch_size = 1000
            for key in keys_iterator:
                keys_to_delete.append(key)
                # Also clear metadata
                keys_to_delete.append(f"{key}:metadata")

                if len(keys_to_delete) >= batch_size:
                    self.redis_client.delete(*keys_to_delete)
                    deleted += len(keys_to_delete) // 2  # Approximate count (key + metadata)
                    keys_to_delete = []

            if keys_to_delete:
                self.redis_client.delete(*keys_to_delete)
                deleted += len(keys_to_delete) // 2

            return deleted

        # Selective clear
        for key in keys_iterator:
            if key.endswith(":metadata"):
                continue

            metadata_key = f"{key}:metadata"
            # Pipeline get to reduce RTT? Or just get.
            # Since we need to check values, we have to fetch.
            metadata = self.redis_client.hgetall(metadata_key)

            match = False
            if function_name and class_name:
                if (metadata.get("function_name") == function_name and
                    metadata.get("class_name") == class_name):
                    match = True
            elif function_name:
                if metadata.get("function_name") == function_name:
                    match = True
            elif class_name:
                if metadata.get("class_name") == class_name:
                    match = True

            if match:
                self.redis_client.delete(key)
                self.redis_client.delete(metadata_key)
                deleted += 1

        return deleted

    def _get_stats_impl(self) -> Dict[str, Any]:
        pattern = "agent_cache:*"
        # Use scan_iter instead of keys
        keys_iterator = self.redis_client.scan_iter(pattern)

        unique_functions = set()
        unique_classes = set()
        oldest = None
        newest = None
        total_entries = 0

        for key in keys_iterator:
            if key.endswith(":metadata"):
                continue

            total_entries += 1
            metadata = self.redis_client.hgetall(f"{key}:metadata")
            if metadata:
                if metadata.get("function_name"):
                    unique_functions.add(metadata["function_name"])
                if metadata.get("class_name"):
                    unique_classes.add(metadata["class_name"])

                created = metadata.get("created_at")
                if created:
                    if oldest is None or created < oldest:
                        oldest = created
                    if newest is None or created > newest:
                        newest = created

        conn_kwargs = self.redis_client.connection_pool.connection_kwargs
        return {
            "backend": "redis",
            "host": conn_kwargs.get("host", "unknown"),
            "port": conn_kwargs.get("port", "unknown"),
            "db": conn_kwargs.get("db", "unknown"),
            "total_entries": total_entries,
            "unique_functions": len(unique_functions),
            "unique_classes": len(unique_classes),
            "oldest_entry": oldest,
            "newest_entry": newest
        }
