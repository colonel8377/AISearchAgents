"""General-purpose caching decorator for agent methods using function signatures (AOP-style).

Supports both Redis and local SQLite cache backends.
"""

import hashlib
import json
import sqlite3
import inspect
import functools
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Union, Protocol
from datetime import datetime
from abc import ABC, abstractmethod

from ..config.settings import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CacheBackend(Protocol):
    """Protocol for cache backend implementations."""
    
    def get(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        class_name: Optional[str] = None
    ) -> Optional[Any]:
        """Get cached result if it exists."""
        ...
    
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
        ...
    
    def clear(self, function_name: Optional[str] = None, class_name: Optional[str] = None):
        """Clear cached results."""
        ...
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        ...


class LocalCache:
    """
    Local SQLite cache backend for agent method results.
    
    Uses SQLite database to store results keyed by hash of function signature
    and all parameters. This saves tokens by avoiding reprocessing of the same
    inputs with the same few_shots/formats.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the local cache.
        
        Args:
            db_path: Path to SQLite database file. Defaults to data/agent_cache.db
        """
        if db_path is None:
            base_dir = Path(__file__).resolve().parents[2]
            db_path = str(base_dir / "data" / "agent_cache.db")
        
        self.db_path = db_path
        # Ensure data directory exists
        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"LocalCache initialized: db_path={db_path}")
    
    def _init_db(self):
        """Initialize the cache database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
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
        
        # Create indexes for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_function_name ON agent_cache(function_name)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_class_function ON agent_cache(class_name, function_name)
        """)
        
        conn.commit()
        conn.close()
    
    def _serialize_parameter(self, value: Any) -> Any:
        """Serialize a parameter value for hashing."""
        if value is None:
            return None
        elif isinstance(value, (str, int, float, bool)):
            return value
        elif isinstance(value, (list, tuple)):
            return [self._serialize_parameter(item) for item in value]
        elif isinstance(value, dict):
            return {k: self._serialize_parameter(v) for k, v in sorted(value.items())}
        elif hasattr(value, '__dict__'):
            return {
                '__class__': type(value).__name__,
                '__module__': getattr(type(value), '__module__', ''),
                '__dict__': self._serialize_parameter(value.__dict__)
            }
        elif hasattr(value, '__slots__'):
            return {
                '__class__': type(value).__name__,
                '__module__': getattr(type(value), '__module__', ''),
                '__slots__': {slot: self._serialize_parameter(getattr(value, slot, None)) 
                             for slot in value.__slots__}
            }
        elif callable(value):
            if hasattr(value, '__name__'):
                return {
                    '__callable__': True,
                    '__name__': value.__name__,
                    '__module__': getattr(value, '__module__', ''),
                    '__qualname__': getattr(value, '__qualname__', '')
                }
            return {'__callable__': True, '__str__': str(value)}
        else:
            try:
                return str(value)
            except Exception:
                return repr(value)
    
    def _generate_cache_key(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        class_name: Optional[str] = None
    ) -> str:
        """Generate a cache key from function signature and parameters."""
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        params_dict = {}
        for param_name, param_value in bound_args.arguments.items():
            if param_name == 'self' and class_name:
                continue
            params_dict[param_name] = self._serialize_parameter(param_value)
        
        cache_data = {
            'function_name': func.__name__,
            'class_name': class_name,
            'module': func.__module__,
            'qualname': getattr(func, '__qualname__', func.__name__),
            'parameters': params_dict
        }
        
        cache_str = json.dumps(cache_data, sort_keys=True, default=str)
        cache_key = hashlib.sha256(cache_str.encode()).hexdigest()
        
        return cache_key
    
    def get(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        class_name: Optional[str] = None
    ) -> Optional[Any]:
        """Get cached result if it exists."""
        cache_key = self._generate_cache_key(func, args, kwargs, class_name)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT result_json, last_accessed FROM agent_cache
                WHERE cache_key = ? AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """, (cache_key,))

            row = cursor.fetchone()

            if row:
                result_json, _ = row
                cursor.execute("""
                    UPDATE agent_cache
                    SET last_accessed = CURRENT_TIMESTAMP
                    WHERE cache_key = ?
                """, (cache_key,))
                conn.commit()
                
                result = json.loads(result_json)
                logger.debug(f"LocalCache hit for {func.__name__} (key: {cache_key[:16]}...)")
                conn.close()
                return result
            else:
                logger.debug(f"LocalCache miss for {func.__name__} (key: {cache_key[:16]}...)")
                conn.close()
                return None
                
        except Exception as e:
            logger.error(f"Error reading from local cache: {e}", exc_info=True)
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
        cache_key = self._generate_cache_key(func, args, kwargs, class_name)
        
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        params_dict = {}
        for param_name, param_value in bound_args.arguments.items():
            if param_name == 'self' and class_name:
                continue
            params_dict[param_name] = self._serialize_parameter(param_value)
        
        params_json = json.dumps(params_dict, sort_keys=True, default=str)
        result_json = json.dumps(result, default=str)

        # Calculate expiration time
        from datetime import datetime, timedelta
        expires_at = datetime.now() + timedelta(seconds=ttl)

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO agent_cache
                (cache_key, function_name, class_name, parameters_json, result_json, created_at, last_accessed, expires_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
            """, (cache_key, func.__name__, class_name, params_json, result_json, expires_at.isoformat()))
            
            conn.commit()
            conn.close()
            logger.debug(f"LocalCache stored result for {func.__name__} (key: {cache_key[:16]}...)")
            
        except Exception as e:
            logger.error(f"Error writing to local cache: {e}", exc_info=True)
    
    def clear(self, function_name: Optional[str] = None, class_name: Optional[str] = None):
        """Clear cached results."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if function_name and class_name:
                cursor.execute("""
                    DELETE FROM agent_cache
                    WHERE function_name = ? AND class_name = ?
                """, (function_name, class_name))
            elif function_name:
                cursor.execute("""
                    DELETE FROM agent_cache
                    WHERE function_name = ?
                """, (function_name,))
            elif class_name:
                cursor.execute("""
                    DELETE FROM agent_cache
                    WHERE class_name = ?
                """, (class_name,))
            else:
                cursor.execute("DELETE FROM agent_cache")
            
            conn.commit()
            conn.close()
            logger.info(f"LocalCache cleared: function={function_name}, class={class_name}")
        except Exception as e:
            logger.error(f"Error clearing local cache: {e}", exc_info=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM agent_cache")
            count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT function_name) FROM agent_cache")
            unique_functions = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT class_name) FROM agent_cache WHERE class_name IS NOT NULL")
            unique_classes = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT MIN(created_at), MAX(created_at)
                FROM agent_cache
            """)
            row = cursor.fetchone()
            oldest = row[0] if row[0] else None
            newest = row[1] if row[1] else None
            
            conn.close()
            
            return {
                "backend": "local",
                "db_path": self.db_path,
                "total_entries": count,
                "unique_functions": unique_functions,
                "unique_classes": unique_classes,
                "oldest_entry": oldest,
                "newest_entry": newest
            }
        except Exception as e:
            logger.error(f"Error getting local cache stats: {e}", exc_info=True)
            return {"error": str(e)}


class RedisCache:
    """
    Redis cache backend for agent method results.
    
    Uses Redis to store results keyed by hash of function signature
    and all parameters. Provides distributed caching capabilities.
    """
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379, 
                 redis_user: str = "default", redis_password: str = "", redis_db: int = 1):
        """
        Initialize the Redis cache.
        
        Args:
            redis_host: Redis host
            redis_port: Redis port
            redis_password: Redis password (if required)
            redis_db: Redis database number (default: 1, separate from vector store)
        """
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
            # Test connection
            self.redis_client.ping()
            logger.info(f"RedisCache initialized: host={redis_host}, port={redis_port}, db={redis_db}")
        except ImportError:
            raise ImportError("redis package is required for Redis cache backend. Install with: pip install redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}", exc_info=True)
            raise
    
    def _serialize_parameter(self, value: Any) -> Any:
        """Serialize a parameter value for hashing (same as LocalCache)."""
        if value is None:
            return None
        elif isinstance(value, (str, int, float, bool)):
            return value
        elif isinstance(value, (list, tuple)):
            return [self._serialize_parameter(item) for item in value]
        elif isinstance(value, dict):
            return {k: self._serialize_parameter(v) for k, v in sorted(value.items())}
        elif hasattr(value, '__dict__'):
            return {
                '__class__': type(value).__name__,
                '__module__': getattr(type(value), '__module__', ''),
                '__dict__': self._serialize_parameter(value.__dict__)
            }
        elif hasattr(value, '__slots__'):
            return {
                '__class__': type(value).__name__,
                '__module__': getattr(type(value), '__module__', ''),
                '__slots__': {slot: self._serialize_parameter(getattr(value, slot, None)) 
                             for slot in value.__slots__}
            }
        elif callable(value):
            if hasattr(value, '__name__'):
                return {
                    '__callable__': True,
                    '__name__': value.__name__,
                    '__module__': getattr(value, '__module__', ''),
                    '__qualname__': getattr(value, '__qualname__', '')
                }
            return {'__callable__': True, '__str__': str(value)}
        else:
            try:
                return str(value)
            except Exception:
                return repr(value)
    
    def _generate_cache_key(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        class_name: Optional[str] = None
    ) -> str:
        """Generate a cache key from function signature and parameters."""
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        params_dict = {}
        for param_name, param_value in bound_args.arguments.items():
            if param_name == 'self' and class_name:
                continue
            params_dict[param_name] = self._serialize_parameter(param_value)
        
        cache_data = {
            'function_name': func.__name__,
            'class_name': class_name,
            'module': func.__module__,
            'qualname': getattr(func, '__qualname__', func.__name__),
            'parameters': params_dict
        }
        
        cache_str = json.dumps(cache_data, sort_keys=True, default=str)
        cache_key = hashlib.sha256(cache_str.encode()).hexdigest()
        
        # Prefix with namespace for Redis
        return f"agent_cache:{cache_key}"
    
    def get(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        class_name: Optional[str] = None
    ) -> Optional[Any]:
        """Get cached result if it exists."""
        cache_key = self._generate_cache_key(func, args, kwargs, class_name)
        
        try:
            result_json = self.redis_client.get(cache_key)
            
            if result_json:
                result = json.loads(result_json)
                # Update last accessed (using separate key for metadata)
                metadata_key = f"{cache_key}:metadata"
                self.redis_client.hset(metadata_key, "last_accessed", datetime.now().isoformat())
                logger.debug(f"RedisCache hit for {func.__name__} (key: {cache_key[:16]}...)")
                return result
            else:
                logger.debug(f"RedisCache miss for {func.__name__} (key: {cache_key[:16]}...)")
                return None
                
        except Exception as e:
            logger.error(f"Error reading from Redis cache: {e}", exc_info=True)
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
        cache_key = self._generate_cache_key(func, args, kwargs, class_name)
        
        try:
            result_json = json.dumps(result, default=str)
            
            # Store result with TTL
            self.redis_client.setex(cache_key, ttl, result_json)
            
            # Store metadata
            metadata_key = f"{cache_key}:metadata"
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            params_dict = {}
            for param_name, param_value in bound_args.arguments.items():
                if param_name == 'self' and class_name:
                    continue
                params_dict[param_name] = self._serialize_parameter(param_value)
            
            metadata = {
                "function_name": func.__name__,
                "class_name": class_name or "",
                "parameters_json": json.dumps(params_dict, sort_keys=True, default=str),
                "created_at": datetime.now().isoformat(),
                "last_accessed": datetime.now().isoformat()
            }
            self.redis_client.hset(metadata_key, mapping=metadata)
            
            logger.debug(f"RedisCache stored result for {func.__name__} (key: {cache_key[:16]}...)")
            
        except Exception as e:
            logger.error(f"Error writing to Redis cache: {e}", exc_info=True)
    
    def clear(self, function_name: Optional[str] = None, class_name: Optional[str] = None):
        """Clear cached results."""
        try:
            pattern = "agent_cache:*"
            if function_name and class_name:
                # More specific pattern - would need to scan keys
                pattern = f"agent_cache:*"
                keys = self.redis_client.keys(pattern)
                deleted = 0
                for key in keys:
                    metadata_key = f"{key}:metadata"
                    metadata = self.redis_client.hgetall(metadata_key)
                    if metadata.get("function_name") == function_name and metadata.get("class_name") == class_name:
                        self.redis_client.delete(key)
                        self.redis_client.delete(metadata_key)
                        deleted += 1
                logger.info(f"RedisCache cleared: {deleted} entries (function={function_name}, class={class_name})")
            elif function_name:
                keys = self.redis_client.keys(pattern)
                deleted = 0
                for key in keys:
                    metadata_key = f"{key}:metadata"
                    metadata = self.redis_client.hgetall(metadata_key)
                    if metadata.get("function_name") == function_name:
                        self.redis_client.delete(key)
                        self.redis_client.delete(metadata_key)
                        deleted += 1
                logger.info(f"RedisCache cleared: {deleted} entries (function={function_name})")
            elif class_name:
                keys = self.redis_client.keys(pattern)
                deleted = 0
                for key in keys:
                    metadata_key = f"{key}:metadata"
                    metadata = self.redis_client.hgetall(metadata_key)
                    if metadata.get("class_name") == class_name:
                        self.redis_client.delete(key)
                        self.redis_client.delete(metadata_key)
                        deleted += 1
                logger.info(f"RedisCache cleared: {deleted} entries (class={class_name})")
            else:
                # Clear all agent_cache keys
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
                    # Also delete metadata keys
                    metadata_keys = self.redis_client.keys("agent_cache:*:metadata")
                    if metadata_keys:
                        self.redis_client.delete(*metadata_keys)
                logger.info(f"RedisCache cleared: all entries")
        except Exception as e:
            logger.error(f"Error clearing Redis cache: {e}", exc_info=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            pattern = "agent_cache:*"
            keys = [k for k in self.redis_client.keys(pattern) if not k.endswith(":metadata")]
            
            unique_functions = set()
            unique_classes = set()
            oldest = None
            newest = None
            
            for key in keys:
                metadata_key = f"{key}:metadata"
                metadata = self.redis_client.hgetall(metadata_key)
                if metadata:
                    if metadata.get("function_name"):
                        unique_functions.add(metadata["function_name"])
                    if metadata.get("class_name"):
                        unique_classes.add(metadata["class_name"])
                    
                    created_at = metadata.get("created_at")
                    if created_at:
                        if oldest is None or created_at < oldest:
                            oldest = created_at
                        if newest is None or created_at > newest:
                            newest = created_at
            
            return {
                "backend": "redis",
                "host": self.redis_client.connection_pool.connection_kwargs.get("host", "unknown"),
                "port": self.redis_client.connection_pool.connection_kwargs.get("port", "unknown"),
                "db": self.redis_client.connection_pool.connection_kwargs.get("db", "unknown"),
                "total_entries": len(keys),
                "unique_functions": len(unique_functions),
                "unique_classes": len(unique_classes),
                "oldest_entry": oldest,
                "newest_entry": newest
            }
        except Exception as e:
            logger.error(f"Error getting Redis cache stats: {e}", exc_info=True)
            return {"error": str(e)}


# Alias for backward compatibility
AgentCache = LocalCache


def create_cache_backend() -> CacheBackend:
    """
    Create cache backend based on settings.
    
    Returns:
        CacheBackend instance (LocalCache or RedisCache)
    """
    cache_backend = getattr(settings, 'cache_backend', 'local')
    
    if cache_backend == "redis":
        try:
            return RedisCache(
                redis_host=settings.redis_host,
                redis_port=settings.redis_port,
                redis_user=settings.redis_user,
                redis_password=settings.redis_password,
                redis_db=getattr(settings, 'cache_redis_db', 1)
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Redis cache, falling back to local: {e}")
            return LocalCache(db_path=getattr(settings, 'cache_db_path', None))
    else:
        # Default to local cache
        return LocalCache(db_path=getattr(settings, 'cache_db_path', None))


# Global cache instance
_cache_instance: Optional[CacheBackend] = None


def get_agent_cache() -> CacheBackend:
    """Get or create the global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = create_cache_backend()
    return _cache_instance


def cached(enabled: bool = True, exclude_class_name: bool = False, ttl: int = 86400):
    """
    Decorator for caching agent method results with TTL support.

    Uses function signature and all parameters to generate cache keys.
    Automatically detects if method is instance method (includes class name by default).
    Works with both Redis and local cache backends based on settings.

    Args:
        enabled: Whether caching is enabled (default: True). Can be disabled via settings.
        exclude_class_name: Whether to exclude class name from cache key (default: False).
                           Useful for one-time evaluations without memory state.
        ttl: Time-to-live in seconds (default: 86400 = 24 hours).

    Example:
        @cached()
        def my_method(self, text: str, mode: str = "default"):
            # Method implementation
            return result

        @cached(ttl=3600)  # 1 hour TTL
        def quick_expiring_method(self, data):
            return result

        @cached(exclude_class_name=True)
        def evaluate_once(self, data):
            # One-time evaluation, no class prefix in cache key
            return result
    """
    def decorator(func: Callable) -> Callable:
        # Check if caching is enabled globally
        if not enabled or not getattr(settings, 'use_chain_cache', True):
            return func
        
        # Determine if this is an instance method
        sig = inspect.signature(func)
        is_instance_method = 'self' in sig.parameters
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_agent_cache()

            # Get class name if instance method and not excluded
            class_name = None
            if is_instance_method and args and not exclude_class_name:
                instance = args[0]
                class_name = type(instance).__name__

            # Check cache
            cached_result = cache.get(func, args, kwargs, class_name)
            if cached_result is not None:
                logger.info(f"Cache hit for {class_name}.{func.__name__ if class_name else func.__name__}")
                return cached_result

            # Execute function
            result = func(*args, **kwargs)

            # Cache result with TTL
            cache.set(func, args, kwargs, result, class_name, ttl)

            return result
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache = get_agent_cache()

            # Get class name if instance method and not excluded
            class_name = None
            if is_instance_method and args and not exclude_class_name:
                instance = args[0]
                class_name = type(instance).__name__

            # Check cache
            cached_result = cache.get(func, args, kwargs, class_name)
            if cached_result is not None:
                logger.info(f"Cache hit for {class_name}.{func.__name__ if class_name else func.__name__}")
                return cached_result

            # Execute async function
            result = await func(*args, **kwargs)

            # Cache result with TTL
            cache.set(func, args, kwargs, result, class_name, ttl)

            return result
        
        # Return appropriate wrapper based on whether function is async
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


def llm_cached(enabled: bool = True, ttl: int = 86400, hash_content: bool = True):
    """
    Specialized decorator for caching LLM API calls to reduce token consumption.

    This decorator is designed specifically for LLM calls where:
    - System prompt and user message are the key parameters
    - Results should be cached based on exact prompt content
    - Hash-based caching for large prompts to avoid key size limits

    Args:
        enabled: Whether caching is enabled (default: True). Can be disabled via settings.
        ttl: Time-to-live in seconds (default: 86400 = 24 hours).
        hash_content: Whether to hash prompt content for cache keys (default: True).
                     Set to False for exact string matching in cache keys.

    Example:
        @llm_cached(ttl=3600)  # Cache for 1 hour
        async def call_llm(self, system_prompt: str, user_message: str) -> str:
            # LLM API call implementation
            return response

        @llm_cached(hash_content=False)  # Exact string matching
        async def call_llm_exact(self, system_prompt: str, user_message: str) -> str:
            return response
    """
    def decorator(func: Callable) -> Callable:
        # Check if caching is enabled globally
        if not enabled or not getattr(settings, 'use_llm_cache', True):
            return func

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache = get_agent_cache()

            # Extract system_prompt and user_message from arguments
            # Assume signature: (self, system_prompt, user_message, ...)
            if len(args) >= 3:  # self, system_prompt, user_message
                system_prompt = args[1]
                user_message = args[2]
            elif 'system_prompt' in kwargs and 'user_message' in kwargs:
                system_prompt = kwargs['system_prompt']
                user_message = kwargs['user_message']
            else:
                # Fallback to original behavior if signature doesn't match
                return await func(*args, **kwargs)

            # Create cache key based on prompts
            if hash_content:
                # Hash the content to avoid key size issues
                content_str = f"{system_prompt}\n---\n{user_message}"
                cache_key_data = {
                    'function_name': func.__name__,
                    'content_hash': hashlib.sha256(content_str.encode()).hexdigest(),
                    'content_length': len(content_str)
                }
            else:
                # Use exact content (be careful with size limits)
                cache_key_data = {
                    'function_name': func.__name__,
                    'system_prompt': system_prompt,
                    'user_message': user_message
                }

            # Generate cache key
            cache_key_str = json.dumps(cache_key_data, sort_keys=True)
            cache_key = f"llm:{hashlib.sha256(cache_key_str.encode()).hexdigest()}"

            # Check cache first
            try:
                cached_result = cache.redis_client.get(cache_key) if hasattr(cache, 'redis_client') else None
                if cached_result is None and hasattr(cache, '_generate_cache_key'):
                    # Try the standard cache interface
                    cached_result = cache.get(func, args, kwargs, "LLMService")

                if cached_result is not None:
                    logger.debug(f"LLM cache hit for {func.__name__}")
                    return cached_result
            except Exception as e:
                logger.debug(f"LLM cache lookup failed: {e}")

            # Execute LLM call
            try:
                result = await func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"LLM call failed: {e}")
                raise

            # Cache the result
            try:
                if hasattr(cache, 'redis_client'):
                    cache.redis_client.setex(cache_key, ttl, json.dumps(result, default=str))
                elif hasattr(cache, 'set'):
                    # Use standard cache interface
                    cache.set(func, args, kwargs, result, "LLMService", ttl)
                logger.debug(f"LLM result cached for {func.__name__}")
            except Exception as e:
                logger.debug(f"LLM cache storage failed: {e}")

            return result

        # Return wrapper (assuming all LLM calls are async)
        return async_wrapper

    return decorator
