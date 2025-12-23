"""Cache utility for web opinion analysis results to save tokens by avoiding reprocessing.

Supports both Redis and local SQLite cache backends.
"""

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from ..config.settings import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class WebOpinionLocalCache:
    """
    Cache for web opinion analysis results.
    
    Uses SQLite database to store results keyed by hash of input parameters.
    This saves tokens by avoiding reprocessing of the same URL with same parameters.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the cache.
        
        Args:
            db_path: Path to SQLite database file. Defaults to data/web_opinion_cache.db
        """
        if db_path is None:
            base_dir = Path(__file__).resolve().parents[2]
            db_path = str(base_dir / "data" / "web_opinion_cache.db")
        
        self.db_path = db_path
        # Ensure data directory exists
        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"WebOpinionLocalCache initialized: db_path={db_path}")
    
    def _init_db(self):
        """Initialize the cache database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_cache (
                cache_key TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_url ON analysis_cache(url)
        """)
        
        conn.commit()
        conn.close()
    
    def _generate_cache_key(
        self,
        url: str,
        mode: str,
        use_mbfc: bool,
        use_few_shots: bool,
        atomizer_shots: Optional[list] = None,
        scorer_shots: Optional[list] = None
    ) -> str:
        """
        Generate a cache key from input parameters.
        
        Args:
            url: URL to analyze
            mode: Logic mode (LOCAL_CHAIN, NO_CHAIN, PURE_ONLINE)
            use_mbfc: Whether MBFC is enabled
            use_few_shots: Whether few-shot examples are enabled
            atomizer_shots: Optional custom atomizer shots
            scorer_shots: Optional custom scorer shots
            
        Returns:
            SHA256 hash of the parameters as cache key
        """
        # Create a dictionary of all parameters
        params = {
            "url": url,
            "mode": mode,
            "use_mbfc": use_mbfc,
            "use_few_shots": use_few_shots,
            "atomizer_shots": json.dumps(atomizer_shots, sort_keys=True) if atomizer_shots else None,
            "scorer_shots": json.dumps(scorer_shots, sort_keys=True) if scorer_shots else None
        }
        
        # Convert to JSON string and hash
        params_str = json.dumps(params, sort_keys=True)
        cache_key = hashlib.sha256(params_str.encode()).hexdigest()
        
        return cache_key
    
    def get(
        self,
        url: str,
        mode: str,
        use_mbfc: bool,
        use_few_shots: bool,
        atomizer_shots: Optional[list] = None,
        scorer_shots: Optional[list] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached result if it exists.
        
        Args:
            url: URL to analyze
            mode: Logic mode
            use_mbfc: Whether MBFC is enabled
            use_few_shots: Whether few-shot examples are enabled
            atomizer_shots: Optional custom atomizer shots
            scorer_shots: Optional custom scorer shots
            
        Returns:
            Cached result dictionary or None if not found
        """
        cache_key = self._generate_cache_key(url, mode, use_mbfc, use_few_shots, atomizer_shots, scorer_shots)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT result_json, last_accessed FROM analysis_cache
                WHERE cache_key = ?
            """, (cache_key,))
            
            row = cursor.fetchone()
            
            if row:
                result_json, _ = row
                # Update last_accessed timestamp
                cursor.execute("""
                    UPDATE analysis_cache
                    SET last_accessed = CURRENT_TIMESTAMP
                    WHERE cache_key = ?
                """, (cache_key,))
                conn.commit()
                
                result = json.loads(result_json)
                logger.info(f"Cache hit for URL: {url} (key: {cache_key[:16]}...)")
                conn.close()
                return result
            else:
                logger.debug(f"Cache miss for URL: {url} (key: {cache_key[:16]}...)")
                conn.close()
                return None
                
        except Exception as e:
            logger.error(f"Error reading from cache: {e}", exc_info=True)
            return None
    
    def set(
        self,
        url: str,
        mode: str,
        use_mbfc: bool,
        use_few_shots: bool,
        result: Dict[str, Any],
        atomizer_shots: Optional[list] = None,
        scorer_shots: Optional[list] = None
    ):
        """
        Store result in cache.
        
        Args:
            url: URL that was analyzed
            mode: Logic mode used
            use_mbfc: Whether MBFC was enabled
            use_few_shots: Whether few-shot examples were enabled
            result: Result dictionary to cache
            atomizer_shots: Optional custom atomizer shots used
            scorer_shots: Optional custom scorer shots used
        """
        cache_key = self._generate_cache_key(url, mode, use_mbfc, use_few_shots, atomizer_shots, scorer_shots)
        result_json = json.dumps(result)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO analysis_cache
                (cache_key, url, result_json, created_at, last_accessed)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """, (cache_key, url, result_json))
            
            conn.commit()
            conn.close()
            logger.info(f"Cached result for URL: {url} (key: {cache_key[:16]}...)")
            
        except Exception as e:
            logger.error(f"Error writing to cache: {e}", exc_info=True)
    
    def clear(self):
        """Clear all cached results."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM analysis_cache")
            conn.commit()
            conn.close()
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}", exc_info=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM analysis_cache")
            count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT url) FROM analysis_cache")
            unique_urls = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT MIN(created_at), MAX(created_at)
                FROM analysis_cache
            """)
            row = cursor.fetchone()
            oldest = row[0] if row[0] else None
            newest = row[1] if row[1] else None
            
            conn.close()
            
            return {
                "total_entries": count,
                "unique_urls": unique_urls,
                "oldest_entry": oldest,
                "newest_entry": newest
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}", exc_info=True)
            return {"error": str(e)}


class WebOpinionRedisCache:
    """
    Redis cache backend for web opinion analysis results.
    
    Uses Redis to store results keyed by hash of input parameters.
    """
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379, 
                 redis_password: str = "", redis_db: int = 2):
        """
        Initialize the Redis cache.
        
        Args:
            redis_host: Redis host
            redis_port: Redis port
            redis_password: Redis password (if required)
            redis_db: Redis database number (default: 2, separate from agent cache)
        """
        try:
            import redis
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                password=redis_password if redis_password else None,
                db=redis_db,
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info(f"WebOpinionRedisCache initialized: host={redis_host}, port={redis_port}, db={redis_db}")
        except ImportError:
            raise ImportError("redis package is required for Redis cache backend. Install with: pip install redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}", exc_info=True)
            raise
    
    def _generate_cache_key(
        self,
        url: str,
        mode: str,
        use_mbfc: bool,
        use_few_shots: bool,
        atomizer_shots: Optional[list] = None,
        scorer_shots: Optional[list] = None
    ) -> str:
        """Generate a cache key from input parameters."""
        params = {
            "url": url,
            "mode": mode,
            "use_mbfc": use_mbfc,
            "use_few_shots": use_few_shots,
            "atomizer_shots": json.dumps(atomizer_shots, sort_keys=True) if atomizer_shots else None,
            "scorer_shots": json.dumps(scorer_shots, sort_keys=True) if scorer_shots else None
        }
        params_str = json.dumps(params, sort_keys=True)
        cache_key = hashlib.sha256(params_str.encode()).hexdigest()
        return f"web_opinion_cache:{cache_key}"
    
    def get(
        self,
        url: str,
        mode: str,
        use_mbfc: bool,
        use_few_shots: bool,
        atomizer_shots: Optional[list] = None,
        scorer_shots: Optional[list] = None
    ) -> Optional[Dict[str, Any]]:
        """Get cached result if it exists."""
        cache_key = self._generate_cache_key(url, mode, use_mbfc, use_few_shots, atomizer_shots, scorer_shots)
        
        try:
            result_json = self.redis_client.get(cache_key)
            if result_json:
                result = json.loads(result_json)
                logger.info(f"RedisCache hit for URL: {url} (key: {cache_key[:16]}...)")
                return result
            else:
                logger.debug(f"RedisCache miss for URL: {url} (key: {cache_key[:16]}...)")
                return None
        except Exception as e:
            logger.error(f"Error reading from Redis cache: {e}", exc_info=True)
            return None
    
    def set(
        self,
        url: str,
        mode: str,
        use_mbfc: bool,
        use_few_shots: bool,
        result: Dict[str, Any],
        atomizer_shots: Optional[list] = None,
        scorer_shots: Optional[list] = None
    ):
        """Store result in cache."""
        cache_key = self._generate_cache_key(url, mode, use_mbfc, use_few_shots, atomizer_shots, scorer_shots)
        result_json = json.dumps(result)
        
        try:
            self.redis_client.set(cache_key, result_json)
            logger.info(f"RedisCache stored result for URL: {url} (key: {cache_key[:16]}...)")
        except Exception as e:
            logger.error(f"Error writing to Redis cache: {e}", exc_info=True)
    
    def clear(self):
        """Clear all cached results."""
        try:
            keys = self.redis_client.keys("web_opinion_cache:*")
            if keys:
                self.redis_client.delete(*keys)
            logger.info("RedisCache cleared")
        except Exception as e:
            logger.error(f"Error clearing Redis cache: {e}", exc_info=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            keys = self.redis_client.keys("web_opinion_cache:*")
            unique_urls = set()
            for key in keys:
                result_json = self.redis_client.get(key)
                if result_json:
                    result = json.loads(result_json)
                    if result.get("url"):
                        unique_urls.add(result["url"])
            
            return {
                "backend": "redis",
                "total_entries": len(keys),
                "unique_urls": len(unique_urls),
            }
        except Exception as e:
            logger.error(f"Error getting Redis cache stats: {e}", exc_info=True)
            return {"error": str(e)}


# Alias for backward compatibility
WebOpinionCache = WebOpinionLocalCache


def create_web_opinion_cache():
    """
    Create web opinion cache backend based on settings.
    
    Returns:
        WebOpinionLocalCache or WebOpinionRedisCache instance
    """
    cache_backend = getattr(settings, 'cache_backend', 'local')
    
    if cache_backend == "redis":
        try:
            return WebOpinionRedisCache(
                redis_host=settings.redis_host,
                redis_port=settings.redis_port,
                redis_password=settings.redis_password,
                redis_db=getattr(settings, 'cache_redis_db', 2)  # Use db 2 for web opinion cache
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Redis cache, falling back to local: {e}")
            return WebOpinionLocalCache()
    else:
        return WebOpinionLocalCache()


# Global cache instance
_cache_instance = None


def get_cache():
    """Get or create the global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = create_web_opinion_cache()
    return _cache_instance

