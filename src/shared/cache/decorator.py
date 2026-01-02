"""
Cache decorator for agent methods.
"""

import functools
import inspect
from typing import Any, Callable, Optional, TypeVar, cast

from .factory import get_agent_cache
from ...shared.config.settings import settings
from ...shared.utils.logger import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def cached(
    enabled: bool = True,
    exclude_class_name: bool = False,
    ttl: int = 86400
) -> Callable[[F], F]:
    """
    Decorator for caching agent method results with TTL support.

    Supports both synchronous and asynchronous methods.
    """
    def decorator(func: F) -> F:
        sig = inspect.signature(func)
        # Heuristic to detect instance methods
        is_instance_method = 'self' in sig.parameters

        def _get_cache_key_parts(args: tuple, kwargs: dict) -> tuple[Optional[str], str]:
            """Helper to determine class name and function name for logging."""
            class_name = None
            if is_instance_method and args and not exclude_class_name:
                instance = args[0]
                class_name = type(instance).__name__

            func_display_name = (
                f"{class_name}.{func.__name__}" if class_name else func.__name__
            )
            return class_name, func_display_name

        def _should_skip_cache() -> bool:
            """Check if caching is disabled via args or global settings."""
            return not enabled or not getattr(settings, 'use_chain_cache', True)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            if _should_skip_cache():
                return func(*args, **kwargs)

            cache = get_agent_cache()
            class_name, func_name = _get_cache_key_parts(args, kwargs)

            # Try to get from cache
            cached_result = cache.get(func, args, kwargs, class_name)
            if cached_result is not None:
                logger.info(f"Cache hit for {func_name}")
                return cached_result

            # Execute and cache
            result = func(*args, **kwargs)
            cache.set(func, args, kwargs, result, class_name, ttl)
            return result

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if _should_skip_cache():
                return await func(*args, **kwargs)

            cache = get_agent_cache()
            class_name, func_name = _get_cache_key_parts(args, kwargs)

            # Try to get from cache
            cached_result = cache.get(func, args, kwargs, class_name)
            if cached_result is not None:
                logger.info(f"Cache hit for {func_name}")
                return cached_result

            # Execute and cache (awaiting the result)
            result = await func(*args, **kwargs)
            cache.set(func, args, kwargs, result, class_name, ttl)
            return result

        if inspect.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        return cast(F, sync_wrapper)

    return decorator
