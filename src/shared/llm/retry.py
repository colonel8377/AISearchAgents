"""
Unified retry decorator for LLM client calls.

This module provides a standardized @llm_retry decorator for all LLM API calls,
ensuring consistent retry behavior across the application.
"""

from functools import wraps
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import asyncio
import inspect


def llm_retry(func):
    """
    Unified retry decorator for LLM client calls.
    
    This decorator applies consistent retry logic to LLM API calls:
    - Maximum 3 attempts
    - Exponential backoff: multiplier=1, min=2s, max=10s
    - Retries on all exceptions
    - Supports both sync and async functions
    
    Usage:
        @llm_retry
        def call_llm(...):
            ...
        
        @llm_retry
        async def call_llm_async(...):
            ...
    
    Args:
        func: The function to decorate (sync or async)
        
    Returns:
        Decorated function with retry logic
    """
    if inspect.iscoroutinefunction(func):
        # Async function
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type(Exception)
        )
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        
        return async_wrapper
    else:
        # Sync function
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type(Exception)
        )
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        return sync_wrapper

