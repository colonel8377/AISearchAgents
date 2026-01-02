"""
Utility functions for cache key generation and parameter serialization.
"""

import hashlib
import json
import inspect
from typing import Any, Dict, Callable, Optional
from functools import lru_cache


def serialize_parameter(value: Any, visited: Optional[set] = None) -> Any:
    """
    Serialize a parameter value for hashing.
    
    Args:
        value: Parameter value to serialize
        visited: Set of visited object IDs to prevent recursion loops

    Returns:
        Serialized representation of the value
    """
    # Optimization: Handle primitives immediately to avoid set creation overhead
    if value is None:
        return None
    elif isinstance(value, (str, int, float, bool)):
        return value

    if visited is None:
        visited = set()

    # Check for circular references
    obj_id = id(value)
    if obj_id in visited:
        return f"<CircularReference {type(value).__name__}>"

    visited.add(obj_id)

    try:
        if isinstance(value, bytes):
            # Convert bytes to hash for consistent caching
            return {
                '__bytes__': True,
                '__hash__': hashlib.sha256(value).hexdigest()
            }
        elif isinstance(value, (list, tuple)):
            return [serialize_parameter(item, visited) for item in value]
        elif isinstance(value, dict):
            # Sort keys safely - handle non-comparable keys (e.g., enums)
            # Convert keys to strings for consistent hashing, but preserve type info
            try:
                # Try to sort by string representation of keys
                sorted_items = sorted(value.items(), key=lambda x: (str(type(x[0]).__name__), str(x[0])))
            except (TypeError, ValueError):
                # If sorting fails (e.g., enum comparison), use original order
                # Python 3.7+ dicts maintain insertion order
                sorted_items = value.items()
            
            # Serialize both keys and values
            result = {}
            for k, v in sorted_items:
                # Serialize key to string representation, but include type info for complex types
                if isinstance(k, (str, int, float, bool)) or k is None:
                    key_str = str(k)
                else:
                    # For complex keys (enums, objects), include type info
                    key_str = f"{type(k).__name__}:{str(k)}"
                result[key_str] = serialize_parameter(v, visited)
            return result
        elif hasattr(value, '__dict__'):
            return {
                '__class__': type(value).__name__,
                '__module__': getattr(type(value), '__module__', ''),
                '__dict__': serialize_parameter(value.__dict__, visited)
            }
        elif hasattr(value, '__slots__'):
            return {
                '__class__': type(value).__name__,
                '__module__': getattr(type(value), '__module__', ''),
                '__slots__': {
                    slot: serialize_parameter(getattr(value, slot, None), visited)
                    for slot in value.__slots__
                }
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
    finally:
        visited.remove(obj_id)


@lru_cache(maxsize=1024)
def _get_signature(func: Callable) -> inspect.Signature:
    """Cache function signature to avoid repeated inspection overhead."""
    return inspect.signature(func)


def generate_cache_key(
    func: Callable,
    args: tuple,
    kwargs: dict,
    class_name: Optional[str] = None
) -> str:
    """
    Generate a cache key from function signature and parameters.
    
    Args:
        func: Function to generate key for
        args: Function positional arguments
        kwargs: Function keyword arguments
        class_name: Optional class name to include in key
        
    Returns:
        SHA256 hash of the cache key
    """
    sig = _get_signature(func)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()
    
    params_dict = {}
    for param_name, param_value in bound_args.arguments.items():
        if param_name == 'self' and class_name:
            continue
        params_dict[param_name] = serialize_parameter(param_value)
    
    cache_data = {
        'function_name': func.__name__,
        'class_name': class_name,
        'module': getattr(func, '__module__', ''),
        'qualname': getattr(func, '__qualname__', func.__name__),
        'parameters': params_dict
    }
    
    cache_str = json.dumps(cache_data, sort_keys=True, default=str)
    cache_key = hashlib.sha256(cache_str.encode()).hexdigest()
    
    return cache_key


def get_parameters_dict(
    func: Callable,
    args: tuple,
    kwargs: dict,
    class_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract parameters dictionary from function call.
    
    Args:
        func: Function to extract parameters from
        args: Function positional arguments
        kwargs: Function keyword arguments
        class_name: Optional class name
        
    Returns:
        Dictionary of parameter names to serialized values
    """
    sig = _get_signature(func)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()
    
    params_dict = {}
    for param_name, param_value in bound_args.arguments.items():
        if param_name == 'self' and class_name:
            continue
        params_dict[param_name] = serialize_parameter(param_value)
    
    return params_dict
