"""
Legacy state manager module - redirects to new modules.

This file is kept for backward compatibility. New code should import from
interfaces and in_memory_state_manager directly.
"""

from .in_memory_state_manager import (
    InMemoryStateManager,
    get_memory_state_manager,
    BaseMemoryStateManager,
    clear_in_memory_state
)

__all__ = [
    "BaseMemoryStateManager",
    "InMemoryStateManager",
    "get_memory_state_manager",
    "clear_in_memory_state"
]
