"""
In-memory state management implementation.

This module provides functionality to clear all in-memory state including:
- Debate service sessions cache and statistics
- Agent manager instances
- Bot manager instances
"""

from typing import Dict, Any, Optional

from .base_memory_state_manager import BaseMemoryStateManager
from ...shared.utils.logger import get_logger


logger = get_logger(__name__)


class InMemoryStateManager(BaseMemoryStateManager):
    """
    Implementation for managing in-memory state cleanup.
    
    This class is responsible for clearing all cached state in the application,
    including debate sessions, agent managers, and bot managers.
    """
    
    def clear_all(self) -> Dict[str, Any]:
        """
        Clear all in-memory state including:
        - Debate service sessions cache and statistics
        - Agent manager instances
        - Bot manager instances
        
        Returns:
            Dict with:
            - success: bool - Whether all state was cleared successfully
            - errors: List[str] - List of error messages (if any)
        """
        result = {
            "success": True,
            "errors": []
        }
        
        def add_error(msg: str):
            """Helper to add error and mark result as failed."""
            logger.error(msg, exc_info=True)
            result["errors"].append(msg)
            result["success"] = False
        
        # Clear debate sessions
        try:
            from ...application.agents.debate.service import DebateService
            debate_service_instance = DebateService()
            debate_service_instance.clear_cache()
            logger.info("Debate sessions cache cleared (in-memory state)")
        except Exception as e:
            logger.warning(f"Failed to clear debate sessions cache: {e}")
            add_error(f"Failed to clear debate sessions cache: {e}")
        
        # Clear agent and bot managers
        try:
            from ...application.agents.manager import AgentManager
            # Note: This creates a new instance to clear any global state.
            # Ideally, there should be a singleton pattern or service locator.
            agent_manager = AgentManager()
            agent_manager.clear_all()
            logger.info("Agent manager cleared (in-memory state)")
        except Exception as e:
            logger.warning(f"Failed to clear agent managers: {e}")
            add_error(f"Failed to clear agent managers: {e}")
        
        return result


# Global memory state manager instance
_memory_state_manager: Optional[InMemoryStateManager] = None


def get_memory_state_manager() -> BaseMemoryStateManager:
    """
    Get or create the global memory state manager instance.
    
    Returns:
        Singleton BaseMemoryStateManager instance
    """
    global _memory_state_manager
    if _memory_state_manager is None:
        _memory_state_manager = InMemoryStateManager()
    return _memory_state_manager


def clear_in_memory_state() -> Dict[str, Any]:
    """
    Clear all in-memory state.
    
    This function provides a unified interface to clear all in-memory state,
    including debate sessions, agent managers, and bot managers.
    
    Returns:
        Dict with:
        - success: bool - Whether all state was cleared successfully
        - errors: List[str] - List of error messages (if any)
    """
    manager = get_memory_state_manager()
    return manager.clear_all()

