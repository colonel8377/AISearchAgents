"""
Abstract interfaces for memory state management.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseMemoryStateManager(ABC):
    """
    Abstract base class for memory state management.
    
    Defines the interface for clearing in-memory state across the application.
    """
    
    @abstractmethod
    def clear_all(self) -> Dict[str, Any]:
        """
        Clear all in-memory state.
        
        Returns:
            Dict with:
            - success: bool - Whether the clear completed successfully
            - errors: List[str] - List of error messages (if any)
        """
        pass

