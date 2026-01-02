"""
Repository module - Data access layer (Repository Pattern).

This module provides repository implementations for data access operations.
Repository pattern abstracts data access logic from business logic.

NOTE:
- This module contains Repository implementations (data access layer)
- Storage implementations are in storage module (persistence layer)
- Agent state management is in agent_state module
"""

from .interfaces import (
    AgentRepositoryInterface,
    AgentProtocol
)
from .agent_repository import AgentRepository

__all__ = [
    # Interfaces
    "AgentRepositoryInterface",
    "AgentProtocol",
    "AgentRepository",
]
