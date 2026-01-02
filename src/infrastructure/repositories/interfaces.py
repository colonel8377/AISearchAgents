"""
Repository interfaces module.

This module defines ALL repository and storage interfaces.
All implementations are in storage or agent_state modules.

NOTE: This module ONLY contains interface definitions. No implementations here.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Protocol, Union

from ...shared.constant.enums import AgentType


# ============================================================================
# Agent-related interfaces
# ============================================================================

class AgentProtocol(Protocol):
    """Protocol for agent instances."""
    def reset(self) -> None:
        """Reset agent state."""
        ...


class AgentRepositoryInterface(ABC):
    """Interface for agent repository operations."""

    @abstractmethod
    def create_agent(self, agent_instance: AgentProtocol, agent_type: AgentType, agent_id: Optional[str] = None) -> str:
        """Create a new agent and return its ID."""
        pass

    @abstractmethod
    def get_agent(self, agent_id: str) -> Optional[AgentProtocol]:
        """Get agent instance by ID."""
        pass

    @abstractmethod
    def get_agent_type(self, agent_id: str) -> Optional[AgentType]:
        """Get agent type by ID."""
        pass

    @abstractmethod
    def exists(self, agent_id: str) -> bool:
        """Check if agent exists."""
        pass

    @abstractmethod
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all agents with their metadata."""
        pass

    @abstractmethod
    def delete_agent(self, agent_id: str) -> bool:
        """Delete agent by ID."""
        pass

    @abstractmethod
    def reset_agent(self, agent_id: str) -> bool:
        """Reset agent state."""
        pass


class BotRepositoryInterface(ABC):
    """Interface for bot repository operations."""

    @abstractmethod
    def save_bot(self, bot_data: Dict[str, Any]) -> bool:
        """Save or update a bot."""
        pass

    @abstractmethod
    def load_bot(self, bot_name: str) -> Optional[Dict[str, Any]]:
        """Load a bot by name."""
        pass

    @abstractmethod
    def load_all_bots(self) -> List[Dict[str, Any]]:
        """Load all bots."""
        pass

    @abstractmethod
    def delete_bot(self, bot_name: str) -> bool:
        """Delete a bot by name."""
        pass

    @abstractmethod
    def bot_exists(self, bot_name: str) -> bool:
        """Check if a bot exists."""
        pass

    @abstractmethod
    def save_conversation_history(self, bot_name: str, history: List[Dict[str, Any]]) -> bool:
        """Save conversation history for a bot."""
        pass

