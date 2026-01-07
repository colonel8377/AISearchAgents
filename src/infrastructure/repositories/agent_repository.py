"""
Agent Repository implementation using AgentManager.

This repository provides a clean data access layer for agent operations,
abstracting the underlying AgentManager implementation.
"""

from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...application.agents.manager import AgentManager

from ...shared.constant.enums import AgentType
from .interfaces import AgentRepositoryInterface, AgentProtocol


class AgentRepository(AgentRepositoryInterface):
    """
    Repository for agent data access operations.

    Wraps AgentManager to provide a clean repository interface
    for agent CRUD operations.
    """

    def __init__(self, agent_manager: "AgentManager"):
        """
        Initialize repository with agent manager.

        Args:
            agent_manager: The agent manager instance
        """
        self.agent_manager = agent_manager

    def create_agent(self, agent_instance: AgentProtocol, agent_type: AgentType, agent_id: Optional[str] = None) -> str:
        """Create a new agent and return its ID."""
        return self.agent_manager.create_agent(
            agent_instance=agent_instance,
            agent_type=agent_type,
            agent_id=agent_id
        )

    def get_agent(self, agent_id: str) -> Optional[AgentProtocol]:
        """Get agent instance by ID."""
        return self.agent_manager.get_agent(agent_id)

    def get_agent_type(self, agent_id: str) -> Optional[AgentType]:
        """Get agent type by ID."""
        return self.agent_manager.get_agent_type(agent_id)

    def exists(self, agent_id: str) -> bool:
        """Check if agent exists."""
        return self.agent_manager.exists(agent_id)

    def list_agents(self) -> List[Dict[str, Any]]:
        """List all agents with their metadata."""
        return self.agent_manager.list_agents()

    def delete_agent(self, agent_id: str) -> bool:
        """Delete agent by ID."""
        return self.agent_manager.delete_agent(agent_id)

    def reset_agent(self, agent_id: str) -> bool:
        """Reset agent state."""
        agent = self.get_agent(agent_id)
        if agent:
            agent.reset()
            return True
        return False