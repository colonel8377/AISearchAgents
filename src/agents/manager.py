"""Agent Manager for managing multiple agent instances."""

from typing import Dict, Optional, Any, List
from enum import Enum

from ..utils.logger import get_logger

logger = get_logger(__name__)


class AgentType(str, Enum):
    """Supported agent types."""
    NUDGE_COLLAPSE = "nudge_collapse"
    SUMMARIZER = "summarizer"
    BOT_CREATOR = "bot_creator"


class AgentManager:
    """
    Manages multiple agent instances with unique IDs.
    
    Allows for creating, retrieving, and managing multiple agents
    of different types simultaneously.
    """
    
    def __init__(self):
        """Initialize the agent manager."""
        self._agents: Dict[str, Dict[str, Any]] = {}
        self._agent_counter = 0
        logger.info("AgentManager initialized")
        
    def create_agent(
        self,
        agent_instance: Any,
        agent_type: AgentType,
        agent_id: Optional[str] = None
    ) -> str:
        """
        Register a new agent instance.
        
        Args:
            agent_instance: The agent instance to register
            agent_type: Type of the agent
            agent_id: Optional custom agent ID, if not provided, auto-generated
            
        Returns:
            The agent ID
        """
        if agent_id is None:
            self._agent_counter += 1
            agent_id = f"agent_{self._agent_counter}"
        
        if agent_id in self._agents:
            logger.error(f"Attempted to create agent with duplicate ID: {agent_id}")
            raise ValueError(f"Agent with ID '{agent_id}' already exists")
        
        self._agents[agent_id] = {
            "instance": agent_instance,
            "type": agent_type,
            "created_at": None  # Could add timestamp if needed
        }
        
        logger.info(f"Agent registered: id={agent_id}, type={agent_type.value}")
        
        return agent_id
    
    def get_agent(self, agent_id: str) -> Optional[Any]:
        """
        Get an agent instance by ID.
        
        Args:
            agent_id: The agent ID
            
        Returns:
            The agent instance or None if not found
        """
        agent_data = self._agents.get(agent_id)
        if agent_data:
            logger.debug(f"Agent retrieved: id={agent_id}")
        else:
            logger.debug(f"Agent not found: id={agent_id}")
        return agent_data["instance"] if agent_data else None
    
    def get_agent_type(self, agent_id: str) -> Optional[AgentType]:
        """
        Get the type of an agent by ID.
        
        Args:
            agent_id: The agent ID
            
        Returns:
            The agent type or None if not found
        """
        agent_data = self._agents.get(agent_id)
        return agent_data["type"] if agent_data else None
    
    def delete_agent(self, agent_id: str) -> bool:
        """
        Delete an agent instance.
        
        Args:
            agent_id: The agent ID
            
        Returns:
            True if deleted, False if not found
        """
        if agent_id in self._agents:
            agent_type = self._agents[agent_id]["type"]
            del self._agents[agent_id]
            logger.info(f"Agent deleted: id={agent_id}, type={agent_type.value}")
            return True
        logger.warning(f"Attempted to delete non-existent agent: id={agent_id}")
        return False
    
    def list_agents(self) -> List[Dict[str, str]]:
        """
        List all registered agents.
        
        Returns:
            List of agent metadata
        """
        return [
            {
                "agent_id": agent_id,
                "agent_type": data["type"].value  # Convert enum to string
            }
            for agent_id, data in self._agents.items()
        ]
    
    def clear_all(self) -> None:
        """Clear all agents."""
        count = len(self._agents)
        self._agents.clear()
        self._agent_counter = 0
        logger.info(f"Cleared all agents: {count} agents removed")
    
    def exists(self, agent_id: str) -> bool:
        """
        Check if an agent exists.
        
        Args:
            agent_id: The agent ID
            
        Returns:
            True if exists, False otherwise
        """
        return agent_id in self._agents
