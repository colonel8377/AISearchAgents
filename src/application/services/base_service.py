"""Base service class for all business logic services."""

from typing import Any, Optional
from abc import ABC

from ...application.agents.manager import AgentManager
from ...shared.constant.enums import AgentType
from ...shared.utils.logger import get_logger

logger = get_logger(__name__)


class BaseService(ABC):
    """
    Base service class that provides common functionality for all services.
    
    Services act as the Controller layer in MVC architecture, handling business logic
    and orchestrating interactions between different components.
    """
    
    def __init__(self, agent_manager: AgentManager):
        """
        Initialize the service with dependencies.
        
        Args:
            agent_manager: The agent manager instance
        """
        self.agent_manager = agent_manager
        logger.debug(f"{self.__class__.__name__} initialized")
    
    def _get_agent(self, agent_id: str) -> Optional[Any]:
        """
        Get an agent instance by ID.
        
        Args:
            agent_id: The agent ID
            
        Returns:
            The agent instance or None if not found
        """
        return self.agent_manager.get_agent(agent_id)
    
    def _get_agent_type(self, agent_id: str) -> Optional[AgentType]:
        """
        Get the type of an agent by ID.
        
        Args:
            agent_id: The agent ID
            
        Returns:
            The agent type or None if not found
        """
        return self.agent_manager.get_agent_type(agent_id)
    
    def _validate_agent_exists(self, agent_id: str) -> None:
        """
        Validate that an agent exists.
        
        Args:
            agent_id: The agent ID
            
        Raises:
            ValueError: If agent does not exist
        """
        if not self.agent_manager.exists(agent_id):
            raise ValueError(f"Agent '{agent_id}' not found")
    
    def _validate_agent_type(self, agent_id: str, expected_type: AgentType) -> None:
        """
        Validate that an agent is of the expected type.
        
        Args:
            agent_id: The agent ID
            expected_type: The expected agent type
            
        Raises:
            ValueError: If agent does not exist or is not of the expected type
        """
        self._validate_agent_exists(agent_id)
        agent_type = self._get_agent_type(agent_id)
        if agent_type != expected_type:
            raise ValueError(
                f"This operation requires a '{expected_type.value}' agent, "
                f"but agent '{agent_id}' is type '{agent_type.value}'"
            )

