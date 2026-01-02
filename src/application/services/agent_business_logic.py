"""
Agent business logic handlers.

This module contains the core business logic for agent operations,
extracted from the service layer to maintain separation of concerns.
"""

from typing import Optional, Dict, Any
from ...shared.constant.enums import AgentType
from ...infrastructure.repositories import AgentRepository
from ...shared.utils.logger import get_logger

logger = get_logger(__name__)


class AgentBusinessLogic:
    """
    Handles agent business logic operations.

    This class contains all the business rules and logic for agent operations,
    keeping it separate from service coordination and data access.
    """

    def __init__(self, agent_repository: AgentRepository):
        """
        Initialize business logic handler.

        Args:
            agent_repository: Repository for agent data access
        """
        self.repository = agent_repository

    def create_agent_logic(
        self,
        agent_type: str,
        agent_id: Optional[str] = None,
        use_memory: bool = False,
        persona_mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Business logic for creating a new agent.

        Args:
            agent_type: Type of agent to create (string value)
            agent_id: Optional custom agent ID
            use_memory: Whether to enable memory
            persona_mode: Optional persona mode for BotCreatorAgent

        Returns:
            Dictionary with agent creation result

        Raises:
            ValueError: If agent type is invalid or creation fails
        """
        # Validate agent type
        try:
            agent_type_enum = AgentType(agent_type)
        except ValueError:
            valid_types = [e.value for e in AgentType]
            raise ValueError(f"Invalid agent type. Must be one of: {valid_types}")

        logger.info(f"Creating agent: type={agent_type}, id={agent_id}, use_memory={use_memory}")

        # Create agent instance using factory
        from ...application.services.agent_factory import AgentFactory
        agent_instance = AgentFactory.create_agent(
            agent_type=agent_type_enum,
            agent_id=agent_id,
            use_memory=use_memory,
            persona_mode=persona_mode
        )

        # Register agent with repository
        registered_agent_id = self.repository.create_agent(
            agent_instance=agent_instance,
            agent_type=agent_type_enum,
            agent_id=agent_id
        )

        logger.info(f"Agent created successfully: id={registered_agent_id}, type={agent_type}")

        # Build response
        response = {
            "agent_id": registered_agent_id,
            "agent_type": agent_type,
            "status": "created",
            "message": f"Agent '{registered_agent_id}' of type '{agent_type}' created successfully"
        }

        # Add persona_mode for BotCreatorAgent (use original request value, default to "system_prompt")
        if agent_type == AgentType.BOT_CREATOR.value:
            response["persona_mode"] = persona_mode if persona_mode is not None else "system_prompt"

        return response

    def list_agents_logic(self) -> Dict[str, Any]:
        """
        Business logic for listing all agents.

        Returns:
            Dictionary with agents list and total_count
        """
        agents = self.repository.list_agents()
        return {
            "agents": agents,
            "total_count": len(agents)
        }

    def get_agent_status_logic(self, agent_id: str) -> Dict[str, Any]:
        """
        Business logic for getting agent status.

        Args:
            agent_id: The agent ID

        Returns:
            Dictionary with agent status information

        Raises:
            ValueError: If agent does not exist
        """
        if not self.repository.exists(agent_id):
            raise ValueError(f"Agent '{agent_id}' not found")

        agent = self.repository.get_agent(agent_id)
        agent_type = self.repository.get_agent_type(agent_id)

        # Get current turn if available
        current_turn = getattr(agent, 'get_current_turn', lambda: None)()

        # Build additional info based on agent type
        additional_info = {}
        if agent_type == AgentType.NUDGE_COLLAPSE:
            additional_info["max_turns"] = getattr(agent, 'max_turns', None)
        elif agent_type == AgentType.SUMMARIZER:
            additional_info["summaries_generated"] = len(getattr(agent, 'summary_history', []))
        elif agent_type == AgentType.BOT_CREATOR:
            additional_info["bots_created"] = len(getattr(agent, 'created_bots', []))

        return {
            "agent_id": agent_id,
            "agent_type": agent_type.value,
            "status": "active",
            "current_turn": current_turn,
            "additional_info": additional_info
        }

    def delete_agent_logic(self, agent_id: str) -> Dict[str, str]:
        """
        Business logic for deleting an agent.

        Args:
            agent_id: The agent ID

        Returns:
            Dictionary with status and message

        Raises:
            ValueError: If agent does not exist
        """
        if not self.repository.exists(agent_id):
            raise ValueError(f"Agent '{agent_id}' not found")

        success = self.repository.delete_agent(agent_id)
        if not success:
            raise ValueError(f"Failed to delete agent '{agent_id}'")

        return {
            "status": "deleted",
            "message": f"Agent '{agent_id}' deleted successfully"
        }

    def reset_agent_logic(
        self,
        agent_id: str,
        reset_conversation: bool = True,
        clear_memory: bool = False
    ) -> Dict[str, Any]:
        """
        Business logic for resetting an agent.

        Args:
            agent_id: The agent ID
            reset_conversation: Whether to reset conversation history
            clear_memory: Whether to clear memory (not implemented yet)

        Returns:
            Dictionary with status, message, and current_turn

        Raises:
            ValueError: If agent does not exist
            Exception: If reset fails
        """
        if not self.repository.exists(agent_id):
            raise ValueError(f"Agent '{agent_id}' not found")

        agent = self.repository.get_agent(agent_id)

        try:
            if reset_conversation and hasattr(agent, 'reset'):
                agent.reset()

            if clear_memory:
                # TODO: Implement memory clearing
                logger.warning("Memory clearing not yet implemented")

            current_turn = getattr(agent, 'get_current_turn', lambda: None)()

            return {
                "status": "success",
                "message": f"Agent '{agent_id}' reset successfully",
                "current_turn": current_turn
            }
        except Exception as e:
            logger.error(f"Failed to reset agent {agent_id}: {e}", exc_info=True)
            raise Exception(f"Failed to reset agent: {str(e)}")
