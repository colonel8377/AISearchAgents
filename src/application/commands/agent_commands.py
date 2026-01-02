"""
Agent Command classes and handlers for CQRS pattern.

Commands represent write operations that modify system state.
Command handlers process these commands and ensure business rules are enforced.
"""

from typing import Optional
from dataclasses import dataclass


# Command Classes
@dataclass
class CreateAgentCommand:
    """Command to create a new agent."""
    agent_type: str
    agent_id: Optional[str] = None
    use_memory: bool = False
    persona_mode: Optional[str] = None


@dataclass
class DeleteAgentCommand:
    """Command to delete an agent."""
    agent_id: str


@dataclass
class ResetAgentCommand:
    """Command to reset an agent state."""
    agent_id: str
    reset_conversation: bool = True
    clear_memory: bool = False


# Command Results
@dataclass
class AgentCommandResult:
    """Result of agent command execution."""
    success: bool
    agent_id: Optional[str] = None
    message: str = ""
    error_message: Optional[str] = None


class AgentCommandHandler:
    """
    Handles agent write operations (Commands) in CQRS pattern.

    Commands are write operations that modify state and may return minimal result data.
    They focus on business logic validation, state changes, and side effects.
    """

    def __init__(self, business_logic):
        """
        Initialize command handler.

        Args:
            business_logic: Agent business logic instance
        """
        self.business_logic = business_logic

    async def handle_create_agent(self, command: CreateAgentCommand) -> AgentCommandResult:
        """
        Handle agent creation command.

        Args:
            command: Create agent command

        Returns:
            Command result with success status and agent ID
        """
        try:
            result = self.business_logic.create_agent_logic(
                agent_type=command.agent_type,
                agent_id=command.agent_id,
                use_memory=command.use_memory,
                persona_mode=command.persona_mode
            )

            return AgentCommandResult(
                success=True,
                agent_id=result["agent_id"],
                message=result["message"]
            )

        except Exception as e:
            return AgentCommandResult(
                success=False,
                message="Failed to create agent",
                error_message=str(e)
            )

    async def handle_delete_agent(self, command: DeleteAgentCommand) -> AgentCommandResult:
        """
        Handle agent deletion command.

        Args:
            command: Delete agent command

        Returns:
            Command result with success status
        """
        try:
            result = self.business_logic.delete_agent_logic(command.agent_id)

            return AgentCommandResult(
                success=True,
                agent_id=command.agent_id,
                message=result["message"]
            )

        except Exception as e:
            return AgentCommandResult(
                success=False,
                agent_id=command.agent_id,
                message="Failed to delete agent",
                error_message=str(e)
            )

    async def handle_reset_agent(self, command: ResetAgentCommand) -> AgentCommandResult:
        """
        Handle agent reset command.

        Args:
            command: Reset agent command

        Returns:
            Command result with success status
        """
        try:
            result = self.business_logic.reset_agent_logic(
                agent_id=command.agent_id,
                reset_conversation=command.reset_conversation,
                clear_memory=command.clear_memory
            )

            return AgentCommandResult(
                success=True,
                agent_id=command.agent_id,
                message=result["message"]
            )

        except Exception as e:
            return AgentCommandResult(
                success=False,
                agent_id=command.agent_id,
                message="Failed to reset agent",
                error_message=str(e)
            )