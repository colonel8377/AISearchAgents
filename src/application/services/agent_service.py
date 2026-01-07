"""Agent Service - CQRS-based service layer for agent management."""

from typing import Optional

from . import BaseService
from .agent_business_logic import AgentBusinessLogic
from ...infrastructure.repositories import AgentRepository
from ..commands.agent_commands import AgentCommandHandler
from ..queries.agent_queries import AgentQueryHandler
from ..dto.agent_dto import (
    CreateAgentRequestDTO, ResetAgentRequestDTO,
    ListAgentsRequestDTO, CommandResultDTO, QueryResultDTO, AgentDTOConverter, AgentCreationDataDTO
)
# Removed direct storage dependencies - use repositories instead
from ...shared.utils.logger import get_logger

logger = get_logger(__name__)


class AgentService(BaseService):
    """
    CQRS-based service for managing agent lifecycle and operations.

    Acts as Controller layer between Router (View) and Business Logic/Repository (Model).
    Uses CQRS pattern to separate commands (writes) from queries (reads).
    Uses DTOs for clean data transformation between layers.

    Responsibilities:
    - Coordinate between command/query handlers
    - Handle cross-cutting concerns (logging, error handling)
    - Transform data using DTOs
    """

    def __init__(self, agent_repository: AgentRepository):
        """
        Initialize agent service with CQRS components.

        Args:
            agent_repository: Repository for agent data access
        """
        # Initialize business logic handler
        self.business_logic = AgentBusinessLogic(agent_repository)

        # Initialize CQRS handlers
        self.command_handler = AgentCommandHandler(self.business_logic)
        self.query_handler = AgentQueryHandler(self.business_logic)

        logger.info("AgentService initialized with CQRS pattern")
    
    async def create_agent(self, request: CreateAgentRequestDTO) -> CommandResultDTO:
        """
        Create a new agent instance using CQRS Command pattern.

        Acts as Controller: converts DTO to Command, executes via CommandHandler,
        and returns result as DTO.

        Args:
            request: Create agent request DTO

        Returns:
            Command result DTO with success status and agent info
        """
        logger.info(f"Controller: Creating agent via CQRS Command")

        try:
            # Convert DTO directly to Command
            command = AgentDTOConverter.creation_data_to_command(
                AgentCreationDataDTO(
                    agent_type=request.agent_type,
                    agent_id=request.agent_id,
                    use_memory=request.use_memory,
                    persona_mode=request.persona_mode
                )
            )

            # Execute command via CommandHandler
            result = await self.command_handler.handle_create_agent(command)

            # Agent metadata is handled by AgentRepository
            # No need for additional storage operations here

            # Convert result to DTO
            return CommandResultDTO(
                success=result.success,
                message=result.message,
                agent_id=result.agent_id,
                error_details=result.error_message
            )

        except Exception as e:
            logger.error(f"Failed to create agent: {e}", exc_info=True)
            return CommandResultDTO(
                success=False,
                message="Failed to create agent",
                error_details=str(e)
            )
    
    async def list_agents(self, request: Optional[ListAgentsRequestDTO] = None) -> QueryResultDTO:
        """
        List agents using CQRS Query pattern.

        Args:
            request: Optional list agents request DTO with filters

        Returns:
            Query result DTO with agent list
        """
        logger.debug("Controller: Listing agents via CQRS Query")

        try:
            # Convert DTO to Query
            query = AgentDTOConverter.list_request_to_query(request or ListAgentsRequestDTO())

            # Execute query via QueryHandler
            result = await self.query_handler.handle_list_agents(query)

            if result.success and isinstance(result.data, object) and hasattr(result.data, 'agents'):
                # Return data directly as QueryResultDTO
                return QueryResultDTO(
                    success=True,
                    data={
                        "agents": getattr(result.data, 'agents', []),
                        "total_count": getattr(result.data, 'total_count', 0)
                    }
                )
            else:
                return QueryResultDTO(
                    success=False,
                    error_message="Invalid query result format"
                )

        except Exception as e:
            logger.error(f"Failed to list agents: {e}", exc_info=True)
            return QueryResultDTO(
                success=False,
                error_message=str(e)
            )
    
    async def get_agent_status(self, agent_id: str) -> QueryResultDTO:
        """
        Get agent status using CQRS Query pattern.

        Args:
            agent_id: The agent ID

        Returns:
            Query result DTO with agent status info
        """
        logger.debug(f"Controller: Getting agent status for {agent_id} via CQRS Query")

        try:
            # Convert to Query
            query = AgentDTOConverter.agent_id_to_status_query(agent_id)

            # Execute query via QueryHandler
            result = await self.query_handler.handle_get_agent_status(query)

            if result.success and result.data:
                # Convert AgentInfo to dict for response
                agent_data = {
                    "agent_id": result.data.agent_id,
                    "agent_type": result.data.agent_type.value if hasattr(result.data.agent_type, 'value') else str(result.data.agent_type),
                    "status": result.data.status,
                    "current_turn": result.data.current_turn,
                    "additional_info": result.data.additional_info
                }

                return QueryResultDTO(success=True, data=agent_data)
            else:
                return QueryResultDTO(
                    success=False,
                    error_message=result.error_message or "Agent not found"
                )

        except Exception as e:
            logger.error(f"Failed to get agent status: {e}", exc_info=True)
            return QueryResultDTO(
                success=False,
                error_message=str(e)
            )
    
    async def delete_agent(self, agent_id: str) -> CommandResultDTO:
        """
        Delete an agent using CQRS Command pattern.

        Args:
            agent_id: The agent ID

        Returns:
            Command result DTO with success status
        """
        logger.debug(f"Controller: Deleting agent {agent_id} via CQRS Command")

        try:
            # Convert to Command
            command = AgentDTOConverter.agent_id_to_delete_command(agent_id)

            # Execute command via CommandHandler
            result = await self.command_handler.handle_delete_agent(command)

            # Remove from storage is handled by repository/command handler
            # No need to call storage_strategy_manager here as it was removed

            return CommandResultDTO(
                success=result.success,
                message=result.message,
                agent_id=result.agent_id,
                error_details=result.error_message
            )

        except Exception as e:
            logger.error(f"Failed to delete agent: {e}", exc_info=True)
            return CommandResultDTO(
                success=False,
                message="Failed to delete agent",
                agent_id=agent_id,
                error_details=str(e)
            )

    async def reset_agent(self, agent_id: str, request: ResetAgentRequestDTO) -> CommandResultDTO:
        """
        Reset an agent using CQRS Command pattern.

        Args:
            agent_id: The agent ID
            request: Reset agent request DTO

        Returns:
            Command result DTO with success status
        """
        logger.debug(f"Controller: Resetting agent {agent_id} via CQRS Command")

        try:
            # Convert to Command
            command = AgentDTOConverter.reset_request_to_command(agent_id, request)

            # Execute command via CommandHandler
            result = await self.command_handler.handle_reset_agent(command)

            return CommandResultDTO(
                success=result.success,
                message=result.message,
                agent_id=result.agent_id,
                error_details=result.error_message
            )

        except Exception as e:
            logger.error(f"Failed to reset agent: {e}", exc_info=True)
            return CommandResultDTO(
                success=False,
                message="Failed to reset agent",
                agent_id=agent_id,
                error_details=str(e)
            )
