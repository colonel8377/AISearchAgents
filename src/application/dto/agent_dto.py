"""
Data Transfer Objects for Agent operations.

DTOs provide a clean separation between:
- HTTP request/response formats (external API)
- Internal business data formats
- Storage data formats

This simplifies data transformation and maintains clear boundaries.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel
from ...shared.constant.enums import AgentType


# Request DTOs (from HTTP to internal)
class CreateAgentRequestDTO(BaseModel):
    """DTO for agent creation requests."""
    agent_type: str
    agent_id: Optional[str] = None
    use_memory: bool = False
    persona_mode: Optional[str] = None


class DeleteAgentRequestDTO(BaseModel):
    """DTO for agent deletion requests."""
    agent_id: str


class ResetAgentRequestDTO(BaseModel):
    """DTO for agent reset requests."""
    reset_conversation: bool = True
    clear_memory: bool = False


class ListAgentsRequestDTO(BaseModel):
    """DTO for agent listing requests."""
    filter_type: Optional[str] = None
    limit: Optional[int] = None


# Response DTOs (from internal to HTTP)


class CommandResultDTO(BaseModel):
    """DTO for command execution results."""
    success: bool
    message: str
    agent_id: Optional[str] = None
    error_details: Optional[str] = None


class QueryResultDTO(BaseModel):
    """DTO for query results."""
    success: bool
    data: Optional[Any] = None
    error_message: Optional[str] = None


# Internal DTOs (for business logic)
class AgentCreationDataDTO:
    """Internal DTO for agent creation data."""
    def __init__(self, agent_type: str, agent_id: Optional[str] = None,
                 use_memory: bool = False, persona_mode: Optional[str] = None):
        self.agent_type = agent_type
        self.agent_id = agent_id
        self.use_memory = use_memory
        self.persona_mode = persona_mode




# DTO Converters
class AgentDTOConverter:
    """Converters between different DTO formats."""

    @staticmethod
    def request_to_creation_data(dto: CreateAgentRequestDTO) -> AgentCreationDataDTO:
        """Convert request DTO to internal creation data."""
        return AgentCreationDataDTO(
            agent_type=dto.agent_type,
            agent_id=dto.agent_id,
            use_memory=dto.use_memory,
            persona_mode=dto.persona_mode
        )


    @staticmethod
    def creation_data_to_command(dto: AgentCreationDataDTO):
        """Convert creation data to command object."""
        from ..commands.agent_commands import CreateAgentCommand
        return CreateAgentCommand(
            agent_type=dto.agent_type,
            agent_id=dto.agent_id,
            use_memory=dto.use_memory,
            persona_mode=dto.persona_mode
        )

    @staticmethod
    def agent_id_to_delete_command(agent_id: str):
        """Convert agent ID to delete command."""
        from ..commands.agent_commands import DeleteAgentCommand
        return DeleteAgentCommand(agent_id=agent_id)

    @staticmethod
    def reset_request_to_command(agent_id: str, dto: ResetAgentRequestDTO):
        """Convert reset request to command."""
        from ..commands.agent_commands import ResetAgentCommand
        return ResetAgentCommand(
            agent_id=agent_id,
            reset_conversation=dto.reset_conversation,
            clear_memory=dto.clear_memory
        )

    @staticmethod
    def list_request_to_query(dto: ListAgentsRequestDTO):
        """Convert list request to query."""
        from ..queries.agent_queries import ListAgentsQuery
        from ...shared.constant.enums import AgentType

        filter_type = None
        if dto.filter_type:
            try:
                filter_type = AgentType(dto.filter_type)
            except ValueError:
                filter_type = None

        return ListAgentsQuery(
            filter_type=filter_type,
            limit=dto.limit
        )

    @staticmethod
    def agent_id_to_status_query(agent_id: str):
        """Convert agent ID to status query."""
        from ..queries.agent_queries import GetAgentStatusQuery
        return GetAgentStatusQuery(agent_id=agent_id)