"""Agent management API routes - MVC View layer (only HTTP request/response handling)."""

from fastapi import APIRouter, Depends, HTTPException

from src.application.dto.agent_dto import CreateAgentRequestDTO, CommandResultDTO, ListAgentsRequestDTO, QueryResultDTO, \
    ResetAgentRequestDTO
from src.shared.utils.logger import get_logger

from ..common import get_api_key, agent_service
from ..schemas import (
    CreateAgentRequest, AgentIdResponse, ListAgentsResponse,
    AgentStatusResponse, ResetAgentRequest, _convert_agent_type_enum
)


logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/agents", tags=["Agent Management"])


@router.post("/create", response_model=AgentIdResponse)
async def create_agent(
    request: CreateAgentRequest,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Create a new agent instance using CQRS and DTO patterns.

    Args:
        request: Agent creation request
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Agent ID and metadata
    """
    try:
        # Convert HTTP request to DTO
        dto_request = CreateAgentRequestDTO(
            agent_type=request.agent_type,
            agent_id=request.agent_id,
            use_memory=request.use_memory,
            persona_mode=request.persona_mode
        )

        # Execute via CQRS service
        result: CommandResultDTO = await agent_service.create_agent(dto_request)

        if not result.success:
            raise HTTPException(
                status_code=400,
                detail=result.error_details or result.message
            )

        return AgentIdResponse(
            agent_id=result.agent_id,
            agent_type=_convert_agent_type_enum(request.agent_type),
            status="created",
            message=result.message,
            persona_mode=request.persona_mode
        )
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error creating agent: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create agent: {str(e)}")


@router.get("/list", response_model=ListAgentsResponse)
async def list_agents(_api_key: str = Depends(get_api_key)):  # Authentication dependency (value not used)
    """
    List all active agent instances using CQRS Query pattern.

    Args:
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        List of agents with their IDs and types
    """
    try:
        # Execute via CQRS service
        dto_request = ListAgentsRequestDTO()
        result: QueryResultDTO = await agent_service.list_agents(dto_request)

        if not result.success:
            raise HTTPException(
                status_code=500,
                detail=result.error_message or "Failed to list agents"
            )

        return ListAgentsResponse(
            agents=result.data["agents"] if isinstance(result.data, dict) else getattr(result.data, "agents", []),
            total_count=result.data["total_count"] if isinstance(result.data, dict) else getattr(result.data, "total_count", 0)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list agents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list agents: {str(e)}")


@router.get("/{agent_id}/status", response_model=AgentStatusResponse)
async def get_agent_status(
    agent_id: str,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Get status of a specific agent using CQRS Query pattern.

    Args:
        agent_id: The agent's unique identifier
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Agent status information
    """
    try:
        # Execute via CQRS service
        result: QueryResultDTO = await agent_service.get_agent_status(agent_id)

        if not result.success:
            if "not found" in (result.error_message or "").lower():
                raise HTTPException(status_code=404, detail=result.error_message)
            else:
                raise HTTPException(
                    status_code=500,
                    detail=result.error_message or "Failed to get agent status"
                )

        agent_data = result.data
        return AgentStatusResponse(
            agent_id=agent_data["agent_id"] if isinstance(agent_data, dict) else agent_data.agent_id,
            agent_type=_convert_agent_type_enum(
                agent_data["agent_type"] if isinstance(agent_data, dict) else str(agent_data.agent_type)
            ),
            status=agent_data["status"] if isinstance(agent_data, dict) else agent_data.status,
            current_turn=agent_data.get("current_turn") if isinstance(agent_data, dict) else agent_data.current_turn,
            additional_info=agent_data.get("additional_info", {}) if isinstance(agent_data, dict) else (agent_data.additional_info or {})
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get agent status: {str(e)}")


@router.delete("/{agent_id}")
async def delete_agent(
    agent_id: str,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Delete an agent instance using CQRS Command pattern.

    Args:
        agent_id: The agent's unique identifier
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Deletion confirmation
    """
    try:
        # Execute via CQRS service
        result: CommandResultDTO = await agent_service.delete_agent(agent_id)

        if not result.success:
            if "not found" in (result.error_details or "").lower():
                raise HTTPException(status_code=404, detail=result.error_details)
            else:
                raise HTTPException(
                    status_code=500,
                    detail=result.error_details or result.message
                )

        return {
            "status": "deleted",
            "message": result.message
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete agent: {str(e)}")


@router.post("/{agent_id}/reset")
async def reset_agent(
    agent_id: str,
    request: ResetAgentRequest,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Reset an agent state using CQRS Command pattern.

    Args:
        agent_id: The agent's unique identifier
        request: Reset options
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Reset confirmation
    """
    try:
        # Convert HTTP request to DTO
        dto_request = ResetAgentRequestDTO(
            reset_conversation=request.reset_conversation,
            clear_memory=request.clear_memory
        )

        # Execute via CQRS service
        result: CommandResultDTO = await agent_service.reset_agent(agent_id, dto_request)

        if not result.success:
            if "not found" in (result.error_details or "").lower():
                raise HTTPException(status_code=404, detail=result.error_details)
            else:
                raise HTTPException(
                    status_code=500,
                    detail=result.error_details or result.message
                )

        return {
            "status": "success",
            "message": result.message
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reset agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reset agent: {str(e)}")
