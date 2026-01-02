"""Nudge Collapse API routes - MVC View layer (only HTTP request/response handling)."""

from fastapi import APIRouter, Depends, HTTPException

from src.shared.utils import get_logger
from ..common import get_api_key, nudge_collapse_service
from ..schemas import (
    GenerateTurnRequest, TurnResponse, ConversationHistoryResponse
)

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/agent", tags=["Nudge Collapse"])


@router.post("/{agent_id}/nudge-collapse/generate", response_model=TurnResponse)
async def generate_turn(
    agent_id: str,
    request: GenerateTurnRequest,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Generate a response turn for the Nudge-Collapse agent.

    Args:
        agent_id: The agent's unique identifier
        request: Generation request with user query and search context
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Turn response with agent's reply
    """
    try:
        result = nudge_collapse_service.generate_turn(
            agent_id=agent_id,
            user_query=request.user_query,
            search_summary=request.search_summary,
            search_urls=request.search_urls,
            history_mode=request.history_mode,
            use_few_shots=request.use_few_shots,
            custom_few_shots=request.custom_few_shots
        )
        
        if "error" in result:
            logger.warning(f"Agent {agent_id} generate_turn error: {result['error']}")
            raise HTTPException(status_code=400, detail=result["error"])
        
        return TurnResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate turn for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate turn: {str(e)}")


@router.get("/{agent_id}/nudge-collapse/history", response_model=ConversationHistoryResponse)
async def get_conversation_history(
    agent_id: str,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Get conversation history for a Nudge-Collapse agent.

    Args:
        agent_id: The agent's unique identifier
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Conversation history
    """
    try:
        result = nudge_collapse_service.get_conversation_history(agent_id)
        return ConversationHistoryResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get history for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")
