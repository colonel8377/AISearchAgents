"""Nudge Collapse API routes."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from ..common import get_api_key, agent_manager
from ..schemas import (
    GenerateTurnRequest, TurnResponse, ConversationHistoryResponse
)
from ...agents.manager import AgentType
from ...config.settings import settings
from ...utils.logger import get_logger

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

    agent = agent_manager.get_agent(agent_id)

    if not agent:

        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    agent_type = agent_manager.get_agent_type(agent_id)

    if agent_type != AgentType.NUDGE_COLLAPSE:

        raise HTTPException(

            status_code=400,

            detail=f"This endpoint requires a 'nudge_collapse' agent, but agent '{agent_id}' is type '{agent_type}'"

        )

    try:        

        history_mode = request.history_mode if request.history_mode is not None else settings.default_history_mode

        result = agent.generate_turn(

            user_query=request.user_query,

            search_summary=request.search_summary,

            search_urls=request.search_urls,

            history_mode=history_mode,

            use_few_shots=request.use_few_shots,

            custom_few_shots=request.custom_few_shots

        )

        if "error" in result:

            logger.warning(f"Agent {agent_id} generate_turn error: {result['error']}")

            raise HTTPException(status_code=400, detail=result["error"])

        return TurnResponse(**result)

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

    agent = agent_manager.get_agent(agent_id)

    if not agent:

        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    agent_type = agent_manager.get_agent_type(agent_id)

    if agent_type != AgentType.NUDGE_COLLAPSE:

        raise HTTPException(

            status_code=400,

            detail=f"This endpoint requires a 'nudge_collapse' agent, but agent '{agent_id}' is type '{agent_type}'"

        )

    try:

        return ConversationHistoryResponse(

            current_turn=agent.get_current_turn(),

            max_turns=agent.max_turns,

            history=agent.get_conversation_history()

        )

    except Exception as e:

        logger.error(f"Failed to get history for agent {agent_id}: {e}", exc_info=True)

        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")

