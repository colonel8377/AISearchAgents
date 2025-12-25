"""Bot API routes."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from ..common import get_api_key, bot_manager
from ..schemas import (
    CreateBotRequest, BotCreationResponse, ChatWithBotRequest, ChatWithBotResponse,
    DeleteConversationTurnRequest
)
from ...utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Bot"])

@router.post("/bot/create", response_model=BotCreationResponse, tags=["Bot Management"])
async def create_bot(
    request: CreateBotRequest,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):

    """

    Create a new bot with UUID.

    Args:

        request: Bot creation request with persona prompt and history_mode

        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:

        Bot creation response with UUID

    """

    logger.info(f"Creating bot with persona_prompt length: {len(request.persona_prompt)}, history_mode: {request.history_mode}")

    # Validate persona_prompt is not empty

    if not request.persona_prompt or not request.persona_prompt.strip():

        logger.warning("Empty persona_prompt provided")

        raise HTTPException(status_code=400, detail="persona_prompt cannot be empty")

    try:

        result = bot_manager.create_bot(

            persona_prompt=request.persona_prompt,

            bot_name=request.bot_name,

            history_mode=request.history_mode,

            execution_mode=request.execution_mode,

            use_few_shots=request.use_few_shots,

            custom_few_shots=request.custom_few_shots

        )

        if "error" in result:

            logger.warning(f"Create bot error: {result['error']}")

            raise HTTPException(status_code=400, detail=result["error"])

        logger.info(f"Successfully created bot {result.get('bot_id')}")

        return BotCreationResponse(**result)

    except HTTPException:

        raise

    except Exception as e:

        logger.error(f"Failed to create bot: {e}", exc_info=True)

        raise HTTPException(status_code=500, detail=f"Failed to create bot: {str(e)}")

@router.get("/bot/list", tags=["Bot Management"])
async def list_bots(
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):

    """

    List all bots.

    Args:

        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:

        List of all bots with their UUIDs and basic info

    """

    try:

        bots = bot_manager.list_bots()

        return {"bots": bots, "total_count": len(bots)}

    except Exception as e:

        logger.error(f"Failed to list bots: {e}", exc_info=True)

        raise HTTPException(status_code=500, detail=f"Failed to list bots: {str(e)}")

@router.post("/bot/chat", response_model=ChatWithBotResponse, tags=["Bot Management"])
async def chat_with_bot(
    request: ChatWithBotRequest,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)

):

    """

    Chat with a bot. Conversation history is automatically managed if bot is in stateful mode.

    Args:

        request: Chat request with bot_id (UUID) and message

        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:

        Bot's response and updated conversation history

    """

    logger.info(f"Chat with bot request: bot_id={request.bot_id}")

    try:

        result = bot_manager.chat_with_bot(

            bot_id=request.bot_id,

            user_message=request.message

        )

        if "error" in result:

            logger.warning(f"Chat with bot error: {result['error']}")

            raise HTTPException(status_code=400, detail=result["error"])

        logger.info(f"Successfully chatted with bot {request.bot_id}")

        return ChatWithBotResponse(**result)

    except HTTPException:

        raise

    except Exception as e:

        logger.error(f"Failed to chat with bot {request.bot_id}: {e}", exc_info=True)

        raise HTTPException(status_code=500, detail=f"Failed to chat with bot: {str(e)}")

@router.delete("/bot/{bot_id}", tags=["Bot Management"])
async def delete_bot(
    bot_id: str,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)

):

    """

    Delete a bot by UUID.

    Args:

        bot_id: UUID of the bot to delete

        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:

        Success message

    """

    logger.info(f"Deleting bot: {bot_id}")

    try:

        deleted = bot_manager.delete_bot(bot_id)

        if not deleted:

            raise HTTPException(status_code=404, detail=f"Bot '{bot_id}' not found")

        return {

            "message": f"Bot '{bot_id}' deleted successfully",

            "bot_id": bot_id

        }

    except HTTPException:

        raise

    except Exception as e:

        logger.error(f"Failed to delete bot {bot_id}: {e}", exc_info=True)

        raise HTTPException(status_code=500, detail=f"Failed to delete bot: {str(e)}")

@router.get("/bot/{bot_id}/conversation", tags=["Bot Management"])
async def get_conversation_history(
    bot_id: str,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Get conversation history for a bot with turn structure information.
    
    Returns the conversation history organized by turns, making it clear how turns are structured.
    Each turn contains a user message and an assistant response (2 entries in the history list).
    
    Args:
        bot_id: UUID of the bot
        _api_key: Authentication dependency (value not used, required for auth check)
    
    Returns:
        Dictionary containing:
        - bot_id: Bot UUID
        - history_mode: "stateless" or "stateful"
        - total_turns: Total number of conversation turns
        - turns: List of turns, each containing turn_index, user_message, and assistant_response
        - raw_history: Raw conversation history list (for reference)
    """
    logger.info(f"Getting conversation history for bot {bot_id}")
    
    bot = bot_manager.get_bot(bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail=f"Bot '{bot_id}' not found")
    
    if bot["history_mode"] != "stateful":
        return {
            "bot_id": bot_id,
            "history_mode": bot["history_mode"],
            "total_turns": 0,
            "turns": [],
            "raw_history": [],
            "message": "Bot is in stateless mode, no conversation history stored"
        }
    
    history = bot.get("conversation_history", [])
    total_turns = len(history) // 2
    
    # Organize history into turns
    turns = []
    for i in range(total_turns):
        user_idx = i * 2
        assistant_idx = i * 2 + 1
        
        if user_idx < len(history) and assistant_idx < len(history):
            user_entry = history[user_idx]
            assistant_entry = history[assistant_idx]
            
            turns.append({
                "turn_index": i,
                "user_message": user_entry.get("content", ""),
                "assistant_response": assistant_entry.get("content", ""),
                "user_role": user_entry.get("role", "user"),
                "assistant_role": assistant_entry.get("role", "assistant")
            })
    
    return {
        "bot_id": bot_id,
        "history_mode": bot["history_mode"],
        "total_turns": total_turns,
        "turns": turns,
        "raw_history": history
    }


@router.delete("/bot/{bot_id}/conversation/turn", tags=["Bot Management"])
async def reset_conversation_turn(
    bot_id: str,
    request: DeleteConversationTurnRequest,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Reset (delete) a specific conversation turn from bot's history.
    
    Each turn consists of a user message and an assistant response (2 entries in the history list).
    Turn index 0 refers to the first turn, index 1 to the second turn, etc.
    
    Args:
        bot_id: UUID of the bot
        request: Request with turn_index to reset/delete
        _api_key: Authentication dependency (value not used, required for auth check)
    
    Returns:
        Success message with deleted entries
    """
    logger.info(f"Resetting conversation turn {request.turn_index} for bot {bot_id}")
    
    try:
        result = bot_manager.delete_conversation_turn(
            bot_id=bot_id,
            turn_index=request.turn_index
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Update message to use "reset" terminology
        if "message" in result:
            result["message"] = result["message"].replace("deleted", "reset")
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reset conversation turn: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reset conversation turn: {str(e)}")


@router.post("/bot/{bot_id}/conversation/reset", tags=["Bot Management"])
async def reset_conversation(
    bot_id: str,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Reset (clear) all conversation history for a bot.
    
    This operation clears all conversation turns and resets the bot's conversation history to empty.
    The bot configuration and persona remain unchanged.
    
    Args:
        bot_id: UUID of the bot
        _api_key: Authentication dependency (value not used, required for auth check)
    
    Returns:
        Success message
    """
    logger.info(f"Resetting conversation history for bot {bot_id}")
    
    try:
        result = bot_manager.clear_conversation_history(bot_id)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Update message to use "reset" terminology
        result["message"] = "Conversation history reset successfully"
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reset conversation history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reset conversation history: {str(e)}")


@router.post("/bot/{bot_id}/conversation/summarize", tags=["Bot Management"])
async def summarize_bot_conversation(
    bot_id: str,
    turn: Optional[int] = Query(default=None, description="Turn index (0-based) to summarize up to. If None, summarizes all turns."),
    execution_mode: Optional[str] = Query(default=None, description="Execution mode: 'chain_online', 'chain_local', or 'no_chain'"),
    use_few_shots: bool = Query(default=True, description="Whether to use few-shot examples"),
    custom_few_shots: Optional[str] = Query(default=None, description="Optional custom few-shot examples"),
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Summarize bot conversation history up to a specific turn.
    
    If turn is specified, summarizes conversation up to and including that turn (0-indexed).
    If turn is None, summarizes the entire conversation history.
    
    Each turn consists of a user message and an assistant response.
    Turn index 0 refers to the first turn, index 1 to the second turn, etc.
    
    Args:
        bot_id: UUID of the bot
        turn: Optional turn index (0-based). If None, summarizes all turns.
        execution_mode: Optional execution mode ('chain_online', 'chain_local', 'no_chain')
        use_few_shots: Whether to use few-shot examples (default: True)
        custom_few_shots: Optional custom few-shot examples
        _api_key: Authentication dependency (value not used, required for auth check)
    
    Returns:
        Summary response with summary text and metadata
    """
    logger.info(f"Summarizing conversation for bot {bot_id}, turn={turn}")
    
    bot = bot_manager.get_bot(bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail=f"Bot '{bot_id}' not found")
    
    if bot["history_mode"] != "stateful":
        raise HTTPException(
            status_code=400,
            detail=f"Bot '{bot_id}' is in stateless mode, no conversation history to summarize"
        )
    
    history = bot.get("conversation_history", [])
    if not history:
        raise HTTPException(
            status_code=400,
            detail="No conversation history found for this bot"
        )
    
    # Convert history to conversation records format expected by summarizer
    conversation_records = []
    total_turns = len(history) // 2
    
    # Determine how many turns to include
    if turn is not None:
        if turn < 0 or turn >= total_turns:
            raise HTTPException(
                status_code=400,
                detail=f"Turn index {turn} out of range. Bot has {total_turns} turns (0-{total_turns-1})"
            )
        turns_to_include = turn + 1  # Include up to and including the specified turn
    else:
        turns_to_include = total_turns
    
    # Build conversation records from history
    # Format expected by summarizer: [{"user": "...", "assistant": "...", "turn": i}, ...]
    for i in range(turns_to_include):
        user_idx = i * 2
        assistant_idx = i * 2 + 1
        
        if user_idx < len(history) and assistant_idx < len(history):
            user_entry = history[user_idx]
            assistant_entry = history[assistant_idx]
            
            # Format as conversation records expected by summarizer
            conversation_records.append({
                "user": user_entry.get("content", ""),
                "assistant": assistant_entry.get("content", ""),
                "turn": i
            })
    
    try:
        from ...agents.summarizer.agent import SummarizerAgent
        
        # Create a temporary summarizer agent instance
        summarizer = SummarizerAgent()
        
        # Summarize the conversation
        result = summarizer.summarize_conversation(
            conversation_records=conversation_records,
            execution_mode=execution_mode,
            use_few_shots=use_few_shots,
            custom_few_shots=custom_few_shots
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Add bot and turn information to the result metadata
        if "metadata" not in result:
            result["metadata"] = {}
        result["metadata"]["bot_id"] = bot_id
        result["metadata"]["turns_included"] = turns_to_include
        result["metadata"]["total_turns"] = total_turns
        result["metadata"]["turn_limit"] = turn
        
        from ..schemas import SummaryResponse
        return SummaryResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to summarize conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to summarize conversation: {str(e)}")


# ===========================

# Multi-Agent Debate Endpoints

# ===========================
