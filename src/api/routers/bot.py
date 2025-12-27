"""Bot API routes."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from ..common import get_api_key, bot_manager
from ..schemas import (
    CreateBotRequest, BotCreationResponse, ChatWithBotRequest, ChatWithBotResponse,
    CreateConversationRequest, RenameConversationRequest, BotListResponse,
    ConversationListResponse, ConversationDetailResponse, SummaryResponse
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

        request: Bot creation request

        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:

        Bot creation response with UUID

    """

    logger.info(f"Creating bot with name: {request.bot_name}")

    try:

        result = bot_manager.create_bot(

            bot_name=request.bot_name,

            use_few_shots=request.use_few_shots

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

@router.get("/bot/list", response_model=BotListResponse, tags=["Bot Management"])
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

    Chat with a bot. Supports two modes: conversation threads and incognito mode.

    Conversation Mode (when conversation_id is provided):
    - Bot remembers all previous messages in this conversation thread
    - Each conversation maintains its own persistent context and history
    - Like ChatGPT conversation threads - bot knows the full conversation history
    - Conversation threads are saved and can be resumed later

    Incognito Mode (when conversation_id is not provided):
    - Creates a new temporary conversation with no prior context
    - Bot responds only to the current message with no memory
    - Like ChatGPT's temporary chats - each request is completely independent
    - Temporary conversations are not saved

    Args:
        request: Chat request with bot_id, message, and optional conversation_id
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Bot's response with conversation info

    """

    logger.info(f"Chat with bot request: bot_id={request.bot_id}, conversation_id={request.conversation_id}, mode={'conversation' if request.conversation_id else 'incognito'}, title={request.conversation_title}")

    try:

        result = bot_manager.chat_with_bot(

            bot_id=request.bot_id,

            user_message=request.message,

            conversation_id=request.conversation_id,

            conversation_title=request.conversation_title

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





@router.delete("/bot/{bot_id}/conversations", tags=["Bot Management"])
async def clear_all_conversations(
    bot_id: str,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Clear all conversation threads for a bot.

    This operation deletes all conversation threads and their message history.
    The bot configuration and persona remain unchanged.

    Args:
        bot_id: UUID of the bot
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Success message
    """
    logger.info(f"Clearing all conversations for bot {bot_id}")

    try:
        result = bot_manager.clear_all_conversations(bot_id)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear conversations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to clear conversations: {str(e)}")




@router.post("/bot/{bot_id}/conversation/{conversation_id}/summarize", response_model=SummaryResponse, tags=["Bot Management"])
async def summarize_conversation(
    bot_id: str,
    conversation_id: str,
    turn: Optional[int] = Query(default=None, description="Turn index (0-based) to summarize up to. If None, summarizes all turns."),
    use_few_shots: bool = Query(default=True, description="Whether to use few-shot examples"),
    custom_few_shots: Optional[str] = Query(default=None, description="Optional custom few-shot examples"),
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Summarize a specific conversation thread up to a specific turn.

    If turn is specified, summarizes conversation up to and including that turn (0-indexed).
    If turn is None, summarizes the entire conversation thread.

    Each turn consists of a user message and an assistant response.
    Turn index 0 refers to the first turn, index 1 to the second turn, etc.

    Args:
        bot_id: UUID of the bot
        conversation_id: UUID of the conversation thread
        turn: Optional turn index (0-based). If None, summarizes all turns.
        use_few_shots: Whether to use few-shot examples (default: True)
        custom_few_shots: Optional custom few-shot examples
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Summary response with summary text and metadata
    """
    logger.info(f"Summarizing conversation {conversation_id} for bot {bot_id}, turn={turn}")

    bot = bot_manager.get_bot(bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail=f"Bot '{bot_id}' not found")

    conversations = bot.get("conversations", {})
    conversation = conversations.get(conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation '{conversation_id}' not found for bot '{bot_id}'"
        )

    history = conversation.get("history", [])
    if not history:
        raise HTTPException(
            status_code=400,
            detail="No conversation history found for this conversation"
        )

    # Convert history to conversation records format expected by summarizer
    conversation_records = []
    total_turns = len(history) // 2

    # Determine how many turns to include
    if turn is not None:
        if turn < 0 or turn >= total_turns:
            raise HTTPException(
                status_code=400,
                detail=f"Turn index {turn} out of range. Conversation has {total_turns} turns (0-{total_turns-1})"
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
            use_few_shots=use_few_shots,
            custom_few_shots=custom_few_shots
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        # Add bot and conversation information to the result metadata
        if "metadata" not in result:
            result["metadata"] = {}
        result["metadata"]["bot_id"] = bot_id
        result["metadata"]["conversation_id"] = conversation_id
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

# Conversation Management Endpoints

# ===========================

@router.post("/bot/{bot_id}/conversation", tags=["Bot Management"])
async def create_conversation(
    bot_id: str,
    request: CreateConversationRequest,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Create a new conversation thread for a bot.

    This creates a new persistent conversation thread that will maintain its own
    message history across multiple chat interactions. Unlike incognito mode,
    conversation threads are saved and can be resumed later.

    Args:
        bot_id: UUID of the bot
        request: Conversation creation request with optional title and system prompt
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Conversation creation result with conversation_id
    """
    logger.info(f"Creating new conversation for bot {bot_id} with title: {request.title}")

    bot = bot_manager.get_bot(bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail=f"Bot '{bot_id}' not found")

    try:
        result = bot_manager.create_conversation(
            bot_id=bot_id,
            title=request.title,
            system_prompt=request.system_prompt
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create conversation: {str(e)}")


@router.get("/bot/{bot_id}/conversations", response_model=ConversationListResponse, tags=["Bot Management"])
async def list_conversations(
    bot_id: str,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    List all conversation threads for a bot.

    Returns all persistent conversation threads created for this bot,
    including their titles, creation times, and turn counts.

    Args:
        bot_id: UUID of the bot
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        List of conversations with metadata
    """
    logger.info(f"Listing conversations for bot {bot_id}")

    bot = bot_manager.get_bot(bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail=f"Bot '{bot_id}' not found")

    try:
        conversations = bot_manager.list_conversations(bot_id)
        return {
            "bot_id": bot_id,
            "conversations": conversations,
            "total_count": len(conversations)
        }

    except Exception as e:
        logger.error(f"Failed to list conversations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list conversations: {str(e)}")


@router.delete("/bot/{bot_id}/conversation/{conversation_id}", tags=["Bot Management"])
async def delete_conversation(
    bot_id: str,
    conversation_id: str,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Delete a conversation thread.

    This permanently removes a conversation thread and all its message history.
    The bot configuration remains unchanged.

    Args:
        bot_id: UUID of the bot
        conversation_id: UUID of the conversation to delete
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Success message
    """
    logger.info(f"Deleting conversation {conversation_id} for bot {bot_id}")

    try:
        result = bot_manager.delete_conversation(bot_id=bot_id, conversation_id=conversation_id)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete conversation: {str(e)}")


@router.put("/bot/{bot_id}/conversation/{conversation_id}", tags=["Bot Management"])
async def rename_conversation(
    bot_id: str,
    conversation_id: str,
    request: RenameConversationRequest,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Rename a conversation thread.

    Updates the title of an existing conversation thread.

    Args:
        bot_id: UUID of the bot
        conversation_id: UUID of the conversation to rename
        request: Rename request with new title
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Success message with updated conversation info
    """
    logger.info(f"Renaming conversation {conversation_id} for bot {bot_id} to: {request.title}")

    try:
        result = bot_manager.rename_conversation(
            bot_id=bot_id,
            conversation_id=conversation_id,
            new_title=request.title
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to rename conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to rename conversation: {str(e)}")


@router.get("/bot/{bot_id}/conversation/{conversation_id}", response_model=ConversationDetailResponse, tags=["Bot Management"])
async def get_conversation(
    bot_id: str,
    conversation_id: str,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Get details of a specific conversation thread.

    Returns the conversation metadata and full message history organized by turns.

    Args:
        bot_id: UUID of the bot
        conversation_id: UUID of the conversation
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Conversation details with message history organized by turns
    """
    logger.info(f"Getting conversation {conversation_id} for bot {bot_id}")

    bot = bot_manager.get_bot(bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail=f"Bot '{bot_id}' not found")

    try:
        result = bot_manager.get_conversation(bot_id=bot_id, conversation_id=conversation_id)

        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get conversation: {str(e)}")


# ===========================

# Multi-Agent Debate Endpoints

# ===========================
