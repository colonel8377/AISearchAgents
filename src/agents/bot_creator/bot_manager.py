"""Global Bot Manager for managing all bots independently."""

import uuid
from datetime import datetime
from typing import Dict, Optional, Any, List
from .agent import BotCreatorAgent
from ...utils.logger import get_logger
from ...config.settings import settings
from ...storage import get_storage

logger = get_logger(__name__)


class BotManager:
    """Global manager for all bots, independent of agent instances."""

    def __init__(self):
        """Initialize the bot manager."""
        # Create a shared BotCreatorAgent instance for bot creation
        self._bot_creator = BotCreatorAgent(
            model_name=settings.openai_model,
            api_key=settings.openai_api_key,
            api_base=settings.openai_api_base,
            temperature=settings.agent_temperature,
            persona_mode="system_prompt"
        )

        # Initialize storage connection if persistence is enabled
        self._storage = get_storage() if settings.enable_persistence else None
        self._database = self._storage.get_database() if self._storage else None

        logger.info("BotManager initialized")
    
    def create_bot(
        self,
        bot_name: Optional[str] = None,
        execution_mode: Optional[str] = None,
        use_few_shots: bool = True
    ) -> Dict[str, Any]:
        """
        Create a new bot with UUID.

        Args:
            bot_name: Optional name for the bot
            execution_mode: Execution mode for bot creation
            use_few_shots: Whether to use few-shot examples

        Returns:
            Bot creation result with UUID
        """

        # Create bot using BotCreatorAgent
        result = self._bot_creator.create_bot(
            bot_name=bot_name,
            execution_mode=execution_mode,
            use_few_shots=use_few_shots
        )

        if "error" in result:
            return result

        # Generate UUID for bot
        bot_uuid = str(uuid.uuid4())

        # Prepare bot entry
        bot_entry = {
            "bot_id": bot_uuid,
            "bot_name": result["bot_name"],
            "bot_configuration": result["bot_configuration"],
            "status": "initialized",
            "conversations": {},  # Dictionary to store conversation threads
            "persona_mode": result.get("persona_mode", "system_prompt"),
            "execution_mode": result.get("execution_mode"),
            "metadata": result.get("metadata", {})
        }

        # Save to database if persistence is enabled
        if self._database:
            success = self._database.save_bot(bot_entry)
            if not success:
                logger.warning(f"Failed to persist bot {bot_uuid}, continuing without persistence")

        logger.info(f"Bot created with UUID: {bot_uuid}")

        return {
            "bot_id": bot_uuid,
            "bot_name": bot_entry["bot_name"],
            "status": "initialized",
            "persona_mode": bot_entry["persona_mode"],
            "execution_mode": bot_entry["execution_mode"],
            "bot_configuration": result["bot_configuration"],
            "message": f"Bot '{bot_entry['bot_name']}' created successfully"
        }
    
    def get_bot(self, bot_id: str) -> Optional[Dict[str, Any]]:
        """Get bot by UUID."""
        if self._database:
            return self._database.load_bot(bot_id)
        return None
    
    def list_bots(self) -> List[Dict[str, Any]]:
        """List all bots."""
        if self._database:
            bots = self._database.load_all_bots()
            return [
                {
                    "bot_id": bot["bot_id"],
                    "bot_name": bot["bot_name"],
                    "status": bot["status"],
                    "conversations_count": len(bot.get("conversations", {})),
                    "created_at": bot.get("created_at")
                }
                for bot in bots
            ]
        return []
    
    def delete_bot(self, bot_id: str) -> bool:
        """
        Delete a bot by UUID.

        Args:
            bot_id: Bot UUID

        Returns:
            True if deleted, False if not found
        """
        if self._database:
            success = self._database.delete_bot(bot_id)
            if success:
                logger.info(f"Bot {bot_id} deleted")
                return True
            return False
        return False
    
    def chat_with_bot(
        self,
        bot_id: str,
        user_message: str,
        conversation_id: Optional[str] = None,
        conversation_title: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Chat with a bot. Supports conversation threads and incognito mode.

        Args:
            bot_id: Bot UUID
            user_message: User message
            conversation_id: Conversation ID for conversation thread. If provided, bot remembers
                           previous messages in this thread. If None, creates temporary incognito conversation.
            conversation_title: Optional title for new conversations

        Returns:
            Bot response with conversation info
        """
        bot = self.get_bot(bot_id)
        if not bot:
            return {
                "error": f"Bot '{bot_id}' not found",
                "bot_id": bot_id
            }

        # Determine mode and setup conversation
        if conversation_id:
            # Conversation thread mode: bot remembers previous messages
            # Initialize conversations dict if it doesn't exist
            if "conversations" not in bot:
                bot["conversations"] = {}

            conversations = bot["conversations"]
            conversation = conversations.get(conversation_id)
            if not conversation:
                return {
                    "error": f"Conversation '{conversation_id}' not found for bot '{bot_id}'",
                    "bot_id": bot_id,
                    "conversation_id": conversation_id
                }

            conversation_history = conversation.get("history", [])
            history_mode = True
            mode = "conversation"
            title = conversation.get("title")
        else:
            # Incognito mode: completely independent, no conversation system interaction
            # Don't even initialize or access conversations dict for maximum isolation
            conversations = None  # Don't access conversations at all
            conversation_history = None  # No history for incognito
            history_mode = False
            mode = "incognito"
            title = None  # No title for incognito mode

        # Chat with bot using BotCreatorAgent
        result = self._bot_creator.chat_with_bot_by_config(
            bot_configuration=bot["bot_configuration"],
            bot_name=bot["bot_name"],
            user_message=user_message,
            conversation_history=conversation_history,
            history_mode=history_mode
        )

        # Add metadata to result
        result["bot_id"] = bot_id
        result["bot_name"] = bot["bot_name"]
        result["mode"] = mode

        # Only add conversation-related fields in conversation mode
        if conversation_id:
            result["conversation_id"] = conversation_id
            result["conversation_title"] = title
            result["turn_count"] = len(result.get("conversation_history", [])) // 2
        else:
            # In incognito mode, don't include conversation-related fields
            # Remove conversation_history from result for cleaner incognito response
            if "conversation_history" in result:
                del result["conversation_history"]

        # Update conversation thread if in conversation mode
        if conversation_id and self._database:
            conversations[conversation_id]["history"] = result["conversation_history"]
            conversations[conversation_id]["turn_count"] = result["turn_count"]
            conversations[conversation_id]["updated_at"] = str(datetime.utcnow())

            # Save updated bot data
            success = self._database.save_bot(bot)
            if not success:
                logger.warning(f"Failed to persist conversation for bot {bot_id}")

        if "error" in result:
            return result

        return result
    
    
    def clear_all_conversations(self, bot_id: str) -> Dict[str, Any]:
        """
        Clear all conversation threads for a bot.

        Args:
            bot_id: Bot UUID

        Returns:
            Success message or error
        """
        bot = self.get_bot(bot_id)
        if not bot:
            return {
                "error": f"Bot '{bot_id}' not found",
                "bot_id": bot_id
            }

        # Clear all conversations
        conversations_count = len(bot.get("conversations", {}))
        bot["conversations"] = {}

        # Clear all conversation-related cache/memory for this bot
        try:
            # Clear agent cache entries related to this bot's conversations
            from ..utils.agent_cache import get_agent_cache
            cache = get_agent_cache()

            # Clear cache entries that contain this bot_id in parameters
            # This ensures LLM calls related to this bot's conversations are not cached anymore
            cache.clear()  # For simplicity, clear all cache (could be more selective)

            logger.info(f"Cleared cache/memory for all conversations of bot {bot_id}")

        except Exception as e:
            logger.warning(f"Failed to clear cache for bot {bot_id}: {e}")

        # Persist changes to database
        if self._database:
            success = self._database.save_bot(bot)
            if not success:
                return {
                    "error": "Failed to clear conversations from database",
                    "bot_id": bot_id
                }
        else:
            return {
                "error": "Persistence not enabled",
                "bot_id": bot_id
            }

        logger.info(f"Cleared {conversations_count} conversations for bot {bot_id} with memory/cache cleanup")
        return {
            "message": f"All conversations cleared successfully (memory and cache cleared, {conversations_count} conversations removed)",
            "bot_id": bot_id,
            "conversations_cleared": conversations_count,
            "cache_cleared": True
        }

    def update_conversation_history(self, bot_id: str, conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Update a bot's conversation history.

        Args:
            bot_id: Bot UUID
            conversation_history: New conversation history

        Returns:
            Success message or error
        """
        bot = self.get_bot(bot_id)
        if not bot:
            return {
                "error": f"Bot '{bot_id}' not found",
                "bot_id": bot_id
            }

        if bot["history_mode"] != "stateful":
            return {
                "error": f"Bot '{bot_id}' is in stateless mode, cannot update conversation history",
                "bot_id": bot_id
            }

        # Update in-memory bot object
        bot["conversation_history"] = conversation_history

        # Persist to database if available
        if self._database:
            success = self._database.save_conversation_history(bot_id, conversation_history)
            if not success:
                logger.warning(f"Failed to persist conversation history for bot {bot_id}")
                return {
                    "error": "Failed to persist conversation history to database",
                    "bot_id": bot_id
                }

        logger.info(f"Updated conversation history for bot {bot_id}")
        return {
            "message": "Conversation history updated successfully",
            "bot_id": bot_id
        }

    def create_conversation(
        self,
        bot_id: str,
        title: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new conversation thread for a bot.

        Args:
            bot_id: Bot UUID
            title: Optional title for the conversation
            system_prompt: Optional system prompt to initialize the conversation

        Returns:
            Conversation creation result
        """
        bot = self.get_bot(bot_id)
        if not bot:
            return {
                "error": f"Bot '{bot_id}' not found",
                "bot_id": bot_id
            }

        # Generate unique conversation ID
        conversation_id = str(uuid.uuid4())

        # Initialize conversation
        conversation = {
            "conversation_id": conversation_id,
            "title": title or f"Conversation {conversation_id[:8]}",
            "created_at": str(datetime.utcnow()),
            "updated_at": str(datetime.utcnow()),
            "turn_count": 0,
            "history": []
        }

        # Add system prompt if provided
        if system_prompt:
            conversation["history"].append({
                "role": "system",
                "content": system_prompt
            })

        # Initialize conversations dict if needed
        if "conversations" not in bot:
            bot["conversations"] = {}

        # Add conversation to bot
        bot["conversations"][conversation_id] = conversation

        # Persist to database
        if self._database:
            success = self._database.save_bot(bot)
            if not success:
                logger.warning(f"Failed to persist new conversation for bot {bot_id}")
                return {
                    "error": "Failed to persist conversation to database",
                    "bot_id": bot_id
                }

        logger.info(f"Created conversation {conversation_id} for bot {bot_id}")
        return {
            "message": "Conversation created successfully",
            "bot_id": bot_id,
            "conversation_id": conversation_id,
            "title": conversation["title"],
            "created_at": conversation["created_at"]
        }

    def list_conversations(self, bot_id: str) -> List[Dict[str, Any]]:
        """
        List all conversation threads for a bot.

        Args:
            bot_id: Bot UUID

        Returns:
            List of conversations with metadata
        """
        bot = self.get_bot(bot_id)
        if not bot:
            return []

        conversations = bot.get("conversations", {})

        # Convert conversations dict to list format
        conversation_list = []
        for conv_id, conv_data in conversations.items():
            conversation_list.append({
                "conversation_id": conv_id,
                "title": conv_data.get("title", f"Conversation {conv_id[:8]}"),
                "created_at": conv_data.get("created_at"),
                "updated_at": conv_data.get("updated_at"),
                "turn_count": conv_data.get("turn_count", 0),
                "last_message": conv_data.get("history", [])[-1].get("content", "") if conv_data.get("history") else ""
            })

        # Sort by updated_at descending (most recent first)
        conversation_list.sort(key=lambda x: x.get("updated_at", ""), reverse=True)

        return conversation_list

    def delete_conversation(self, bot_id: str, conversation_id: str) -> Dict[str, Any]:
        """
        Delete a conversation thread.

        Args:
            bot_id: Bot UUID
            conversation_id: Conversation UUID

        Returns:
            Success or error message
        """
        bot = self.get_bot(bot_id)
        if not bot:
            return {
                "error": f"Bot '{bot_id}' not found",
                "bot_id": bot_id
            }

        conversations = bot.get("conversations", {})
        if conversation_id not in conversations:
            return {
                "error": f"Conversation '{conversation_id}' not found for bot '{bot_id}'",
                "bot_id": bot_id,
                "conversation_id": conversation_id
            }

        # Remove conversation
        deleted_conversation = conversations.pop(conversation_id)

        # Clear conversation-related cache/memory
        try:
            # Clear agent cache entries related to this conversation
            from ..utils.agent_cache import get_agent_cache
            cache = get_agent_cache()

            # Clear cache entries that contain this conversation_id in parameters
            # This ensures LLM calls related to this conversation are not cached anymore
            cache.clear()  # For simplicity, clear all cache (could be more selective)

            logger.info(f"Cleared cache/memory for conversation {conversation_id}")

        except Exception as e:
            logger.warning(f"Failed to clear cache for conversation {conversation_id}: {e}")

        # Persist changes to database
        if self._database:
            success = self._database.save_bot(bot)
            if not success:
                logger.warning(f"Failed to persist conversation deletion for bot {bot_id}")
                return {
                    "error": "Failed to persist conversation deletion to database",
                    "bot_id": bot_id
                }

        logger.info(f"Deleted conversation {conversation_id} for bot {bot_id} with memory/cache cleanup")
        return {
            "message": "Conversation deleted successfully (memory and cache cleared)",
            "bot_id": bot_id,
            "conversation_id": conversation_id,
            "deleted_title": deleted_conversation.get("title"),
            "cache_cleared": True
        }

    def rename_conversation(self, bot_id: str, conversation_id: str, new_title: str) -> Dict[str, Any]:
        """
        Rename a conversation thread.

        Args:
            bot_id: Bot UUID
            conversation_id: Conversation UUID
            new_title: New title for the conversation

        Returns:
            Success or error message
        """
        bot = self.get_bot(bot_id)
        if not bot:
            return {
                "error": f"Bot '{bot_id}' not found",
                "bot_id": bot_id
            }

        conversations = bot.get("conversations", {})
        if conversation_id not in conversations:
            return {
                "error": f"Conversation '{conversation_id}' not found for bot '{bot_id}'",
                "bot_id": bot_id,
                "conversation_id": conversation_id
            }

        # Update title and timestamp
        old_title = conversations[conversation_id].get("title")
        conversations[conversation_id]["title"] = new_title
        conversations[conversation_id]["updated_at"] = str(datetime.utcnow())

        # Persist changes
        if self._database:
            success = self._database.save_bot(bot)
            if not success:
                logger.warning(f"Failed to persist conversation rename for bot {bot_id}")
                return {
                    "error": "Failed to persist conversation rename to database",
                    "bot_id": bot_id
                }

        logger.info(f"Renamed conversation {conversation_id} for bot {bot_id}: '{old_title}' -> '{new_title}'")
        return {
            "message": "Conversation renamed successfully",
            "bot_id": bot_id,
            "conversation_id": conversation_id,
            "old_title": old_title,
            "new_title": new_title
        }

    def get_conversation(self, bot_id: str, conversation_id: str) -> Dict[str, Any]:
        """
        Get details of a specific conversation thread.

        Args:
            bot_id: Bot UUID
            conversation_id: Conversation UUID

        Returns:
            Conversation details with organized message history
        """
        bot = self.get_bot(bot_id)
        if not bot:
            return {
                "error": f"Bot '{bot_id}' not found",
                "bot_id": bot_id
            }

        conversations = bot.get("conversations", {})
        conversation = conversations.get(conversation_id)
        if not conversation:
            return {
                "error": f"Conversation '{conversation_id}' not found for bot '{bot_id}'",
                "bot_id": bot_id,
                "conversation_id": conversation_id
            }

        # Organize history into turns
        history = conversation.get("history", [])
        total_turns = len(history) // 2

        # Build turns list (excluding system messages)
        turns = []
        for i in range(total_turns):
            user_idx = i * 2
            assistant_idx = i * 2 + 1

            if user_idx < len(history) and assistant_idx < len(history):
                user_entry = history[user_idx]
                assistant_entry = history[assistant_idx]

                # Skip system messages in turns
                if user_entry.get("role") == "user" and assistant_entry.get("role") == "assistant":
                    turns.append({
                        "turn_index": i,
                        "user_message": user_entry.get("content", ""),
                        "assistant_response": assistant_entry.get("content", ""),
                        "user_role": user_entry.get("role", "user"),
                        "assistant_role": assistant_entry.get("role", "assistant")
                    })

        return {
            "bot_id": bot_id,
            "conversation_id": conversation_id,
            "title": conversation.get("title"),
            "created_at": conversation.get("created_at"),
            "updated_at": conversation.get("updated_at"),
            "total_turns": total_turns,
            "turns": turns,
            "raw_history": history
        }


# Global bot manager instance
bot_manager = BotManager()

