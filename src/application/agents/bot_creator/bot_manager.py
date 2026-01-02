"""Global Bot Manager for managing all bots independently."""

import uuid
from datetime import datetime
from typing import Dict, Optional, Any, List

from src.shared.cache.factory import reset_agent_cache
from .agent import BotCreatorAgent
from ....infrastructure.storage.persistence import get_storage
from ....infrastructure.repositories.bot_repository import BotRepository
from ....shared.config.settings import settings
from ....shared.utils.logger import get_logger

logger = get_logger(__name__)


class BotManager:
    """Global manager for all bots, independent of agent instances."""

    def __init__(self):
        """Initialize the bot manager."""
        # Validate API key before creating BotCreatorAgent
        self._bot_creator = BotCreatorAgent(
            model_name=settings.openai_model,
            api_key=settings.openai_api_key,
            api_base=settings.openai_api_base,
            temperature=settings.agent_temperature,
            persona_mode="system_prompt"
        )

        # Initialize storage persistence if persistence is enabled
        self._storage = get_storage() if settings.enable_persistence else None
        self._database = self._storage.get_database() if self._storage else None

        # Initialize bot repository if persistence is enabled
        self._bot_repository = BotRepository(self._database) if self._database else None

        logger.info("BotManager initialized")
    
    def create_bot(
        self,
        bot_name: Optional[str] = None,
        execution_mode: Optional[str] = None,
        use_few_shots: bool = True
    ) -> Dict[str, Any]:
        """
        Create a new bot.

        Args:
            bot_name: Optional name for the bot
            execution_mode: Execution mode for bot creation
            use_few_shots: Whether to use few-shot examples

        Returns:
            Bot creation result
        """

        # Create bot using BotCreatorAgent
        result = self._bot_creator.create_bot(
            bot_name=bot_name,
            execution_mode=execution_mode,
            use_few_shots=use_few_shots
        )

        if "error" in result:
            return result

        bot_name_final = result["bot_name"]

        # Check if bot name already exists
        if self._database and self._bot_repository.bot_exists(bot_name_final):
            return {
                "error": f"Bot with name '{bot_name_final}' already exists",
                "bot_name": bot_name_final
            }

        # Prepare bot entry
        bot_entry = {
            "bot_name": bot_name_final,
            "bot_configuration": result["bot_configuration"],
            "status": "initialized",
            "conversations": {},  # Dictionary to store conversation threads
            "persona_mode": result.get("persona_mode", "system_prompt"),
            "execution_mode": result.get("execution_mode"),
            "metadata": result.get("metadata", {})
        }

        # Save to persistence if persistence is enabled
        if self._bot_repository:
            success = self._bot_repository.save_bot(bot_entry)
            if not success:
                logger.warning(f"Failed to persist bot {bot_name_final}, continuing without persistence")

        logger.info(f"Bot created with name: {bot_name_final}")

        return {
            "bot_name": bot_name_final,
            "status": "initialized",
            "persona_mode": bot_entry["persona_mode"],
            "execution_mode": bot_entry["execution_mode"],
            "bot_configuration": result["bot_configuration"],
            "message": f"Bot '{bot_name_final}' created successfully"
        }
    
    def get_bot(self, bot_name: str) -> Optional[Dict[str, Any]]:
        """Get bot by name."""
        if self._database:
            return self._bot_repository.load_bot(bot_name)
        return None

    def get_bot_by_name(self, bot_name: str) -> Optional[Dict[str, Any]]:
        """Get bot by name (alias for get_bot)."""
        return self.get_bot(bot_name)

    def list_bots(self) -> List[Dict[str, Any]]:
        """List all bots."""
        if self._database:
            bots = self._bot_repository.load_all_bots()
            return [
                {
                    "bot_name": bot["bot_name"],
                    "status": bot["status"],
                    "conversations_count": len(bot.get("conversations", {})),
                    "created_at": bot.get("created_at")
                }
                for bot in bots
            ]
        return []
    
    def delete_bot(self, bot_name: str) -> bool:
        """
        Delete a bot by name.

        Args:
            bot_name: Bot name

        Returns:
            True if deleted, False if not found
        """
        if self._database:
            success = self._bot_repository.delete_bot(bot_name)
            if success:
                logger.info(f"Bot {bot_name} deleted")
                return True
            return False
        return False
    
    def _generate_conversation_title(self, user_message: str, assistant_response: str) -> str:
        """
        Generate a conversation title based on the first message exchange.
        Uses LLM directly to create a concise title from the first turn.

        Args:
            user_message: First user message
            assistant_response: First assistant response

        Returns:
            Generated conversation title
        """
        try:
            # Import here to avoid circular dependencies and cache serialization issues
            from langchain_core.messages import HumanMessage
            
            # Use BotCreatorAgent's LLM directly to generate title
            # This avoids creating SummarizerAgent instance which causes cache serialization issues
            prompt = f"""Based on this conversation exchange, generate a concise title (maximum 50 characters, no quotes):

User: {user_message[:200]}
Assistant: {assistant_response[:200]}

Generate a short, descriptive title that captures the main topic or question:"""
            
            # Use the same LLM as BotCreatorAgent
            response = self._bot_creator.llm.invoke([HumanMessage(content=prompt)])
            title = response.content.strip()
            
            # Clean up title (remove quotes if present)
            title = title.strip('"\'')
            
            # Limit length
            if len(title) > 50:
                title = title[:47] + "..."
            
            # Fallback if title is empty or too short
            if not title or len(title) < 3:
                words = user_message.split()[:5]
                title = " ".join(words) + ("..." if len(user_message.split()) > 5 else "")
            
            return title
        except Exception as e:
            logger.warning(f"Failed to generate conversation title: {e}, using fallback")
            # Fallback: use first few words of user message
            words = user_message.split()[:5]
            return " ".join(words) + ("..." if len(user_message.split()) > 5 else "")

    def chat_with_bot(
        self,
        bot_name: str,
        user_message: str,
        conversation_id: Optional[str] = None,
        conversation_title: Optional[str] = None,
        auto_create_conversation: bool = True
    ) -> Dict[str, Any]:
        """
        Chat with a bot. Supports conversation threads, auto-creation, and incognito mode.
        
        ChatGPT-like behavior:
        - If conversation_id is provided: Continue existing conversation
        - If conversation_id is None and auto_create_conversation is True: Auto-create new conversation
        - If conversation_id is None and auto_create_conversation is False: Incognito mode (temporary, not saved)

        Args:
            bot_name: Bot name (unique identifier)
            user_message: User message
            conversation_id: Conversation ID for conversation thread. If provided, bot remembers
                           previous messages in this thread.
            conversation_title: Optional title for new conversations. If None and auto-creating,
                              will auto-generate from first message.
            auto_create_conversation: If True and conversation_id is None, automatically creates
                                    a new conversation (ChatGPT-like). If False, uses incognito mode.

        Returns:
            Bot response with conversation info
        """
        bot = self.get_bot_by_name(bot_name)
        if not bot:
            return {
                "error": f"Bot '{bot_name}' not found",
                "bot_name": bot_name
            }

        # Initialize conversations dict if needed
        if "conversations" not in bot:
            bot["conversations"] = {}

        conversations = bot["conversations"]
        is_new_conversation = False
        conversation_id_to_use = conversation_id

        # Determine mode and setup conversation
        if conversation_id_to_use:
            # Conversation thread mode: bot remembers previous messages
            conversation = conversations.get(conversation_id_to_use)
            if not conversation:
                return {
                    "error": f"Conversation '{conversation_id_to_use}' not found for bot '{bot_name}'",
                    "bot_name": bot_name,
                    "conversation_id": conversation_id_to_use
                }

            conversation_history = conversation.get("history", [])
            history_mode = True
            mode = "conversation"
            title = conversation.get("title")
        elif auto_create_conversation:
            # ChatGPT-like mode: Auto-create new conversation
            conversation_id_to_use = str(uuid.uuid4())
            is_new_conversation = True
            
            # Initialize new conversation
            conversation = {
                "conversation_id": conversation_id_to_use,
                "title": conversation_title or f"Conversation {conversation_id_to_use[:8]}",  # Will be updated after first response
                "created_at": str(datetime.utcnow()),
                "updated_at": str(datetime.utcnow()),
                "turn_count": 0,
                "history": []
            }
            
            conversations[conversation_id_to_use] = conversation
            conversation_history = []
            history_mode = True
            mode = "conversation"
            title = conversation["title"]  # Temporary title, will be updated
        else:
            # Incognito mode: completely independent, no conversation system interaction
            conversation_id_to_use = None
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

        if "error" in result:
            return result

        # Add metadata to result
        result["bot_name"] = bot_name
        result["mode"] = mode

        # Handle conversation mode (existing or newly created)
        if conversation_id_to_use:
            # Update conversation history
            conversation_history_updated = result.get("conversation_history", [])
            conversations[conversation_id_to_use]["history"] = conversation_history_updated
            conversations[conversation_id_to_use]["turn_count"] = len(conversation_history_updated) // 2
            conversations[conversation_id_to_use]["updated_at"] = str(datetime.utcnow())

            # Auto-generate title for new conversations if not provided
            if is_new_conversation and (not conversation_title or conversation_title.startswith("Conversation ")):
                # Generate title from first message exchange
                assistant_response = result.get("response", "")
                if assistant_response:
                    generated_title = self._generate_conversation_title(user_message, assistant_response)
                    conversations[conversation_id_to_use]["title"] = generated_title
                    title = generated_title
                    logger.info(f"Auto-generated conversation title: {generated_title}")

            result["conversation_id"] = conversation_id_to_use
            result["conversation_title"] = title
            result["turn_count"] = conversations[conversation_id_to_use]["turn_count"]

            # Persist conversation if persistence is enabled
            if self._database:
                success = self._bot_repository.save_bot(bot)
                if not success:
                    logger.warning(f"Failed to persist conversation for bot {bot_name}")
        else:
            # In incognito mode, don't include conversation-related fields
            # Remove conversation_history from result for cleaner incognito response
            if "conversation_history" in result:
                del result["conversation_history"]

        return result
    
    
    def clear_all_conversations(self, bot_name: str) -> Dict[str, Any]:
        """
        Clear all conversation threads for a bot.

        Args:
            bot_name: Bot name

        Returns:
            Success message or error
        """
        bot = self.get_bot(bot_name)
        if not bot:
            return {
                "error": f"Bot '{bot_name}' not found",
                "bot_name": bot_name
            }

        # Clear all conversations
        conversations_count = len(bot.get("conversations", {}))
        bot["conversations"] = {}

        reset_agent_cache()  # For simplicity, clear all cache (could be more selective)
        logger.info(f"Cleared cache/memory for all conversations of bot {bot_name}")

        # Persist changes to persistence
        if self._database:
            success = self._bot_repository.save_bot(bot)
            if not success:
                return {
                    "error": "Failed to clear conversations from persistence",
                    "bot_name": bot_name
                }
        else:
            return {
                "error": "Persistence not enabled",
                "bot_name": bot_name
            }

        logger.info(f"Cleared {conversations_count} conversations for bot {bot_name} with memory/cache cleanup")
        return {
            "message": f"All conversations cleared successfully (memory and cache cleared, {conversations_count} conversations removed)",
            "bot_name": bot_name,
            "conversations_cleared": conversations_count,
            "cache_cleared": True
        }

    def update_conversation_history(self, bot_name: str, conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Update a bot's conversation history.

        Args:
            bot_name: Bot name
            conversation_history: New conversation history

        Returns:
            Success message or error
        """
        bot = self.get_bot(bot_name)
        if not bot:
            return {
                "error": f"Bot '{bot_name}' not found",
                "bot_name": bot_name
            }

        if bot["history_mode"] != "stateful":
            return {
                "error": f"Bot '{bot_name}' is in stateless mode, cannot update conversation history",
                "bot_name": bot_name
            }

        # Update in-memory bot object
        bot["conversation_history"] = conversation_history

        # Persist to persistence if available
        if self._bot_repository:
            success = self._bot_repository.save_conversation_history(bot_name, conversation_history)
            if not success:
                logger.warning(f"Failed to persist conversation history for bot {bot_name}")
                return {
                    "error": "Failed to persist conversation history to persistence",
                    "bot_name": bot_name
                }

        logger.info(f"Updated conversation history for bot {bot_name}")
        return {
            "message": "Conversation history updated successfully",
            "bot_name": bot_name
        }

    def create_conversation(
        self,
        bot_name: str,
        title: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new conversation thread for a bot.

        Args:
            bot_name: Bot name
            title: Optional title for the conversation
            system_prompt: Optional system prompt to initialize the conversation

        Returns:
            Conversation creation result
        """
        bot = self.get_bot(bot_name)
        if not bot:
            return {
                "error": f"Bot '{bot_name}' not found",
                "bot_name": bot_name
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

        # Persist to persistence
        if self._database:
            success = self._bot_repository.save_bot(bot)
            if not success:
                logger.warning(f"Failed to persist new conversation for bot {bot_name}")
                return {
                    "error": "Failed to persist conversation to persistence",
                    "bot_name": bot_name
                }

        logger.info(f"Created conversation {conversation_id} for bot {bot_name}")
        return {
            "message": "Conversation created successfully",
            "bot_name": bot_name,
            "conversation_id": conversation_id,
            "title": conversation["title"],
            "created_at": conversation["created_at"]
        }

    def list_conversations(self, bot_name: str) -> List[Dict[str, Any]]:
        """
        List all conversation threads for a bot.

        Args:
            bot_name: Bot name

        Returns:
            List of conversations with metadata
        """
        bot = self.get_bot(bot_name)
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

    def delete_conversation(self, bot_name: str, conversation_id: str) -> Dict[str, Any]:
        """
        Delete a conversation thread.

        Args:
            bot_name: Bot name
            conversation_id: Conversation UUID

        Returns:
            Success or error message
        """
        bot = self.get_bot(bot_name)
        if not bot:
            return {
                "error": f"Bot '{bot_name}' not found",
                "bot_name": bot_name
            }

        conversations = bot.get("conversations", {})
        if conversation_id not in conversations:
            return {
                "error": f"Conversation '{conversation_id}' not found for bot '{bot_name}'",
                "bot_name": bot_name,
                "conversation_id": conversation_id
            }

        # Remove conversation
        deleted_conversation = conversations.pop(conversation_id)

        # Clear conversation-related cache/memory
        reset_agent_cache()  # For simplicity, clear all cache (could be more selective)

        logger.info(f"Cleared cache/memory for conversation {conversation_id}")


        # Persist changes to persistence
        if self._database:
            success = self._bot_repository.save_bot(bot)
            if not success:
                logger.warning(f"Failed to persist conversation deletion for bot {bot_name}")
                return {
                    "error": "Failed to persist conversation deletion to persistence",
                    "bot_name": bot_name
                }

        logger.info(f"Deleted conversation {conversation_id} for bot {bot_name} with memory/cache cleanup")
        return {
            "message": "Conversation deleted successfully (memory and cache cleared)",
            "bot_name": bot_name,
            "conversation_id": conversation_id,
            "deleted_title": deleted_conversation.get("title"),
            "cache_cleared": True
        }

    def rename_conversation(self, bot_name: str, conversation_id: str, new_title: str) -> Dict[str, Any]:
        """
        Rename a conversation thread.

        Args:
            bot_name: Bot name
            conversation_id: Conversation UUID
            new_title: New title for the conversation

        Returns:
            Success or error message
        """
        bot = self.get_bot(bot_name)
        if not bot:
            return {
                "error": f"Bot '{bot_name}' not found",
                "bot_name": bot_name
            }

        conversations = bot.get("conversations", {})
        if conversation_id not in conversations:
            return {
                "error": f"Conversation '{conversation_id}' not found for bot '{bot_name}'",
                "bot_name": bot_name,
                "conversation_id": conversation_id
            }

        # Update title and timestamp
        old_title = conversations[conversation_id].get("title")
        conversations[conversation_id]["title"] = new_title
        conversations[conversation_id]["updated_at"] = str(datetime.utcnow())

        # Persist changes
        if self._database:
            success = self._bot_repository.save_bot(bot)
            if not success:
                logger.warning(f"Failed to persist conversation rename for bot {bot_name}")
                return {
                    "error": "Failed to persist conversation rename to persistence",
                    "bot_name": bot_name
                }

        logger.info(f"Renamed conversation {conversation_id} for bot {bot_name}: '{old_title}' -> '{new_title}'")
        return {
            "message": "Conversation renamed successfully",
            "bot_name": bot_name,
            "conversation_id": conversation_id,
            "old_title": old_title,
            "new_title": new_title
        }

    def get_conversation(self, bot_name: str, conversation_id: str) -> Dict[str, Any]:
        """
        Get details of a specific conversation thread.

        Args:
            bot_name: Bot name
            conversation_id: Conversation UUID

        Returns:
            Conversation details with organized message history
        """
        bot = self.get_bot(bot_name)
        if not bot:
            return {
                "error": f"Bot '{bot_name}' not found",
                "bot_name": bot_name
            }

        conversations = bot.get("conversations", {})
        conversation = conversations.get(conversation_id)
        if not conversation:
            return {
                "error": f"Conversation '{conversation_id}' not found for bot '{bot_name}'",
                "bot_name": bot_name,
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
            "bot_name": bot_name,
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

