"""Global Bot Manager for managing all bots independently."""

import uuid
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
            proxy=settings.openai_proxy,
            persona_mode="system_prompt"
        )

        # Initialize storage connection if persistence is enabled
        self._storage = get_storage() if settings.enable_persistence else None
        self._database = self._storage.get_database() if self._storage else None

        logger.info("BotManager initialized")
    
    def create_bot(
        self,
        persona_prompt: str,
        bot_name: Optional[str] = None,
        history_mode: str = "stateless",  # "stateless" or "stateful"
        execution_mode: Optional[str] = None,
        use_few_shots: bool = True,
        custom_few_shots: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new bot with UUID.

        Args:
            persona_prompt: Persona description for the bot
            bot_name: Optional name for the bot
            history_mode: "stateless" or "stateful" - determines if conversation history is stored
            execution_mode: Execution mode for bot creation
            use_few_shots: Whether to use few-shot examples
            custom_few_shots: Optional custom few-shot examples

        Returns:
            Bot creation result with UUID
        """
        if history_mode not in ["stateless", "stateful"]:
            raise ValueError("history_mode must be 'stateless' or 'stateful'")

        # Create bot using BotCreatorAgent
        result = self._bot_creator.create_bot(
            persona_prompt=persona_prompt,
            bot_name=bot_name,
            execution_mode=execution_mode,
            use_few_shots=use_few_shots,
            custom_few_shots=custom_few_shots
        )

        if "error" in result:
            return result

        # Generate UUID for bot
        bot_uuid = str(uuid.uuid4())

        # Prepare bot entry
        bot_entry = {
            "bot_id": bot_uuid,
            "bot_name": result["bot_name"],
            "persona_prompt": persona_prompt,
            "bot_configuration": result["bot_configuration"],
            "status": "initialized",
            "history_mode": history_mode,
            "conversation_history": [] if history_mode == "stateful" else None,
            "persona_mode": result.get("persona_mode", "system_prompt"),
            "execution_mode": result.get("execution_mode"),
            "metadata": result.get("metadata", {})
        }

        # Save to database if persistence is enabled
        if self._database:
            success = self._database.save_bot(bot_entry)
            if not success:
                logger.warning(f"Failed to persist bot {bot_uuid}, continuing without persistence")

        logger.info(f"Bot created with UUID: {bot_uuid}, history_mode: {history_mode}")

        return {
            "bot_id": bot_uuid,
            "bot_name": bot_entry["bot_name"],
            "status": "initialized",
            "persona_prompt": persona_prompt,
            "persona_mode": bot_entry["persona_mode"],
            "execution_mode": bot_entry["execution_mode"],
            "history_mode": history_mode,
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
                    "history_mode": bot["history_mode"],
                    "persona_prompt": bot["persona_prompt"],
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
        user_message: str
    ) -> Dict[str, Any]:
        """
        Chat with a bot. Automatically manages conversation history if bot is in stateful mode.

        Args:
            bot_id: Bot UUID
            user_message: User message

        Returns:
            Bot response with updated conversation history
        """
        bot = self.get_bot(bot_id)
        if not bot:
            return {
                "error": f"Bot '{bot_id}' not found",
                "bot_id": bot_id
            }

        # Get conversation history based on mode
        conversation_history = None
        if bot["history_mode"] == "stateful":
            conversation_history = bot.get("conversation_history", [])

        # Chat with bot using BotCreatorAgent
        # Pass bot configuration instead of bot_id
        result = self._bot_creator.chat_with_bot_by_config(
            bot_configuration=bot["bot_configuration"],
            bot_name=bot["bot_name"],
            user_message=user_message,
            conversation_history=conversation_history,
            history_mode=(bot["history_mode"] == "stateful")
        )

        # Add bot_id and history_mode to result
        result["bot_id"] = bot_id
        result["history_mode"] = bot["history_mode"]

        if "error" in result:
            return result

        # Save updated conversation history to database if stateful
        if bot["history_mode"] == "stateful" and self._database:
            success = self._database.save_conversation_history(bot_id, result["conversation_history"])
            if not success:
                logger.warning(f"Failed to persist conversation history for bot {bot_id}")

        return result
    
    def delete_conversation_turn(
        self,
        bot_id: str,
        turn_index: int
    ) -> Dict[str, Any]:
        """
        Delete a specific conversation turn from bot's history.

        Args:
            bot_id: Bot UUID
            turn_index: Index of the turn to delete (0-based, each turn is 2 entries: user + assistant)

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
                "error": f"Bot '{bot_id}' is in stateless mode, no conversation history to delete",
                "bot_id": bot_id
            }

        # Get current history length for validation
        history = bot.get("conversation_history", [])
        if not history:
            return {
                "error": "No conversation history found",
                "bot_id": bot_id
            }

        # Each turn consists of user message + assistant response (2 entries)
        start_idx = turn_index * 2
        if start_idx >= len(history):
            return {
                "error": f"Turn index {turn_index} out of range. Bot has {len(history) // 2} turns",
                "bot_id": bot_id
            }

        # Delete the turn from database
        if self._database:
            success = self._database.delete_conversation_turn(bot_id, turn_index)
            if not success:
                return {
                    "error": "Failed to delete conversation turn from database",
                    "bot_id": bot_id
                }
        else:
            return {
                "error": "Persistence not enabled",
                "bot_id": bot_id
            }

        # Reload bot to get updated history
        updated_bot = self.get_bot(bot_id)
        updated_history = updated_bot.get("conversation_history", []) if updated_bot else []

        logger.info(f"Deleted turn {turn_index} from bot {bot_id}")
        return {
            "message": f"Turn {turn_index} deleted successfully",
            "bot_id": bot_id,
            "deleted_entries": history[start_idx:start_idx + 2],  # Original entries
            "remaining_turns": len(updated_history) // 2
        }
    
    def clear_conversation_history(self, bot_id: str) -> Dict[str, Any]:
        """
        Clear all conversation history for a bot.

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

        if bot["history_mode"] != "stateful":
            return {
                "error": f"Bot '{bot_id}' is in stateless mode, no conversation history to clear",
                "bot_id": bot_id
            }

        if self._database:
            success = self._database.clear_conversation_history(bot_id)
            if not success:
                return {
                    "error": "Failed to clear conversation history from database",
                    "bot_id": bot_id
                }
        else:
            return {
                "error": "Persistence not enabled",
                "bot_id": bot_id
            }

        logger.info(f"Cleared conversation history for bot {bot_id}")
        return {
            "message": "Conversation history cleared successfully",
            "bot_id": bot_id
        }


# Global bot manager instance
bot_manager = BotManager()

