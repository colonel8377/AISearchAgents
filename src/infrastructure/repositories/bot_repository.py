"""
Bot Repository - Domain layer data access.

This repository provides business semantics for bot operations.
It uses DatabaseStorage (infrastructure layer) to perform actual persistence.
"""

from typing import Optional, List, Dict, Any

from .interfaces import BotRepositoryInterface
from ...shared.utils.logger import get_logger

logger = get_logger(__name__)

class BotRepository(BotRepositoryInterface):
    """
    Bot repository implementation.
    
    Provides business semantics for bot operations.
    Uses DatabaseStorage (infrastructure layer) for persistence.
    """

    def __init__(self, storage):
        """
        Initialize bot repository.
        
        Args:
            storage: DatabaseStorage instance (infrastructure layer)
        """
        self.storage = storage
        self._init_tables()
        logger.debug("BotRepository initialized")

    def _init_tables(self) -> None:
        """Initialize bot-related tables."""
        if not self.storage.table_exists("bots"):
            self.storage.create_table("bots", """
                bot_name TEXT PRIMARY KEY,
                persona_prompt TEXT,
                bot_configuration TEXT,
                history_mode TEXT,
                persona_mode TEXT,
                execution_mode TEXT,
                metadata TEXT,
                status TEXT DEFAULT 'initialized',
                conversations TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            """)

    def _json_dumps(self, data: Any) -> str:
        """Convert data to JSON string."""
        import json
        return json.dumps(data, ensure_ascii=False)

    def _json_loads(self, data: str) -> Any:
        """Parse JSON string to data."""
        import json
        if data is None or data == "":
            return None
        return json.loads(data)

    def save_bot(self, bot_data: Dict[str, Any]) -> bool:
        """Save or update a bot."""
        try:
            bot_name = bot_data.get("bot_name")
            if not bot_name:
                logger.error("Bot name is required")
                return False

            # Check if bot exists
            existing = self.storage.execute_query(
                "SELECT bot_name FROM bots WHERE bot_name = ?",
                (bot_name,)
            )

            data = (
                bot_data.get("persona_prompt"),
                self._json_dumps(bot_data.get("bot_configuration")),
                bot_data.get("history_mode"),
                bot_data.get("persona_mode"),
                bot_data.get("execution_mode"),
                self._json_dumps(bot_data.get("metadata", {})),
                bot_data.get("status", "initialized"),
                self._json_dumps(bot_data.get("conversations", {}))
            )

            if existing:
                # Update existing bot
                self.storage.execute_update(
                    """UPDATE bots SET
                        persona_prompt = ?, bot_configuration = ?,
                        history_mode = ?, persona_mode = ?, execution_mode = ?,
                        metadata = ?, status = ?, conversations = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE bot_name = ?""",
                    data + (bot_name,)
                )
            else:
                # Insert new bot
                self.storage.execute_update(
                    """INSERT INTO bots (
                        bot_name, persona_prompt, bot_configuration,
                        history_mode, persona_mode, execution_mode, metadata, status, conversations
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (bot_name,) + data
                )

            logger.debug(f"Bot {bot_name} saved successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to save bot {bot_data.get('bot_name')}: {e}")
            return False

    def load_bot(self, bot_name: str) -> Optional[Dict[str, Any]]:
        """Load a bot by name."""
        try:
            rows = self.storage.execute_query(
                "SELECT * FROM bots WHERE bot_name = ?",
                (bot_name,)
            )
            if not rows:
                return None

            row = rows[0]
            return {
                "bot_name": row["bot_name"],
                "persona_prompt": row["persona_prompt"],
                "bot_configuration": self._json_loads(row["bot_configuration"]),
                "history_mode": row["history_mode"],
                "persona_mode": row["persona_mode"],
                "execution_mode": row["execution_mode"],
                "metadata": self._json_loads(row["metadata"]),
                "status": row["status"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "conversations": self._json_loads(row["conversations"]) if row["conversations"] else {}
            }
        except Exception as e:
            logger.error(f"Failed to load bot {bot_name}: {e}")
            return None


    def load_all_bots(self) -> List[Dict[str, Any]]:
        """Load all bots."""
        try:
            rows = self.storage.execute_query(
                "SELECT * FROM bots ORDER BY created_at DESC"
            )
            bots = []
            for row in rows:
                bots.append({
                    "bot_name": row["bot_name"],
                    "persona_prompt": row["persona_prompt"],
                    "bot_configuration": self._json_loads(row["bot_configuration"]),
                    "history_mode": row["history_mode"],
                    "persona_mode": row["persona_mode"],
                    "execution_mode": row["execution_mode"],
                    "metadata": self._json_loads(row["metadata"]),
                    "status": row["status"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "conversations": self._json_loads(row["conversations"]) if row["conversations"] else {}
                })
            return bots
        except Exception as e:
            logger.error(f"Failed to load all bots: {e}")
            return []

    def delete_bot(self, bot_name: str) -> bool:
        """Delete a bot by name."""
        try:
            self.storage.execute_update(
                "DELETE FROM bots WHERE bot_name = ?",
                (bot_name,)
            )
            logger.debug(f"Bot {bot_name} deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to delete bot {bot_name}: {e}")
            return False

    def bot_exists(self, bot_name: str) -> bool:
        """Check if a bot exists."""
        try:
            rows = self.storage.execute_query(
                "SELECT 1 FROM bots WHERE bot_name = ? LIMIT 1",
                (bot_name,)
            )
            return len(rows) > 0
        except Exception as e:
            logger.error(f"Failed to check if bot {bot_name} exists: {e}")
            return False

    def save_conversation_history(self, bot_name: str, history: List[Dict[str, Any]]) -> bool:
        """Save conversation history for a bot."""
        try:
            # First check if conversations table exists, create if not
            if not self.storage.table_exists("conversations"):
                self.storage.create_table("conversations", """
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bot_name TEXT NOT NULL,
                    conversation_id TEXT,
                    turn_index INTEGER,
                    user_message TEXT,
                    assistant_response TEXT,
                    user_role TEXT,
                    assistant_role TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (bot_name) REFERENCES bots (bot_name)
                """)

            # Clear existing conversation history for this bot
            self.storage.execute_update(
                "DELETE FROM conversations WHERE bot_name = ?",
                (bot_name,)
            )

            # Insert new conversation history
            for i, turn in enumerate(history):
                self.storage.execute_update(
                    """INSERT INTO conversations (
                        bot_name, conversation_id, turn_index, user_message,
                        assistant_response, user_role, assistant_role
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        bot_name,
                        turn.get("conversation_id"),
                        i,
                        turn.get("user_message"),
                        turn.get("assistant_response"),
                        turn.get("user_role"),
                        turn.get("assistant_role")
                    )
                )

            logger.debug(f"Saved {len(history)} conversation turns for bot {bot_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to save conversation history for bot {bot_name}: {e}")
            return False

