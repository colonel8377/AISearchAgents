"""
Database management for unified storage module.
Database-centric persistence for bots, conversation history, custom few shots, and debate sessions.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from ..config.settings import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class StorageDatabase:
    """
    SQLite database manager for persistent storage of application data.

    Handles all CRUD operations for:
    - Bots and conversation history
    - Custom few shots
    - Debate sessions
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database connection and create tables if they don't exist.

        Args:
            db_path: Path to SQLite database file. If None, uses default path.
        """
        if db_path is None:
            # Use project root data directory as default (not src/data)
            from ..config.settings import BASE_DIR
            data_dir = BASE_DIR / "data"
            data_dir.mkdir(exist_ok=True)
            db_path = data_dir / "app_storage.db"

        self.db_path = str(db_path)
        self._connection = None
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection (lazy initialization)."""
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path)
            self._connection.execute("PRAGMA foreign_keys = ON")
        return self._connection

    def _init_db(self) -> None:
        """Initialize database tables and indexes."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Create bots table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_id TEXT NOT NULL UNIQUE,
                bot_name TEXT,
                persona_prompt TEXT,
                bot_configuration TEXT,
                history_mode TEXT,
                persona_mode TEXT,
                execution_mode TEXT,
                metadata TEXT,
                status TEXT DEFAULT 'initialized',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create conversation_history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_id TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(bot_id, turn_index, role)
            )
        """)

        # Create custom_few_shots table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS custom_few_shots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_type TEXT NOT NULL UNIQUE,
                few_shots_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create privacy_detection_results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS privacy_detection_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detection_id TEXT NOT NULL UNIQUE,
                conversation_records TEXT NOT NULL,
                detection_result TEXT NOT NULL,
                execution_mode TEXT,
                use_few_shots BOOLEAN DEFAULT 1,
                conversation_length INTEGER,
                analyzed_at TIMESTAMP,
                agent_version TEXT,
                error TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create debate_sessions table (legacy schema for backward compatibility)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS debate_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL UNIQUE,
                topic TEXT NOT NULL,
                personas TEXT,
                agents TEXT,
                vote_history TEXT,
                reasoning_history TEXT,
                statistics TEXT,
                max_rounds INTEGER,
                current_round INTEGER DEFAULT 0,
                user_corpus TEXT,
                detected_user_style TEXT,
                debate_history TEXT,
                current_phase TEXT,
                current_round_reasonings TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bots_bot_id ON bots(bot_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversation_bot_id ON conversation_history(bot_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversation_turn ON conversation_history(turn_index)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_custom_few_shots_agent_type ON custom_few_shots(agent_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_debate_sessions_session_id ON debate_sessions(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_privacy_detection_id ON privacy_detection_results(detection_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_privacy_analyzed_at ON privacy_detection_results(analyzed_at)")

        # Run migrations for existing databases
        self._migrate_debate_sessions()
        self._migrate_privacy_detection_results()

        conn.commit()
        logger.info(f"Database initialized at {self.db_path}")

    def _migrate_debate_sessions(self) -> None:
        """Migrate existing debate_sessions table to support new schema."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Check if new columns exist
            cursor.execute("PRAGMA table_info(debate_sessions)")
            columns = [row[1] for row in cursor.fetchall()]

            new_columns = [
                ("user_corpus", "TEXT"),
                ("detected_user_style", "TEXT"),
                ("debate_history", "TEXT"),
                ("current_phase", "TEXT"),
                ("current_round_reasonings", "TEXT")
            ]

            for column_name, column_type in new_columns:
                if column_name not in columns:
                    logger.info(f"Adding column '{column_name}' to debate_sessions table")
                    cursor.execute(f"ALTER TABLE debate_sessions ADD COLUMN {column_name} {column_type}")

            conn.commit()
            logger.info("Debate sessions migration completed")

        except Exception as e:
            logger.error(f"Failed to migrate debate_sessions table: {e}")
            # Don't fail initialization if migration fails - just log the error

    def _migrate_privacy_detection_results(self) -> None:
        """Migrate existing privacy_detection_results table to support account_id."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Check if account_id column exists
            cursor.execute("PRAGMA table_info(privacy_detection_results)")
            columns = [row[1] for row in cursor.fetchall()]

            if 'account_id' not in columns:
                logger.info("Adding account_id column to privacy_detection_results table")
                cursor.execute("ALTER TABLE privacy_detection_results ADD COLUMN account_id TEXT")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_privacy_account_id ON privacy_detection_results(account_id)")

            conn.commit()
            logger.info("Privacy detection results migration completed")

        except Exception as e:
            logger.error(f"Failed to migrate privacy_detection_results table: {e}")
            # Don't fail initialization if migration fails - just log the error

    def _json_dumps(self, data: Any) -> str:
        """Convert data to JSON string."""
        return json.dumps(data, ensure_ascii=False)

    def _json_loads(self, data: str) -> Any:
        """Parse JSON string to data."""
        if data is None or data == "":
            return None
        return json.loads(data)

    def _update_timestamp(self, table: str, id_field: str, id_value: Any) -> None:
        """Update the updated_at timestamp for a record."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            f"UPDATE {table} SET updated_at = CURRENT_TIMESTAMP WHERE {id_field} = ?",
            (id_value,)
        )
        conn.commit()

    # Bot management methods

    def save_bot(self, bot_data: Dict[str, Any]) -> bool:
        """
        Save or update a bot in the database.

        Args:
            bot_data: Bot data dictionary

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Check if bot exists
            cursor.execute("SELECT id FROM bots WHERE bot_id = ?", (bot_data["bot_id"],))
            existing = cursor.fetchone()

            data = {
                "bot_id": bot_data["bot_id"],
                "bot_name": bot_data.get("bot_name"),
                "persona_prompt": bot_data.get("persona_prompt"),
                "bot_configuration": self._json_dumps(bot_data.get("bot_configuration")),
                "history_mode": bot_data.get("history_mode"),
                "persona_mode": bot_data.get("persona_mode"),
                "execution_mode": bot_data.get("execution_mode"),
                "metadata": self._json_dumps(bot_data.get("metadata", {})),
                "status": bot_data.get("status", "initialized")
            }

            if existing:
                # Update existing bot
                cursor.execute("""
                    UPDATE bots SET
                        bot_name = ?, persona_prompt = ?, bot_configuration = ?,
                        history_mode = ?, persona_mode = ?, execution_mode = ?,
                        metadata = ?, status = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE bot_id = ?
                """, (
                    data["bot_name"], data["persona_prompt"], data["bot_configuration"],
                    data["history_mode"], data["persona_mode"], data["execution_mode"],
                    data["metadata"], data["status"], data["bot_id"]
                ))
            else:
                # Insert new bot
                cursor.execute("""
                    INSERT INTO bots (
                        bot_id, bot_name, persona_prompt, bot_configuration,
                        history_mode, persona_mode, execution_mode, metadata, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data["bot_id"], data["bot_name"], data["persona_prompt"], data["bot_configuration"],
                    data["history_mode"], data["persona_mode"], data["execution_mode"],
                    data["metadata"], data["status"]
                ))

            conn.commit()
            logger.debug(f"Bot {bot_data['bot_id']} saved successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to save bot {bot_data.get('bot_id')}: {e}")
            return False

    def load_bot(self, bot_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a bot from the database, including conversation history.

        Args:
            bot_id: Bot ID

        Returns:
            Bot data with conversation history or None if not found
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Load bot data
            cursor.execute("SELECT * FROM bots WHERE bot_id = ?", (bot_id,))
            bot_row = cursor.fetchone()

            if not bot_row:
                return None

            # Parse bot data
            bot_data = {
                "id": bot_row[0],
                "bot_id": bot_row[1],
                "bot_name": bot_row[2],
                "persona_prompt": bot_row[3],
                "bot_configuration": self._json_loads(bot_row[4]),
                "history_mode": bot_row[5],
                "persona_mode": bot_row[6],
                "execution_mode": bot_row[7],
                "metadata": self._json_loads(bot_row[8]),
                "status": bot_row[9],
                "created_at": bot_row[10],
                "updated_at": bot_row[11]
            }

            # Load conversation history if bot is stateful
            if bot_data["history_mode"] == "stateful":
                cursor.execute(
                    "SELECT turn_index, role, content FROM conversation_history WHERE bot_id = ? ORDER BY turn_index, id",
                    (bot_id,)
                )
                history_rows = cursor.fetchall()
                bot_data["conversation_history"] = [
                    {"role": row[1], "content": row[2]} for row in history_rows
                ]
            else:
                bot_data["conversation_history"] = None

            return bot_data

        except Exception as e:
            logger.error(f"Failed to load bot {bot_id}: {e}")
            return None

    def load_all_bots(self) -> List[Dict[str, Any]]:
        """
        Load all bots from the database.

        Returns:
            List of bot data dictionaries
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT bot_id FROM bots ORDER BY created_at DESC")
            bot_ids = [row[0] for row in cursor.fetchall()]

            bots = []
            for bot_id in bot_ids:
                bot = self.load_bot(bot_id)
                if bot:
                    bots.append(bot)

            return bots

        except Exception as e:
            logger.error(f"Failed to load all bots: {e}")
            return []

    def delete_bot(self, bot_id: str) -> bool:
        """
        Delete a bot and its conversation history from the database.

        Args:
            bot_id: Bot ID

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Delete conversation history first (to maintain data consistency)
            cursor.execute("DELETE FROM conversation_history WHERE bot_id = ?", (bot_id,))

            # Delete bot
            cursor.execute("DELETE FROM bots WHERE bot_id = ?", (bot_id,))

            conn.commit()
            logger.debug(f"Bot {bot_id} and its history deleted successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to delete bot {bot_id}: {e}")
            return False

    def bot_exists(self, bot_id: str) -> bool:
        """
        Check if a bot exists in the database.

        Args:
            bot_id: Bot ID

        Returns:
            bool: True if bot exists, False otherwise
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT 1 FROM bots WHERE bot_id = ? LIMIT 1", (bot_id,))
            return cursor.fetchone() is not None

        except Exception as e:
            logger.error(f"Failed to check if bot {bot_id} exists: {e}")
            return False

    # Conversation history management methods

    def save_conversation_history(self, bot_id: str, history: List[Dict[str, Any]]) -> bool:
        """
        Save conversation history to the database.

        Args:
            bot_id: Bot ID
            history: List of conversation entries with role and content

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Clear existing history for this bot
            cursor.execute("DELETE FROM conversation_history WHERE bot_id = ?", (bot_id,))

            # Insert new history
            for turn_index, entry in enumerate(history):
                cursor.execute("""
                    INSERT INTO conversation_history (bot_id, turn_index, role, content)
                    VALUES (?, ?, ?, ?)
                """, (bot_id, turn_index, entry["role"], entry["content"]))

            conn.commit()
            logger.debug(f"Conversation history saved for bot {bot_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save conversation history for bot {bot_id}: {e}")
            return False

    def load_conversation_history(self, bot_id: str) -> List[Dict[str, Any]]:
        """
        Load conversation history from the database.

        Args:
            bot_id: Bot ID

        Returns:
            List of conversation entries
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                "SELECT turn_index, role, content FROM conversation_history WHERE bot_id = ? ORDER BY turn_index, id",
                (bot_id,)
            )

            history = [
                {"role": row[1], "content": row[2]} for row in cursor.fetchall()
            ]

            return history

        except Exception as e:
            logger.error(f"Failed to load conversation history for bot {bot_id}: {e}")
            return []

    def clear_conversation_history(self, bot_id: str) -> bool:
        """
        Clear conversation history for a bot.

        Args:
            bot_id: Bot ID

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("DELETE FROM conversation_history WHERE bot_id = ?", (bot_id,))
            conn.commit()

            logger.debug(f"Conversation history cleared for bot {bot_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to clear conversation history for bot {bot_id}: {e}")
            return False

    def delete_conversation_turn(self, bot_id: str, turn_index: int) -> bool:
        """
        Delete a specific conversation turn from the database.

        Args:
            bot_id: Bot ID
            turn_index: Turn index to delete

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Delete the turn (turn_index and turn_index+1 for user/assistant pair)
            cursor.execute(
                "DELETE FROM conversation_history WHERE bot_id = ? AND (turn_index = ? OR turn_index = ?)",
                (bot_id, turn_index * 2, turn_index * 2 + 1)
            )

            # Re-index remaining turns
            cursor.execute("""
                UPDATE conversation_history
                SET turn_index = turn_index - 2
                WHERE bot_id = ? AND turn_index > ?
            """, (bot_id, turn_index * 2 + 1))

            conn.commit()
            logger.debug(f"Turn {turn_index} deleted from bot {bot_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete conversation turn {turn_index} for bot {bot_id}: {e}")
            return False

    # Custom few shots management methods

    def save_custom_few_shots(self, agent_type: str, few_shots: Union[str, Dict]) -> bool:
        """
        Save custom few shots to the database.

        Args:
            agent_type: Agent type identifier
            few_shots: Few shots data (string or dict)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            few_shots_json = self._json_dumps(few_shots)

            # Check if exists
            cursor.execute("SELECT id FROM custom_few_shots WHERE agent_type = ?", (agent_type,))
            existing = cursor.fetchone()

            if existing:
                # Update
                cursor.execute("""
                    UPDATE custom_few_shots SET few_shots_json = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE agent_type = ?
                """, (few_shots_json, agent_type))
            else:
                # Insert
                cursor.execute("""
                    INSERT INTO custom_few_shots (agent_type, few_shots_json)
                    VALUES (?, ?)
                """, (agent_type, few_shots_json))

            conn.commit()
            logger.debug(f"Custom few shots saved for agent type {agent_type}")
            return True

        except Exception as e:
            logger.error(f"Failed to save custom few shots for {agent_type}: {e}")
            return False

    def load_custom_few_shots(self, agent_type: str) -> Optional[Union[str, Dict]]:
        """
        Load custom few shots from the database.

        Args:
            agent_type: Agent type identifier

        Returns:
            Few shots data or None if not found
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT few_shots_json FROM custom_few_shots WHERE agent_type = ?", (agent_type,))
            row = cursor.fetchone()

            if not row:
                return None

            return self._json_loads(row[0])

        except Exception as e:
            logger.error(f"Failed to load custom few shots for {agent_type}: {e}")
            return None

    def delete_custom_few_shots(self, agent_type: str) -> bool:
        """
        Delete custom few shots from the database.

        Args:
            agent_type: Agent type identifier

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("DELETE FROM custom_few_shots WHERE agent_type = ?", (agent_type,))
            conn.commit()

            logger.debug(f"Custom few shots deleted for agent type {agent_type}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete custom few shots for {agent_type}: {e}")
            return False

    def load_all_custom_few_shots(self) -> Dict[str, Union[str, Dict]]:
        """
        Load all custom few shots from the database.

        Returns:
            Dictionary mapping agent_type to few_shots data
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT agent_type, few_shots_json FROM custom_few_shots")
            rows = cursor.fetchall()

            result = {}
            for row in rows:
                agent_type, few_shots_json = row
                result[agent_type] = self._json_loads(few_shots_json)

            return result

        except Exception as e:
            logger.error(f"Failed to load all custom few shots: {e}")
            return {}

    # Debate sessions management methods

    def save_debate_session(self, session_data: Dict[str, Any]) -> bool:
        """
        Save or update a debate session in the database.

        Args:
            session_data: Session data dictionary

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            session_id = session_data["session_id"]

            # Check if session exists
            cursor.execute("SELECT id FROM debate_sessions WHERE session_id = ?", (session_id,))
            existing = cursor.fetchone()

            data = {
                "session_id": session_id,
                "topic": session_data["topic"],
                "personas": self._json_dumps(session_data.get("personas", [])),
                "agents": self._json_dumps(session_data.get("agents", {})),
                "vote_history": self._json_dumps(session_data.get("vote_history", [])),
                "reasoning_history": self._json_dumps(session_data.get("reasoning_history", [])),
                "statistics": self._json_dumps(session_data.get("statistics", {})),
                "max_rounds": session_data.get("max_rounds"),
                "current_round": session_data.get("current_round", 0),
                "user_corpus": self._json_dumps(session_data.get("user_corpus", [])),
                "detected_user_style": session_data.get("detected_user_style"),
                "debate_history": self._json_dumps(session_data.get("debate_history", [])),
                "current_phase": session_data.get("current_phase"),
                "current_round_reasonings": self._json_dumps(session_data.get("current_round_reasonings", []))
            }

            if existing:
                # Update
                cursor.execute("""
                    UPDATE debate_sessions SET
                        topic = ?, personas = ?, agents = ?, vote_history = ?,
                        reasoning_history = ?, statistics = ?, max_rounds = ?,
                        current_round = ?, user_corpus = ?, detected_user_style = ?,
                        debate_history = ?, current_phase = ?, current_round_reasonings = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE session_id = ?
                """, (
                    data["topic"], data["personas"], data["agents"], data["vote_history"],
                    data["reasoning_history"], data["statistics"], data["max_rounds"],
                    data["current_round"], data["user_corpus"], data["detected_user_style"],
                    data["debate_history"], data["current_phase"], data["current_round_reasonings"],
                    data["session_id"]
                ))
            else:
                # Insert
                cursor.execute("""
                    INSERT INTO debate_sessions (
                        session_id, topic, personas, agents, vote_history,
                        reasoning_history, statistics, max_rounds, current_round,
                        user_corpus, detected_user_style, debate_history,
                        current_phase, current_round_reasonings
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data["session_id"], data["topic"], data["personas"], data["agents"], data["vote_history"],
                    data["reasoning_history"], data["statistics"], data["max_rounds"], data["current_round"],
                    data["user_corpus"], data["detected_user_style"], data["debate_history"],
                    data["current_phase"], data["current_round_reasonings"]
                ))

            conn.commit()
            logger.debug(f"Debate session {session_id} saved successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to save debate session {session_data.get('session_id')}: {e}")
            return False

    def load_debate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a debate session from the database.

        Args:
            session_id: Session ID

        Returns:
            Session data or None if not found
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM debate_sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()

            if not row:
                return None

            return {
                "id": row[0],
                "session_id": row[1],
                "topic": row[2],
                "personas": self._json_loads(row[3]),
                "agents": self._json_loads(row[4]),
                "vote_history": self._json_loads(row[5]),
                "reasoning_history": self._json_loads(row[6]),
                "statistics": self._json_loads(row[7]),
                "max_rounds": row[8],
                "current_round": row[9],
                "user_corpus": self._json_loads(row[10]),
                "detected_user_style": row[11],
                "debate_history": self._json_loads(row[12]),
                "current_phase": row[13],
                "current_round_reasonings": self._json_loads(row[14]),
                "created_at": row[15],
                "updated_at": row[16]
            }

        except Exception as e:
            logger.error(f"Failed to load debate session {session_id}: {e}")
            return None

    def load_all_debate_sessions(self) -> List[Dict[str, Any]]:
        """
        Load all debate sessions from the database.

        Returns:
            List of session data dictionaries
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT session_id FROM debate_sessions ORDER BY created_at DESC")
            session_ids = [row[0] for row in cursor.fetchall()]

            sessions = []
            for session_id in session_ids:
                session = self.load_debate_session(session_id)
                if session:
                    sessions.append(session)

            return sessions

        except Exception as e:
            logger.error(f"Failed to load all debate sessions: {e}")
            return []

    def update_debate_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update specific fields of a debate session.

        Args:
            session_id: Session ID
            updates: Dictionary of fields to update

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Build update query dynamically
            set_parts = []
            values = []

            for key, value in updates.items():
                if key in ["topic", "max_rounds", "current_round"]:
                    set_parts.append(f"{key} = ?")
                    values.append(value)
                elif key in ["personas", "agents", "vote_history", "reasoning_history", "statistics"]:
                    set_parts.append(f"{key} = ?")
                    values.append(self._json_dumps(value))

            if not set_parts:
                return False

            set_clause = ", ".join(set_parts)
            query = f"UPDATE debate_sessions SET {set_clause}, updated_at = CURRENT_TIMESTAMP WHERE session_id = ?"
            values.append(session_id)

            cursor.execute(query, values)
            conn.commit()

            logger.debug(f"Debate session {session_id} updated successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to update debate session {session_id}: {e}")
            return False

    def delete_debate_session(self, session_id: str) -> bool:
        """
        Delete a debate session from the database.

        Args:
            session_id: Session ID

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("DELETE FROM debate_sessions WHERE session_id = ?", (session_id,))
            conn.commit()

            logger.debug(f"Debate session {session_id} deleted successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to delete debate session {session_id}: {e}")
            return False

    def session_exists(self, session_id: str) -> bool:
        """
        Check if a debate session exists in the database.

        Args:
            session_id: Session ID

        Returns:
            bool: True if session exists, False otherwise
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT 1 FROM debate_sessions WHERE session_id = ? LIMIT 1", (session_id,))
            return cursor.fetchone() is not None

        except Exception as e:
            logger.error(f"Failed to check if session {session_id} exists: {e}")
            return False

    # Clear all data methods

    def clear_all_bots(self) -> bool:
        """Clear all bots from the database."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("DELETE FROM conversation_history")
            cursor.execute("DELETE FROM bots")
            conn.commit()

            logger.info("All bots cleared from database")
            return True

        except Exception as e:
            logger.error(f"Failed to clear all bots: {e}")
            return False

    def clear_all_conversation_history(self) -> bool:
        """Clear all conversation history from the database."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("DELETE FROM conversation_history")
            conn.commit()

            logger.info("All conversation history cleared from database")
            return True

        except Exception as e:
            logger.error(f"Failed to clear all conversation history: {e}")
            return False

    def clear_all_custom_few_shots(self) -> bool:
        """Clear all custom few shots from the database."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("DELETE FROM custom_few_shots")
            conn.commit()

            logger.info("All custom few shots cleared from database")
            return True

        except Exception as e:
            logger.error(f"Failed to clear all custom few shots: {e}")
            return False

    def clear_all_debate_sessions(self) -> bool:
        """Clear all debate sessions from the database."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("DELETE FROM debate_sessions")
            conn.commit()

            logger.info("All debate sessions cleared from database")
            return True

        except Exception as e:
            logger.error(f"Failed to clear all debate sessions: {e}")
            return False

    def clear_all_data(self) -> bool:
        """Clear all data from the database."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("DELETE FROM conversation_history")
            cursor.execute("DELETE FROM bots")
            cursor.execute("DELETE FROM custom_few_shots")
            cursor.execute("DELETE FROM debate_sessions")
            conn.commit()

            logger.info("All data cleared from database")
            return True

        except Exception as e:
            logger.error(f"Failed to clear all data: {e}")
            return False

    # Privacy detection results management methods

    def save_privacy_detection_result(self, result_data: Dict[str, Any]) -> bool:
        """
        Save a privacy detection result to the database.

        Args:
            result_data: Privacy detection result data

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            detection_id = result_data["detection_id"]

            # Check if result exists
            cursor.execute("SELECT id FROM privacy_detection_results WHERE detection_id = ?", (detection_id,))
            existing = cursor.fetchone()

            data = {
                "detection_id": detection_id,
                "conversation_records": self._json_dumps(result_data["conversation_records"]),
                "detection_result": self._json_dumps(result_data["detection_result"]),
                "execution_mode": result_data.get("execution_mode"),
                "use_few_shots": result_data.get("use_few_shots", True),
                "conversation_length": result_data.get("conversation_length"),
                "analyzed_at": result_data.get("analyzed_at"),
                "agent_version": result_data.get("agent_version"),
                "error": result_data.get("error"),
                "account_id": result_data.get("account_id")
            }

            if existing:
                # Update existing result
                cursor.execute("""
                    UPDATE privacy_detection_results SET
                        conversation_records = ?, detection_result = ?, execution_mode = ?,
                        use_few_shots = ?, conversation_length = ?, analyzed_at = ?,
                        agent_version = ?, error = ?, account_id = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE detection_id = ?
                """, (
                    data["conversation_records"], data["detection_result"], data["execution_mode"],
                    data["use_few_shots"], data["conversation_length"], data["analyzed_at"],
                    data["agent_version"], data["error"], data["account_id"], data["detection_id"]
                ))
            else:
                # Insert new result
                cursor.execute("""
                    INSERT INTO privacy_detection_results (
                        detection_id, conversation_records, detection_result, execution_mode,
                        use_few_shots, conversation_length, analyzed_at, agent_version, error, account_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data["detection_id"], data["conversation_records"], data["detection_result"],
                    data["execution_mode"], data["use_few_shots"], data["conversation_length"],
                    data["analyzed_at"], data["agent_version"], data["error"], data["account_id"]
                ))

            conn.commit()
            logger.debug(f"Privacy detection result {detection_id} saved successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to save privacy detection result {result_data.get('detection_id')}: {e}")
            return False

    def load_privacy_detection_result(self, detection_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a privacy detection result from the database.

        Args:
            detection_id: Detection ID

        Returns:
            Privacy detection result data or None if not found
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM privacy_detection_results WHERE detection_id = ?", (detection_id,))
            row = cursor.fetchone()

            if not row:
                return None

            return {
                "id": row[0],
                "detection_id": row[1],
                "conversation_records": self._json_loads(row[2]),
                "detection_result": self._json_loads(row[3]),
                "execution_mode": row[4],
                "use_few_shots": bool(row[5]),
                "conversation_length": row[6],
                "analyzed_at": row[7],
                "agent_version": row[8],
                "error": row[9],
                "account_id": row[10],
                "created_at": row[11],
                "updated_at": row[12]
            }

        except Exception as e:
            logger.error(f"Failed to load privacy detection result {detection_id}: {e}")
            return None

    def load_all_privacy_detection_results(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load all privacy detection results from the database.

        Args:
            limit: Maximum number of results to return
            offset: Number of results to skip

        Returns:
            List of privacy detection result data dictionaries
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            query = "SELECT * FROM privacy_detection_results ORDER BY created_at DESC"
            params = []

            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)

            if offset is not None:
                query += " OFFSET ?"
                params.append(offset)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            results = []
            for row in rows:
                results.append({
                    "id": row[0],
                    "detection_id": row[1],
                    "conversation_records": self._json_loads(row[2]),
                    "detection_result": self._json_loads(row[3]),
                    "execution_mode": row[4],
                    "use_few_shots": bool(row[5]),
                    "conversation_length": row[6],
                    "analyzed_at": row[7],
                    "agent_version": row[8],
                    "error": row[9],
                    "account_id": row[10],
                    "created_at": row[11],
                    "updated_at": row[12]
                })

            return results

        except Exception as e:
            logger.error(f"Failed to load all privacy detection results: {e}")
            return []

    def delete_privacy_detection_result(self, detection_id: str) -> bool:
        """
        Delete a privacy detection result from the database.

        Args:
            detection_id: Detection ID

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("DELETE FROM privacy_detection_results WHERE detection_id = ?", (detection_id,))
            conn.commit()

            logger.debug(f"Privacy detection result {detection_id} deleted successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to delete privacy detection result {detection_id}: {e}")
            return False

    def clear_all_privacy_detection_results(self) -> bool:
        """Clear all privacy detection results from the database."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("DELETE FROM privacy_detection_results")
            conn.commit()

            logger.info("All privacy detection results cleared from database")
            return True

        except Exception as e:
            logger.error(f"Failed to clear all privacy detection results: {e}")
            return False

    def get_privacy_detection_stats(self, account_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive statistics about privacy detection results.

        Args:
            account_id: Optional account ID to filter results by

        Returns:
            Dictionary with comprehensive statistics
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Base query with optional account filter
            account_filter = "WHERE account_id = ?" if account_id else ""
            account_params = [account_id] if account_id else []

            # Total count
            cursor.execute(f"SELECT COUNT(*) FROM privacy_detection_results {account_filter}", account_params)
            total_count = cursor.fetchone()[0]

            # Count by severity (only for detected leaks)
            cursor.execute(f"""
                SELECT
                    json_extract(detection_result, '$.overall_severity') as severity,
                    COUNT(*) as count
                FROM privacy_detection_results
                {account_filter}
                AND json_extract(detection_result, '$.privacy_detected') = 1
                GROUP BY severity
            """, account_params)
            severity_counts = dict(cursor.fetchall())

            # Count by execution mode
            cursor.execute(f"""
                SELECT execution_mode, COUNT(*) as count
                FROM privacy_detection_results
                {account_filter}
                GROUP BY execution_mode
            """, account_params)
            execution_mode_counts = dict(cursor.fetchall())

            # Count by privacy type (aggregate all leaks)
            cursor.execute(f"""
                SELECT
                    json_extract(value, '$.privacy_type') as privacy_type,
                    COUNT(*) as count
                FROM privacy_detection_results,
                json_each(json_extract(detection_result, '$.privacy_leaks'))
                {account_filter}
                GROUP BY privacy_type
            """, account_params)
            privacy_type_counts = dict(cursor.fetchall())

            # Count by confidence level
            cursor.execute(f"""
                SELECT
                    json_extract(value, '$.confidence') as confidence,
                    COUNT(*) as count
                FROM privacy_detection_results,
                json_each(json_extract(detection_result, '$.privacy_leaks'))
                {account_filter}
                GROUP BY confidence
            """, account_params)
            confidence_counts = dict(cursor.fetchall())

            # Account distribution (only if no specific account filter)
            account_distribution = {}
            if not account_id:
                cursor.execute("""
                    SELECT account_id, COUNT(*) as count
                    FROM privacy_detection_results
                    WHERE account_id IS NOT NULL
                    GROUP BY account_id
                    ORDER BY count DESC
                    LIMIT 20
                """)
                account_distribution = dict(cursor.fetchall())

            # Error rate
            cursor.execute(f"""
                SELECT COUNT(*) FROM privacy_detection_results
                {account_filter}
                AND error IS NOT NULL AND error != ''
            """, account_params)
            error_count = cursor.fetchone()[0]

            # Average conversation length
            cursor.execute(f"""
                SELECT AVG(conversation_length) FROM privacy_detection_results
                {account_filter}
                WHERE conversation_length IS NOT NULL
            """, account_params)
            avg_conversation_length = cursor.fetchone()[0] or 0

            # Detection rate (percentage of conversations with leaks)
            cursor.execute(f"""
                SELECT COUNT(*) FROM privacy_detection_results
                {account_filter}
                AND json_extract(detection_result, '$.privacy_detected') = 1
            """, account_params)
            detected_count = cursor.fetchone()[0]

            detection_rate = (detected_count / total_count * 100) if total_count > 0 else 0

            # Recent activity (last 24 hours)
            cursor.execute(f"""
                SELECT COUNT(*) FROM privacy_detection_results
                {account_filter}
                AND created_at >= datetime('now', '-1 day')
            """, account_params)
            recent_count = cursor.fetchone()[0]

            # Time-based statistics (last 7 days)
            cursor.execute(f"""
                SELECT
                    DATE(created_at) as date,
                    COUNT(*) as count,
                    SUM(CASE WHEN json_extract(detection_result, '$.privacy_detected') = 1 THEN 1 ELSE 0 END) as detected_count
                FROM privacy_detection_results
                {account_filter}
                AND created_at >= datetime('now', '-7 days')
                GROUP BY DATE(created_at)
                ORDER BY date
            """, account_params)
            daily_stats = cursor.fetchall()

            # Most common leak types
            cursor.execute(f"""
                SELECT
                    json_extract(value, '$.privacy_type') as privacy_type,
                    COUNT(*) as count
                FROM privacy_detection_results,
                json_each(json_extract(detection_result, '$.privacy_leaks'))
                {account_filter}
                GROUP BY privacy_type
                ORDER BY count DESC
                LIMIT 10
            """, account_params)
            top_leak_types = cursor.fetchall()

            return {
                "total_detections": total_count,
                "detection_rate_percent": round(detection_rate, 2),
                "error_rate_percent": round((error_count / total_count * 100) if total_count > 0 else 0, 2),
                "severity_distribution": severity_counts,
                "privacy_type_distribution": privacy_type_counts,
                "confidence_distribution": confidence_counts,
                "execution_mode_distribution": execution_mode_counts,
                "account_distribution": account_distribution,
                "average_conversation_length": round(avg_conversation_length, 2),
                "recent_detections_24h": recent_count,
                "daily_stats_last_7_days": [
                    {
                        "date": row[0],
                        "total_detections": row[1],
                        "privacy_detections": row[2]
                    } for row in daily_stats
                ],
                "top_leak_types": [
                    {"privacy_type": row[0], "count": row[1]} for row in top_leak_types
                ],
                "metadata": {
                    "account_filter": account_id,
                    "generated_at": self._get_timestamp()
                }
            }

        except Exception as e:
            logger.error(f"Failed to get privacy detection stats: {e}")
            return {}

    def close(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.debug("Database connection closed")
