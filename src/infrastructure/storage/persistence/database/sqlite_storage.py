"""
Generic database storage implementation.

This module provides generic database storage operations without business semantics.
Repository layer will use this to implement business-specific operations.
"""
import sqlite3
import json
from typing import Dict, List, Optional, Any
from src.infrastructure.storage.persistence.database_storage import DatabaseStorage
from src.shared.utils.logger import get_logger

logger = get_logger(__name__)


class SQLiteStorage(DatabaseStorage):
    """
    SQLite database storage implementation.
    
    Provides generic database operations. Business-specific operations
    should be implemented in Repository layer.
    """

    def __init__(self, connection: Any):
        """
        Initialize database storage.
        
        Args:
            connection: SQLite database connection
        """
        self._connection = connection
        logger.debug("SQLiteStorage initialized")

    def get_connection(self) -> Any:
        """Get database connection."""
        return self._connection

    def close(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close()
            logger.debug("Database connection closed")

    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        Execute a SELECT query and return results.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List of result dictionaries
        """
        try:
            cursor = self._connection.cursor()
            cursor.execute(query, params or ())
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            cursor.close()
            return results
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

    def execute_update(self, query: str, params: Optional[tuple] = None) -> int:
        """
        Execute an INSERT, UPDATE, or DELETE query.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Number of affected rows
        """
        try:
            cursor = self._connection.cursor()
            cursor.execute(query, params or ())
            self._connection.commit()
            affected_rows = cursor.rowcount
            cursor.close()
            return affected_rows
        except Exception as e:
            logger.error(f"Update execution failed: {e}")
            raise

    def begin_transaction(self) -> None:
        """Begin a database transaction."""
        self._connection.execute("BEGIN")

    def commit_transaction(self) -> None:
        """Commit the current database transaction."""
        self._connection.commit()

    def rollback_transaction(self) -> None:
        """Rollback the current database transaction."""
        self._connection.rollback()

    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists.
        
        Args:
            table_name: Name of the table to check
            
        Returns:
            True if table exists, False otherwise
        """
        result = self.execute_query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        return len(result) > 0

    def create_table(self, table_name: str, schema: str) -> bool:
        """
        Create a table with the given schema.
        
        Args:
            table_name: Name of the table to create
            schema: SQL schema definition
            
        Returns:
            True if created successfully, False otherwise
        """
        try:
            self.execute_update(f"CREATE TABLE IF NOT EXISTS {table_name} ({schema})")
            return True
        except Exception as e:
            logger.error(f"Failed to create table {table_name}: {e}")
            return False

    def _json_dumps(self, data: Any) -> str:
        """Convert data to JSON string."""
        return json.dumps(data, ensure_ascii=False)

    def _json_loads(self, data: str) -> Any:
        """Parse JSON string to data."""
        if data is None or data == "":
            return None
        return json.loads(data)

    def save_custom_few_shots(self, agent_type: str, few_shots: Optional[str]) -> bool:
        """
        Save custom few-shot examples for an agent type.
        
        Args:
            agent_type: Type of agent (e.g., 'bot_creator', 'summarizer')
            few_shots: Custom few-shot examples as JSON string, or None to clear
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Ensure custom_few_shots table exists
            if not self.table_exists("custom_few_shots"):
                self.create_table("custom_few_shots", """
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_type TEXT NOT NULL UNIQUE,
                    few_shots_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                """)
            
            if few_shots is None:
                # Delete existing few shots
                self.execute_update(
                    "DELETE FROM custom_few_shots WHERE agent_type = ?",
                    (agent_type,)
                )
            else:
                # Insert or update
                existing = self.execute_query(
                    "SELECT id FROM custom_few_shots WHERE agent_type = ?",
                    (agent_type,)
                )
                
                if existing:
                    # Update existing
                    self.execute_update(
                        """UPDATE custom_few_shots 
                           SET few_shots_json = ?, updated_at = CURRENT_TIMESTAMP 
                           WHERE agent_type = ?""",
                        (few_shots, agent_type)
                    )
                else:
                    # Insert new
                    self.execute_update(
                        """INSERT INTO custom_few_shots (agent_type, few_shots_json) 
                           VALUES (?, ?)""",
                        (agent_type, few_shots)
                    )
            
            logger.debug(f"Saved custom few shots for agent type: {agent_type}")
            return True
        except Exception as e:
            logger.error(f"Failed to save custom few shots for {agent_type}: {e}")
            return False

    def load_custom_few_shots(self, agent_type: str) -> Optional[str]:
        """
        Load custom few-shot examples for an agent type.
        
        Args:
            agent_type: Type of agent (e.g., 'bot_creator', 'summarizer')
            
        Returns:
            Custom few-shot examples as JSON string, or None if not found
        """
        try:
            if not self.table_exists("custom_few_shots"):
                return None
            
            results = self.execute_query(
                "SELECT few_shots_json FROM custom_few_shots WHERE agent_type = ?",
                (agent_type,)
            )
            
            if results:
                few_shots = results[0].get("few_shots_json")
                logger.debug(f"Loaded custom few shots for agent type: {agent_type}")
                return few_shots
            else:
                logger.debug(f"No custom few shots found for agent type: {agent_type}")
                return None
        except Exception as e:
            logger.error(f"Failed to load custom few shots for {agent_type}: {e}")
            return None

    def save_privacy_detection_result(self, result_data: Dict[str, Any]) -> bool:
        """
        Save privacy detection result to database.
        
        Args:
            result_data: Dictionary containing detection result data
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Ensure privacy_detection_results table exists
            if not self.table_exists("privacy_detection_results"):
                self.create_table("privacy_detection_results", """
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
                    account_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                """)
            
            detection_id = result_data.get("detection_id")
            if not detection_id:
                logger.error("detection_id is required")
                return False
            
            # Check if detection exists
            existing = self.execute_query(
                "SELECT detection_id FROM privacy_detection_results WHERE detection_id = ?",
                (detection_id,)
            )
            
            # Prepare data
            data = (
                self._json_dumps(result_data.get("conversation_records", [])),
                self._json_dumps(result_data.get("detection_result", {})),
                result_data.get("execution_mode"),
                1 if result_data.get("use_few_shots", True) else 0,
                result_data.get("conversation_length"),
                result_data.get("analyzed_at"),
                result_data.get("agent_version"),
                result_data.get("error"),
                result_data.get("account_id")
            )
            
            if existing:
                # Update existing
                self.execute_update(
                    """UPDATE privacy_detection_results SET
                        conversation_records = ?, detection_result = ?,
                        execution_mode = ?, use_few_shots = ?,
                        conversation_length = ?, analyzed_at = ?,
                        agent_version = ?, error = ?, account_id = ?,
                        updated_at = CURRENT_TIMESTAMP
                        WHERE detection_id = ?""",
                    data + (detection_id,)
                )
            else:
                # Insert new
                self.execute_update(
                    """INSERT INTO privacy_detection_results (
                        detection_id, conversation_records, detection_result,
                        execution_mode, use_few_shots, conversation_length,
                        analyzed_at, agent_version, error, account_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (detection_id,) + data
                )
            
            logger.debug(f"Saved privacy detection result: {detection_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save privacy detection result: {e}")
            return False

    def load_privacy_detection_result(self, detection_id: str) -> Optional[Dict[str, Any]]:
        """
        Load privacy detection result by ID.
        
        Args:
            detection_id: Detection ID
            
        Returns:
            Detection result dictionary or None if not found
        """
        try:
            if not self.table_exists("privacy_detection_results"):
                return None
            
            results = self.execute_query(
                "SELECT * FROM privacy_detection_results WHERE detection_id = ?",
                (detection_id,)
            )
            
            if not results:
                return None
            
            row = results[0]
            return {
                "detection_id": row["detection_id"],
                "conversation_records": self._json_loads(row["conversation_records"]),
                "detection_result": self._json_loads(row["detection_result"]),
                "execution_mode": row["execution_mode"],
                "use_few_shots": bool(row["use_few_shots"]),
                "conversation_length": row["conversation_length"],
                "analyzed_at": row["analyzed_at"],
                "agent_version": row["agent_version"],
                "error": row["error"],
                "account_id": row["account_id"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"]
            }
        except Exception as e:
            logger.error(f"Failed to load privacy detection result {detection_id}: {e}")
            return None

    def load_all_privacy_detection_results(
        self,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Load all privacy detection results with pagination.
        
        Args:
            limit: Maximum number of results to return
            offset: Number of results to skip
            
        Returns:
            List of detection result dictionaries
        """
        try:
            if not self.table_exists("privacy_detection_results"):
                return []
            
            results = self.execute_query(
                """SELECT detection_id, account_id, analyzed_at, created_at
                   FROM privacy_detection_results
                   ORDER BY created_at DESC
                   LIMIT ? OFFSET ?""",
                (limit, offset)
            )
            
            return results
        except Exception as e:
            logger.error(f"Failed to load privacy detection results: {e}")
            return []

    def get_privacy_detection_stats(self, account_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get privacy detection statistics.
        
        Args:
            account_id: Optional account ID to filter by
            
        Returns:
            Statistics dictionary
        """
        try:
            if not self.table_exists("privacy_detection_results"):
                return {}
            
            if account_id:
                # Filter by account_id
                total = self.execute_query(
                    "SELECT COUNT(*) as count FROM privacy_detection_results WHERE account_id = ?",
                    (account_id,)
                )
                with_leaks = self.execute_query(
                    """SELECT COUNT(*) as count FROM privacy_detection_results
                       WHERE account_id = ? AND detection_result LIKE '%"privacy_detected": true%'""",
                    (account_id,)
                )
            else:
                # All results
                total = self.execute_query(
                    "SELECT COUNT(*) as count FROM privacy_detection_results"
                )
                with_leaks = self.execute_query(
                    """SELECT COUNT(*) as count FROM privacy_detection_results
                       WHERE detection_result LIKE '%"privacy_detected": true%'"""
                )
            
            total_count = total[0]["count"] if total else 0
            leaks_count = with_leaks[0]["count"] if with_leaks else 0
            
            return {
                "total_detections": total_count,
                "detections_with_leaks": leaks_count,
                "detections_without_leaks": total_count - leaks_count,
                "account_id": account_id
            }
        except Exception as e:
            logger.error(f"Failed to get privacy detection stats: {e}")
            return {}

    def delete_privacy_detection_result(self, detection_id: str) -> bool:
        """
        Delete privacy detection result by ID.
        
        Args:
            detection_id: Detection ID
            
        Returns:
            True if deleted, False otherwise
        """
        try:
            if not self.table_exists("privacy_detection_results"):
                return False
            
            affected = self.execute_update(
                "DELETE FROM privacy_detection_results WHERE detection_id = ?",
                (detection_id,)
            )
            
            return affected > 0
        except Exception as e:
            logger.error(f"Failed to delete privacy detection result {detection_id}: {e}")
            return False

