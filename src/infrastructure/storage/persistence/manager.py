"""
Storage manager for unified persistence-centric persistence.

This module provides a centralized manager for persistence operations, using the
abstract base class pattern to support different persistence backends.

NOTE: Storage layer provides generic database operations.
Business-specific operations should use Repository layer.
"""

from typing import Optional, Dict, Any

from src.infrastructure.storage.persistence.database_storage import DatabaseStorage
from .database.sqlite_storage import SQLiteStorage
from src.shared.config.settings import settings
from src.shared.utils.logger import get_logger

logger = get_logger(__name__)


class StorageManager:
    """
    Singleton manager for persistence storage operations.

    Provides centralized access to persistence operations with lazy initialization.
    Uses the abstract base class pattern to support different persistence backends
    (SQLite, PostgreSQL, etc.) while maintaining a consistent interface.

    NOTE: This manager provides generic database storage (infrastructure layer).
    Business-specific operations should use Repository layer.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        connection: Optional[Any] = None,
        database_class: Optional[type] = None
    ):
        """
        Initialize storage manager.

        Args:
            db_path: Optional database path. If None, uses settings.storage_db_path or default.
            connection: Optional database connection object. If provided, db_path is ignored.
            database_class: Optional database implementation class. If None, uses SQLiteStorage.
                          Must be a subclass of DatabaseStorage.
        """
        # Use provided database class or default to SQLiteStorage
        if database_class is None:
            database_class = SQLiteStorage

        # Ensure database_class is a subclass of DatabaseStorage
        if not issubclass(database_class, DatabaseStorage):
            raise TypeError(
                f"database_class must be a subclass of DatabaseStorage, "
                f"got {database_class}"
            )

        if connection is None:
            if db_path is None:
                db_path = getattr(settings, 'storage_db_path', None)

            if db_path is None:
                # Default path: use project root data directory
                from src.shared.config.settings import BASE_DIR
                data_dir = BASE_DIR / "data"
                data_dir.mkdir(exist_ok=True)
                db_path = str(data_dir / "app_storage.db")

            # Create default SQLite connection
            import sqlite3
            connection = sqlite3.connect(db_path)
            connection.execute("PRAGMA foreign_keys = ON")
            logger.info(f"Created SQLite connection to {db_path}")

        self._database: DatabaseStorage = database_class(connection)
        logger.info(
            f"StorageManager initialized with {database_class.__name__}"
        )

    def get_database(self) -> DatabaseStorage:
        """
        Get the database storage instance.

        Returns:
            DatabaseStorage: Database instance implementing the storage interface.
                           Returns SQLiteStorage instance by default.
        """
        return self._database

    def reset_all(self) -> Dict[str, Any]:
        """
        Reset all application storage.
        
        NOTE: This method is deprecated. Use Repository layer for business-specific operations.
        This method only provides basic database clearing operations.
        
        Returns:
            Dict with:
            - success: bool - Whether the reset completed successfully
            - errors: List[str] - List of error messages (if any)
        """
        result = {
            "success": True,
            "errors": []
        }
        
        def add_error(msg: str):
            logger.error(msg, exc_info=True)
            result["errors"].append(msg)
            result["success"] = False
        
        try:
            if settings.enable_persistence:
                # NOTE: DatabaseStorage interface doesn't have reset() method
                # This should be handled by Repository layer
                logger.warning("reset_all() is deprecated. Use Repository layer instead.")
                # For now, just close and recreate connection
                self._database.close()
                # Reinitialize
                import sqlite3
                from src.shared.config.settings import BASE_DIR
                data_dir = BASE_DIR / "data"
                db_path = str(data_dir / "app_storage.db")
                connection = sqlite3.connect(db_path)
                connection.execute("PRAGMA foreign_keys = ON")
                self._database = SQLiteStorage(connection)
                logger.info("Database storage reset (reinitialized)")
            else:
                logger.info("Persistence disabled, skipping database clearing")
        except Exception as e:
            add_error(f"Failed to reset database storage: {e}")
        
        return result

    def close(self) -> None:
        """Close persistence persistence."""
        if self._database:
            self._database.close()
            logger.debug("StorageManager persistence persistence closed")
