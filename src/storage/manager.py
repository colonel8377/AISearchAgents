"""
Storage manager for unified database-centric persistence.
"""

from typing import Optional
from .database import StorageDatabase
from ..config.settings import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class StorageManager:
    """
    Singleton manager for database storage operations.

    Provides centralized access to database operations with lazy initialization.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize storage manager.

        Args:
            db_path: Optional database path. If None, uses settings.storage_db_path or default.
        """
        if db_path is None:
            db_path = getattr(settings, 'storage_db_path', None)

        if db_path is None:
            # Default path: use project root data directory
            from ..config.settings import BASE_DIR
            data_dir = BASE_DIR / "data"
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "app_storage.db")

        self._database = StorageDatabase(db_path)
        logger.info(f"StorageManager initialized with database at {db_path}")

    def get_database(self) -> StorageDatabase:
        """
        Get the database instance.

        Returns:
            StorageDatabase instance
        """
        return self._database

    def close(self) -> None:
        """Close database connection."""
        if self._database:
            self._database.close()
            logger.debug("StorageManager database connection closed")
