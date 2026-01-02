"""
Unified Storage Module - Database-Centric Persistence

This module provides a unified interface for persistence storage operations,
supporting different persistence backends through an abstract base class.

NOTE: Storage layer provides generic database operations.
Business-specific operations should use Repository layer.
"""

from typing import Optional
from .manager import StorageManager

__all__ = [
    "StorageManager",
    "get_storage",
    "get_database"
]



# Global storage instance
_storage_instance: Optional[StorageManager] = None


def get_storage() -> StorageManager:
    """
    Get the global storage manager instance (singleton).

    Returns:
        StorageManager: The global storage manager instance
    """
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = StorageManager()
    return _storage_instance


def get_database():
    """
    Get the persistence instance from the storage manager.

    Returns:
        DatabaseStorage: Database instance implementing the storage interface.
    """
    return get_storage().get_database()
