"""
Unified Storage Module - Database-Centric Persistence
"""

import os
from typing import Optional
from .manager import StorageManager
from .database import StorageDatabase

__all__ = [
    "StorageManager",
    "StorageDatabase",
    "get_storage",
    "get_database"
]

# Global storage instance
_storage_instance: Optional[StorageManager] = None

def get_storage() -> StorageManager:
    """Get the global storage manager instance (singleton)"""
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = StorageManager()
    return _storage_instance

def get_database() -> StorageDatabase:
    """Get the database instance from the storage manager"""
    return get_storage().get_database()
