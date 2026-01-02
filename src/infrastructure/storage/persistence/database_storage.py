"""
Storage interfaces - Infrastructure layer.

This module defines storage interfaces for infrastructure layer.
Storage provides generic storage operations, not business-specific methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class DatabaseStorage(ABC):
    """
    Generic database storage interface.
    
    Provides generic database operations without business semantics.
    This is infrastructure layer - Repository layer will use this to implement
    business-specific operations.
    """

    @abstractmethod
    def get_connection(self) -> Any:
        """Get database connection."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close database connection."""
        pass

    @abstractmethod
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        Execute a SELECT query and return results.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List of result dictionaries
        """
        pass

    @abstractmethod
    def execute_update(self, query: str, params: Optional[tuple] = None) -> int:
        """
        Execute an INSERT, UPDATE, or DELETE query.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Number of affected rows
        """
        pass

    @abstractmethod
    def begin_transaction(self) -> None:
        """Begin a database transaction."""
        pass

    @abstractmethod
    def commit_transaction(self) -> None:
        """Commit the current database transaction."""
        pass

    @abstractmethod
    def rollback_transaction(self) -> None:
        """Rollback the current database transaction."""
        pass

    @abstractmethod
    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists.
        
        Args:
            table_name: Name of the table to check
            
        Returns:
            True if table exists, False otherwise
        """
        pass

    @abstractmethod
    def create_table(self, table_name: str, schema: str) -> bool:
        """
        Create a table with the given schema.
        
        Args:
            table_name: Name of the table to create
            schema: SQL schema definition
            
        Returns:
            True if created successfully, False otherwise
        """
        pass

