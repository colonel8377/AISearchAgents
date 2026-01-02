"""System Service - Handles system management business logic."""

import shutil
from pathlib import Path
from typing import Dict, Callable, Any

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from ..few_shots import reset_all_agent_few_shots
from ...infrastructure.state import clear_in_memory_state
from ...infrastructure.storage.persistence import get_storage
from ...shared.cache import reset_all_caches
from ...shared.config.settings import settings
from ...shared.utils.logger import get_logger

logger = get_logger(__name__)


class SystemService:
    """
    Service for system management operations.
    
    Single Responsibility: Handle system-level operations like reset.
    """
    
    @staticmethod
    def reset_system() -> Dict[str, Any]:
        """
        Reset the entire system: clear all persistence state, caches, and memory state.
        
        Returns:
            Dictionary with success status, errors, and cleared_items
        """
        reset_results = {
            "success": True,
            "errors": [],
            "cleared_items": []
        }
        
        def add_error(msg: str) -> None:
            """Helper to add error and log it."""
            logger.error(msg, exc_info=True)
            reset_results["errors"].append(msg)
            reset_results["success"] = False
        
        try:
            logger.warning("System reset initiated - this will clear all caches and state")
            
            # 1. Clear all caches (agent cache, web opinion cache, Redis)
            try:
                cache_result = reset_all_caches()
                reset_results["cleared_items"].extend(cache_result.get("cleared_items", []))
                reset_results["errors"].extend(cache_result.get("errors", []))
                if not cache_result.get("success", True):
                    reset_results["success"] = False
            except Exception as e:
                add_error(f"Failed to clear caches: {e}")
            
            # 2. Clear application storage persistence (app_storage.db)
            try:
                storage = get_storage()
                storage_result = storage.reset_all()
                reset_results["errors"].extend(storage_result.get("errors", []))
                if not storage_result.get("success", True):
                    reset_results["success"] = False
            except Exception as e:
                add_error(f"Failed to reset application storage persistence: {e}")

            SystemService._clear_vector_store(reset_results, add_error)
            
            # 4. Clear in-memory state
            try:
                memory_result = clear_in_memory_state()
                reset_results["errors"].extend(memory_result.get("errors", []))
                if not memory_result.get("success", True):
                    reset_results["success"] = False
            except Exception as e:
                add_error(f"Failed to clear in-memory state: {e}")
            
            # 5. Reset all agent custom few shots
            try:
                few_shots_result = reset_all_agent_few_shots()
                reset_results["errors"].extend(few_shots_result.get("errors", []))
                if not few_shots_result.get("success", True):
                    reset_results["success"] = False
            except Exception as e:
                add_error(f"Failed to reset agent custom few shots: {e}")
            
            logger.info("Chain cache info noted (instance-level, will be cleared on next agent creation)")
            
            # Finalize results
            if reset_results["errors"]:
                logger.warning(f"System reset completed with {len(reset_results['errors'])} errors")
            else:
                logger.info("System reset completed successfully")
            
            return reset_results
            
        except Exception as e:
            error_msg = f"System reset failed: {e}"
            logger.error(error_msg, exc_info=True)
            reset_results["success"] = False
            reset_results["errors"].append(error_msg)
            return reset_results
    
    @staticmethod
    def _clear_vector_store(reset_results: Dict, add_error: Callable) -> None:
        """Clear vector store data (Chroma/PostgreSQL)."""
        try:
            if settings.vector_store_type == "chroma":
                SystemService._clear_chroma_store(reset_results, add_error)
            elif settings.vector_store_type == "postgres":
                SystemService._clear_postgres_store(reset_results, add_error)
        except Exception as e:
            msg = f"Failed to clear vector store: {e}"
            add_error(msg)

    @staticmethod
    def _clear_chroma_store(reset_results: Dict, add_error: Callable) -> None:
        """Clear Chroma vector store."""
        chroma_dir = Path(settings.chroma_persist_directory)
        if chroma_dir.exists():
            # Delete all contents of the directory (files and subdirectories)
            for item in chroma_dir.iterdir():
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                    logger.debug(f"Deleted {item}")
                except Exception as e:
                    msg = f"Failed to delete {item}: {e}"
                    logger.warning(msg)
                    add_error(msg)

            reset_results["cleared_items"].append("chroma_vector_store")
            logger.info("Chroma vector store cleared")

    @staticmethod
    def _clear_postgres_store(reset_results: Dict, add_error: Callable) -> None:
        """Clear PostgreSQL vector store."""
        try:
            conn = psycopg2.connect(
                host=settings.postgres_host,
                port=settings.postgres_port,
                user=settings.postgres_user,
                password=settings.postgres_password,
                database=settings.postgres_db
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()

            # Truncate tables instead of DELETE for performance and resetting identity
            # Assuming standard LangChain/PGVector tables
            tables_to_clear = ["langchain_pg_embedding", "langchain_pg_collection"]

            for table in tables_to_clear:
                try:
                    cursor.execute(f"TRUNCATE TABLE {table} CASCADE")
                    logger.debug(f"Truncated table {table}")
                except psycopg2.errors.UndefinedTable:
                    logger.debug(f"Table {table} does not exist, skipping")
                except Exception as e:
                    logger.warning(f"Failed to truncate table {table}: {e}")
                    # Don't fail hard if table doesn't exist or other minor issue, but log it

            cursor.close()
            conn.close()
            reset_results["cleared_items"].append("postgres_vector_store")
            logger.info("PostgreSQL vector store cleared")

        except ImportError:
            msg = "psycopg2 not available, skipping PostgreSQL vector store clearing"
            logger.warning(msg)
            add_error(msg)
        except Exception as e:
            msg = f"Failed to clear PostgreSQL vector store: {e}"
            add_error(msg)
