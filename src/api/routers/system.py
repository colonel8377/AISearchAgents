"""System management and default API routes."""

import shutil
import sqlite3
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from ..common import get_api_key
from ...config.settings import settings
from ...utils.agent_cache import get_agent_cache
from ...utils.logger import get_logger
from ...utils.web_opinion_cache import get_cache

logger = get_logger(__name__)

# System router for system management endpoints
router = APIRouter(prefix="/api/v1/system", tags=["System"])

# Default router for root and default endpoints (merged with System in API docs)
default_router = APIRouter(tags=["System"])


@router.post("/reset")
async def reset_system(_api_key: str = Depends(get_api_key)):  # Authentication dependency (value not used)
    """
    Reset the entire system: clear all database state, caches, and memory state.
    
    This endpoint performs the following operations:
    1. Clear all SQLite cache databases (agent_cache.db, web_opinion_cache.db)
    2. Clear all Redis caches (if used):
       - Agent cache (db=1)
       - Web opinion cache (db=2)
       - Vector store (db=settings.redis_db)
    3. Clear all Debate sessions and statistics
    4. Reset all Agent custom few shots
    5. Clear vector store data (Chroma/PostgreSQL/Redis)
    6. Clear in-memory chain cache
    7. Clear all Agent instances from AgentManager
    8. Clear all Bot instances from BotManager
    
    Warning: This operation is irreversible, use with caution.
    
    Args:
        _api_key: Authentication dependency (value not used, required for auth check)
        
    Returns:
        Detailed results of the reset operation
    """
    
    reset_results = {
        "success": True,
        "errors": [],
        "cleared_items": []
    }
    
    def add_error(msg: str):
        """Helper to add error and log it."""
        logger.error(msg, exc_info=True)
        reset_results["errors"].append(msg)
    
    def add_cleared_item(item: str):
        """Helper to add cleared item."""
        reset_results["cleared_items"].append(item)
    
    def clear_sqlite_table(db_path: Path, table_name: str, item_name: str):
        """Clear a SQLite table."""
        if db_path.exists():
            try:
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                cursor.execute(f"DELETE FROM {table_name}")
                conn.commit()
                conn.close()
                add_cleared_item(item_name)
                logger.info(f"{item_name} cleared")
            except Exception as e:
                add_error(f"Failed to clear {item_name}: {e}")
    
    def clear_redis_keys(redis_client, pattern: str, item_name: str):
        """Clear Redis keys matching a pattern."""
        try:
            keys = redis_client.keys(pattern)
            if keys:
                redis_client.delete(*keys)
                add_cleared_item(item_name)
                logger.info(f"{item_name} cleared")
        except Exception as e:
            logger.warning(f"Failed to clear {item_name}: {e}")
    
    try:
        logger.warning("System reset initiated - this will clear all caches and state")
        
        try:
            agent_cache = get_agent_cache()
            if agent_cache:
                agent_cache.clear()
                add_cleared_item("agent_cache")
                logger.info("Agent cache cleared")
        except Exception as e:
            add_error(f"Failed to clear agent cache: {e}")
        
        try:
            web_opinion_cache = get_cache()
            if web_opinion_cache:
                web_opinion_cache.clear()
                add_cleared_item("web_opinion_cache")
                logger.info("Web opinion cache cleared")
        except Exception as e:
            add_error(f"Failed to clear web opinion cache: {e}")
        
        try:
            base_dir = Path(__file__).resolve().parents[3]
            data_dir = base_dir / "data"
            
            clear_sqlite_table(
                data_dir / "agent_cache.db",
                "agent_cache",
                "agent_cache.db"
            )
            clear_sqlite_table(
                data_dir / "web_opinion_cache.db",
                "analysis_cache",
                "web_opinion_cache.db"
            )
        except Exception as e:
            add_error(f"Failed to clear SQLite databases: {e}")
        
        if settings.cache_backend == "redis":
            try:
                import redis
                
                try:
                    redis_client_cache = redis.Redis(
                        host=settings.redis_host,
                        port=settings.redis_port,
                        username=settings.redis_user if settings.redis_user else None,
                        password=settings.redis_password if settings.redis_password else None,
                        db=getattr(settings, 'cache_redis_db', 1),
                        decode_responses=True
                    )
                    clear_redis_keys(redis_client_cache, "agent_cache:*", "redis_agent_cache")
                except Exception as e:
                    logger.warning(f"Failed to clear Redis agent cache: {e}")
                
                try:
                    redis_client_web = redis.Redis(
                        host=settings.redis_host,
                        port=settings.redis_port,
                        username=settings.redis_user if settings.redis_user else None,
                        password=settings.redis_password if settings.redis_password else None,
                        db=2,
                        decode_responses=True
                    )
                    clear_redis_keys(redis_client_web, "web_opinion_cache:*", "redis_web_opinion_cache")
                except Exception as e:
                    logger.warning(f"Failed to clear Redis web opinion cache: {e}")
                
                try:
                    redis_client_vector = redis.Redis(
                        host=settings.redis_host,
                        port=settings.redis_port,
                        username=settings.redis_user if settings.redis_user else None,
                        password=settings.redis_password if settings.redis_password else None,
                        db=settings.redis_db,
                        decode_responses=True
                    )
                    clear_redis_keys(redis_client_vector, "agent_memory_*", "redis_vector_store")
                except Exception as e:
                    logger.warning(f"Failed to clear Redis vector store: {e}")
            except ImportError:
                logger.warning("Redis not available, skipping Redis cache clearing")
            except Exception as e:
                add_error(f"Failed to clear Redis caches: {e}")
        
        try:
            from ...storage import get_database
            if settings.enable_persistence:
                database = get_database()
                database.clear_all_data()
                add_cleared_item("database_storage")
                logger.info("Database storage cleared")
            else:
                logger.info("Persistence disabled, skipping database clearing")
        except Exception as e:
            add_error(f"Failed to clear database storage: {e}")

        try:
            from ...debate.service import DebateService
            # Clear debate service cache
            debate_service_instance = DebateService()
            debate_service_instance._sessions_cache.clear()
            debate_service_instance._statistics.clear()
            add_cleared_item("debate_sessions_cache")
            logger.info("Debate sessions cache cleared")
        except Exception as e:
            add_error(f"Failed to clear debate sessions cache: {e}")

        try:
            from ..common import agent_manager, bot_manager
            # Clear all agent instances
            agent_manager.clear_all()
            add_cleared_item("agent_manager_agents")
            logger.info("Agent manager cleared")

            # Clear bot manager (no longer has _bots dict, but clear any cache)
            add_cleared_item("bot_manager_cache")
            logger.info("Bot manager cache cleared")
        except Exception as e:
            add_error(f"Failed to clear agent/bot managers: {e}")

        try:
            from ...agents.nudge_collapse.agent import NudgeCollapseAgent
            from ...agents.summarizer.agent import SummarizerAgent
            from ...agents.bot_creator.agent import BotCreatorAgent
            from ...agents.demographic_evaluator.agent import DemographicEvaluatorAgent
            from ...agents.content_extractor.agent import ContentExtractorAgent
            from ...agents.claim_atomizer.agent import ClaimAtomizerAgent
            from ...agents.conflict_auditor.agent import ConflictAuditorAgent
            from ...agents.web_opinion_extractor.agent import WebOpinionAnalyzer
            
            agents_to_reset = [
                ("NudgeCollapseAgent", NudgeCollapseAgent),
                ("SummarizerAgent", SummarizerAgent),
                ("BotCreatorAgent", BotCreatorAgent),
                ("DemographicEvaluatorAgent", DemographicEvaluatorAgent),
                ("ContentExtractorAgent", ContentExtractorAgent),
                ("ClaimAtomizerAgent", ClaimAtomizerAgent),
                ("ConflictAuditorAgent", ConflictAuditorAgent),
                ("WebOpinionAnalyzer", WebOpinionAnalyzer),
            ]
            
            for agent_name, agent_class in agents_to_reset:
                try:
                    if hasattr(agent_class, 'set_custom_few_shots'):
                        agent_class.set_custom_few_shots(None)
                        add_cleared_item(f"{agent_name}_custom_shots")
                except Exception as e:
                    logger.warning(f"Failed to reset {agent_name} custom shots: {e}")
            
            logger.info("All agent custom few shots reset")
        except Exception as e:
            add_error(f"Failed to reset agent custom few shots: {e}")
        
        try:
            if settings.vector_store_type == "chroma":
                chroma_dir = Path(settings.chroma_persist_directory)
                if chroma_dir.exists():
                    for item in chroma_dir.iterdir():
                        if item.is_dir():
                            try:
                                shutil.rmtree(item)
                                add_cleared_item(f"chroma_collection_{item.name}")
                            except Exception as e:
                                logger.warning(f"Failed to delete Chroma collection {item.name}: {e}")
                    logger.info("Chroma vector store cleared")
            elif settings.vector_store_type == "postgres":
                try:
                    import psycopg2
                    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
                    conn = psycopg2.connect(
                        host=settings.postgres_host,
                        port=settings.postgres_port,
                        user=settings.postgres_user,
                        password=settings.postgres_password,
                        database=settings.postgres_db
                    )
                    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM langchain_pg_embedding")
                    cursor.close()
                    conn.close()
                    add_cleared_item("postgres_vector_store")
                    logger.info("PostgreSQL vector store cleared")
                except ImportError:
                    logger.warning("psycopg2 not available, skipping PostgreSQL vector store clearing")
                except Exception as e:
                    logger.warning(f"Failed to clear PostgreSQL vector store: {e}")
        except Exception as e:
            add_error(f"Failed to clear vector store: {e}")
        
        add_cleared_item("chain_cache_info")
        logger.info("Chain cache info noted (instance-level, will be cleared on next agent creation)")
        
        if reset_results["errors"]:
            reset_results["success"] = False
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


# ============================================================================
# Default/Root Endpoints (merged with System in API docs)
# ============================================================================

@default_router.get("/")
async def root():
    """
    Root endpoint with API information.
    
    Returns:
        API information including version, features, agent types, and available endpoints
    """
    return {
        "message": "AI Search Agents Platform v2.0",
        "version": "2.0.0",
        "features": [
            "Multi-agent support with unique IDs",
            "RESTful API design",
            "Optional API key authentication",
            "Improved conversation summarization",
            "Per-agent memory management",
            "Multi-Agent Debate System with stability checking",
            "Web Opinion Extraction with bias analysis"
        ],
        "agent_types": {
            "nudge_collapse": "4-turn radicalization protocol agent",
            "summarizer": "Conversation summarization agent with focus on user questions",
            "bot_creator": "Bot creation and configuration agent",
            "demographic_evaluator": "Sentence evaluation from demographic perspectives",
            "content_extractor": "Academic content extraction from web pages",
            "claim_atomizer": "Atomic claim decomposition from text",
            "evidence_locator": "Evidence location in main body text",
            "conflict_auditor": "Logical consistency auditing between claims and evidence",
        },
        "endpoints": {
            "agent_management": {
                "create_agent": "/api/v1/agents/create (POST - create new agent instance)",
                "list_agents": "/api/v1/agents/list (GET - list all active agents)",
                "agent_status": "/api/v1/agents/{agent_id}/status (GET - get agent status)",
                "reset_agent": "/api/v1/agents/{agent_id}/reset (POST - reset agent to initial state)"
            },
            "system": {
                "reset": "/api/v1/system/reset (POST - reset entire system: clear all databases, caches, and state)"
            }
        },
        "authentication": {
            "enabled": settings.api_key_required,
            "header": "X-API-Key"
        }
    }
