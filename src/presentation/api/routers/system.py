"""System management and default API routes - MVC View layer (only HTTP request/response handling)."""

from fastapi import APIRouter, Depends

from src.shared.config.settings import settings
from src.shared.utils import get_logger
from ..common import get_api_key, system_service

logger = get_logger(__name__)

# System router for system management endpoints
router = APIRouter(prefix="/api/v1/system", tags=["System"])

# Default router for root and default endpoints (merged with System in API docs)
default_router = APIRouter(tags=["System"])


@router.post("/reset")
async def reset_system(_api_key: str = Depends(get_api_key)):  # Authentication dependency (value not used)
    """
    Reset the entire system: clear all persistence state, caches, and memory state.
    
    This endpoint orchestrates the reset by calling appropriate modules:
    1. Clear all caches (agent cache, web opinion cache, Redis caches) - via agent_cache module
    2. Clear application storage persistence (app_storage.db) - via storage manager
    3. Clear vector store (Chroma/PostgreSQL) - local helper
    4. Clear in-memory state (managers, debate sessions) - via memory module
    5. Reset all Agent custom few shots - via few_shots module
    
    Warning: This operation is irreversible, use with caution.
    
    Args:
        _api_key: Authentication dependency (value not used, required for auth check)
        
    Returns:
        Detailed results of the reset operation with:
        - success: bool - Whether the reset completed successfully
        - errors: List[str] - List of error messages (if any)
        - cleared_items: List[str] - List of cache items that were cleared
    """
    return system_service.reset_system()


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
