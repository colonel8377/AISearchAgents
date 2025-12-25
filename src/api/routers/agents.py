"""Agent management API routes."""

from fastapi import APIRouter, Depends, HTTPException
from langchain_openai import OpenAIEmbeddings

from ..common import get_api_key, agent_manager
from ..schemas import (
    CreateAgentRequest, AgentIdResponse, ListAgentsResponse,
    AgentStatusResponse, ResetAgentRequest
)
from ...agents.bot_creator.agent import BotCreatorAgent
from ...agents.claim_atomizer.agent import ClaimAtomizerAgent
from ...agents.conflict_auditor.agent import ConflictAuditorAgent
from ...agents.content_extractor.agent import ContentExtractorAgent
from ...agents.demographic_evaluator.agent import DemographicEvaluatorAgent
from ...agents.manager import AgentType
from ...agents.nudge_collapse.agent import NudgeCollapseAgent
from ...agents.summarizer.agent import SummarizerAgent
from ...config.settings import settings
from ...memory.factory import VectorStoreFactory
from ...utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/agents", tags=["Agent Management"])

@router.post("/create", response_model=AgentIdResponse)
async def create_agent(
    request: CreateAgentRequest,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Create a new agent instance.
    
    Args:
        request: Agent creation request
        _api_key: Authentication dependency (value not used, required for auth check)
        
    Returns:
        Agent ID and metadata
    """
    logger.info(f"Creating agent: type={request.agent_type}, id={request.agent_id}, use_memory={request.use_memory}")
    
    try:
        if request.agent_type not in [e.value for e in AgentType]:
            logger.warning(f"Invalid agent type requested: {request.agent_type}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid agent type. Must be one of: {[e.value for e in AgentType]}"
            )
        
        vector_store = None
        if request.use_memory:
            logger.debug(f"Creating vector store: type={settings.vector_store_type}")
            embeddings = OpenAIEmbeddings(
                api_key=settings.openai_api_key,
                base_url=settings.openai_api_base
            )
            
            if settings.vector_store_type == "redis":
                if settings.redis_password:
                    redis_url = f"redis://:{settings.redis_password}@{settings.redis_host}:{settings.redis_port}/{settings.redis_db}"
                else:
                    redis_url = f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}"
                
                vector_store_kwargs = {
                    "redis_url": redis_url,
                    "index_name": f"agent_memory_{request.agent_id or 'auto'}"
                }
                logger.debug(f"Redis vector store config: host={settings.redis_host}, port={settings.redis_port}")
            elif settings.vector_store_type == "postgres":
                vector_store_kwargs = {
                    "connection_string": f"postgresql://{settings.postgres_user}:{settings.postgres_password}@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}",
                    "collection_name": f"agent_memory_{request.agent_id or 'auto'}"
                }
                logger.debug(f"Postgres vector store config: host={settings.postgres_host}, port={settings.postgres_port}")
            else:  # chroma
                vector_store_kwargs = {
                    "persist_directory": settings.chroma_persist_directory,
                    "collection_name": f"agent_memory_{request.agent_id or 'auto'}"
                }
                logger.debug(f"Chroma vector store config: persist_directory={settings.chroma_persist_directory}")
            
            vector_store = VectorStoreFactory.create_vector_store(
                store_type=settings.vector_store_type,
                embeddings=embeddings,
                **vector_store_kwargs
            )
            logger.info(f"Vector store created successfully for agent type: {request.agent_type}")
        
        logger.debug(f"Instantiating agent: type={request.agent_type}, model={settings.openai_model}")
        proxy = settings.openai_proxy if settings.openai_proxy else None
        if request.agent_type == AgentType.NUDGE_COLLAPSE:
            agent_instance = NudgeCollapseAgent(
                model_name=settings.openai_model,
                api_key=settings.openai_api_key,
                api_base=settings.openai_api_base,
                temperature=settings.agent_temperature,
                vector_store=vector_store,
                proxy=proxy
            )
        elif request.agent_type == AgentType.SUMMARIZER:
            agent_instance = SummarizerAgent(
                model_name=settings.openai_model,
                api_key=settings.openai_api_key,
                api_base=settings.openai_api_base,
                temperature=settings.agent_temperature,
                vector_store=vector_store,
                proxy=proxy
            )
        elif request.agent_type == AgentType.BOT_CREATOR:
            persona_mode = request.persona_mode or "system_prompt"
            if persona_mode not in ["system_prompt", "user_instruction"]:
                raise ValueError(f"Invalid persona_mode. Must be 'system_prompt' or 'user_instruction'")
            
            agent_instance = BotCreatorAgent(
                model_name=settings.openai_model,
                api_key=settings.openai_api_key,
                api_base=settings.openai_api_base,
                temperature=settings.agent_temperature,
                vector_store=vector_store,
                proxy=proxy,
                persona_mode=persona_mode
            )
        elif request.agent_type == AgentType.DEMOGRAPHIC_EVALUATOR:
            agent_instance = DemographicEvaluatorAgent(
                model_name=settings.openai_model,
                api_key=settings.openai_api_key,
                api_base=settings.openai_api_base,
                temperature=settings.agent_temperature,
                proxy=proxy
            )
        elif request.agent_type == AgentType.CONTENT_EXTRACTOR:
            agent_instance = ContentExtractorAgent(
                model_name=settings.openai_model,
                api_key=settings.openai_api_key,
                api_base=settings.openai_api_base,
                temperature=settings.agent_temperature,
                proxy=proxy
            )
        elif request.agent_type == AgentType.CLAIM_ATOMIZER:
            agent_instance = ClaimAtomizerAgent(
                model_name=settings.openai_model,
                api_key=settings.openai_api_key,
                api_base=settings.openai_api_base,
                temperature=settings.agent_temperature,
                proxy=proxy,
                execution_mode=settings.default_execution_mode
            )
        elif request.agent_type == AgentType.CONFLICT_AUDITOR:
            agent_instance = ConflictAuditorAgent(
                model_name=settings.openai_model,
                api_key=settings.openai_api_key,
                api_base=settings.openai_api_base,
                temperature=settings.agent_temperature,
                proxy=proxy,
                execution_mode=settings.default_execution_mode
            )
        else:
            raise ValueError(f"Unsupported agent type: {request.agent_type}")
        
        agent_id = agent_manager.create_agent(
            agent_instance=agent_instance,
            agent_type=AgentType(request.agent_type),
            agent_id=request.agent_id
        )
        
        response_persona_mode = None
        if request.agent_type == AgentType.BOT_CREATOR.value:
            response_persona_mode = request.persona_mode or "system_prompt"
        
        logger.info(f"Agent created successfully: id={agent_id}, type={request.agent_type}")
        
        return AgentIdResponse(
            agent_id=agent_id,
            agent_type=request.agent_type,
            status="created",
            message=f"Agent '{agent_id}' of type '{request.agent_type}' created successfully",
            persona_mode=response_persona_mode
        )
        
    except ValueError as e:
        logger.error(f"Validation error creating agent: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create agent: {str(e)}")


@router.get("/list", response_model=ListAgentsResponse)
async def list_agents(_api_key: str = Depends(get_api_key)):  # Authentication dependency (value not used)
    """
    List all active agent instances.
    
    Args:
        _api_key: Authentication dependency (value not used, required for auth check)
    
    Returns:
        List of agents with their IDs and types
    """
    agents = agent_manager.list_agents()
    return ListAgentsResponse(
        agents=agents,
        total_count=len(agents)
    )


@router.get("/{agent_id}/status", response_model=AgentStatusResponse)
async def get_agent_status(
    agent_id: str,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Get status of a specific agent.
    
    Args:
        agent_id: The agent's unique identifier
        _api_key: Authentication dependency (value not used, required for auth check)
        
    Returns:
        Agent status information
    """
    agent = agent_manager.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    
    agent_type = agent_manager.get_agent_type(agent_id)
    current_turn = getattr(agent, 'get_current_turn', lambda: None)()
    
    additional_info = {}
    if agent_type == AgentType.NUDGE_COLLAPSE:
        additional_info["max_turns"] = getattr(agent, 'max_turns', None)
    elif agent_type == AgentType.SUMMARIZER:
        additional_info["summaries_generated"] = len(getattr(agent, 'summary_history', []))
    elif agent_type == AgentType.BOT_CREATOR:
        additional_info["bots_created"] = len(getattr(agent, 'created_bots', []))
    
    return AgentStatusResponse(
        agent_id=agent_id,
        agent_type=agent_type,
        status="active",
        current_turn=current_turn,
        additional_info=additional_info
    )


@router.delete("/{agent_id}")
async def delete_agent(
    agent_id: str,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Delete an agent instance.
    
    Args:
        agent_id: The agent's unique identifier
        _api_key: Authentication dependency (value not used, required for auth check)
        
    Returns:
        Deletion confirmation
    """
    if not agent_manager.exists(agent_id):
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    
    agent_manager.delete_agent(agent_id)
    return {
        "status": "deleted",
        "message": f"Agent '{agent_id}' deleted successfully"
    }


@router.post("/{agent_id}/reset")
async def reset_agent(
    agent_id: str,
    request: ResetAgentRequest,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Reset an agent's state.
    
    Args:
        agent_id: The agent's unique identifier
        request: Reset options
        _api_key: Authentication dependency (value not used, required for auth check)
        
    Returns:
        Reset confirmation
    """
    agent = agent_manager.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    
    try:
        if request.reset_conversation and hasattr(agent, 'reset'):
            agent.reset()
        
        if request.clear_memory:
            pass
        
        current_turn = getattr(agent, 'get_current_turn', lambda: None)()
        
        return {
            "status": "success",
            "message": f"Agent '{agent_id}' reset successfully",
            "current_turn": current_turn
        }
        
    except Exception as e:
        logger.error(f"Failed to reset agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reset agent: {str(e)}")


# Nudge-Collapse Agent Endpoints
