"""FastAPI application for AI Search Agents Platform - Optimized Version."""

from typing import List, Optional, Dict, Any, Literal
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field, model_validator
from langchain_openai import OpenAIEmbeddings

from ..config.settings import settings, ExecutionMode, HistoryMode
from ..utils.logger import configure_app_logging, get_logger
from ..memory.factory import VectorStoreFactory
from ..agents.nudge_collapse.agent import NudgeCollapseAgent
from ..agents.summarizer.agent import SummarizerAgent
from ..agents.bot_creator.agent import BotCreatorAgent
from ..agents.manager import AgentManager, AgentType
from .auth import verify_api_key
from ..debate.schemas import PersonaConfig, AgentMetadata, InitRequest, InteractRequest, VoteResponse
from ..debate.service import DebateService, generate_default_personas
from ..agents.web_opinion_extractor import (
    WebOpinionAnalyzer,
    WebOpinionEngine,
    LogicMode,
    OpinionExtractionResult,
    AtomicOpinion,
    BiasDistribution
)

# Configure centralized logging
configure_app_logging(
    log_level=settings.log_level,
    log_file=settings.log_file,
    enable_debug=settings.enable_debug
)
logger = get_logger(__name__)


# Pydantic models for request/response
class CreateAgentRequest(BaseModel):
    """Request model for creating a new agent instance."""
    agent_type: str = Field(..., description="Type of agent: 'nudge_collapse', 'summarizer', or 'bot_creator'")
    agent_id: Optional[str] = Field(default=None, description="Custom agent ID (auto-generated if not provided)")
    use_memory: bool = Field(default=False, description="Whether to use vector memory for this agent")
    persona_mode: Optional[str] = Field(default="system_prompt", description="Persona mode for bot_creator: 'system_prompt' or 'user_instruction'")


class AgentIdResponse(BaseModel):
    """Response model for agent creation."""
    agent_id: str
    agent_type: str
    status: str
    message: str
    persona_mode: Optional[str] = None


class GenerateTurnRequest(BaseModel):
    """Request model for generating a turn (nudge_collapse agent)."""
    user_query: str = Field(..., description="User's query or input")
    search_summary: str = Field(default="", description="Summary from search engine")
    search_urls: Optional[List[str]] = Field(default=None, description="URLs from search results")
    history_mode: Optional[str] = Field(
        default=None,
        description="History mode: 'full' (include conversation history), 'none' (stateless). Defaults to system setting."
    )
    use_few_shots: bool = Field(
        default=True,
        description="Whether to use few-shot examples in the prompt (default: True)"
    )
    custom_few_shots: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional custom few-shot examples. Should be a dict with keys 'turn_0', 'turn_1', 'turn_2', 'turn_3'. If provided, use_few_shots must be True."
    )


class ResetAgentRequest(BaseModel):
    """Request model for resetting an agent."""
    reset_conversation: bool = Field(default=True, description="Whether to reset conversation history")
    clear_memory: bool = Field(default=False, description="Whether to clear vector memory")


class TurnResponse(BaseModel):
    """Response model for a turn."""
    turn: int
    query: str
    response: str
    search_summary: str
    search_urls: List[str]
    strategy: str
    history_mode: Optional[str] = None


class ConversationHistoryResponse(BaseModel):
    """Response model for conversation history."""
    current_turn: int
    max_turns: int
    history: List[Dict[str, Any]]


class AgentStatusResponse(BaseModel):
    """Response model for agent status."""
    agent_id: str
    agent_type: str
    status: str
    current_turn: Optional[int] = None
    additional_info: Optional[Dict[str, Any]] = None


class SummarizeRequest(BaseModel):
    """Request model for summarizing conversations."""
    conversation_records: List[Dict[str, str]] = Field(..., description="List of conversation records to summarize")
    execution_mode: Optional[ExecutionMode] = Field(
        default=None,
        description="Execution mode: 'chain_online' (LLM chains), 'chain_local' (local decomposition), 'no_chain' (pure prompt). Defaults to system setting."
    )
    use_few_shots: bool = Field(
        default=True,
        description="Whether to use few-shot examples in the prompt (default: True)"
    )
    custom_few_shots: Optional[str] = Field(
        default=None,
        description="Optional custom few-shot examples to use instead of defaults. If provided, use_few_shots must be True."
    )


class SummaryResponse(BaseModel):
    """Response model for summary."""
    summary: str
    conversation_length: int
    original_length: int
    truncated: bool
    execution_mode: str
    metadata: Dict[str, Any]


class CreateBotRequest(BaseModel):
    """Request model for creating a bot."""
    persona_prompt: str = Field(..., description="Persona prompt corpus for the bot")
    bot_name: Optional[str] = Field(default=None, description="Optional name for the bot")
    execution_mode: Optional[ExecutionMode] = Field(
        default=None,
        description="Execution mode: 'chain_online' (LLM chains), 'chain_local' (local decomposition), 'no_chain' (pure prompt). Defaults to system setting."
    )
    use_few_shots: bool = Field(
        default=True,
        description="Whether to use few-shot examples in the prompt (default: True)"
    )
    custom_few_shots: Optional[str] = Field(
        default=None,
        description="Optional custom few-shot examples to use instead of defaults. If provided, use_few_shots must be True."
    )


class BotCreationResponse(BaseModel):
    """Response model for bot creation."""
    bot_id: str
    bot_name: str
    status: str
    persona_prompt: str
    persona_mode: Optional[str] = None
    execution_mode: Optional[str] = None
    bot_configuration: str
    message: str


class ListAgentsResponse(BaseModel):
    """Response model for listing agents."""
    agents: List[Dict[str, str]]
    total_count: int


# Initialize FastAPI app
app = FastAPI(
    title="AI Search Agents Platform",
    description="Platform for experimenting with AI search agents with multi-agent support and authentication",
    version="2.0.0"
)

# Global agent manager
agent_manager = AgentManager()

# Global debate service
debate_service = DebateService()


@app.get("/")
async def root():
    """Root endpoint with API information."""
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
            "bot_creator": "Bot creation and configuration agent"
        },
        "endpoints": {
            "agents": "/api/v1/agents (POST to create, GET to list)",
            "agent_details": "/api/v1/agents/{agent_id} (GET status, DELETE to remove)",
            "agent_reset": "/api/v1/agents/{agent_id}/reset",
            "nudge_collapse": "/api/v1/agents/{agent_id}/nudge-collapse/*",
            "nudge_collapse_default_shots": "/api/v1/agents/nudge-collapse/default-shots (GET - get default few-shot examples)",
            "summarizer": "/api/v1/agents/{agent_id}/summarizer/*",
            "summarizer_default_shots": "/api/v1/agents/summarizer/default-shots (GET - get default few-shot examples)",
            "bot_creator": "/api/v1/agents/{agent_id}/bot-creator/*",
            "bot_creator_default_shots": "/api/v1/agents/bot-creator/default-shots (GET - get default few-shot examples)",
            "debate_init": "/debate/init (POST to create debate session)",
            "debate_chat": "/agent/{agent_id}/chat (POST to interact with agent)",
            "debate_stability": "/debate/{session_id}/stability_check (POST to check stability)",
            "web_opinion_extractandclean": "/api/v1/web-opinion/extractandclean (POST - extract HTML from URL and clean to text)",
            "web_opinion_extract_opinions": "/api/v1/web-opinion/extract-opinions (POST - extract atomic opinions from text)",
            "web_opinion_analyze": "/api/v1/web-opinion/analyze (POST - complete analysis from URL with WebOpinionEngine)",
            "web_opinion_bias_score": "/api/v1/web-opinion/bias-score (POST - get overall bias score from URL)",
            "web_opinion_default_shots": "/api/v1/web-opinion/default-shots (GET - get default few-shot examples)"
        },
        "authentication": {
            "enabled": settings.api_key_required,
            "header": "X-API-Key"
        }
    }


@app.post("/api/v1/agents", response_model=AgentIdResponse)
async def create_agent(
    request: CreateAgentRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Create a new agent instance.
    
    Args:
        request: Agent creation request
        api_key: API key for authentication
        
    Returns:
        Agent ID and metadata
    """
    logger.info(f"Creating agent: type={request.agent_type}, id={request.agent_id}, use_memory={request.use_memory}")
    
    try:
        # Validate agent type
        if request.agent_type not in [e.value for e in AgentType]:
            logger.warning(f"Invalid agent type requested: {request.agent_type}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid agent type. Must be one of: {[e.value for e in AgentType]}"
            )
        
        # Create vector store if memory is enabled
        vector_store = None
        if request.use_memory:
            logger.debug(f"Creating vector store: type={settings.vector_store_type}")
            embeddings = OpenAIEmbeddings(
                api_key=settings.openai_api_key,
                base_url=settings.openai_api_base
            )
            
            # Build kwargs for vector store based on type
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
        
        # Create agent instance
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
            # Validate persona_mode
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
        else:
            raise ValueError(f"Unsupported agent type: {request.agent_type}")
        
        # Register agent
        agent_id = agent_manager.create_agent(
            agent_instance=agent_instance,
            agent_type=AgentType(request.agent_type),
            agent_id=request.agent_id
        )
        
        # Get persona_mode for response (only for bot_creator)
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


@app.get("/api/v1/agents", response_model=ListAgentsResponse)
async def list_agents(api_key: str = Depends(verify_api_key)):
    """
    List all active agent instances.
    
    Returns:
        List of agents with their IDs and types
    """
    agents = agent_manager.list_agents()
    return ListAgentsResponse(
        agents=agents,
        total_count=len(agents)
    )


@app.get("/api/v1/agents/{agent_id}", response_model=AgentStatusResponse)
async def get_agent_status(
    agent_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Get status of a specific agent.
    
    Args:
        agent_id: The agent's unique identifier
        
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


@app.delete("/api/v1/agents/{agent_id}")
async def delete_agent(
    agent_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Delete an agent instance.
    
    Args:
        agent_id: The agent's unique identifier
        
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


@app.post("/api/v1/agents/{agent_id}/reset")
async def reset_agent(
    agent_id: str,
    request: ResetAgentRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Reset an agent's state.
    
    Args:
        agent_id: The agent's unique identifier
        request: Reset options
        
    Returns:
        Reset confirmation
    """
    agent = agent_manager.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    
    try:
        if request.reset_conversation and hasattr(agent, 'reset'):
            agent.reset()
        
        # Note: Clearing vector memory would require additional implementation
        if request.clear_memory:
            # This is a placeholder - actual implementation would vary by vector store
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
@app.post("/api/v1/agents/{agent_id}/nudge-collapse/generate", response_model=TurnResponse)
async def generate_turn(
    agent_id: str,
    request: GenerateTurnRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Generate a response turn for the Nudge-Collapse agent.
    
    Args:
        agent_id: The agent's unique identifier
        request: Generation request with user query and search context
        
    Returns:
        Turn response with agent's reply
    """
    agent = agent_manager.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    
    agent_type = agent_manager.get_agent_type(agent_id)
    if agent_type != AgentType.NUDGE_COLLAPSE:
        raise HTTPException(
            status_code=400,
            detail=f"This endpoint requires a 'nudge_collapse' agent, but agent '{agent_id}' is type '{agent_type}'"
        )
    
    try:
        result = agent.generate_turn(
            user_query=request.user_query,
            search_summary=request.search_summary,
            search_urls=request.search_urls,
            history_mode=request.history_mode,
            use_few_shots=request.use_few_shots,
            custom_few_shots=request.custom_few_shots
        )
        
        if "error" in result:
            logger.warning(f"Agent {agent_id} generate_turn error: {result['error']}")
            raise HTTPException(status_code=400, detail=result["error"])
        
        return TurnResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate turn for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate turn: {str(e)}")


@app.get("/api/v1/agents/{agent_id}/nudge-collapse/history", response_model=ConversationHistoryResponse)
async def get_conversation_history(
    agent_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Get conversation history for a Nudge-Collapse agent.
    
    Args:
        agent_id: The agent's unique identifier
        
    Returns:
        Conversation history
    """
    agent = agent_manager.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    
    agent_type = agent_manager.get_agent_type(agent_id)
    if agent_type != AgentType.NUDGE_COLLAPSE:
        raise HTTPException(
            status_code=400,
            detail=f"This endpoint requires a 'nudge_collapse' agent, but agent '{agent_id}' is type '{agent_type}'"
        )
    
    try:
        return ConversationHistoryResponse(
            current_turn=agent.get_current_turn(),
            max_turns=agent.max_turns,
            history=agent.get_conversation_history()
        )
    except Exception as e:
        logger.error(f"Failed to get history for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@app.get("/api/v1/agents/nudge-collapse/default-shots")
async def get_nudge_collapse_default_shots(
    turn: Optional[int] = None,
    api_key: str = Depends(verify_api_key)
):
    """
    Get default few-shot examples for Nudge-Collapse agent.
    
    Args:
        turn: Optional turn number (0-3). If provided, returns few shots for that turn only.
              If None, returns all turns as a dictionary.
        api_key: API key for authentication
        
    Returns:
        Default few-shot examples for the specified turn or all turns
    """
    try:
        from ..agents.nudge_collapse.agent import NudgeCollapseAgent
        few_shots = NudgeCollapseAgent.get_default_few_shots(turn=turn)
        return {
            "agent_type": "nudge_collapse",
            "turn": turn,
            "few_shots": few_shots
        }
    except Exception as e:
        logger.error(f"Failed to get default few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get default few shots: {str(e)}")


# Summarizer Agent Endpoints
@app.post("/api/v1/agents/{agent_id}/summarizer/summarize", response_model=SummaryResponse)
async def summarize_conversation(
    agent_id: str,
    request: SummarizeRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Summarize conversation records using the Summarizer agent.
    
    This endpoint now focuses on user questions and handles long conversations
    by truncating and optimizing the input.
    
    Args:
        agent_id: The agent's unique identifier
        request: Summarization request with conversation records
        
    Returns:
        Summary response with metadata about truncation
    """
    agent = agent_manager.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    
    agent_type = agent_manager.get_agent_type(agent_id)
    if agent_type != AgentType.SUMMARIZER:
        raise HTTPException(
            status_code=400,
            detail=f"This endpoint requires a 'summarizer' agent, but agent '{agent_id}' is type '{agent_type}'"
        )
    
    try:
        result = agent.summarize_conversation(
            request.conversation_records,
            execution_mode=request.execution_mode,
            use_few_shots=request.use_few_shots,
            custom_few_shots=request.custom_few_shots
        )
        
        if "error" in result:
            logger.warning(f"Agent {agent_id} summarize error: {result['error']}")
            raise HTTPException(status_code=400, detail=result["error"])
        
        return SummaryResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to summarize conversation for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to summarize conversation: {str(e)}")


@app.get("/api/v1/agents/{agent_id}/summarizer/history")
async def get_summary_history(
    agent_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Get the history of summaries generated by this agent.
    
    Args:
        agent_id: The agent's unique identifier
        
    Returns:
        List of summary metadata
    """
    agent = agent_manager.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    
    agent_type = agent_manager.get_agent_type(agent_id)
    if agent_type != AgentType.SUMMARIZER:
        raise HTTPException(
            status_code=400,
            detail=f"This endpoint requires a 'summarizer' agent, but agent '{agent_id}' is type '{agent_type}'"
        )
    
    try:
        history = agent.get_summary_history()
        # Return only metadata, not full records
        return {
            "summaries": [
                {
                    "conversation_length": entry["conversation_length"],
                    "summary": entry["summary"]
                }
                for entry in history
            ],
            "total_summaries": len(history)
        }
    except Exception as e:
        logger.error(f"Failed to get summary history for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get summary history: {str(e)}")


@app.get("/api/v1/agents/summarizer/default-shots")
async def get_summarizer_default_shots(
    api_key: str = Depends(verify_api_key)
):
    """
    Get default few-shot examples for Summarizer agent.
    
    Args:
        api_key: API key for authentication
        
    Returns:
        Default few-shot examples
    """
    try:
        from ..agents.summarizer.agent import SummarizerAgent
        few_shots = SummarizerAgent.get_default_few_shots()
        return {
            "agent_type": "summarizer",
            "few_shots": few_shots
        }
    except Exception as e:
        logger.error(f"Failed to get default few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get default few shots: {str(e)}")


# Bot Creator Agent Endpoints
@app.post("/api/v1/agents/{agent_id}/bot-creator/create", response_model=BotCreationResponse)
async def create_bot(
    agent_id: str,
    request: CreateBotRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Create a new bot using the Bot Creator agent.
    
    Args:
        agent_id: The agent's unique identifier
        request: Bot creation request with persona prompt
        
    Returns:
        Bot creation response
    """
    logger.info(f"Creating bot for agent {agent_id} with persona_prompt length: {len(request.persona_prompt)}")
    
    agent = agent_manager.get_agent(agent_id)
    if not agent:
        logger.warning(f"Agent {agent_id} not found")
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    
    agent_type = agent_manager.get_agent_type(agent_id)
    if agent_type != AgentType.BOT_CREATOR:
        logger.warning(f"Wrong agent type for bot creation: {agent_type}")
        raise HTTPException(
            status_code=400,
            detail=f"This endpoint requires a 'bot_creator' agent, but agent '{agent_id}' is type '{agent_type}'"
        )
    
    # Validate persona_prompt is not empty
    if not request.persona_prompt or not request.persona_prompt.strip():
        logger.warning(f"Empty persona_prompt provided for agent {agent_id}")
        raise HTTPException(status_code=400, detail="persona_prompt cannot be empty")
    
    try:
        result = agent.create_bot(
            persona_prompt=request.persona_prompt,
            bot_name=request.bot_name,
            execution_mode=request.execution_mode,
            use_few_shots=request.use_few_shots,
            custom_few_shots=request.custom_few_shots
        )
        
        if "error" in result:
            logger.warning(f"Agent {agent_id} create_bot error: {result['error']}")
            raise HTTPException(status_code=400, detail=result["error"])
        
        logger.info(f"Successfully created bot {result.get('bot_id')} for agent {agent_id}")
        return BotCreationResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create bot for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create bot: {str(e)}")


@app.get("/api/v1/agents/{agent_id}/bot-creator/bots")
async def list_bots(
    agent_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    List all bots created by this Bot Creator agent.
    
    Args:
        agent_id: The agent's unique identifier
        
    Returns:
        List of bot configurations
    """
    agent = agent_manager.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    
    agent_type = agent_manager.get_agent_type(agent_id)
    if agent_type != AgentType.BOT_CREATOR:
        raise HTTPException(
            status_code=400,
            detail=f"This endpoint requires a 'bot_creator' agent, but agent '{agent_id}' is type '{agent_type}'"
        )
    
    try:
        bots = agent.list_bots()
        return {"bots": bots, "total_count": len(bots)}
    except Exception as e:
        logger.error(f"Failed to list bots for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list bots: {str(e)}")


class ChatWithBotRequest(BaseModel):
    """Request model for chatting with a bot."""
    bot_id: str = Field(..., description="ID of the bot to chat with")
    message: str = Field(..., description="User message to send to the bot")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Previous conversation history (list of {role: 'user'|'assistant', content: str})"
    )
    history_mode: Optional[str] = Field(
        default=None,
        description="History mode: 'full' (include conversation history), 'none' (stateless). Defaults to system setting."
    )


class ChatWithBotResponse(BaseModel):
    """Response model for bot chat."""
    bot_id: str
    bot_name: str
    response: str
    conversation_history: List[Dict[str, str]]
    history_mode: Optional[str] = None


@app.post("/api/v1/agents/{agent_id}/bot-creator/chat", response_model=ChatWithBotResponse)
async def chat_with_bot(
    agent_id: str,
    request: ChatWithBotRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Chat with a bot created by the Bot Creator agent.
    
    This endpoint allows you to have conversations with bots that have been
    previously created using the /bot-creator/create endpoint. Each bot uses
    its configured persona to respond to messages.
    
    Args:
        agent_id: The Bot Creator agent's unique identifier
        request: Chat request with bot_id, message, and optional conversation history
        
    Returns:
        Bot's response and updated conversation history
    """
    logger.info(f"Chat with bot request: agent_id={agent_id}, bot_id={request.bot_id}")
    
    agent = agent_manager.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    
    agent_type = agent_manager.get_agent_type(agent_id)
    if agent_type != AgentType.BOT_CREATOR:
        raise HTTPException(
            status_code=400,
            detail=f"This endpoint requires a 'bot_creator' agent, but agent '{agent_id}' is type '{agent_type}'"
        )
    
    try:
        result = agent.chat_with_bot(
            bot_id=request.bot_id,
            user_message=request.message,
            conversation_history=request.conversation_history,
            history_mode=request.history_mode
        )
        
        if "error" in result:
            logger.warning(f"Chat with bot error: {result['error']}")
            raise HTTPException(status_code=400, detail=result["error"])
        
        logger.info(f"Successfully chatted with bot {request.bot_id}")
        return ChatWithBotResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to chat with bot {request.bot_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to chat with bot: {str(e)}")


@app.get("/api/v1/agents/bot-creator/default-shots")
async def get_bot_creator_default_shots(
    api_key: str = Depends(verify_api_key)
):
    """
    Get default few-shot examples for Bot Creator agent.
    
    Args:
        api_key: API key for authentication
        
    Returns:
        Default few-shot examples
    """
    try:
        from ..agents.bot_creator.agent import BotCreatorAgent
        few_shots = BotCreatorAgent.get_default_few_shots()
        return {
            "agent_type": "bot_creator",
            "few_shots": few_shots
        }
    except Exception as e:
        logger.error(f"Failed to get default few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get default few shots: {str(e)}")


# ===========================
# Multi-Agent Debate Endpoints
# ===========================

class InitDebateResponse(BaseModel):
    """Response model for debate initialization."""
    session_id: str
    agents: List[AgentMetadata]
    topic: str
    max_rounds: int = Field(default=10, description="Maximum number of debate rounds")
    execution_mode: Optional[str] = Field(default=None, description="Persona generation mode used")


class StabilityCheckRequest(BaseModel):
    """Request model for stability check."""
    votes: List[int] = Field(..., description="List of votes from the current round")
    reasonings: Optional[List[str]] = Field(default=None, description="Optional reasoning strings from each agent")


class RoundStatistics(BaseModel):
    """Statistics for a single debate round."""
    round_number: int
    votes: List[int]
    vote_distribution: Dict[str, int]
    consensus_level: float
    vote_mean: float
    vote_variance: float
    ks_statistic: Optional[float] = None


class StabilityCheckResponse(BaseModel):
    """Response model for stability check with comprehensive statistics."""
    stable: bool
    current_round: int
    max_rounds: int
    max_rounds_reached: bool
    should_continue: bool
    continue_reason: str
    round_stats: Optional[Dict[str, Any]] = None


class DebateStatisticsResponse(BaseModel):
    """Response model for debate statistics."""
    session_id: str
    topic: str
    total_rounds: int
    max_rounds: int
    max_rounds_reached: bool
    converged: bool
    convergence_round: Optional[int] = None
    final_consensus_level: Optional[float] = None
    final_vote_distribution: Optional[Dict[str, int]] = None
    winner: Optional[int] = None
    winner_vote_count: Optional[int] = None
    total_agents: int
    convergence_history: List[float]
    agent_analysis: Dict[str, Any]
    round_by_round: List[Dict[str, Any]]


class GeneratePersonasRequest(BaseModel):
    """Request model for generating personas with chain modes."""
    topic: str = Field(..., description="Debate topic for persona generation")
    context: str = Field(default="", description="Additional context for persona generation")
    num_agents: int = Field(default=3, ge=2, le=10, description="Number of agents to generate")
    execution_mode: Optional[ExecutionMode] = Field(
        default=None,
        description="Persona generation mode: 'chain_online', 'chain_local', or 'no_chain'"
    )


class GeneratePersonasResponse(BaseModel):
    """Response model for generated personas."""
    personas: List[PersonaConfig]
    execution_mode: str
    topic: str


class DebateContinueResponse(BaseModel):
    """Response model for debate continue check."""
    continue_debate: bool
    reason: str
    current_round: int
    rounds_remaining: Optional[int] = None
    max_rounds: int


@app.post("/debate/generate_personas", response_model=GeneratePersonasResponse)
async def generate_personas(
    request: GeneratePersonasRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Generate appropriate personas for a debate topic using chain reasoning.
    
    This endpoint uses LLM-based persona generation to create debate participants
    tailored to the specific topic. Three modes are available:
    - chain_online: LLM performs all reasoning and analysis
    - chain_local: Task is decomposed into subtasks locally
    - no_chain: Simple prompt-based generation
    
    Args:
        request: GeneratePersonasRequest with topic, context, and options
        api_key: API key for authentication
    
    Returns:
        List of generated PersonaConfig objects
    """
    # Validate request
    if not request.topic or not request.topic.strip():
        raise HTTPException(
            status_code=400,
            detail="Topic cannot be empty. Please provide a valid debate topic."
        )
    
    if request.num_agents < 2:
        raise HTTPException(
            status_code=400,
            detail="num_agents must be at least 2 for a meaningful debate."
        )
    
    try:
        execution_mode = request.execution_mode or settings.default_execution_mode
        
        personas = await debate_service.generate_personas_for_topic(
            topic=request.topic,
            context=request.context,
            num_agents=request.num_agents,
            execution_mode=execution_mode
        )
        
        return GeneratePersonasResponse(
            personas=personas,
            execution_mode=execution_mode,
            topic=request.topic
        )
    
    except ConnectionError as e:
        logger.error(f"LLM connection error during persona generation: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail="Failed to connect to LLM service. Please try again later."
        )
    except TimeoutError as e:
        logger.error(f"LLM timeout during persona generation: {e}", exc_info=True)
        raise HTTPException(
            status_code=504,
            detail="LLM request timed out. Please try again with a simpler topic."
        )
    except ValueError as e:
        logger.error(f"Invalid input for persona generation: {e}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Failed to generate personas: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate personas: {str(e)}"
        )


@app.post("/debate/init", response_model=InitDebateResponse)
async def init_debate(
    request: InitRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Initialize a new debate session.
    
    Creates agents based on custom personas or auto-generates them using
    LLM-based persona generation with the specified execution mode.
    Each agent gets a specialized system prompt and few-shot example.
    
    Args:
        request: InitRequest with topic, persona configuration, and options
        api_key: API key for authentication
        
    Returns:
        Session ID, list of agent metadata, and debate configuration
    """
    try:
        execution_mode = None
        
        # Determine personas to use
        if request.custom_personas:
            personas = request.custom_personas
        elif request.auto_agent_count > 0:
            # Use chain-based persona generation if execution_mode is specified
            if request.execution_mode:
                execution_mode = request.execution_mode
                personas = await debate_service.generate_personas_for_topic(
                    topic=request.topic,
                    context=request.context or "",
                    num_agents=request.auto_agent_count,
                    execution_mode=execution_mode
                )
            else:
                personas = generate_default_personas(request.auto_agent_count)
        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide either custom_personas or auto_agent_count > 0"
            )
        
        # Create session with max_rounds
        session_id, agents = debate_service.create_session(
            request.topic,
            personas,
            max_rounds=request.max_rounds
        )
        
        # Get max_rounds from session
        session = debate_service.get_session(session_id)
        max_rounds = session.get("max_rounds", 10) if session else 10
        
        return InitDebateResponse(
            session_id=session_id,
            agents=agents,
            topic=request.topic,
            max_rounds=max_rounds,
            execution_mode=execution_mode
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to initialize debate: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initialize debate: {str(e)}")


@app.post("/agent/{agent_id}/chat", response_model=VoteResponse)
async def agent_chat(
    agent_id: str,
    request: InteractRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Interact with a specific debate agent.
    
    The agent receives context about what others have said and responds
    with a vote and reasoning. This is a placeholder that calls the LLM
    (currently mocked with await call_llm).
    
    Args:
        agent_id: UUID of the agent
        request: InteractRequest with session_id and history_context
        api_key: API key for authentication
        
    Returns:
        VoteResponse with agent's verdict and reasoning
    """
    try:
        from uuid import UUID
        
        # Parse agent_id as UUID
        try:
            agent_uuid = UUID(agent_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid agent_id format. Must be a valid UUID.")
        
        # Verify agent exists in session
        agent_metadata = debate_service.get_agent_metadata(request.session_id, agent_uuid)
        if not agent_metadata:
            raise HTTPException(
                status_code=404,
                detail=f"Agent '{agent_id}' not found in session '{request.session_id}'"
            )
        
        # Get session info
        session = debate_service.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session '{request.session_id}' not found")
        
        # Prepare the message for the LLM
        user_message = f"""Topic: {session['topic']}

Context from other agents:
{request.history_context}

Please provide your vote (as an integer or descriptive string) and reasoning in JSON format.
{agent_metadata.few_shot_example}"""
        
        # Call the LLM (mocked for now)
        response = await debate_service.call_llm(
            agent_metadata.system_prompt,
            user_message
        )
        
        # Parse the response (in real implementation, parse JSON from LLM)
        # For now, return a mock response
        import json
        try:
            parsed = json.loads(response)
            return VoteResponse(
                agent_id=agent_uuid,
                verdict=parsed.get("verdict", 1),
                reasoning=parsed.get("reasoning", "No reasoning provided")
            )
        except json.JSONDecodeError as e:
            # Fallback if LLM doesn't return valid JSON
            logger.warning(f"Failed to parse LLM response as JSON for agent {agent_id}: {e}")
            return VoteResponse(
                agent_id=agent_uuid,
                verdict=1,
                reasoning=response
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process agent chat for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process agent chat: {str(e)}")


@app.post("/debate/{session_id}/stability_check", response_model=StabilityCheckResponse)
async def check_stability(
    session_id: str,
    request: StabilityCheckRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Check if the debate has reached stability.
    
    Uses KS Statistic logic to compare vote distributions between rounds.
    If the difference is < 0.05 for 2 consecutive rounds, returns True.
    
    Also returns comprehensive statistics about the debate progress including
    whether to continue, current round, max rounds, and round statistics.
    
    Args:
        session_id: Session identifier
        request: StabilityCheckRequest with current round votes and optional reasonings
        api_key: API key for authentication
        
    Returns:
        StabilityCheckResponse with stability status and statistics
    """
    try:
        # Verify session exists
        session = debate_service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        
        # Get agent IDs for tracking
        agent_ids = list(session.get("agents", {}).keys())
        
        # Add the new round of votes with statistics
        round_result = debate_service.add_vote_round(
            session_id,
            request.votes,
            reasonings=request.reasonings,
            agent_ids=agent_ids
        )
        
        # Calculate stability
        is_stable = debate_service.calculate_stability(session_id)
        
        # Check if should continue
        continue_info = debate_service.should_continue_debate(session_id)
        
        # Get updated session info
        session = debate_service.get_session(session_id)
        current_round = session.get("current_round", 0)
        max_rounds = session.get("max_rounds", 10)
        
        return StabilityCheckResponse(
            stable=is_stable,
            current_round=current_round,
            max_rounds=max_rounds,
            max_rounds_reached=round_result.get("max_rounds_reached", False),
            should_continue=continue_info.get("continue", False),
            continue_reason=continue_info.get("reason", "Unknown"),
            round_stats=round_result.get("round_stats")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to check stability for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to check stability: {str(e)}")


@app.get("/debate/{session_id}/statistics", response_model=DebateStatisticsResponse)
async def get_debate_statistics(
    session_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Get comprehensive statistics for a debate session.
    
    Returns detailed information about the debate including:
    - Total rounds completed
    - Convergence status and round
    - Vote distributions and history
    - Per-agent analysis (vote changes, consistency)
    - Round-by-round breakdown
    
    Args:
        session_id: Session identifier
        api_key: API key for authentication
        
    Returns:
        DebateStatisticsResponse with comprehensive statistics
    """
    try:
        # Verify session exists
        session = debate_service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        
        # Get statistics
        stats = debate_service.get_debate_statistics(session_id)
        
        if "error" in stats:
            raise HTTPException(status_code=404, detail=stats["error"])
        
        return DebateStatisticsResponse(**stats)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get statistics for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


@app.get("/debate/{session_id}/should_continue", response_model=DebateContinueResponse)
async def check_debate_continue(
    session_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Check if the debate should continue.
    
    Returns whether to continue the debate and the reason why.
    This helps manage non-converging debates by enforcing max rounds.
    
    Args:
        session_id: Session identifier
        api_key: API key for authentication
        
    Returns:
        DebateContinueResponse with continue status and reason
    """
    try:
        # Verify session exists
        session = debate_service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        
        # Check if should continue
        continue_info = debate_service.should_continue_debate(session_id)
        
        return DebateContinueResponse(
            continue_debate=continue_info.get("continue", False),
            reason=continue_info.get("reason", "Unknown"),
            current_round=continue_info.get("current_round", 0),
            rounds_remaining=continue_info.get("rounds_remaining"),
            max_rounds=session.get("max_rounds", 10)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to check continue status for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to check continue status: {str(e)}")


# ===========================
# Web Opinion Extract Endpoints
# ===========================

class ExtractAndCleanRequest(BaseModel):
    """Request model for combined HTML extraction and cleaning from URL."""
    url: str = Field(..., description="The URL to fetch and clean")


class ExtractAndCleanResponse(BaseModel):
    """Response model for combined HTML extraction and cleaning."""
    url: str
    text: Optional[str] = None
    title: Optional[str] = None
    text_length: Optional[int] = None
    error: Optional[str] = None
    error_message: Optional[str] = None


class ExtractOpinionsRequest(BaseModel):
    """Request model for extracting atomic opinions from text or URL."""
    url: Optional[str] = Field(default=None, description="URL to fetch and analyze (alternative to text)")
    text: Optional[str] = Field(default=None, description="Text content to analyze (alternative to URL)")
    title: Optional[str] = Field(default=None, description="Optional title (used when text is provided)")
    execution_mode: Optional[ExecutionMode] = Field(
        default=None,
        description="Execution mode: 'chain_online', 'chain_local', or 'no_chain'"
    )
    
    @model_validator(mode='after')
    def validate_url_or_text(self):
        """Ensure either URL or text is provided."""
        if not self.url and not self.text:
            raise ValueError("Either 'url' or 'text' must be provided")
        if self.url and self.text:
            raise ValueError("Cannot provide both 'url' and 'text'. Provide either URL or text+title.")
        return self


class BiasDistributionResponse(BaseModel):
    """Response model for bias distribution."""
    left: float = Field(..., description="Probability of Left/Progressive bias (0.0 to 1.0)")
    right: float = Field(..., description="Probability of Right/Conservative bias (0.0 to 1.0)")
    neutral: float = Field(..., description="Probability of Neutral/Centrist bias (0.0 to 1.0)")
    dominant_bias: str = Field(..., description="The dominant bias category")
    bias_score: float = Field(..., description="Single bias score (-1.0 left to +1.0 right)")


class AtomicOpinionResponse(BaseModel):
    """Response model for a single atomic opinion."""
    text: str
    opinion_type: str
    bias_probabilities: BiasDistributionResponse
    original_sentence: Optional[str] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None


class ExtractOpinionsResponse(BaseModel):
    """Response model for opinion extraction."""
    url: Optional[str] = None
    title: Optional[str] = None
    atomic_opinions: List[AtomicOpinionResponse]
    facts: List[AtomicOpinionResponse]
    opinions: List[AtomicOpinionResponse]
    overall_bias_distribution: Optional[BiasDistributionResponse] = None
    text_length: int
    truncated: bool
    error: Optional[str] = None
    error_message: Optional[str] = None


class AnalyzeUrlRequest(BaseModel):
    """Request model for complete URL analysis."""
    url: str = Field(..., description="The URL to analyze")
    mode: Optional[str] = Field(
        default="LOCAL_CHAIN",
        description="Logic mode: 'LOCAL_CHAIN', 'NO_CHAIN', or 'PURE_ONLINE'"
    )
    use_mbfc: bool = Field(
        default=True,
        description="Whether to use MBFC database for prior probability"
    )
    use_few_shots: bool = Field(
        default=True,
        description="Whether to use few-shot examples to optimize the agent"
    )
    atomizer_shots: Optional[List[dict]] = Field(
        default=None,
        description="Optional custom few-shot examples for atomization (overrides defaults)"
    )
    scorer_shots: Optional[List[dict]] = Field(
        default=None,
        description="Optional custom few-shot examples for bias scoring (overrides defaults)"
    )


class BiasScoreRequest(BaseModel):
    """Request model for bias score from URL."""
    url: str = Field(..., description="The URL to analyze for bias")
    execution_mode: Optional[ExecutionMode] = Field(
        default=None,
        description="Execution mode: 'chain_online', 'chain_local', or 'no_chain'"
    )


class BiasScoreResponse(BaseModel):
    """Response model for overall bias score."""
    url: str
    overall_bias_distribution: Optional[BiasDistributionResponse] = None
    opinions_count: int
    facts_count: int
    error: Optional[str] = None
    error_message: Optional[str] = None


class DefaultShotsResponse(BaseModel):
    """Response model for default few-shot examples."""
    atomizer_shots: List[dict] = Field(default_factory=list, description="Default atomizer few-shot examples")
    scorer_shots: List[dict] = Field(default_factory=list, description="Default scorer few-shot examples")


def _convert_bias_distribution(bias: BiasDistribution) -> BiasDistributionResponse:
    """Convert BiasDistribution to response model."""
    return BiasDistributionResponse(
        left=bias.left,
        right=bias.right,
        neutral=bias.neutral,
        dominant_bias=bias.dominant_bias,
        bias_score=bias.bias_score
    )


def _convert_atomic_opinion(opinion: AtomicOpinion) -> AtomicOpinionResponse:
    """Convert AtomicOpinion to response model."""
    return AtomicOpinionResponse(
        text=opinion.text,
        opinion_type=opinion.opinion_type,
        bias_probabilities=_convert_bias_distribution(opinion.bias_probabilities),
        original_sentence=opinion.original_sentence,
        confidence=opinion.confidence,
        reasoning=opinion.reasoning
    )


@app.post("/api/v1/web-opinion/extractandclean", response_model=ExtractAndCleanResponse)
async def extract_and_clean_from_url(
    request: ExtractAndCleanRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Combined API: Extract HTML from URL and clean to text in one step.
    
    This endpoint combines HTML extraction and cleaning into a single API call,
    saving tokens by avoiding the need to pass large HTML content between calls.
    It fetches HTML from the URL and directly returns cleaned text using BeautifulSoup.
    
    If a proxy is needed, configure it using the OPENAI_PROXY setting or 
    HTTP_PROXY/HTTPS_PROXY environment variables.
    
    Args:
        request: ExtractAndCleanRequest with URL
        api_key: API key for authentication
        
    Returns:
        ExtractAndCleanResponse with cleaned text and title or error
    """
    logger.info(f"Extracting and cleaning HTML from URL: {request.url}")
    
    try:
        # Reuse OpenAI proxy settings if configured
        proxy = settings.openai_proxy or None
        
        analyzer = WebOpinionAnalyzer(
            execution_mode=settings.default_execution_mode,
            proxy=proxy
        )
        
        # Step 1: Extract HTML from URL
        html = analyzer.extract_html(request.url)
        
        if html is None:
            return ExtractAndCleanResponse(
                url=request.url,
                error="fetch_failed",
                error_message="Failed to fetch HTML from URL"
            )
        
        # Step 2: Clean HTML to extract text
        text, title = analyzer.clean_html(html)
        
        if text is None:
            return ExtractAndCleanResponse(
                url=request.url,
                error="cleaning_failed",
                error_message="Failed to clean HTML content"
            )
        
        return ExtractAndCleanResponse(
            url=request.url,
            text=text,
            title=title,
            text_length=len(text)
        )
        
    except Exception as e:
        logger.error(f"Failed to extract and clean from {request.url}: {e}", exc_info=True)
        return ExtractAndCleanResponse(
            url=request.url,
            error="extraction_failed",
            error_message=str(e)
        )


@app.post("/api/v1/web-opinion/extract-opinions", response_model=ExtractOpinionsResponse)
async def extract_atomic_opinions(
    request: ExtractOpinionsRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Extract atomic opinions from URL or text content.
    
    This endpoint can analyze content in two ways:
    1. Provide a URL: The endpoint will fetch HTML, extract text, and analyze it
    2. Provide text + title: The endpoint will analyze the provided text directly
    
    Each atomic opinion includes:
    - Text of the opinion
    - Opinion type (fact or opinion)
    - Bias probability distribution (left, right, neutral)
    - Optional reasoning (if CoT mode is enabled)
    
    Args:
        request: ExtractOpinionsRequest with either URL or text+title
        api_key: API key for authentication
        
    Returns:
        ExtractOpinionsResponse with extracted opinions and bias scores
    """
    try:
        execution_mode = request.execution_mode or settings.default_execution_mode
        
        analyzer = WebOpinionAnalyzer(
            execution_mode=execution_mode,
            proxy=settings.openai_proxy if settings.openai_proxy else None
        )
        
        # Handle URL input: fetch and extract content
        if request.url:
            logger.info(f"Extracting opinions from URL: {request.url}")
            
            # Step 1: Fetch HTML
            html = analyzer.extract_html(request.url)
            if html is None:
                return ExtractOpinionsResponse(
                    url=request.url,
                    title=None,
                    atomic_opinions=[],
                    facts=[],
                    opinions=[],
                    text_length=0,
                    truncated=False,
                    error="fetch_failed",
                    error_message="Failed to fetch HTML from URL"
                )
            
            # Step 2: Clean HTML to extract text and title
            text, title = analyzer.clean_html(html)
            if text is None:
                return ExtractOpinionsResponse(
                    url=request.url,
                    title=None,
                    atomic_opinions=[],
                    facts=[],
                    opinions=[],
                    text_length=0,
                    truncated=False,
                    error="cleaning_failed",
                    error_message="Failed to clean HTML content"
                )
            
            # Step 3: Analyze the extracted text
            result = analyzer.analyze_text(text, url=request.url, title=title)
            
        else:
            # Handle direct text input
            logger.info(f"Extracting opinions from text ({len(request.text)} chars)")
            result = analyzer.analyze_text(request.text, url=None, title=request.title)
        
        # Check for errors
        if result.extraction_metadata and "error" in result.extraction_metadata:
            return ExtractOpinionsResponse(
                url=request.url if request.url else None,
                title=request.title if not request.url else result.title,
                atomic_opinions=[],
                facts=[],
                opinions=[],
                text_length=len(request.text) if request.text else 0,
                truncated=False,
                error=result.extraction_metadata["error"],
                error_message=result.extraction_metadata.get("error_message", "Unknown error")
            )
        
        # Convert opinions to response format
        atomic_opinions = [_convert_atomic_opinion(op) for op in result.atomic_opinions]
        facts = [_convert_atomic_opinion(op) for op in result.facts]
        opinions = [_convert_atomic_opinion(op) for op in result.opinions]
        
        overall_bias = None
        if result.overall_bias_distribution:
            overall_bias = _convert_bias_distribution(result.overall_bias_distribution)
        
        return ExtractOpinionsResponse(
            url=result.url,
            title=result.title,
            atomic_opinions=atomic_opinions,
            facts=facts,
            opinions=opinions,
            overall_bias_distribution=overall_bias,
            text_length=result.text_length,
            truncated=result.truncated
        )
        
    except ValueError as e:
        # Handle validation errors (e.g., missing URL or text)
        logger.error(f"Validation error: {e}", exc_info=True)
        return ExtractOpinionsResponse(
            url=request.url if request.url else None,
            title=request.title if request.title else None,
            atomic_opinions=[],
            facts=[],
            opinions=[],
            text_length=0,
            truncated=False,
            error="validation_failed",
            error_message=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to extract opinions: {e}", exc_info=True)
        return ExtractOpinionsResponse(
            url=request.url if request.url else None,
            title=request.title if request.title else None,
            atomic_opinions=[],
            facts=[],
            opinions=[],
            text_length=0,
            truncated=False,
            error="extraction_failed",
            error_message=str(e)
        )


@app.post("/api/v1/web-opinion/analyze", response_model=ExtractOpinionsResponse)
async def analyze_url_complete(
    request: AnalyzeUrlRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Complete analysis pipeline: Extract and analyze opinions from URL using WebOpinionEngine.
    
    This endpoint uses the new WebOpinionEngine with configurable:
    - Logic modes: LOCAL_CHAIN, NO_CHAIN, PURE_ONLINE
    - MBFC prior: Optional database lookup for bias prior
    - Few-shot examples: User can enable/disable or provide custom examples
    
    Args:
        request: AnalyzeUrlRequest with URL, mode, use_mbfc, use_few_shots, and optional shots
        api_key: API key for authentication
        
    Returns:
        ExtractOpinionsResponse with all extracted opinions and bias scores
    """
    logger.info(f"Analyzing URL: {request.url}, mode={request.mode}, use_mbfc={request.use_mbfc}, use_few_shots={request.use_few_shots}")
    
    try:
        # Parse mode
        try:
            mode = LogicMode(request.mode.upper())
        except ValueError:
            mode = LogicMode.LOCAL_CHAIN
            logger.warning(f"Invalid mode '{request.mode}', using LOCAL_CHAIN")
        
        # Initialize engine (db_path comes from settings)
        engine = WebOpinionEngine(
            proxy=settings.openai_proxy if settings.openai_proxy else None
        )
        
        # Run pipeline
        result = engine.run(
            url=request.url,
            mode=mode,
            use_mbfc=request.use_mbfc,
            use_few_shots=request.use_few_shots,
            atomizer_shots=request.atomizer_shots,
            scorer_shots=request.scorer_shots
        )
        
        # Convert to response format
        atomic_opinions = []
        facts = []
        opinions = []
        
        if result.get("atomic_units"):
            for unit in result["atomic_units"]:
                # Create a simple bias distribution for atomic units (neutral by default)
                bias_dist = BiasDistributionResponse(
                    left=0.33,
                    right=0.33,
                    neutral=0.34,
                    dominant_bias="neutral",
                    bias_score=0.0
                )
                opinion_resp = AtomicOpinionResponse(
                    text=unit["statement"],
                    opinion_type=unit["type"],
                    bias_probabilities=bias_dist,
                    original_sentence=unit.get("original_sentence"),
                    confidence=unit.get("confidence"),
                    reasoning=unit.get("reasoning")
                )
                atomic_opinions.append(opinion_resp)
                if unit["type"] == "fact":
                    facts.append(opinion_resp)
                else:
                    opinions.append(opinion_resp)
        
        # Get overall bias from result
        overall_bias = None
        if result.get("bias_analysis") and result["bias_analysis"].get("distribution"):
            dist = result["bias_analysis"]["distribution"]
            overall_bias = BiasDistributionResponse(
                left=dist["left"],
                right=dist["right"],
                neutral=dist["neutral"],
                dominant_bias=result["bias_analysis"].get("dominant_bias", "neutral"),
                bias_score=dist["left"] * -1.0 + dist["right"] * 1.0
            )
        
        return ExtractOpinionsResponse(
            url=result["url"],
            title=result["article"].get("title"),
            atomic_opinions=atomic_opinions,
            facts=facts,
            opinions=opinions,
            overall_bias_distribution=overall_bias,
            text_length=result["article"].get("text_length", 0),
            truncated=False
        )
        
    except Exception as e:
        logger.error(f"Failed to analyze URL {request.url}: {e}", exc_info=True)
        return ExtractOpinionsResponse(
            url=request.url,
            title=None,
            atomic_opinions=[],
            facts=[],
            opinions=[],
            text_length=0,
            truncated=False,
            error="analysis_failed",
            error_message=str(e)
        )


@app.get("/api/v1/web-opinion/default-shots", response_model=DefaultShotsResponse)
async def get_default_shots(
    api_key: str = Depends(verify_api_key)
):
    """
    Get default few-shot examples from the server.
    
    Returns the default few-shot examples used for atomization and bias scoring.
    Users can use these as a reference or modify them for custom shots.
    
    Args:
        api_key: API key for authentication
        
    Returns:
        DefaultShotsResponse with atomizer_shots and scorer_shots
    """
    logger.info("Getting default few-shot examples")
    
    try:
        engine = WebOpinionEngine(
            proxy=settings.openai_proxy if settings.openai_proxy else None
        )
        
        atomizer_shots = engine.get_default_atomizer_shots()
        scorer_shots = engine.get_default_scorer_shots()
        
        return DefaultShotsResponse(
            atomizer_shots=atomizer_shots,
            scorer_shots=scorer_shots
        )
        
    except Exception as e:
        logger.error(f"Failed to get default shots: {e}", exc_info=True)
        return DefaultShotsResponse(
            atomizer_shots=[],
            scorer_shots=[]
        )


@app.post("/api/v1/web-opinion/bias-score", response_model=BiasScoreResponse)
async def get_overall_bias_score(
    request: BiasScoreRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Get overall bias score from URL.
    
    This endpoint provides a simplified API that returns only the overall
    bias distribution for a URL, without detailed opinion breakdowns.
    
    Input: URL only
    Output: Overall bias distribution (left, right, neutral probabilities)
    
    Args:
        request: BiasScoreRequest with URL
        api_key: API key for authentication
        
    Returns:
        BiasScoreResponse with overall bias score
    """
    logger.info(f"Getting bias score for URL: {request.url}")
    
    try:
        execution_mode = request.execution_mode or settings.default_execution_mode
        
        analyzer = WebOpinionAnalyzer(
            execution_mode=execution_mode,
            proxy=settings.openai_proxy if settings.openai_proxy else None
        )
        
        result = analyzer.extract_and_analyze(request.url)
        
        # Check for errors
        if result.extraction_metadata and "error" in result.extraction_metadata:
            return BiasScoreResponse(
                url=request.url,
                overall_bias_distribution=None,
                opinions_count=0,
                facts_count=0,
                error=result.extraction_metadata["error"],
                error_message=result.extraction_metadata.get("error_message", "Unknown error")
            )
        
        overall_bias = None
        if result.overall_bias_distribution:
            overall_bias = _convert_bias_distribution(result.overall_bias_distribution)
        
        return BiasScoreResponse(
            url=request.url,
            overall_bias_distribution=overall_bias,
            opinions_count=len(result.opinions),
            facts_count=len(result.facts)
        )
        
    except Exception as e:
        logger.error(f"Failed to get bias score for {request.url}: {e}", exc_info=True)
        return BiasScoreResponse(
            url=request.url,
            overall_bias_distribution=None,
            opinions_count=0,
            facts_count=0,
            error="analysis_failed",
            error_message=str(e)
        )
