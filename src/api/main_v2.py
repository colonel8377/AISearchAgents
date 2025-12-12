"""FastAPI application for AI Search Agents Platform - Optimized Version."""

from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings

from ..config.settings import settings
from ..memory.factory import VectorStoreFactory
from ..agents.nudge_collapse.agent import NudgeCollapseAgent
from ..agents.summarizer.agent import SummarizerAgent
from ..agents.bot_creator.agent import BotCreatorAgent
from ..agents.manager import AgentManager, AgentType
from .auth import verify_api_key


# Pydantic models for request/response
class CreateAgentRequest(BaseModel):
    """Request model for creating a new agent instance."""
    agent_type: str = Field(..., description="Type of agent: 'nudge_collapse', 'summarizer', or 'bot_creator'")
    agent_id: Optional[str] = Field(default=None, description="Custom agent ID (auto-generated if not provided)")
    use_memory: bool = Field(default=False, description="Whether to use vector memory for this agent")


class AgentIdResponse(BaseModel):
    """Response model for agent creation."""
    agent_id: str
    agent_type: str
    status: str
    message: str


class GenerateTurnRequest(BaseModel):
    """Request model for generating a turn (nudge_collapse agent)."""
    user_query: str = Field(..., description="User's query or input")
    search_summary: str = Field(default="", description="Summary from search engine")
    search_urls: Optional[List[str]] = Field(default=None, description="URLs from search results")


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


class SummaryResponse(BaseModel):
    """Response model for summary."""
    summary: str
    conversation_length: int
    original_length: int
    truncated: bool
    metadata: Dict[str, Any]


class CreateBotRequest(BaseModel):
    """Request model for creating a bot."""
    persona_prompt: str = Field(..., description="Persona prompt corpus for the bot")
    bot_name: Optional[str] = Field(default=None, description="Optional name for the bot")


class BotCreationResponse(BaseModel):
    """Response model for bot creation."""
    bot_id: str
    bot_name: str
    status: str
    persona_prompt: str
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
            "Per-agent memory management"
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
            "summarizer": "/api/v1/agents/{agent_id}/summarizer/*",
            "bot_creator": "/api/v1/agents/{agent_id}/bot-creator/*"
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
    try:
        # Validate agent type
        if request.agent_type not in [e.value for e in AgentType]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid agent type. Must be one of: {[e.value for e in AgentType]}"
            )
        
        # Create vector store if memory is enabled
        vector_store = None
        if request.use_memory:
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
            elif settings.vector_store_type == "postgres":
                vector_store_kwargs = {
                    "connection_string": f"postgresql://{settings.postgres_user}:{settings.postgres_password}@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}",
                    "collection_name": f"agent_memory_{request.agent_id or 'auto'}"
                }
            else:  # chroma
                vector_store_kwargs = {
                    "persist_directory": settings.chroma_persist_directory,
                    "collection_name": f"agent_memory_{request.agent_id or 'auto'}"
                }
            
            vector_store = VectorStoreFactory.create_vector_store(
                store_type=settings.vector_store_type,
                embeddings=embeddings,
                **vector_store_kwargs
            )
        
        # Create agent instance
        if request.agent_type == AgentType.NUDGE_COLLAPSE:
            agent_instance = NudgeCollapseAgent(
                model_name=settings.openai_model,
                api_key=settings.openai_api_key,
                api_base=settings.openai_api_base,
                temperature=settings.agent_temperature,
                vector_store=vector_store
            )
        elif request.agent_type == AgentType.SUMMARIZER:
            agent_instance = SummarizerAgent(
                model_name=settings.openai_model,
                api_key=settings.openai_api_key,
                api_base=settings.openai_api_base,
                temperature=settings.agent_temperature,
                vector_store=vector_store
            )
        elif request.agent_type == AgentType.BOT_CREATOR:
            agent_instance = BotCreatorAgent(
                model_name=settings.openai_model,
                api_key=settings.openai_api_key,
                api_base=settings.openai_api_base,
                temperature=settings.agent_temperature,
                vector_store=vector_store
            )
        else:
            raise ValueError(f"Unsupported agent type: {request.agent_type}")
        
        # Register agent
        agent_id = agent_manager.create_agent(
            agent_instance=agent_instance,
            agent_type=AgentType(request.agent_type),
            agent_id=request.agent_id
        )
        
        return AgentIdResponse(
            agent_id=agent_id,
            agent_type=request.agent_type,
            status="created",
            message=f"Agent '{agent_id}' of type '{request.agent_type}' created successfully"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
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
            search_urls=request.search_urls
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return TurnResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
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
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


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
        result = agent.summarize_conversation(request.conversation_records)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return SummaryResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
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
        raise HTTPException(status_code=500, detail=f"Failed to get summary history: {str(e)}")


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
        result = agent.create_bot(
            persona_prompt=request.persona_prompt,
            bot_name=request.bot_name
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return BotCreationResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
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
        raise HTTPException(status_code=500, detail=f"Failed to list bots: {str(e)}")
