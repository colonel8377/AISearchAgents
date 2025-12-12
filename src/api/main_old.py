"""FastAPI application for AI Search Agents Platform."""

from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings

from ..config.settings import settings
from ..memory.factory import VectorStoreFactory
from ..agents.nudge_collapse.agent import NudgeCollapseAgent
from ..agents.summarizer.agent import SummarizerAgent
from ..agents.bot_creator.agent import BotCreatorAgent


# Pydantic models for request/response
class InitAgentRequest(BaseModel):
    """Request model for initializing an agent."""
    agent_type: str = Field(default="nudge_collapse", description="Type of agent to initialize")
    use_memory: bool = Field(default=False, description="Whether to use vector memory")
    

class GenerateTurnRequest(BaseModel):
    """Request model for generating a turn."""
    user_query: str = Field(..., description="User's query or input")
    search_summary: str = Field(default="", description="Summary from search engine")
    search_urls: Optional[List[str]] = Field(default=None, description="URLs from search results")


class ResetRequest(BaseModel):
    """Request model for resetting the agent."""
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


class StatusResponse(BaseModel):
    """Response model for status."""
    status: str
    message: str
    current_turn: Optional[int] = None


class SummarizeRequest(BaseModel):
    """Request model for summarizing conversations."""
    conversation_records: List[Dict[str, str]] = Field(..., description="List of conversation records to summarize")


class SummaryResponse(BaseModel):
    """Response model for summary."""
    summary: str
    conversation_length: int
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


# Initialize FastAPI app
app = FastAPI(
    title="AI Search Agents Platform",
    description="Platform for experimenting with AI search agents, including the Nudge-and-Collapse protocol",
    version="1.0.0"
)

# Global agent instance (can be NudgeCollapseAgent, SummarizerAgent, or BotCreatorAgent)
agent: Optional[Any] = None
agent_type: Optional[str] = None


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AI Search Agents Platform",
        "version": "1.0.0",
        "agents": {
            "nudge_collapse": "4-turn radicalization protocol agent",
            "summarizer": "Conversation summarization agent",
            "bot_creator": "Bot creation and configuration agent"
        },
        "endpoints": {
            "initialize": "/agent/initialize",
            "generate": "/agent/generate (nudge_collapse only)",
            "summarize": "/agent/summarize (summarizer only)",
            "create_bot": "/agent/create_bot (bot_creator only)",
            "list_bots": "/agent/list_bots (bot_creator only)",
            "reset": "/agent/reset",
            "history": "/agent/history (nudge_collapse only)",
            "status": "/agent/status"
        }
    }


@app.post("/agent/initialize", response_model=StatusResponse)
async def initialize_agent(request: InitAgentRequest):
    """
    Initialize an agent instance.
    
    Args:
        request: Initialization request with agent type and memory settings
        
    Returns:
        Status response
    """
    global agent, agent_type
    
    try:
        # Create vector store if memory is enabled
        vector_store = None
        if request.use_memory:
            embeddings = OpenAIEmbeddings(
                api_key=settings.openai_api_key,
                base_url=settings.openai_api_base
            )
            
            # Build kwargs for vector store based on type
            if settings.vector_store_type == "redis":
                # Build Redis URL conditionally based on password
                if settings.redis_password:
                    redis_url = f"redis://:{settings.redis_password}@{settings.redis_host}:{settings.redis_port}/{settings.redis_db}"
                else:
                    redis_url = f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}"
                
                vector_store_kwargs = {
                    "redis_url": redis_url,
                    "index_name": "agent_memory"
                }
            elif settings.vector_store_type == "postgres":
                vector_store_kwargs = {
                    "connection_string": f"postgresql://{settings.postgres_user}:{settings.postgres_password}@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}",
                    "collection_name": "agent_memory"
                }
            else:  # chroma
                vector_store_kwargs = {
                    "persist_directory": settings.chroma_persist_directory,
                    "collection_name": "agent_memory"
                }
            
            vector_store = VectorStoreFactory.create_vector_store(
                store_type=settings.vector_store_type,
                embeddings=embeddings,
                **vector_store_kwargs
            )
        
        # Create agent
        if request.agent_type == "nudge_collapse":
            agent = NudgeCollapseAgent(
                model_name=settings.openai_model,
                api_key=settings.openai_api_key,
                api_base=settings.openai_api_base,
                temperature=settings.agent_temperature,
                vector_store=vector_store
            )
            agent_type = "nudge_collapse"
            return StatusResponse(
                status="success",
                message=f"Agent '{request.agent_type}' initialized successfully",
                current_turn=agent.get_current_turn()
            )
        elif request.agent_type == "summarizer":
            agent = SummarizerAgent(
                model_name=settings.openai_model,
                api_key=settings.openai_api_key,
                api_base=settings.openai_api_base,
                temperature=settings.agent_temperature,
                vector_store=vector_store
            )
            agent_type = "summarizer"
            return StatusResponse(
                status="success",
                message=f"Agent '{request.agent_type}' initialized successfully"
            )
        elif request.agent_type == "bot_creator":
            agent = BotCreatorAgent(
                model_name=settings.openai_model,
                api_key=settings.openai_api_key,
                api_base=settings.openai_api_base,
                temperature=settings.agent_temperature,
                vector_store=vector_store
            )
            agent_type = "bot_creator"
            return StatusResponse(
                status="success",
                message=f"Agent '{request.agent_type}' initialized successfully"
            )
        else:
            raise ValueError(f"Unsupported agent type: {request.agent_type}")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize agent: {str(e)}")


@app.post("/agent/generate", response_model=TurnResponse)
async def generate_turn(request: GenerateTurnRequest):
    """
    Generate a response for the current turn.
    
    Args:
        request: Generation request with query and search context
        
    Returns:
        Turn response with agent's reply
    """
    global agent
    
    if agent is None:
        raise HTTPException(
            status_code=400,
            detail="Agent not initialized. Please call /agent/initialize first"
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


@app.post("/agent/reset", response_model=StatusResponse)
async def reset_agent(request: ResetRequest):
    """
    Reset the agent to initial state.
    
    Args:
        request: Reset request with options
        
    Returns:
        Status response
    """
    global agent
    
    if agent is None:
        raise HTTPException(
            status_code=400,
            detail="Agent not initialized. Please call /agent/initialize first"
        )
    
    try:
        agent.reset()
        
        # Note: Clearing vector memory would require additional implementation
        # as it depends on the specific vector store being used
        if request.clear_memory:
            # This is a placeholder - actual implementation would vary by vector store
            pass
        
        current_turn = getattr(agent, 'get_current_turn', lambda: None)()
        
        return StatusResponse(
            status="success",
            message="Agent reset successfully",
            current_turn=current_turn
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset agent: {str(e)}")


@app.get("/agent/history", response_model=ConversationHistoryResponse)
async def get_history():
    """
    Get the conversation history (for nudge_collapse agent).
    
    Returns:
        Conversation history response
    """
    global agent, agent_type
    
    if agent is None:
        raise HTTPException(
            status_code=400,
            detail="Agent not initialized. Please call /agent/initialize first"
        )
    
    if agent_type != "nudge_collapse":
        raise HTTPException(
            status_code=400,
            detail=f"This endpoint is only available for nudge_collapse agent, current agent is {agent_type}"
        )
    
    try:
        return ConversationHistoryResponse(
            current_turn=agent.get_current_turn(),
            max_turns=agent.max_turns,
            history=agent.get_conversation_history()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@app.get("/agent/status", response_model=StatusResponse)
async def get_status():
    """
    Get the current agent status.
    
    Returns:
        Status response
    """
    global agent
    
    if agent is None:
        return StatusResponse(
            status="not_initialized",
            message="Agent not initialized"
        )
    
    current_turn = getattr(agent, 'get_current_turn', lambda: None)()
    
    return StatusResponse(
        status="ready",
        message=f"Agent '{agent_type}' ready" + (f" at turn {current_turn}" if current_turn is not None else ""),
        current_turn=current_turn
    )


@app.post("/agent/summarize", response_model=SummaryResponse)
async def summarize_conversation(request: SummarizeRequest):
    """
    Summarize conversation records using the Summarizer agent.
    
    Args:
        request: Summarize request with conversation records
        
    Returns:
        Summary response
    """
    global agent, agent_type
    
    if agent is None:
        raise HTTPException(
            status_code=400,
            detail="Agent not initialized. Please call /agent/initialize with agent_type='summarizer' first"
        )
    
    if agent_type != "summarizer":
        raise HTTPException(
            status_code=400,
            detail=f"This endpoint requires a 'summarizer' agent, but current agent is '{agent_type}'"
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


@app.post("/agent/create_bot", response_model=BotCreationResponse)
async def create_bot(request: CreateBotRequest):
    """
    Create a new bot using the Bot Creator agent.
    
    Args:
        request: Bot creation request with persona prompt
        
    Returns:
        Bot creation response
    """
    global agent, agent_type
    
    if agent is None:
        raise HTTPException(
            status_code=400,
            detail="Agent not initialized. Please call /agent/initialize with agent_type='bot_creator' first"
        )
    
    if agent_type != "bot_creator":
        raise HTTPException(
            status_code=400,
            detail=f"This endpoint requires a 'bot_creator' agent, but current agent is '{agent_type}'"
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


@app.get("/agent/list_bots")
async def list_bots():
    """
    List all bots created by the Bot Creator agent.
    
    Returns:
        List of bot configurations
    """
    global agent, agent_type
    
    if agent is None:
        raise HTTPException(
            status_code=400,
            detail="Agent not initialized. Please call /agent/initialize with agent_type='bot_creator' first"
        )
    
    if agent_type != "bot_creator":
        raise HTTPException(
            status_code=400,
            detail=f"This endpoint requires a 'bot_creator' agent, but current agent is '{agent_type}'"
        )
    
    try:
        bots = agent.list_bots()
        return {"bots": bots}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list bots: {str(e)}")
