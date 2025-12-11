"""FastAPI application for AI Search Agents Platform."""

from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings

from ..config.settings import settings
from ..memory.factory import VectorStoreFactory
from ..agents.nudge_collapse.agent import NudgeCollapseAgent


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


# Initialize FastAPI app
app = FastAPI(
    title="AI Search Agents Platform",
    description="Platform for experimenting with AI search agents, including the Nudge-and-Collapse protocol",
    version="1.0.0"
)

# Global agent instance
agent: Optional[NudgeCollapseAgent] = None


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AI Search Agents Platform",
        "version": "1.0.0",
        "endpoints": {
            "initialize": "/agent/initialize",
            "generate": "/agent/generate",
            "reset": "/agent/reset",
            "history": "/agent/history",
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
    global agent
    
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
                vector_store_kwargs = {
                    "redis_url": f"redis://:{settings.redis_password}@{settings.redis_host}:{settings.redis_port}/{settings.redis_db}",
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
        else:
            raise ValueError(f"Unsupported agent type: {request.agent_type}")
        
        return StatusResponse(
            status="success",
            message=f"Agent '{request.agent_type}' initialized successfully",
            current_turn=agent.get_current_turn()
        )
        
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
        
        return StatusResponse(
            status="success",
            message="Agent reset successfully",
            current_turn=agent.get_current_turn()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset agent: {str(e)}")


@app.get("/agent/history", response_model=ConversationHistoryResponse)
async def get_history():
    """
    Get the conversation history.
    
    Returns:
        Conversation history response
    """
    global agent
    
    if agent is None:
        raise HTTPException(
            status_code=400,
            detail="Agent not initialized. Please call /agent/initialize first"
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
    
    return StatusResponse(
        status="ready",
        message=f"Agent ready at turn {agent.get_current_turn()}",
        current_turn=agent.get_current_turn()
    )
