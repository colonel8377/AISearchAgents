"""
Data models for Multi-Agent Debate System.

Incorporates concepts from ChatEval (Personas) and Adaptive Stability (Stopping Logic).
"""

from typing import List, Optional, Union
from uuid import UUID
from pydantic import BaseModel, Field


class PersonaConfig(BaseModel):
    """Configuration for a debate agent persona."""
    name: str = Field(..., description="Name of the persona")
    description: str = Field(..., description="Description of the persona's characteristics")
    style: str = Field(..., description="Style of the persona (e.g., 'Skeptical', 'Optimistic', 'Neutral')")


class AgentMetadata(BaseModel):
    """Metadata for a debate agent."""
    agent_id: UUID = Field(..., description="Unique identifier for the agent")
    role_name: str = Field(..., description="Role name of the agent")
    system_prompt: str = Field(..., description="System prompt for the agent")
    few_shot_example: str = Field(..., description="Static example showing the expected JSON output format")


class InitRequest(BaseModel):
    """Request model for initializing a debate session."""
    topic: str = Field(..., description="Topic of the debate")
    custom_personas: List[PersonaConfig] = Field(default_factory=list, description="Custom personas for agents")
    auto_agent_count: int = Field(default=0, description="Number of agents to auto-generate if custom_personas is empty")


class InteractRequest(BaseModel):
    """Request model for agent interaction."""
    session_id: str = Field(..., description="Session identifier for the debate")
    agent_id: UUID = Field(..., description="Agent identifier")
    history_context: str = Field(..., description="String summary of what others said")


class VoteResponse(BaseModel):
    """Response model for agent vote."""
    agent_id: UUID = Field(..., description="Agent identifier")
    verdict: Union[int, str] = Field(..., description="The agent's verdict/vote")
    reasoning: str = Field(..., description="Reasoning behind the verdict")
