"""
Data models for Multi-Agent Debate System.

Incorporates concepts from ChatEval (Personas) and Adaptive Stability (Stopping Logic).
"""

from typing import List, Optional, Union, Literal, Dict, Any
from uuid import UUID

from pydantic import BaseModel, Field


class PersonaConfig(BaseModel):
    """Configuration for a debate agent persona."""
    name: str = Field(..., description="Name of the persona")
    description: str = Field(..., description="Description of the persona's characteristics")
    style: Optional[str] = Field(default=None, description="Optional style of the persona (e.g., 'Skeptical', 'Optimistic', 'Neutral') - auto-detected if not provided")
    corpus: Optional[List[str]] = Field(default=None, description="Optional corpus of statements specific to this persona's background and expertise")


class AgentMetadata(BaseModel):
    """Metadata for a debate agent."""
    agent_id: UUID = Field(..., description="Unique identifier for the agent")
    role_name: str = Field(..., description="Role name of the agent")
    system_prompt: str = Field(..., description="System prompt for the agent")
    few_shot_example: str = Field(..., description="Static example showing the expected JSON output format")
    corpus: Optional[List[str]] = Field(default=None, description="Optional corpus of statements specific to this agent's background and expertise")


class InitRequest(BaseModel):
    """Request model for initializing a debate session."""
    topic: str = Field(..., description="Topic of the debate")
    custom_personas: List[PersonaConfig] = Field(default_factory=list, description="Custom personas for agents")
    auto_agent_count: int = Field(default=0, description="Number of agents to auto-generate if custom_personas is empty")
    max_rounds: Optional[int] = Field(default=10, ge=1, le=50, description="Maximum number of debate rounds")
    context: str = Field(default="", description="Additional context for persona generation")
    corpus: Optional[List[str]] = Field(default=None, description="Optional user history statements for style detection and few-shot examples")


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


class AgentReasoning(BaseModel):
    """Response model for a single agent's reasoning in a round."""
    agent_id: UUID = Field(..., description="Agent identifier")
    statement: str = Field(..., description="Agent's position statement")
    reasoning: str = Field(..., description="Detailed reasoning behind the statement")
    timestamp: float = Field(..., description="Timestamp when reasoning was generated")
    word_count: int = Field(..., description="Word count of the combined statement and reasoning")


class AgentVote(BaseModel):
    """Response model for a single agent's vote in a round."""
    agent_id: UUID = Field(..., description="Agent identifier")
    verdict: Union[int, str] = Field(..., description="The agent's verdict/vote")
    reasoning: str = Field(..., description="Reasoning behind the verdict")
    timestamp: float = Field(..., description="Timestamp when vote was cast")
    word_count: int = Field(..., description="Word count of the reasoning")


class RoundReasoningResponse(BaseModel):
    """Response model for a complete reasoning round."""
    round_number: int = Field(..., description="Current round number")
    reasonings: List[AgentReasoning] = Field(..., description="All agents' reasoning for this round")
    timestamp: float = Field(..., description="Timestamp when round was completed")
    total_word_count: int = Field(..., description="Total word count across all reasonings")


class RoundVotingResponse(BaseModel):
    """Response model for a complete voting round."""
    round_number: int = Field(..., description="Current round number")
    votes: List[AgentVote] = Field(..., description="All agents' votes for this round")
    timestamp: float = Field(..., description="Timestamp when round was completed")
    total_word_count: int = Field(..., description="Total word count across all vote reasonings")
    statistics: Dict[str, Any] = Field(..., description="Round statistics including consensus level, distributions, etc.")
