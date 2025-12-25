"""Multi-Agent Debate System module."""

from .schemas import (
    PersonaConfig,
    AgentMetadata,
    InitRequest,
    InteractRequest,
    VoteResponse
)
from .service import (
    DebateService,
    DebateStatistics,
    PersonaGenerator,
    generate_default_personas,
    generate_system_prompt,
    get_few_shot_example
)

__all__ = [
    # Schemas
    "PersonaConfig",
    "AgentMetadata",
    "InitRequest",
    "InteractRequest",
    "VoteResponse",
    # Service
    "DebateService",
    "DebateStatistics",
    "PersonaGenerator",
    # Functions
    "generate_default_personas",
    "generate_system_prompt",
    "get_few_shot_example",
]
