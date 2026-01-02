"""Multi-Agent Debate System module."""

from .service import (
    DebateService,
    DebateStatistics,
    PersonaGenerator,
    generate_default_personas,
    generate_system_prompt,
    get_few_shot_example
)

__all__ = [
    # Service
    "DebateService",
    "DebateStatistics",
    "PersonaGenerator",
    # Functions
    "generate_default_personas",
    "generate_system_prompt",
    "get_few_shot_example",
]
