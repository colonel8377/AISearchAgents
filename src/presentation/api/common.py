"""Common dependencies and utility functions for API routes."""

from fastapi import Depends, HTTPException
from .auth import verify_api_key
from ...application.agents.manager import AgentManager
from ...application.agents.bot_creator.bot_manager import BotManager
from ...infrastructure.repositories import AgentRepository
from ...application.services import (
    AgentService,
    SummarizerService,
    NudgeCollapseService,
    SystemService
)
from ...shared.utils.logger import get_logger

logger = get_logger(__name__)

# Shared service instances
# Note: These are module-level singletons to maintain state across requests
agent_manager = AgentManager()
bot_manager = BotManager()

# Repository layer instances
agent_repository = AgentRepository(agent_manager)

# Service layer instances (CQRS Controller layer)
agent_service = AgentService(agent_repository)
summarizer_service = SummarizerService(agent_manager)
nudge_collapse_service = NudgeCollapseService(agent_manager)
system_service = SystemService()

# debate_service is imported where needed to avoid circular dependencies


# Common dependency for API key verification
def get_api_key(api_key: str = Depends(verify_api_key)) -> str:
    """
    Common dependency for API key verification.
    
    This dependency wraps verify_api_key and can be used in route handlers.
    The actual verification logic is handled by verify_api_key.
    
    Args:
        api_key: Verified API key from verify_api_key dependency
        
    Returns:
        The verified API key (or "no-auth-required" if auth is disabled)
    """
    return api_key


def handle_exception(e: Exception, operation: str) -> HTTPException:
    """Common exception handler that logs and converts to HTTPException."""
    if isinstance(e, HTTPException):
        return e
    logger.error(f"{operation} failed: {e}", exc_info=True)
    return HTTPException(status_code=500, detail=f"{operation} failed: {str(e)}")


def split_text_into_paragraphs(text: str):
    """
    Split text into paragraphs with robust paragraph detection.

    Args:
        text: The text to split

    Returns:
        List of ParagraphResponse objects with index, text, and text_length
    """
    from .schemas import ParagraphResponse
    import re

    if not text or not text.strip():
        return []

    # Split by double newlines (common paragraph separator)
    paragraphs = re.split(r'\n\s*\n', text.strip())

    # Filter out empty paragraphs and clean up
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # Filter out very short paragraphs (likely headers or separators)
    paragraphs = [p for p in paragraphs if len(p) > 10]

    # Convert to ParagraphResponse objects
    paragraph_responses = []
    for i, para_text in enumerate(paragraphs):
        paragraph_responses.append(ParagraphResponse(
            index=i,
            text=para_text,
            text_length=len(para_text)
        ))

    return paragraph_responses

