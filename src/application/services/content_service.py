"""Content Service - Handles content extraction business logic.

This service layer orchestrates content extraction operations by:
1. Coordinating with ContentExtractorAgent for agent-level operations
2. Coordinating with ClaimAtomizerAgent for claim atomization
3. Formatting results for API responses
4. Managing business logic flow
"""

import asyncio
from typing import Dict, Any, Optional

from src.application.agents.content_extractor.agent import ContentExtractorAgent
from src.application.services import BaseService
from src.application.services.agent_factory import AgentFactory
from src.infrastructure.repositories import AgentProtocol
from src.shared.config.settings import settings
from src.shared.constant.enums import CoTMode, AgentType
from src.shared.utils import get_logger

logger = get_logger(__name__)

# Fixed agent IDs for stateless agents
DEFAULT_CONTENT_EXTRACTOR_ID = "default_content_extractor"
DEFAULT_CLAIM_ATOMIZER_ID = "default_claim_atomizer"


class ContentService(BaseService):
    """Service for content extraction operations."""

    def _get_content_agent(self) -> AgentProtocol:
        """Get or create ContentExtractorAgent instance."""
        agent = self.agent_manager.get_agent(DEFAULT_CONTENT_EXTRACTOR_ID)
        if agent is None:
            agent = AgentFactory.create_agent(AgentType.CONTENT_EXTRACTOR)
            self.agent_manager.create_agent(agent, AgentType.CONTENT_EXTRACTOR, DEFAULT_CONTENT_EXTRACTOR_ID)
        return agent

    def _get_atomizer_agent(self):
        """Get or create ClaimAtomizerAgent instance."""
        agent = self.agent_manager.get_agent(DEFAULT_CLAIM_ATOMIZER_ID)
        if agent is None:
            agent = AgentFactory.create_agent(AgentType.CLAIM_ATOMIZER)
            self.agent_manager.create_agent(agent, AgentType.CLAIM_ATOMIZER, DEFAULT_CLAIM_ATOMIZER_ID)
        return agent

    async def extract_content(
        self,
        url: Optional[str] = None,
        text: Optional[str] = None,
        title: Optional[str] = None,
        use_llm: bool = False,
        use_cot: CoTMode = CoTMode.NO_CHAIN,
        custom_few_shots: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract content from URL or text.

        Args:
            url: URL to extract content from (if provided)
            text: Plain text content (if url is not provided)
            title: Optional title for text extraction
            use_llm: Whether to use LLM for content refinement
            use_cot: Chain of Thought mode
            custom_few_shots: Optional custom few-shot examples

        Returns:
            ContentExtractionResult as dictionary
        """
        agent = self._get_content_agent()

        if url:
            result = await agent.extract_from_url(
                url=url,
                use_llm=use_llm,
                use_cot=use_cot,
                custom_few_shots=custom_few_shots
            )
        else:
            result = await agent.extract_from_text(
                text=text,
                url=None,
                title=title,
                use_llm=use_llm,
                use_cot=use_cot,
                custom_few_shots=custom_few_shots
            )

        return result.__dict__

    async def atomize_claims(
        self,
        text: str,
        use_cot: CoTMode = CoTMode.NO_CHAIN,
        split_into_paragraphs: bool = False,
        use_few_shots: bool = True
    ) -> Dict[str, Any]:
        """
        Decompose text into atomic claims.

        Args:
            text: Text to atomize
            use_cot: Chain of Thought mode
            split_into_paragraphs: Whether to split text into paragraphs before atomization
            use_few_shots: Whether to use few-shot examples

        Returns:
            ClaimAtomizationResult as dictionary
        """
        agent = self._get_atomizer_agent()

        result = await agent.atomize_text(
            text=text,
            use_cot=use_cot,
            split_into_paragraphs=split_into_paragraphs,
            use_few_shots=use_few_shots
        )

        return {
            "atomic_claims": result.atomic_claims,
            "paragraphs": result.paragraphs,
            "original_text": result.original_text,
            "metadata": result.metadata
        }

