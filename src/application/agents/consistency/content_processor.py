"""Content processing module for consistency checking."""

import asyncio
from typing import Dict, Any, Optional

from src.application.agents.claim_atomizer.agent import ClaimAtomizerAgent
from src.application.agents.content_extractor.agent import ContentExtractorAgent
from src.shared.constant.enums import CoTMode
from src.shared.utils import get_logger

logger = get_logger(__name__)


class ContentProcessor:
    """Utility class for processing content (extraction and atomization)."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.1
    ):
        """
        Initialize the ContentProcessor.

        Args:
            model_name: Name of the LLM model to use
            api_key: OpenAI API key
            api_base: Base URL for the API
            temperature: Temperature for LLM responses
        """
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature

    def _create_content_agent(self) -> ContentExtractorAgent:
        """Create and return a ContentExtractorAgent instance."""
        return ContentExtractorAgent(
            model_name=self.model_name,
            api_key=self.api_key,
            api_base=self.api_base,
            temperature=self.temperature
        )

    def _create_atomizer_agent(self) -> ClaimAtomizerAgent:
        """Create and return a ClaimAtomizerAgent instance."""
        return ClaimAtomizerAgent(
            model_name=self.model_name,
            api_key=self.api_key,
            api_base=self.api_base,
            temperature=self.temperature
        )

    async def extract_and_atomize(
        self,
        url: str,
        summary: Optional[str] = None,
        use_cot_atomization: CoTMode = CoTMode.NO_CHAIN,
        split_url_into_paragraphs: bool = True,
        use_llm_extraction: bool = False
    ) -> Dict[str, Any]:
        """
        Extract content from URL and atomize both summary and URL content.

        Args:
            url: URL to extract content from
            summary: Optional summary text to atomize
            use_cot_atomization: Chain of Thought mode for atomization
            split_url_into_paragraphs: Whether to split URL content into paragraphs
            use_llm_extraction: Whether to use LLM for content extraction

        Returns:
            Dict with content_result, summary_atomization, and url_atomization
        """
        # Step 1: Extract content from URL
        content_agent = self._create_content_agent()
        content_result = await content_agent.extract_from_url(url, use_llm=use_llm_extraction)

        if not content_result.main_body:
            raise ValueError("Could not extract content from URL")

        # Step 2 & 3: Atomize summary and URL content concurrently
        summary_atomizer = self._create_atomizer_agent()
        url_atomizer = self._create_atomizer_agent()

        if summary:
            # Atomize summary and URL content concurrently for better performance
            summary_atomization, url_atomization = await asyncio.gather(
                summary_atomizer.atomize_text(
                    summary,
                    use_cot=use_cot_atomization,
                    split_into_paragraphs=False
                ),
                url_atomizer.atomize_text(
                    content_result.main_body,
                    use_cot=use_cot_atomization,
                    split_into_paragraphs=split_url_into_paragraphs
                )
            )
        else:
            # Only atomize URL content
            url_atomization = await url_atomizer.atomize_text(
                content_result.main_body,
                use_cot=use_cot_atomization,
                split_into_paragraphs=split_url_into_paragraphs
            )
            summary_atomization = None

        return {
            "content_result": content_result,
            "summary_atomization": summary_atomization,
            "url_atomization": url_atomization
        }

