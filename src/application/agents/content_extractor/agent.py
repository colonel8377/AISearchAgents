"""Content Extractor Agent for academic content extraction.

This agent processes web content to extract the main title and body text,
focusing on academic/informational content while excluding navigation,
advertisements, and other non-content elements.
"""

import re
from typing import Optional, Union

import httpx
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from src.application.agents.content_extractor.model import ContentExtractionResult
from src.application.few_shots.content_extractor.few_shots import CONTENT_EXTRACTOR_FEW_SHOTS
from src.infrastructure.storage.persistence import get_database
from src.shared.cache import cached
from src.shared.config.settings import settings
from src.shared.constant.enums import CoTMode
from src.shared.llm.llm_manager import llm_manager
from src.shared.parsers.html_extractor import HTMLExtractor
from src.shared.utils.logger import get_logger

logger = get_logger(__name__)


class ContentExtractorAgent(HTMLExtractor):
    """
    Content Extractor Agent that processes web content to extract
    the main title and body text for academic content analysis.

    Features:
    - Robust HTML extraction with BeautifulSoup
    - Cleaning of non-content tags (script, style, nav, footer, iframe)
    - Network error handling (timeouts, 404s)
    - Academic content focus (main title + main body)
    """

    # Class variable for caching custom few shots (optional performance optimization)
    _custom_few_shots_cache: Optional[str] = None

    SYSTEM_PROMPT = """You are a Precise Web Data Auditor specializing in Academic Content Extraction.

Your task is to process the provided raw data of the URL and distinguish and extract the primary academic/informational substance.

CRITICAL INSTRUCTIONS:

1. Identify and extract the Main Title: This should be the core headline of the article/report.

2. Identify and extract the Main Body: Focus exclusively on the textual substance. Strip away navigation menus, footer links, sidebars, advertisement text, and recommended reading links.

3. Preserve Paragraph Structure: Maintain the logical separation of paragraphs to ensure context is not lost.

4. Verification: If multiple text blocks exist, prioritize the one with the highest information density and coherent semantic flow.

CONSTRAINT: Do not summarize. Do not alter any words. Provide a clean, verbatim extraction.

OUTPUT FORMAT:
- TITLE: [Extracted Title]
- MAIN BODY: [Extracted Full Text]"""

    SYSTEM_PROMPT_COT = """You are a Precise Web Data Auditor specializing in Academic Content Extraction.

Your task is to process the provided raw data of the URL and distinguish and extract the primary academic/informational substance.

CRITICAL INSTRUCTIONS:

1. Identify and extract the Main Title: This should be the core headline of the article/report.

2. Identify and extract the Main Body: Focus exclusively on the textual substance. Strip away navigation menus, footer links, sidebars, advertisement text, and recommended reading links.

3. Preserve Paragraph Structure: Maintain the logical separation of paragraphs to ensure context is not lost.

4. Verification: If multiple text blocks exist, prioritize the one with the highest information density and coherent semantic flow.

Chain of Thought Reasoning Process:
1. First, analyze the HTML structure to identify potential content areas
2. Evaluate information density of different text blocks
3. Check semantic coherence and academic relevance
4. Verify that extracted content maintains logical flow
5. Ensure no alteration of original wording

CONSTRAINT: Do not summarize. Do not alter any words. Provide a clean, verbatim extraction.

OUTPUT FORMAT:
- TITLE: [Extracted Title]
- MAIN BODY: [Extracted Full Text]"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.1,
        request_timeout: float = 30.0,
        execution_mode: Optional[str] = None
    ):
        """
        Initialize the ContentExtractorAgent.

        Args:
            model_name: Name of the LLM model to use (defaults to settings.openai_model)
            api_key: OpenAI API key or compatible API key
            api_base: Base URL for the API
            temperature: Temperature for LLM responses (lower for more consistent extraction)
            request_timeout: Timeout for HTTP requests in seconds
            execution_mode: Execution mode (\'chain_online\' for CoT, \'no_chain\' for direct)
        """
        # Initialize parent HTMLExtractor
        super().__init__(request_timeout=request_timeout)

        model_name = model_name or settings.openai_model
        logger.info(f"Initializing ContentExtractorAgent: model={model_name}, temperature={temperature}")

        # Use shared HTTP client for LLM
        http_client = llm_manager.get_http_client()

        self.llm = ChatOpenAI(
            model_name=model_name,
            api_key=api_key or settings.openai_api_key,
            base_url=api_base or settings.openai_api_base,
            temperature=temperature,
            max_retries=settings.openai_max_retries,
            timeout=settings.openai_timeout,
            http_client=http_client
        )

        self.execution_mode = execution_mode or "no_chain"

        # Initialize custom few shots
        self._custom_few_shots = None
        if settings.enable_persistence:
            try:
                database = get_database()
                self._custom_few_shots = database.load_custom_few_shots("content_extractor")
            except Exception as e:
                logger.warning(f"Failed to load custom few shots: {e}")

        logger.debug(f"ContentExtractorAgent initialized successfully with execution_mode={self.execution_mode}")


    async def extract_from_url(self, url: str, use_llm: bool = False, use_cot: Union[CoTMode, bool] = CoTMode.NO_CHAIN, custom_few_shots: Optional[str] = None) -> ContentExtractionResult:
        """Async version of extract_from_url."""
        logger.info(f"Extracting content from URL: {url} (use_llm={use_llm})")

        try:
            html = self.fetch_html(url)
        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP error fetching {url}: {e.response.status_code} - {e.response.reason_phrase}")
            return ContentExtractionResult(
                url=url,
                title=None,
                main_body="",
                text_length=0,
                truncated=False,
                extraction_metadata={
                    "error": "http_error",
                    "status_code": e.response.status_code,
                    "reason": e.response.reason_phrase
                }
            )
        except (httpx.TimeoutException, httpx.RequestError) as e:
            logger.warning(f"Network error fetching {url}: {type(e).__name__}: {e}")
            return ContentExtractionResult(
                url=url,
                title=None,
                main_body="",
                text_length=0,
                truncated=False,
                extraction_metadata={
                    "error": "network_error",
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )

        main_body, title = self.clean_html(html)

        if not main_body or len(main_body.strip()) < 50:
            logger.warning(f"Insufficient content extracted from {url}")
            return ContentExtractionResult(
                url=url,
                title=title,
                main_body="",
                text_length=0,
                truncated=False,
                extraction_metadata={"error": "insufficient_content"}
            )

        if use_llm:
            logger.info("Using LLM for content refinement (this consumes tokens)")
            title, main_body = await self._extract_with_llm(main_body, use_cot=use_cot, custom_few_shots=custom_few_shots)

        main_body, was_truncated = self.truncate_text(main_body)

        result = ContentExtractionResult(
            url=url,
            title=title,
            main_body=main_body,
            text_length=len(main_body),
            truncated=was_truncated,
            extraction_metadata={
                "model": self.llm.model_name if use_llm else None,
                "temperature": self.llm.temperature if use_llm else None,
                "use_llm": use_llm,
                "extraction_method": "llm_refinement" if use_llm else "html_parsing_direct"
            }
        )

        logger.info(f"Content extraction complete: title='{title[:50] if title else None}...', body_length={len(main_body)}, method={'LLM' if use_llm else 'HTML parsing'}")
        return result

    @cached()
    async def _extract_with_llm(self, text: str, use_cot: Union[CoTMode, bool] = CoTMode.NO_CHAIN, custom_few_shots: Optional[str] = None) -> tuple[Optional[str], str]:
        """Async version of _extract_with_llm."""
        if isinstance(use_cot, CoTMode):
            effective_cot = use_cot.value in ("chain_local", "chain_online")
        else:
            effective_cot = bool(use_cot)

        logger.warning(f"Using LLM for content extraction (consuming tokens): text_length={len(text)}, CoT={effective_cot}")

        if len(text) > 8000:
            logger.warning(f"Truncating input text from {len(text)} to 8000 chars to save tokens")
            text = text[:8000] + "...[TRUNCATED]"

        if effective_cot:
            system_prompt = self.SYSTEM_PROMPT_COT
        else:
            system_prompt = self.SYSTEM_PROMPT

        if custom_few_shots:
            system_prompt = f"{custom_few_shots}\n\n{system_prompt}"
        elif self._custom_few_shots:
            system_prompt = f"{self._custom_few_shots}\n\n{system_prompt}"

        user_message = f"""Please extract the main title and body content from the following text:

{text}

Return your extraction in the exact format specified:
- TITLE: [Extracted Title]
- MAIN BODY: [Extracted Full Text]"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]

        try:
            response = await self.llm.ainvoke(messages)
            response_text = response.content

            title_match = re.search(r'- TITLE:\s*(.+?)(?=\n-|$)', response_text, re.DOTALL)
            body_match = re.search(r'- MAIN BODY:\s*(.+?)(?=\n-|$)', response_text, re.DOTALL)

            title = title_match.group(1).strip() if title_match else None
            main_body = body_match.group(1).strip() if body_match else text

            logger.info(f"LLM extraction successful: extracted_title='{title[:30] if title else None}...', output_length={len(main_body)}")
            return title, main_body

        except Exception as e:
            logger.error(f"LLM extraction failed: {e}", exc_info=True)
            return None, text

    async def extract_from_text(self, text: str, url: Optional[str] = None, title: Optional[str] = None, use_llm: bool = True, use_cot: Union[CoTMode, bool] = CoTMode.NO_CHAIN, custom_few_shots: Optional[str] = None) -> ContentExtractionResult:
        """
        Extract content from plain text (assumes text is already cleaned) - async version.

        Args:
            text: Plain text content
            url: Optional source URL for metadata
            title: Optional title (if not provided, LLM will try to extract it)
            use_llm: Whether to use LLM for title extraction and content refinement
            use_cot: Whether to use Chain of Thought reasoning (only when use_llm=True)
            custom_few_shots: Optional custom few-shot examples (only when use_llm=True)

        Returns:
            ContentExtractionResult with extracted title and body
        """
        logger.info(f"Extracting content from text ({len(text)} chars)")

        # Truncate if necessary
        text, was_truncated = self.truncate_text(text)

        # Use LLM for refinement if requested
        if use_llm:
            extracted_title, main_body = await self._extract_with_llm(text, use_cot=use_cot, custom_few_shots=custom_few_shots)
            # Use provided title if LLM didn't find one
            final_title = extracted_title or title
        else:
            final_title = title
            main_body = text

        result = ContentExtractionResult(
            url=url,
            title=final_title,
            main_body=main_body,
            text_length=len(main_body),
            truncated=was_truncated,
            extraction_metadata={
                "model": self.llm.model_name if use_llm else None,
                "temperature": self.llm.temperature if use_llm else None,
                "use_llm": use_llm,
                "extraction_method": "llm" if use_llm else "direct"
            }
        )

        logger.info(f"Text content extraction complete: title='{final_title[:50]}...', body_length={len(main_body)}")

        return result

    @staticmethod
    def get_default_few_shots() -> str:
        """
        Get default few-shot examples for content extraction.

        Returns:
            String containing few-shot examples
        """
        return CONTENT_EXTRACTOR_FEW_SHOTS

    @classmethod
    def set_custom_few_shots(cls, custom_few_shots: Optional[str] = None) -> None:
        """
        Set custom few-shot examples for content extraction.

        Args:
            custom_few_shots: Custom few-shot examples string. If None, clears custom few shots.
        """
        if settings.enable_persistence:
            database = get_database()
            success = database.save_custom_few_shots("content_extractor", custom_few_shots)
            if success:
                cls._custom_few_shots_cache = custom_few_shots  # Update cache
                logger.info(f"Custom few shots saved for ContentExtractorAgent: {custom_few_shots is not None}")
            else:
                logger.warning("Failed to save custom few shots to persistence")
        else:
            cls._custom_few_shots_cache = custom_few_shots
            logger.info(f"Custom few shots set for ContentExtractorAgent (no persistence): {custom_few_shots is not None}")

    @classmethod
    def get_custom_few_shots(cls) -> Optional[str]:
        """
        Get currently set custom few-shot examples.

        Returns:
            Custom few-shot examples string or None if not set
        """
        if settings.enable_persistence:
            database = get_database()
            few_shots = database.load_custom_few_shots("content_extractor")
            # Update cache
            if isinstance(few_shots, str) or few_shots is None:
                cls._custom_few_shots_cache = few_shots
            return few_shots
        else:
            return cls._custom_few_shots_cache

    @classmethod
    def get_effective_few_shots(cls) -> str:
        """
        Get effective few-shot examples (custom if set, otherwise default).

        Returns:
            Effective few-shot examples string
        """
        custom_few_shots = cls.get_custom_few_shots()
        return custom_few_shots if custom_few_shots is not None else cls.get_default_few_shots()

    def reset(self) -> None:
        """Reset the agent to initial state."""
        logger.info("Resetting ContentExtractorAgent state")
        # No state to reset for this agent
        logger.debug("Agent reset complete")
