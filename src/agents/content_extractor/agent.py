"""Content Extractor Agent for academic content extraction.

This agent processes web content to extract the main title and body text,
focusing on academic/informational content while excluding navigation,
advertisements, and other non-content elements.
"""

import re
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

import httpx

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from ..html_extractor import HTMLExtractor
from ...config.settings import settings
from ...utils.llm_client import llm_manager
from ...utils.logger import get_logger
from ...utils.smart_memory import SmartMemory
from ...utils.agent_cache import cached

logger = get_logger(__name__)


@dataclass
class ContentExtractionResult:
    """Result of content extraction."""
    url: Optional[str] = None
    title: Optional[str] = None
    main_body: Optional[str] = None
    text_length: int = 0
    truncated: bool = False
    extraction_metadata: Optional[Dict[str, Any]] = None


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

    # Class variable to store custom few shots (persistent across instances)
    _custom_few_shots: Optional[str] = None

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
        proxy: Optional[str] = None,
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
            proxy: Optional HTTP proxy for API requests
            request_timeout: Timeout for HTTP requests in seconds
            execution_mode: Execution mode (\'chain_online\' for CoT, \'no_chain\' for direct)
        """
        # Initialize parent HTMLExtractor
        super().__init__(request_timeout=request_timeout)

        model_name = model_name or settings.openai_model
        logger.info(f"Initializing ContentExtractorAgent: model={model_name}, temperature={temperature}")

        # Use shared HTTP client for LLM
        http_client = llm_manager.get_http_client(proxy=proxy)

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

        # Initialize smart memory if enabled
        self.smart_memory = SmartMemory(llm=self.llm, vector_store=None) if settings.smart_memory_enabled else None

        logger.debug(f"ContentExtractorAgent initialized successfully with execution_mode={self.execution_mode}")

    @cached()
    def _extract_with_llm(self, text: str, use_cot: bool = False, custom_few_shots: Optional[str] = None) -> Tuple[str, str]:
        """
        Use LLM to extract title and main body from cleaned text.

        WARNING: This method consumes significant tokens. Only use when HTML parsing
        is insufficient and you need LLM-level understanding for content refinement.

        Args:
            text: Cleaned text content (already extracted from HTML)
            use_cot: Whether to use Chain of Thought reasoning
            custom_few_shots: Optional custom few-shot examples

        Returns:
            Tuple of (title, main_body)
        """
        logger.warning(f"Using LLM for content extraction (consuming tokens): text_length={len(text)}, CoT={use_cot}")

        # Truncate input if too long to save tokens
        if len(text) > 8000:  # Keep under ~2000 tokens for input
            logger.warning(f"Truncating input text from {len(text)} to 8000 chars to save tokens")
            text = text[:8000] + "...[TRUNCATED]"

        # Select system prompt based on CoT mode
        if use_cot:
            system_prompt = self.SYSTEM_PROMPT_COT
        else:
            system_prompt = self.SYSTEM_PROMPT

        # Add few-shot examples
        if custom_few_shots:
            # Use explicitly provided custom few shots
            system_prompt = f"{custom_few_shots}\n\n{system_prompt}"
        elif self._custom_few_shots:
            # Use stored custom few shots
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
            response = self.llm.invoke(messages)
            response_text = response.content

            # Parse the response
            title_match = re.search(r'- TITLE:\s*(.+?)(?=\n-|$)', response_text, re.DOTALL)
            body_match = re.search(r'- MAIN BODY:\s*(.+?)(?=\n-|$|$)', response_text, re.DOTALL)

            title = title_match.group(1).strip() if title_match else None
            main_body = body_match.group(1).strip() if body_match else text

            logger.info(f"LLM extraction successful: extracted_title='{title[:30] if title else None}...', output_length={len(main_body)}")
            return title, main_body

        except Exception as e:
            logger.error(f"LLM extraction failed: {e}", exc_info=True)
            # Fallback: return None for title and use full text as body
            return None, text

    def extract_from_url(self, url: str, use_llm: bool = False, use_cot: bool = False, custom_few_shots: Optional[str] = None) -> ContentExtractionResult:
        """
        Extract content from a URL.

        This method first uses HTML parsing to extract title and main body directly from the DOM structure,
        which is much more token-efficient than sending raw HTML to LLM. LLM is only used for refinement
        when explicitly requested.

        Args:
            url: The URL to process
            use_llm: Whether to use LLM for additional content refinement (use sparingly to save tokens)
            use_cot: Whether to use Chain of Thought reasoning (only when use_llm=True)
            custom_few_shots: Optional custom few-shot examples (only when use_llm=True)

        Returns:
            ContentExtractionResult with extracted title and body. If extraction fails,
            returns a result with error metadata instead of raising exceptions.
            httpx.TimeoutException: If the request times out
            httpx.HTTPStatusError: If the response has an error status
            httpx.RequestError: If there\'s a network error
        """
        logger.info(f"Extracting content from URL: {url} (use_llm={use_llm})")

        # Fetch HTML with error handling
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

        # Use HTML parsing to extract structured content (token-efficient)
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

        # Use LLM for refinement only if explicitly requested (saves tokens)
        if use_llm:
            logger.info("Using LLM for content refinement (this consumes tokens)")
            title, main_body = self._extract_with_llm(main_body, use_cot=use_cot, custom_few_shots=custom_few_shots)
        # Otherwise, use the already structured content from HTML parsing

        # Truncate if necessary
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

    def extract_from_html(self, html: str, url: Optional[str] = None, use_llm: bool = False, use_cot: bool = False, custom_few_shots: Optional[str] = None) -> ContentExtractionResult:
        """
        Extract content from raw HTML content.

        This method uses HTML parsing to extract title and main body directly from the DOM structure,
        which is much more token-efficient than sending raw HTML to LLM.

        Args:
            html: Raw HTML string
            url: Optional source URL for metadata
            use_llm: Whether to use LLM for additional content refinement (use sparingly to save tokens)
            use_cot: Whether to use Chain of Thought reasoning (only when use_llm=True)
            custom_few_shots: Optional custom few-shot examples (only when use_llm=True)

        Returns:
            ContentExtractionResult with extracted title and body
        """
        logger.info(f"Extracting content from HTML content (use_llm={use_llm})")

        # Use HTML parsing to extract structured content (token-efficient)
        main_body, title = self.clean_html(html)

        if not main_body or len(main_body.strip()) < 50:
            logger.warning("Insufficient content in HTML")
            return ContentExtractionResult(
                url=url,
                title=title,
                main_body="",
                text_length=0,
                truncated=False,
                extraction_metadata={"error": "insufficient_content"}
            )

        # Use LLM for refinement only if explicitly requested (saves tokens)
        if use_llm:
            logger.info("Using LLM for HTML content refinement (this consumes tokens)")
            title, main_body = self._extract_with_llm(main_body, use_cot=use_cot, custom_few_shots=custom_few_shots)
        # Otherwise, use the already structured content from HTML parsing

        # Truncate if necessary
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

        logger.info(f"HTML content extraction complete: title='{title[:50] if title else None}...', body_length={len(main_body)}, method={'LLM' if use_llm else 'HTML parsing'}")

        return result

    def extract_from_text(self, text: str, url: Optional[str] = None, title: Optional[str] = None, use_llm: bool = True, use_cot: bool = False, custom_few_shots: Optional[str] = None) -> ContentExtractionResult:
        """
        Extract content from plain text (assumes text is already cleaned).

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
            extracted_title, main_body = self._extract_with_llm(text, use_cot=use_cot, custom_few_shots=custom_few_shots)
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
    @staticmethod
    def get_default_few_shots() -> str:
        """
        Get default few-shot examples for content extraction.

        Returns:
            String containing few-shot examples
        """
        return """Example 1 - News Article:
Input: "[Complex HTML with navigation, ads, and article content]"

Output:
- TITLE: Biden Administration Announces New Climate Policy
- MAIN BODY: The Biden administration today announced a comprehensive new climate policy aimed at reducing carbon emissions by 50% by 2030. The policy includes investments in renewable energy and stricter regulations on fossil fuel industries.

Example 2 - Academic Paper Abstract:
Input: "[HTML with paper metadata and abstract]"

Output:
- TITLE: Machine Learning Approaches to Natural Language Processing
- MAIN BODY: This paper presents a comprehensive survey of machine learning techniques applied to natural language processing tasks. We review recent advances in transformer architectures, attention mechanisms, and their applications to text classification, named entity recognition, and machine translation."""

    @classmethod
    def set_custom_few_shots(cls, custom_few_shots: Optional[str] = None) -> None:
        """
        Set custom few-shot examples for content extraction.

        Args:
            custom_few_shots: Custom few-shot examples string. If None, clears custom few shots.
        """
        cls._custom_few_shots = custom_few_shots
        logger.info(f"Custom few shots set for ContentExtractorAgent: {custom_few_shots is not None}")

    @classmethod
    def get_custom_few_shots(cls) -> Optional[str]:
        """
        Get currently set custom few-shot examples.

        Returns:
            Custom few-shot examples string or None if not set
        """
        return cls._custom_few_shots

    @classmethod
    def get_effective_few_shots(cls) -> str:
        """
        Get effective few-shot examples (custom if set, otherwise default).

        Returns:
            Effective few-shot examples string
        """
        return cls._custom_few_shots if cls._custom_few_shots is not None else cls.get_default_few_shots()

    def reset(self) -> None:
        """Reset the agent to initial state."""
        logger.info("Resetting ContentExtractorAgent state")
        # No state to reset for this agent
        logger.debug("Agent reset complete")
