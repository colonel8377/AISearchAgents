"""Web Opinion Extractor Agent implementation.

This agent processes web content to extract atomic opinions, distinguish
facts from opinions, and calculate political bias scores.
"""

import json
import re
from typing import Optional, List, Dict, Any

import httpx
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ...utils.logger import get_logger
from ...config.settings import settings
from ...utils.llm_client import llm_manager
from .models import AtomicOpinion, OpinionExtractionResult
from .exceptions import NetworkError, ContentExtractionError

logger = get_logger(__name__)

# Tags to strip from HTML for cleaning
NON_CONTENT_TAGS = ["script", "style", "nav", "footer", "iframe", "noscript", "header", "aside"]

# Maximum text length before truncation (characters)
# Roughly 4 chars per token, targeting ~8000 tokens for context
MAX_TEXT_LENGTH = 32000


class WebOpinionExtractor:
    """
    Web Opinion Extractor Agent that processes web content to extract
    atomic opinions with fact/opinion classification and bias scoring.
    
    Features:
    - Robust HTML extraction with BeautifulSoup
    - Cleaning of non-content tags (script, style, nav, footer, iframe)
    - Network error handling (timeouts, 404s)
    - Fact vs Opinion separation using LLM
    - Atomic viewpoint extraction (breaking compound sentences)
    - Bias scoring (-1.0 to +1.0)
    - Context window handling (truncation)
    """
    
    SYSTEM_PROMPT = """You are an expert content analyst specializing in extracting and analyzing opinions from text.
Your task is to process the given text and extract ALL viewpoints as atomic opinions.

CRITICAL INSTRUCTIONS:

1. FACT vs OPINION SEPARATION:
   - FACTS are verifiable, objective statements (e.g., "The bill was passed on January 5th")
   - OPINIONS are subjective viewpoints, beliefs, or judgments (e.g., "The policy is harmful to workers")

2. ATOMIC OPINIONS (CRUCIAL):
   - Break down compound sentences into SEPARATE atomic records
   - Example: "I support the tax cut but oppose the trade war" becomes TWO atomic opinions:
     * "Support for the tax cut" (one atomic opinion)
     * "Opposition to the trade war" (another atomic opinion)
   - Each atomic opinion must express ONE and only ONE stance or viewpoint

3. BIAS SCORING:
   - Score each opinion from -1.0 to +1.0
   - -1.0 = Left/Progressive viewpoint
   - 0.0 = Neutral/Centrist
   - +1.0 = Right/Conservative viewpoint
   - Use decimal precision (e.g., -0.7, 0.3, 0.8)

OUTPUT FORMAT:
Return a JSON object with this exact structure:
{
  "atomic_opinions": [
    {
      "text": "The atomic opinion text",
      "opinion_type": "fact" or "opinion",
      "bias_score": <float between -1.0 and 1.0>,
      "original_sentence": "The original sentence this was extracted from",
      "confidence": <float between 0.0 and 1.0>
    }
  ]
}

Extract ALL viewpoints, even subtle ones. Be thorough but precise."""

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.3,
        proxy: Optional[str] = None,
        request_timeout: float = 30.0
    ):
        """
        Initialize the WebOpinionExtractor.
        
        Args:
            model_name: Name of the LLM model to use
            api_key: OpenAI API key or compatible API key
            api_base: Base URL for the API
            temperature: Temperature for LLM responses (lower for more consistent extraction)
            proxy: Optional HTTP proxy for API requests
            request_timeout: Timeout for HTTP requests in seconds
        """
        logger.info(f"Initializing WebOpinionExtractor: model={model_name}, temperature={temperature}")
        
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
        
        self.request_timeout = request_timeout
        self._extraction_history: List[OpinionExtractionResult] = []
        
        logger.debug("WebOpinionExtractor initialized successfully")
    
    def _fetch_html(self, url: str) -> str:
        """
        Fetch HTML content from a URL with error handling.
        
        Args:
            url: The URL to fetch
            
        Returns:
            HTML content as string
            
        Raises:
            NetworkError: If the request fails (timeout, 404, connection error)
        """
        logger.info(f"Fetching HTML from: {url}")
        
        try:
            with httpx.Client(timeout=self.request_timeout) as client:
                response = client.get(url, follow_redirects=True)
                response.raise_for_status()
                logger.debug(f"Successfully fetched {len(response.text)} characters from {url}")
                return response.text
                
        except httpx.TimeoutException as e:
            logger.error(f"Timeout fetching {url}: {e}")
            raise NetworkError(
                message=f"Request timed out after {self.request_timeout} seconds",
                url=url
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching {url}: {e}")
            raise NetworkError(
                message=f"HTTP error: {e.response.status_code}",
                url=url,
                status_code=e.response.status_code
            )
        except httpx.RequestError as e:
            logger.error(f"Request error fetching {url}: {e}")
            raise NetworkError(
                message=f"Request failed: {str(e)}",
                url=url
            )
    
    def _clean_html(self, html: str) -> tuple[str, Optional[str]]:
        """
        Parse and clean HTML content, removing non-content tags.
        
        Args:
            html: Raw HTML string
            
        Returns:
            Tuple of (cleaned_text, page_title)
        """
        logger.debug("Cleaning HTML content")
        
        soup = BeautifulSoup(html, "html.parser")
        
        # Extract title before cleaning
        title = None
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)
        
        # Remove non-content tags
        for tag_name in NON_CONTENT_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()
        
        # Extract text content
        text = soup.get_text(separator="\n", strip=True)
        
        # Clean up whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Reduce multiple newlines
        text = re.sub(r'[ \t]+', ' ', text)  # Reduce multiple spaces
        
        logger.debug(f"Cleaned text length: {len(text)} characters, title: {title}")
        
        return text, title
    
    def _truncate_text(self, text: str, max_length: int = MAX_TEXT_LENGTH) -> tuple[str, bool]:
        """
        Truncate text to fit within context window limits.
        
        Args:
            text: The text to truncate
            max_length: Maximum length in characters
            
        Returns:
            Tuple of (truncated_text, was_truncated)
        """
        if len(text) <= max_length:
            return text, False
        
        logger.warning(f"Text length ({len(text)}) exceeds max ({max_length}), truncating")
        
        # Truncate at a sentence boundary if possible
        truncated = text[:max_length]
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')
        
        # Use the later boundary to preserve more context
        boundary = max(last_period, last_newline)
        if boundary > max_length * 0.8:  # Only if we're not losing too much
            truncated = truncated[:boundary + 1]
        
        return truncated, True
    
    def _extract_opinions_with_llm(self, text: str) -> List[Dict[str, Any]]:
        """
        Use LLM to extract atomic opinions from text.
        
        Args:
            text: Cleaned text content
            
        Returns:
            List of opinion dictionaries
        """
        logger.debug(f"Extracting opinions from text ({len(text)} chars)")
        
        user_message = f"""Analyze the following text and extract all atomic opinions.
Remember to:
1. Separate facts from opinions
2. Break compound sentences into atomic viewpoints
3. Score each opinion's political bias from -1.0 (Left) to +1.0 (Right)

TEXT TO ANALYZE:
{text}

Return your analysis as a JSON object."""

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=user_message)
        ]
        
        try:
            response = self.llm.invoke(messages)
            response_text = response.content
            
            # Parse JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if not json_match:
                logger.warning("No JSON found in LLM response")
                return []
            
            result = json.loads(json_match.group())
            opinions = result.get("atomic_opinions", [])
            
            logger.info(f"Extracted {len(opinions)} atomic opinions")
            return opinions
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}", exc_info=True)
            raise
    
    def _create_atomic_opinion(self, data: Dict[str, Any]) -> AtomicOpinion:
        """
        Create an AtomicOpinion from extracted data with validation.
        
        Args:
            data: Dictionary with opinion data
            
        Returns:
            AtomicOpinion instance
        """
        # Ensure bias_score is a float within bounds
        bias_score = data.get("bias_score", 0.0)
        if isinstance(bias_score, str):
            try:
                bias_score = float(bias_score)
            except ValueError:
                bias_score = 0.0
        bias_score = max(-1.0, min(1.0, float(bias_score)))
        
        # Determine opinion type
        opinion_type = data.get("opinion_type", "opinion").lower()
        if opinion_type not in ("fact", "opinion"):
            opinion_type = "opinion"
        
        # Handle confidence
        confidence = data.get("confidence")
        if confidence is not None:
            try:
                confidence = max(0.0, min(1.0, float(confidence)))
            except (ValueError, TypeError):
                confidence = None
        
        return AtomicOpinion(
            text=str(data.get("text", "")),
            opinion_type=opinion_type,
            bias_score=bias_score,
            original_sentence=data.get("original_sentence"),
            confidence=confidence
        )
    
    def extract_from_url(self, url: str) -> OpinionExtractionResult:
        """
        Extract opinions from a URL.
        
        Args:
            url: The URL to process
            
        Returns:
            OpinionExtractionResult with extracted opinions
            
        Raises:
            NetworkError: If the URL cannot be fetched
            ContentExtractionError: If content cannot be extracted
        """
        logger.info(f"Extracting opinions from URL: {url}")
        
        # Fetch HTML
        html = self._fetch_html(url)
        
        # Clean and extract text
        text, title = self._clean_html(html)
        
        if not text or len(text.strip()) < 50:
            logger.warning(f"Insufficient content extracted from {url}")
            raise ContentExtractionError(
                message="Insufficient content extracted from page",
                url=url
            )
        
        # Process with LLM
        result = self.extract_from_text(text, url=url, title=title)
        
        return result
    
    def extract_from_html(
        self,
        html: str,
        url: Optional[str] = None
    ) -> OpinionExtractionResult:
        """
        Extract opinions from raw HTML content.
        
        Args:
            html: Raw HTML string
            url: Optional source URL for metadata
            
        Returns:
            OpinionExtractionResult with extracted opinions
        """
        logger.info("Extracting opinions from HTML content")
        
        # Clean and extract text
        text, title = self._clean_html(html)
        
        if not text or len(text.strip()) < 50:
            logger.warning("Insufficient content in HTML")
            return OpinionExtractionResult(
                url=url,
                title=title,
                atomic_opinions=[],
                facts=[],
                opinions=[],
                text_length=len(text) if text else 0,
                truncated=False,
                extraction_metadata={"error": "Insufficient content"}
            )
        
        return self.extract_from_text(text, url=url, title=title)
    
    def extract_from_text(
        self,
        text: str,
        url: Optional[str] = None,
        title: Optional[str] = None
    ) -> OpinionExtractionResult:
        """
        Extract opinions from plain text.
        
        Args:
            text: Plain text content
            url: Optional source URL for metadata
            title: Optional page title
            
        Returns:
            OpinionExtractionResult with extracted opinions
        """
        logger.info(f"Extracting opinions from text ({len(text)} chars)")
        
        # Truncate if necessary
        text, was_truncated = self._truncate_text(text)
        
        # Extract opinions using LLM
        raw_opinions = self._extract_opinions_with_llm(text)
        
        # Convert to AtomicOpinion objects
        atomic_opinions = []
        for data in raw_opinions:
            try:
                opinion = self._create_atomic_opinion(data)
                atomic_opinions.append(opinion)
            except Exception as e:
                logger.warning(f"Failed to create atomic opinion: {e}")
                continue
        
        # Separate facts and opinions
        facts = [op for op in atomic_opinions if op.opinion_type == "fact"]
        opinions = [op for op in atomic_opinions if op.opinion_type == "opinion"]
        
        # Calculate overall bias
        overall_bias = None
        if opinions:
            overall_bias = sum(op.bias_score for op in opinions) / len(opinions)
        
        result = OpinionExtractionResult(
            url=url,
            title=title,
            atomic_opinions=atomic_opinions,
            facts=facts,
            opinions=opinions,
            overall_bias_score=overall_bias,
            text_length=len(text),
            truncated=was_truncated,
            extraction_metadata={
                "model": self.llm.model_name,
                "temperature": self.llm.temperature,
                "total_extracted": len(atomic_opinions),
                "facts_count": len(facts),
                "opinions_count": len(opinions)
            }
        )
        
        # Store in history
        self._extraction_history.append(result)
        
        bias_str = f"{overall_bias:.2f}" if overall_bias is not None else "N/A"
        logger.info(
            f"Extraction complete: {len(facts)} facts, {len(opinions)} opinions, "
            f"overall bias: {bias_str}"
        )
        
        return result
    
    def get_extraction_history(self) -> List[OpinionExtractionResult]:
        """Get the history of all extractions performed."""
        return self._extraction_history
    
    def reset(self) -> None:
        """Reset the agent to initial state."""
        logger.info("Resetting WebOpinionExtractor state")
        self._extraction_history = []
        logger.debug("Agent reset complete")
