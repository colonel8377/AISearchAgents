"""Web Opinion Extractor Agent implementation.

This agent processes web content to extract atomic opinions, distinguish
facts from opinions, and calculate political bias scores.
"""

import json
import re
from typing import Optional, List, Dict, Any, Tuple

import httpx
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ...utils.logger import get_logger
from ...config.settings import settings
from ...utils.llm_client import llm_manager
from .models import AtomicOpinion, OpinionExtractionResult, BiasDistribution
from .exceptions import NetworkError, ContentExtractionError

logger = get_logger(__name__)

# Tags to strip from HTML for cleaning
NON_CONTENT_TAGS = [
    "script", "style", "nav", "footer", "iframe", "noscript", "header", "aside",
    "form", "button", "input", "select", "textarea", "label",  # Form elements
    "img", "figure", "figcaption", "picture", "video", "audio", "source", "track",  # Media
    "svg", "canvas", "map", "area",  # Graphics
    "menu", "menuitem",  # Menu elements
    "dialog", "details", "summary",  # Interactive elements
    "template", "slot",  # Template elements
]

# Tags that often contain navigation/non-content links
NAVIGATION_TAGS = ["ul", "ol", "li"]

# Common class/id patterns for main content areas
MAIN_CONTENT_SELECTORS = [
    "article",
    "[role='main']",
    "main",
    ".article-content",
    ".article-body",
    ".post-content",
    ".post-body",
    ".entry-content",
    ".content-body",
    ".story-body",
    ".news-body",
    "#article-body",
    "#main-content",
    "#content",
    ".main-content",
]

# Common class/id patterns for article titles
TITLE_SELECTORS = [
    "h1",
    ".article-title",
    ".post-title",
    ".entry-title",
    ".headline",
    "[itemprop='headline']",
]

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
    - Bias probability distribution (left, right, neutral)
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

3. BIAS PROBABILITY DISTRIBUTION:
   - For each opinion, provide a probability distribution across three categories:
     * "left": Probability of Left/Progressive bias (0.0 to 1.0)
     * "right": Probability of Right/Conservative bias (0.0 to 1.0)
     * "neutral": Probability of Neutral/Centrist stance (0.0 to 1.0)
   - The three probabilities MUST sum to 1.0
   - Example: {"left": 0.7, "right": 0.1, "neutral": 0.2} for a left-leaning opinion
   - Example: {"left": 0.1, "right": 0.8, "neutral": 0.1} for a right-leaning opinion
   - Example: {"left": 0.2, "right": 0.2, "neutral": 0.6} for a mostly neutral statement

OUTPUT FORMAT:
Return a JSON object with this exact structure:
{
  "atomic_opinions": [
    {
      "text": "The atomic opinion text",
      "opinion_type": "fact" or "opinion",
      "bias_probabilities": {
        "left": <float 0.0-1.0>,
        "right": <float 0.0-1.0>,
        "neutral": <float 0.0-1.0>
      },
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
    
    def _clean_html(self, html: str) -> Tuple[str, Optional[str]]:
        """
        Parse and clean HTML content, extracting only the main article text.
        
        This method focuses on extracting the main content body (like news article text)
        while excluding navigation, buttons, links, images, and other non-content elements.
        
        Args:
            html: Raw HTML string
            
        Returns:
            Tuple of (cleaned_text, page_title)
        """
        logger.debug("Cleaning HTML content")
        
        soup = BeautifulSoup(html, "html.parser")
        
        # Extract page title from <title> tag
        page_title = None
        title_tag = soup.find("title")
        if title_tag:
            page_title = title_tag.get_text(strip=True)
        
        # Try to find article title from common heading selectors
        article_title = None
        for selector in TITLE_SELECTORS:
            title_element = soup.select_one(selector)
            if title_element:
                article_title = title_element.get_text(strip=True)
                break
        
        # Use article title if found, otherwise fall back to page title
        title = article_title or page_title
        
        # Remove non-content tags first
        for tag_name in NON_CONTENT_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()
        
        # Remove all anchor tags (links) but keep their text
        for a_tag in soup.find_all("a"):
            a_tag.unwrap()
        
        # Try to find main content area using common selectors
        main_content = None
        for selector in MAIN_CONTENT_SELECTORS:
            main_content = soup.select_one(selector)
            if main_content:
                logger.debug(f"Found main content using selector: {selector}")
                break
        
        # If main content area found, extract text from it
        if main_content:
            # Remove navigation-style lists that might be inside main content
            self._remove_navigation_lists(main_content)
            text = self._extract_article_text(main_content, title)
        else:
            # Fallback: try to extract from body, removing obvious non-content
            body = soup.find("body")
            if body:
                self._remove_navigation_lists(body)
                text = self._extract_article_text(body, title)
            else:
                # Last resort: extract from entire cleaned soup
                text = soup.get_text(separator="\n", strip=True)
        
        # Clean up whitespace
        text = re.sub(r'\n\s*\n+', '\n\n', text)  # Reduce multiple newlines to double
        text = re.sub(r'[ \t]+', ' ', text)  # Reduce multiple spaces to single
        text = text.strip()
        
        logger.debug(f"Cleaned text length: {len(text)} characters, title: {title}")
        
        return text, title
    
    def _remove_navigation_lists(self, element) -> None:
        """
        Remove list elements that appear to be navigation menus.
        
        Navigation lists typically have many links and short text items.
        """
        for ul in element.find_all(["ul", "ol"]):
            # Count links vs total list items
            list_items = ul.find_all("li")
            if not list_items:
                continue
            
            links_count = len(ul.find_all("a"))
            items_count = len(list_items)
            
            # If most items are links, it's likely navigation
            if items_count > 0 and links_count / items_count > 0.7:
                # Check if items are short (navigation-like)
                avg_text_len = sum(len(li.get_text(strip=True)) for li in list_items) / items_count
                if avg_text_len < 50:  # Short items suggest navigation
                    ul.decompose()
    
    def _extract_article_text(self, element, title: Optional[str] = None) -> str:
        """
        Extract article text from an element, focusing on paragraph content.
        
        Args:
            element: BeautifulSoup element to extract text from
            title: Optional title to prepend
            
        Returns:
            Extracted text content
        """
        text_parts = []
        
        # Add title if provided
        if title:
            text_parts.append(title)
        
        # Extract text from paragraph-like elements (excluding h1 if it matches title)
        content_tags = ["p", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre"]
        
        for tag in element.find_all(content_tags):
            tag_text = tag.get_text(strip=True)
            if tag_text and len(tag_text) > 10:  # Skip very short fragments
                # Skip if this is the same as the title we already added
                if title and tag_text == title:
                    continue
                text_parts.append(tag_text)
        
        # If no paragraphs found, fall back to full text extraction
        if len(text_parts) <= 1:  # Only title or nothing
            fallback_text = element.get_text(separator="\n", strip=True)
            if fallback_text:
                text_parts.append(fallback_text)
        
        return "\n\n".join(text_parts)
    
    def _truncate_text(self, text: str, max_length: int = MAX_TEXT_LENGTH) -> Tuple[str, bool]:
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
3. For each opinion, provide bias probabilities (left, right, neutral) that sum to 1.0

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
        # Handle bias probabilities
        bias_probs = data.get("bias_probabilities", {})
        
        # If old format (bias_score) is provided, convert to probabilities
        if not bias_probs and "bias_score" in data:
            bias_score = data.get("bias_score", 0.0)
            if isinstance(bias_score, str):
                try:
                    bias_score = float(bias_score)
                except ValueError:
                    bias_score = 0.0
            bias_score = max(-1.0, min(1.0, float(bias_score)))
            
            # Convert single score to probability distribution
            # Score of -1.0 = 100% left, 0.0 = 100% neutral, +1.0 = 100% right
            if bias_score < 0:
                # Left-leaning: distribute between left and neutral
                left_prob = abs(bias_score)
                neutral_prob = 1.0 - left_prob
                right_prob = 0.0
            elif bias_score > 0:
                # Right-leaning: distribute between right and neutral
                right_prob = bias_score
                neutral_prob = 1.0 - right_prob
                left_prob = 0.0
            else:
                # Neutral
                left_prob = 0.0
                right_prob = 0.0
                neutral_prob = 1.0
            
            bias_probs = {
                "left": left_prob,
                "right": right_prob,
                "neutral": neutral_prob
            }
        
        # Extract and validate probability values
        def safe_float(val, default=0.0):
            if val is None:
                return default
            try:
                return max(0.0, min(1.0, float(val)))
            except (ValueError, TypeError):
                return default
        
        left = safe_float(bias_probs.get("left"), 0.33)
        right = safe_float(bias_probs.get("right"), 0.33)
        neutral = safe_float(bias_probs.get("neutral"), 0.34)
        
        # Normalize to sum to 1.0
        total = left + right + neutral
        if total > 0:
            left = left / total
            right = right / total
            neutral = neutral / total
        else:
            left = 0.33
            right = 0.33
            neutral = 0.34
        
        bias_distribution = BiasDistribution(left=left, right=right, neutral=neutral)
        
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
            bias_probabilities=bias_distribution,
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
        
        # Calculate overall bias distribution
        overall_bias_distribution = None
        if opinions:
            total_left = sum(op.bias_probabilities.left for op in opinions)
            total_right = sum(op.bias_probabilities.right for op in opinions)
            total_neutral = sum(op.bias_probabilities.neutral for op in opinions)
            n = len(opinions)
            overall_bias_distribution = BiasDistribution(
                left=total_left / n,
                right=total_right / n,
                neutral=total_neutral / n
            )
        
        result = OpinionExtractionResult(
            url=url,
            title=title,
            atomic_opinions=atomic_opinions,
            facts=facts,
            opinions=opinions,
            overall_bias_distribution=overall_bias_distribution,
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
        
        if overall_bias_distribution:
            bias_str = f"left={overall_bias_distribution.left:.2f}, right={overall_bias_distribution.right:.2f}, neutral={overall_bias_distribution.neutral:.2f}"
        else:
            bias_str = "N/A"
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
