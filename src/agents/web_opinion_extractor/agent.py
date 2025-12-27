"""Web Opinion Extractor Agent implementation.

This agent processes web content to extract atomic opinions, distinguish
facts from opinions, and calculate political bias scores.
"""

import json
import re
from typing import Optional, List, Dict, Any, Tuple, Union

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..html_extractor import HTMLExtractor
from .exceptions import NetworkError, ContentExtractionError
from .models import AtomicOpinion, OpinionExtractionResult, BiasDistribution, CoTMode
from ...config.settings import settings, ExecutionMode
from ...utils.llm_client import llm_manager
from ...utils.logger import get_logger
from ...utils.agent_cache import cached
from ...storage import get_database

logger = get_logger(__name__)


class WebOpinionExtractor(HTMLExtractor):
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

4. CHAIN OF THOUGHT REASONING (CRUCIAL):
   - For each atomic opinion, provide step-by-step reasoning in the "reasoning" field
   - Explain WHY you classified it as fact/opinion
   - Explain HOW you determined the bias probabilities
   - Show your analytical process clearly
   - Example reasoning: "This statement expresses a subjective preference for tax cuts. The use of 'support' indicates personal stance rather than objective fact. The framing aligns with conservative fiscal policy (lower taxes), hence higher right probability (0.7). Some neutral probability (0.2) acknowledges economic complexity. Left probability is low (0.1) as the statement doesn't align with progressive tax policies."

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
      "reasoning": "Step-by-step explanation of the analysis",
      "original_sentence": "The original sentence this was extracted from",
      "confidence": <float between 0.0 and 1.0>
    }
  ]
}

Extract ALL viewpoints, even subtle ones. Be thorough but precise."""
    
    SYSTEM_PROMPT_NO_COT = """You are an expert content analyst specializing in extracting and analyzing opinions from text.
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
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.3,
        request_timeout: float = 30.0,
        execution_mode: Optional[ExecutionMode] = None
    ):
        """
        Initialize the WebOpinionExtractor.
        
        Args:
            model_name: Name of the LLM model to use (defaults to settings.openai_model)
            api_key: OpenAI API key or compatible API key
            api_base: Base URL for the API
            temperature: Temperature for LLM responses (lower for more consistent extraction)
            request_timeout: Timeout for HTTP requests in seconds
            execution_mode: Execution mode for CoT ('chain_online', 'chain_local', 'no_chain')
        """
        # Initialize parent HTMLExtractor
        super().__init__(request_timeout=request_timeout)

        model_name = model_name or settings.openai_model
        logger.info(f"Initializing WebOpinionExtractor: model={model_name}, temperature={temperature}, execution_mode={execution_mode}")

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
        
        self.request_timeout = request_timeout
        self.execution_mode = execution_mode or settings.default_execution_mode
        self._extraction_history: List[OpinionExtractionResult] = []
        
        logger.debug(f"WebOpinionExtractor initialized successfully with execution_mode={self.execution_mode}")
    
    @cached()
    def _extract_opinions_with_llm(self, text: str, use_cot: Union[CoTMode, bool] = CoTMode.NO_CHAIN, custom_few_shots: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Use LLM to extract atomic opinions from text.

        Args:
            text: Cleaned text content
            use_cot: Whether to use Chain of Thought reasoning
            custom_few_shots: Optional custom few-shot examples

        Returns:
            List of opinion dictionaries
        """
        # Determine effective CoT mode
        if isinstance(use_cot, CoTMode):
            effective_cot = use_cot.value in ("chain_local", "chain_online")
        else:
            effective_cot = bool(use_cot)

        logger.debug(f"Extracting opinions from text ({len(text)} chars) with CoT={effective_cot}")

        # Select system prompt based on CoT mode
        system_prompt = self.SYSTEM_PROMPT if effective_cot else self.SYSTEM_PROMPT_NO_COT

        # Add custom few-shot examples if provided
        if custom_few_shots:
            system_prompt = f"{custom_few_shots}\n\n{system_prompt}"
        
        # Adjust user message based on mode
        if use_cot:
            user_message = f"""Analyze the following text and extract all atomic opinions.
Remember to:
1. Separate facts from opinions
2. Break compound sentences into atomic viewpoints
3. For each opinion, provide bias probabilities (left, right, neutral) that sum to 1.0
4. Include step-by-step reasoning in the "reasoning" field explaining your analysis

TEXT TO ANALYZE:
{text}

Return your analysis as a JSON object."""
        else:
            user_message = f"""Analyze the following text and extract all atomic opinions.
Remember to:
1. Separate facts from opinions
2. Break compound sentences into atomic viewpoints
3. For each opinion, provide bias probabilities (left, right, neutral) that sum to 1.0

TEXT TO ANALYZE:
{text}

Return your analysis as a JSON object."""

        messages = [
            SystemMessage(content=system_prompt),
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
            confidence=confidence,
            reasoning=data.get("reasoning")
        )
    
    def extract_from_url(self, url: str, use_llm: bool = True, use_cot: bool = False, custom_few_shots: Optional[str] = None) -> OpinionExtractionResult:
        """
        Extract opinions from a URL.

        Args:
            url: The URL to process
            use_llm: Whether to use LLM for opinion extraction and analysis (default True for opinion extraction)
            use_cot: Whether to use Chain of Thought reasoning (only when use_llm=True)
            custom_few_shots: Optional custom few-shot examples (only when use_llm=True)

        Returns:
            OpinionExtractionResult with extracted opinions

        Raises:
            NetworkError: If the URL cannot be fetched
            ContentExtractionError: If content cannot be extracted
        """
        logger.info(f"Extracting opinions from URL: {url} (use_llm={use_llm})")

        # Fetch HTML
        html = self.fetch_html(url)

        # Clean and extract text using HTML parsing (token-efficient)
        text, title = self.clean_html(html)

        if not text or len(text.strip()) < 50:
            logger.warning(f"Insufficient content extracted from {url}")
            raise ContentExtractionError(
                message="Insufficient content extracted from page",
                url=url
            )

        # Process with LLM for opinion extraction (always needed for this agent)
        result = self.extract_from_text(text, url=url, title=title, use_llm=use_llm, use_cot=use_cot, custom_few_shots=custom_few_shots)

        return result
    
    def extract_from_html(
        self,
        html: str,
        url: Optional[str] = None,
        use_llm: bool = True,
        use_cot: bool = False,
        custom_few_shots: Optional[str] = None
    ) -> OpinionExtractionResult:
        """
        Extract opinions from raw HTML content.

        Args:
            html: Raw HTML string
            url: Optional source URL for metadata
            use_llm: Whether to use LLM for opinion extraction and analysis (default True for opinion extraction)
            use_cot: Whether to use Chain of Thought reasoning (only when use_llm=True)
            custom_few_shots: Optional custom few-shot examples (only when use_llm=True)

        Returns:
            OpinionExtractionResult with extracted opinions
        """
        logger.info(f"Extracting opinions from HTML content (use_llm={use_llm})")

        # Clean and extract text using HTML parsing (token-efficient)
        text, title = self.clean_html(html)

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

        return self.extract_from_text(text, url=url, title=title, use_llm=use_llm, use_cot=use_cot, custom_few_shots=custom_few_shots)
    
    def extract_from_text(
        self,
        text: str,
        url: Optional[str] = None,
        title: Optional[str] = None,
        use_llm: bool = True,
        use_cot: bool = False,
        custom_few_shots: Optional[str] = None
    ) -> OpinionExtractionResult:
        """
        Extract opinions from plain text.

        Args:
            text: Plain text content
            url: Optional source URL for metadata
            title: Optional page title
            use_llm: Whether to use LLM for opinion extraction and analysis (default True for opinion extraction)
            use_cot: Whether to use Chain of Thought reasoning (only when use_llm=True)
            custom_few_shots: Optional custom few-shot examples (only when use_llm=True)

        Returns:
            OpinionExtractionResult with extracted opinions
        """
        logger.info(f"Extracting opinions from text ({len(text)} chars, use_llm={use_llm})")

        # For opinion extraction, LLM is typically required
        if not use_llm:
            logger.warning("Opinion extraction requires LLM analysis, but use_llm=False. Returning empty result.")
            return OpinionExtractionResult(
                url=url,
                title=title,
                atomic_opinions=[],
                facts=[],
                opinions=[],
                text_length=len(text),
                truncated=False,
                extraction_metadata={
                    "error": "LLM required for opinion extraction",
                    "use_llm": False
                }
            )

        # Truncate if necessary
        text, was_truncated = self.truncate_text(text)

        # Extract opinions using LLM
        raw_opinions = self._extract_opinions_with_llm(text, use_cot=use_cot, custom_few_shots=custom_few_shots)
        
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
                "model": self.llm.model_name if use_llm else None,
                "temperature": self.llm.temperature if use_llm else None,
                "use_llm": use_llm,
                "use_cot": use_cot,
                "total_extracted": len(atomic_opinions),
                "facts_count": len(facts),
                "opinions_count": len(opinions),
                "extraction_method": "llm_analysis" if use_llm else "none"
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
    
    # ========== HIGH-LEVEL PUBLIC API METHODS ==========
    
    def extract_and_analyze(self, url: str, use_llm: bool = True, use_cot: bool = False, custom_few_shots: Optional[str] = None) -> OpinionExtractionResult:
        """
        High-level API: Complete pipeline to extract and analyze opinions from a URL.

        This is the main entry point that encapsulates the entire complexity:
        1. Fetch HTML from URL
        2. Clean and extract main content
        3. Analyze with LLM
        4. Parse and return results

        Args:
            url: The URL to process
            use_llm: Whether to use LLM for opinion extraction and analysis (default True for opinion extraction)
            use_cot: Whether to use Chain of Thought reasoning (only when use_llm=True)
            custom_few_shots: Optional custom few-shot examples (only when use_llm=True)

        Returns:
            OpinionExtractionResult with extracted opinions and metadata

        Raises:
            NetworkError: If the URL cannot be fetched
            ContentExtractionError: If content cannot be extracted

        Example:
            >>> extractor = WebOpinionExtractor()
            >>> result = extractor.extract_and_analyze("https://example.com/article")
            >>> print(f"Found {len(result.opinions)} opinions")
            >>> for opinion in result.opinions:
            ...     print(f"  - {opinion.text}")
            ...     print(f"    Bias: {opinion.bias_probabilities.dominant_bias}")
            ...     if opinion.reasoning:
            ...         print(f"    Reasoning: {opinion.reasoning}")
        """
        logger.info(f"[extract_and_analyze] Processing URL: {url} (use_llm={use_llm})")

        try:
            result = self.extract_from_url(url, use_llm=use_llm, use_cot=use_cot, custom_few_shots=custom_few_shots)
            logger.info(f"[extract_and_analyze] Successfully processed {url}")
            return result
        except (NetworkError, ContentExtractionError) as e:
            logger.error(f"[extract_and_analyze] Failed to process {url}: {e}")
            raise
        except Exception as e:
            logger.error(f"[extract_and_analyze] Unexpected error processing {url}: {e}", exc_info=True)
            raise ContentExtractionError(
                message=f"Unexpected error during extraction: {str(e)}",
                url=url
            )
    
    def extract_html(self, url: str) -> str:
        """
        Step 1 API: Fetch HTML content from a URL.
        
        This method is exposed to verify the HTML fetching step independently.
        
        Args:
            url: The URL to fetch
            
        Returns:
            Raw HTML content as string
            
        Raises:
            NetworkError: If the request fails
            
        Example:
            >>> extractor = WebOpinionExtractor()
            >>> html = extractor.extract_html("https://example.com/article")
            >>> print(f"Fetched {len(html)} characters")
        """
        logger.info(f"[extract_html] Fetching HTML from: {url}")
        try:
            html = self.fetch_html(url)
            logger.info(f"[extract_html] Successfully fetched {len(html)} characters")
            return html
        except NetworkError as e:
            logger.error(f"[extract_html] Failed to fetch {url}: {e}")
            raise
    
    def clean_html(self, html: str) -> Tuple[str, Optional[str]]:
        """
        Step 2 API: Clean HTML and extract main content.
        
        This method is exposed to verify the HTML cleaning step independently.
        
        Args:
            html: Raw HTML string
            
        Returns:
            Tuple of (cleaned_text, page_title)
            
        Example:
            >>> extractor = WebOpinionExtractor()
            >>> html = extractor.extract_html("https://example.com/article")
            >>> text, title = extractor.clean_html(html)
            >>> print(f"Title: {title}")
            >>> print(f"Cleaned text length: {len(text)} characters")
        """
        logger.info(f"[clean_html] Cleaning HTML ({len(html)} chars)")
        text, title = self.clean_html(html)
        logger.info(f"[clean_html] Extracted {len(text)} chars, title: {title}")
        return text, title

    def analyze_text(
        self,
        text: str,
        url: Optional[str] = None,
        title: Optional[str] = None,
        use_llm: bool = True,
        use_cot: Union[CoTMode, bool] = CoTMode.NO_CHAIN,
        custom_few_shots: Optional[str] = None
    ) -> OpinionExtractionResult:
        """
        Step 3 API: Analyze cleaned text with LLM.

        This method is exposed to verify the LLM analysis step independently.

        Args:
            text: Cleaned text content to analyze
            url: Optional source URL for metadata
            title: Optional page title for metadata
            use_llm: Whether to use LLM for opinion extraction and analysis (default True for opinion extraction)
            use_cot: Whether to use Chain of Thought reasoning (only when use_llm=True)
            custom_few_shots: Optional custom few-shot examples (only when use_llm=True)

        Returns:
            OpinionExtractionResult with extracted opinions

        Example:
            >>> extractor = WebOpinionExtractor()
            >>> text = "I strongly support this policy. It will help workers."
            >>> result = extractor.analyze_text(text)
            >>> print(f"Found {len(result.opinions)} opinions")
            >>> for opinion in result.opinions:
            ...     print(f"  {opinion.text}: {opinion.bias_probabilities.dominant_bias}")
        """
        logger.info(f"[analyze_text] Analyzing text ({len(text)} chars, use_llm={use_llm})")
        result = self.extract_from_text(text, url=url, title=title, use_llm=use_llm, use_cot=use_cot, custom_few_shots=custom_few_shots)
        logger.info(f"[analyze_text] Analysis complete: {len(result.opinions)} opinions extracted")
        return result
    
    def reset(self) -> None:
        """Reset the agent to initial state."""
        logger.info("Resetting WebOpinionExtractor state")
        self._extraction_history = []
        logger.debug("Agent reset complete")


class WebOpinionAnalyzer:
    """
    High-level WebOpinionAnalyzer module providing a clean, production-ready API.
    
    This is a convenience wrapper around WebOpinionExtractor that provides:
    - Simple API: extract_and_analyze(url) as main entry point
    - Robust error handling: returns error states instead of crashing
    - Step verification: Individual methods to verify each processing step
    - Chain of Thought support: Configurable CoT reasoning modes
    
    Example Usage:
        >>> analyzer = WebOpinionAnalyzer(execution_mode="chain_local")
        >>> result = analyzer.extract_and_analyze("https://example.com/article")
        >>> if result.extraction_metadata and "error" in result.extraction_metadata:
        ...     print(f"Error: {result.extraction_metadata['error']}")
        >>> else:
        ...     print(f"Found {len(result.opinions)} opinions")
        ...     for opinion in result.opinions:
        ...         print(f"  {opinion.text}")
        ...         print(f"    Bias: {opinion.bias_probabilities.dominant_bias}")
        ...         if opinion.reasoning:
        ...             print(f"    CoT: {opinion.reasoning[:100]}...")
    """

    # Class variables for custom few shots
    _custom_few_shots_cache: Optional[str] = None

    @classmethod
    def get_default_few_shots(cls) -> str:
        """
        Get default few-shot examples for opinion extraction.

        Returns:
            Default few-shot examples string
        """
        return """EXAMPLE 1 - Opinion Classification:
Text: "I believe the new tax policy will devastate small businesses and hurt our economy."
Analysis:
- Opinion: "the new tax policy will devastate small businesses and hurt our economy"
- Type: opinion
- Bias: left-leaning (concern for small businesses and economy)

EXAMPLE 2 - Fact vs Opinion:
Text: "The unemployment rate rose to 4.8% last month, which is the highest in three years."
Analysis:
- Fact: "The unemployment rate rose to 4.8% last month"
- Fact: "which is the highest in three years"
- No opinions identified

EXAMPLE 3 - Multiple Opinions:
Text: "This policy is absolutely brilliant and will solve all our problems. The politicians who oppose it are clearly incompetent."
Analysis:
- Opinion: "This policy is absolutely brilliant and will solve all our problems"
- Type: opinion, bias: positive/supportive
- Opinion: "The politicians who oppose it are clearly incompetent"
- Type: opinion, bias: negative/critical"""

    @classmethod
    def set_custom_few_shots(cls, custom_few_shots: Optional[str] = None) -> None:
        """
        Set custom few-shot examples for opinion extraction.

        Args:
            custom_few_shots: Custom few-shot examples string. If None, clears custom few shots.
        """
        if settings.enable_persistence:
            database = get_database()
            success = database.save_custom_few_shots("web_opinion_extractor", custom_few_shots)
            if success:
                cls._custom_few_shots_cache = custom_few_shots  # Update cache
                logger.info(f"Custom few shots saved for WebOpinionAnalyzer: {custom_few_shots is not None}")
            else:
                logger.warning("Failed to save custom few shots to database")
        else:
            cls._custom_few_shots_cache = custom_few_shots
            logger.info(f"Custom few shots set for WebOpinionAnalyzer (no persistence): {custom_few_shots is not None}")

    @classmethod
    def get_custom_few_shots(cls) -> Optional[str]:
        """
        Get currently set custom few-shot examples.

        Returns:
            Custom few-shot examples string or None if not set
        """
        if settings.enable_persistence:
            database = get_database()
            few_shots = database.load_custom_few_shots("web_opinion_extractor")
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

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.3,
        request_timeout: float = 30.0,
        execution_mode: Optional[ExecutionMode] = None
    ):
        """
        Initialize the WebOpinionAnalyzer.
        
        Args:
            model_name: Name of the LLM model to use (defaults to settings.openai_model)
            api_key: OpenAI API key or compatible API key
            api_base: Base URL for the API
            temperature: Temperature for LLM responses
            request_timeout: Timeout for HTTP requests in seconds
            execution_mode: Execution mode for CoT:
                - 'chain_online': LLM handles full CoT reasoning
                - 'chain_local': Local task decomposition with CoT
                - 'no_chain': No CoT reasoning (fastest)
        """
        self._extractor = WebOpinionExtractor(
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            proxy=proxy,
            request_timeout=request_timeout,
            execution_mode=execution_mode
        )
        exec_mode = getattr(self._extractor, 'execution_mode', execution_mode or 'chain_local')
        logger.info(f"WebOpinionAnalyzer initialized with mode={exec_mode}")
    
    def extract_and_analyze(self, url: str, use_llm: bool = True, use_cot: bool = False, custom_few_shots: Optional[str] = None) -> OpinionExtractionResult:
        """
        Main entry point: Extract and analyze opinions from a URL.

        This method encapsulates the complete pipeline and provides robust error handling.
        Instead of raising exceptions, it returns a valid OpinionExtractionResult with
        error information in the extraction_metadata field when failures occur.

        Args:
            url: The URL to process
            use_llm: Whether to use LLM for opinion extraction and analysis (default True for opinion extraction)
            use_cot: Whether to use Chain of Thought reasoning (only when use_llm=True)
            custom_few_shots: Optional custom few-shot examples (only when use_llm=True)

        Returns:
            OpinionExtractionResult - Always returns a valid result object.
            Check extraction_metadata["error"] to detect failures.

        Example:
            >>> analyzer = WebOpinionAnalyzer()
            >>> result = analyzer.extract_and_analyze("https://example.com/article")
            >>> if result.extraction_metadata and "error" in result.extraction_metadata:
            ...     print(f"Failed: {result.extraction_metadata['error']}")
            ... else:
            ...     print(f"Success! Found {len(result.opinions)} opinions")
        """
        try:
            return self._extractor.extract_and_analyze(url, use_llm=use_llm, use_cot=use_cot, custom_few_shots=custom_few_shots)
        except NetworkError as e:
            logger.warning(f"Network error for {url}: {e}")
            return OpinionExtractionResult(
                url=url,
                atomic_opinions=[],
                facts=[],
                opinions=[],
                text_length=0,
                truncated=False,
                extraction_metadata={
                    "error": "network_error",
                    "error_message": str(e.message),
                    "status_code": e.status_code
                }
            )
        except ContentExtractionError as e:
            logger.warning(f"Content extraction error for {url}: {e}")
            return OpinionExtractionResult(
                url=url,
                atomic_opinions=[],
                facts=[],
                opinions=[],
                text_length=0,
                truncated=False,
                extraction_metadata={
                    "error": "content_extraction_error",
                    "error_message": str(e.message)
                }
            )
        except Exception as e:
            logger.error(f"Unexpected error for {url}: {e}", exc_info=True)
            return OpinionExtractionResult(
                url=url,
                atomic_opinions=[],
                facts=[],
                opinions=[],
                text_length=0,
                truncated=False,
                extraction_metadata={
                    "error": "unexpected_error",
                    "error_message": str(e)
                }
            )
    
    def extract_html(self, url: str) -> Optional[str]:
        """
        Step 1 verification: Fetch HTML from URL.
        
        Returns None on error instead of raising exceptions.
        
        Args:
            url: The URL to fetch
            
        Returns:
            HTML string or None on error
        """
        try:
            return self._extractor.extract_html(url)
        except Exception as e:
            logger.error(f"Failed to fetch HTML from {url}: {e}")
            return None
    
    def clean_html(self, html: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Step 2 verification: Clean HTML and extract main content.
        
        Returns (None, None) on error instead of raising exceptions.
        
        Args:
            html: Raw HTML string
            
        Returns:
            Tuple of (cleaned_text, title) or (None, None) on error
        """
        try:
            return self._extractor.clean_html(html)
        except Exception as e:
            logger.error(f"Failed to clean HTML: {e}")
            return None, None
    
    def analyze_text(
        self,
        text: str,
        url: Optional[str] = None,
        title: Optional[str] = None,
        use_llm: bool = True,
        use_cot: Union[CoTMode, bool] = CoTMode.NO_CHAIN,
        custom_few_shots: Optional[str] = None
    ) -> OpinionExtractionResult:
        """
        Step 3 verification: Analyze text with LLM.

        Returns an error result on failure instead of raising exceptions.

        Args:
            text: Cleaned text to analyze
            url: Optional source URL
            title: Optional page title
            use_llm: Whether to use LLM for opinion extraction and analysis (default True for opinion extraction)
            use_cot: Whether to use Chain of Thought reasoning (only when use_llm=True)
            custom_few_shots: Optional custom few-shot examples (only when use_llm=True)

        Returns:
            OpinionExtractionResult - Check extraction_metadata["error"] for failures
        """
        try:
            return self._extractor.analyze_text(text, url=url, title=title, use_llm=use_llm, use_cot=use_cot, custom_few_shots=custom_few_shots)
        except Exception as e:
            logger.error(f"Failed to analyze text: {e}")
            return OpinionExtractionResult(
                url=url,
                title=title,
                atomic_opinions=[],
                facts=[],
                opinions=[],
                text_length=len(text),
                truncated=False,
                extraction_metadata={
                    "error": "llm_analysis_error",
                    "error_message": str(e)
                }
            )
    
    def get_extraction_history(self) -> List[OpinionExtractionResult]:
        """Get the history of all extractions performed."""
        return self._extractor.get_extraction_history()
    
    def reset(self) -> None:
        """Reset the analyzer to initial state."""
        self._extractor.reset()
