"""Consistency Agent for checking consistency between summaries and content.

This agent orchestrates consistency checking operations by coordinating
with specialized modules for content processing, similarity calculation,
and claim comparison.
"""

from typing import Dict, Any, List, Optional

from src.infrastructure.repositories import AgentProtocol
from src.shared.config.settings import settings
from src.shared.constant.enums import CoTMode
from src.shared.utils import get_logger
from .claim_comparison import ClaimComparison
from .content_processor import ContentProcessor
from .similarity import SimilarityCalculator

logger = get_logger(__name__)


class ConsistencyAgent(AgentProtocol):
    """
    Consistency Agent that orchestrates consistency checking operations.
    
    This agent coordinates with specialized modules to:
    1. Extract content from URLs
    2. Atomize text into claims
    3. Calculate similarity between claims
    4. Compare claims for consistency
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.1
    ):
        """
        Initialize the ConsistencyAgent.

        Args:
            model_name: Name of the LLM model to use (defaults to settings.openai_model)
            api_key: OpenAI API key or compatible API key
            api_base: Base URL for the API
            temperature: Temperature for LLM responses
        """
        model_name = model_name or settings.openai_model
        logger.info(f"Initializing ConsistencyAgent: model={model_name}, temperature={temperature}")
        
        self.model_name = model_name
        self.api_key = api_key or settings.openai_api_key
        self.api_base = api_base or settings.openai_api_base
        self.temperature = temperature
        
        # Initialize specialized modules
        self.content_processor = ContentProcessor(
            model_name=self.model_name,
            api_key=self.api_key,
            api_base=self.api_base,
            temperature=self.temperature
        )
        
        self.claim_comparison = ClaimComparison(
            model_name=self.model_name,
            api_key=self.api_key,
            api_base=self.api_base,
            temperature=self.temperature
        )
        
        logger.debug("ConsistencyAgent initialized successfully")

    async def compare_two_claims(
        self,
        summary_claim: str,
        url_claim: str,
        url_content: Optional[str] = None,
        use_cot: CoTMode = CoTMode.NO_CHAIN
    ) -> Dict[str, Any]:
        """
        Compare two specific claims for consistency.

        Args:
            summary_claim: Claim from the summary
            url_claim: Claim from the URL content
            url_content: Optional full URL content for context
            use_cot: Chain of Thought mode

        Returns:
            Dict with consistency analysis: status, confidence, and concise reason
        """
        return await self.claim_comparison.compare_two_claims(
            summary_claim=summary_claim,
            url_claim=url_claim,
            url_content=url_content,
            use_cot=use_cot
        )

    async def calculate_similarity_matrix(
        self,
        summary_claims: List[Any],
        url_claims: List[Any]
    ):
        """
        Calculate similarity matrix between summary and URL claims using embeddings.

        Args:
            summary_claims: List of summary claim objects with .text attribute
            url_claims: List of URL claim objects with .text attribute

        Returns:
            Similarity matrix as numpy array
        """
        return await SimilarityCalculator.calculate_similarity_matrix(
            summary_claims=summary_claims,
            url_claims=url_claims
        )

    async def analyze_claim_pairs(
        self,
        summary_claims: List[Any],
        url_claims: List[Any],
        similarity_matrix,
        threshold: float,
        enable_deep_analysis: bool,
        use_cot_audit: CoTMode
    ) -> List[Dict[str, Any]]:
        """
        Analyze claim pairs above similarity threshold.

        Args:
            summary_claims: List of summary claim objects
            url_claims: List of URL claim objects
            similarity_matrix: Pre-calculated similarity matrix
            threshold: Minimum similarity score for analysis
            enable_deep_analysis: Whether to enable deep LLM analysis
            use_cot_audit: Chain of Thought mode for audit

        Returns:
            List of comparison results
        """
        return await self.claim_comparison.analyze_claim_pairs(
            summary_claims=summary_claims,
            url_claims=url_claims,
            similarity_matrix=similarity_matrix,
            threshold=threshold,
            enable_deep_analysis=enable_deep_analysis,
            use_cot_audit=use_cot_audit
        )

    async def extract_and_atomize(
        self,
        url: str,
        summary: Optional[str] = None,
        use_cot_atomization: CoTMode = CoTMode.NO_CHAIN,
        split_url_into_paragraphs: bool = True
    ) -> Dict[str, Any]:
        """
        Extract content from URL and atomize both summary and URL content.

        Args:
            url: URL to extract content from
            summary: Optional summary text to atomize
            use_cot_atomization: Chain of Thought mode for atomization
            split_url_into_paragraphs: Whether to split URL content into paragraphs

        Returns:
            Dict with content_result, summary_atomization, and url_atomization
        """
        return await self.content_processor.extract_and_atomize(
            url=url,
            summary=summary,
            use_cot_atomization=use_cot_atomization,
            split_url_into_paragraphs=split_url_into_paragraphs
        )
