"""Claim comparison module for consistency checking."""

import asyncio
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from src.application.agents.conflict_auditor.agent import ConflictAuditorAgent
from src.shared.constant.enums import CoTMode
from src.shared.utils import get_logger
from .utils import ClaimFormatter

logger = get_logger(__name__)


class ClaimComparison:
    """Utility class for comparing claims."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.1
    ):
        """
        Initialize the ClaimComparison.

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

    def _create_auditor_agent(self) -> ConflictAuditorAgent:
        """Create and return a ConflictAuditorAgent instance."""
        return ConflictAuditorAgent(
            model_name=self.model_name,
            api_key=self.api_key,
            api_base=self.api_base,
            temperature=self.temperature
        )

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
        auditor_agent = self._create_auditor_agent()
        return await auditor_agent.acompare_two_claims(
            summary_claim=summary_claim,
            url_claim=url_claim,
            url_content=url_content,
            use_cot=use_cot
        )

    async def analyze_claim_pairs(
        self,
        summary_claims: List[Any],
        url_claims: List[Any],
        similarity_matrix: np.ndarray,
        threshold: float,
        enable_deep_analysis: bool,
        use_cot_audit: CoTMode
    ) -> tuple[Any] | list[Any]:
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
        auditor_agent = self._create_auditor_agent()
        tasks = []

        async def analyze_pair(summary_claim, url_claim, similarity_score):
            """Analyze a single claim pair."""
            if not enable_deep_analysis:
                return ClaimFormatter.format_claim_pair(
                    summary_claim=summary_claim,
                    url_claim=url_claim,
                    similarity_score=similarity_score,
                    relationship="potential_match" if similarity_score > 0.5 else "neutral",
                    confidence=float(similarity_score),
                    reason="Deep analysis disabled. Relationship inferred from similarity score."
                )

            try:
                analysis_result = await auditor_agent.compare_two_claims(
                    summary_claim=summary_claim.text,
                    url_claim=url_claim.text,
                    use_cot=use_cot_audit
                )

                return ClaimFormatter.format_claim_pair(
                    summary_claim=summary_claim,
                    url_claim=url_claim,
                    similarity_score=similarity_score,
                    relationship=analysis_result["status"],
                    confidence=analysis_result["confidence"],
                    reason=analysis_result["reason"]
                )
            except Exception as e:
                logger.warning(f"Failed to analyze claim pair {summary_claim.id} vs {url_claim.id}: {e}")
                return ClaimFormatter.format_claim_pair(
                    summary_claim=summary_claim,
                    url_claim=url_claim,
                    similarity_score=similarity_score,
                    relationship="error",
                    confidence=0.0,
                    reason=f"Analysis failed: {str(e)}"
                )

        for i, summary_claim in enumerate(summary_claims):
            for j, url_claim in enumerate(url_claims):
                similarity_score = similarity_matrix[i, j]
                if similarity_score >= threshold:
                    tasks.append(analyze_pair(summary_claim, url_claim, similarity_score))

        if tasks:
            logger.info(f"Executing {len(tasks)} claim comparisons concurrently...")
            return await asyncio.gather(*tasks)
        return []

