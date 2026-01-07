"""Consistency Service - Handles consistency checking business logic.

This service layer orchestrates consistency checking operations by:
1. Coordinating with ConsistencyAgent for agent-level operations
2. Formatting results for API responses
3. Calculating statistics
4. Managing business logic flow
"""

import time
from typing import Dict, Any, Optional

from langchain_classic.agents.types import AGENT_TYPE

from src.application.agents.consistency.agent import ConsistencyAgent
from src.application.agents.consistency.utils import ClaimFormatter, StatisticsCalculator
from src.application.services import BaseService
from src.shared.config.settings import settings
from src.shared.constant.enums import CoTMode, AgentType
from src.shared.utils import get_logger

logger = get_logger(__name__)

# Fixed agent ID for stateless consistency agent
DEFAULT_CONSISTENCY_AGENT_ID = "default_consistency_agent"
DEFAULT_CONSISTENCY_AGENT_TYPE = "default_consistency_agent"

class ConsistencyService(BaseService):
    """Service for consistency checking operations."""

    def __init__(self, agent_manager=None):
        """Initialize ConsistencyService with optional agent manager."""
        from ...application.agents.manager import AgentManager
        super().__init__(agent_manager or AgentManager())
        self._cached_consistency_agent: Optional[ConsistencyAgent] = None

    def _get_consistency_agent(self):
        """Get or create ConsistencyAgent instance."""
        if self._cached_consistency_agent is None:
            self._cached_consistency_agent = ConsistencyAgent(
                model_name=settings.openai_model,
                api_key=settings.openai_api_key,
                api_base=settings.openai_api_base,
                temperature=settings.agent_temperature
            )
            logger.info(f"ConsistencyAgent created (stateless)")
        return self._cached_consistency_agent

    async def compare_two_claims(
        self,
        summary_claim: str,
        url_claim: str,
        url_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare two specific claims for consistency.

        Args:
            summary_claim: Claim from the summary
            url_claim: Claim from the URL content
            url_content: Optional full URL content for context

        Returns:
            Dict with consistency analysis: status, confidence, and concise reason
        """
        start_time = time.time()

        consistency_agent = self._get_consistency_agent()
        comparison_result = await consistency_agent.compare_two_claims(
            summary_claim=summary_claim,
            url_claim=url_claim,
            url_content=url_content
        )

        result = {
            "summary_claim": summary_claim,
            "url_claim": url_claim,
            "comparison": comparison_result,
            "processing_time": time.time() - start_time
        }

        logger.info(
            f"Claim comparison completed: {comparison_result['status']} "
            f"(confidence: {comparison_result['confidence']:.2f})"
        )
        return result

    async def check_summary_url_consistency(
        self,
        summary: str,
        url: str,
        enable_deep_analysis: bool = True,
        similarity_threshold: float = 0.2,
        use_cot_atomization: CoTMode = CoTMode.NO_CHAIN,
        use_cot_audit: CoTMode = CoTMode.NO_CHAIN
    ) -> Dict[str, Any]:
        """
        Check consistency between website summary and full content.

        Atomize both summary and URL content into claims, then compare each summary claim
        against relevant URL claims to identify consistencies and contradictions.

        Args:
            summary: Summary text to check
            url: URL to extract content from
            enable_deep_analysis: Whether to enable deep LLM analysis
            similarity_threshold: Minimum similarity score for LLM analysis
            use_cot_atomization: Chain of Thought mode for atomization
            use_cot_audit: Chain of Thought mode for audit

        Returns:
            Dict with consistency analysis results
        """
        start_time = time.time()
        logger.info(f"Starting summary vs URL consistency check: summary({len(summary)} chars) vs URL({url})")

        consistency_result = {
            "summary": summary,
            "url": url,
            "summary_claims": [],
            "url_claims": [],
            "claim_comparisons": [],
            "processing_time": None
        }

        try:
            consistency_agent = self._get_consistency_agent()

            # Step 1-3: Extract content and atomize both summary and URL content
            extraction_result = await consistency_agent.extract_and_atomize(
                url=url,
                summary=summary,
                use_cot_atomization=use_cot_atomization,
                split_url_into_paragraphs=True
            )

            content_result = extraction_result["content_result"]
            summary_atomization = extraction_result["summary_atomization"]
            url_atomization = extraction_result["url_atomization"]

            if not content_result or not content_result.main_body:
                raise ValueError("Could not extract content from URL")

            if not summary_atomization:
                raise ValueError("Could not extract content from summary")

            if not url_atomization:
                raise ValueError("Could not extract content from url")

            # Format claims for response
            consistency_result["summary_claims"] = [
                ClaimFormatter.to_simple_dict(c) for c in summary_atomization.atomic_claims
            ]
            consistency_result["url_claims"] = [
                ClaimFormatter.to_dict(c) for c in url_atomization.atomic_claims
            ]

            # Step 4: Calculate similarities and analyze claim pairs
            logger.info("Calculating similarities between all claim pairs...")
            similarity_matrix = await consistency_agent.calculate_similarity_matrix(
                summary_atomization.atomic_claims,
                url_atomization.atomic_claims
            )

            # Count total possible pairs before filtering
            total_possible_pairs = len(summary_atomization.atomic_claims) * len(url_atomization.atomic_claims)
            
            logger.info(
                f"Finding claim pairs above similarity threshold {similarity_threshold} "
                f"(total possible pairs: {total_possible_pairs})"
            )
            
            # Analyze claim pairs (already filtered by threshold in analyze_claim_pairs)
            comparisons = await consistency_agent.analyze_claim_pairs(
                summary_atomization.atomic_claims,
                url_atomization.atomic_claims,
                similarity_matrix,
                similarity_threshold,
                enable_deep_analysis,
                use_cot_audit
            )

            # Verify all comparisons meet threshold requirement (safety check)
            filtered_comparisons = [
                comp for comp in comparisons 
                if comp.get("similarity_score", 0.0) >= similarity_threshold
            ]
            
            if len(filtered_comparisons) != len(comparisons):
                logger.warning(
                    f"Found {len(comparisons) - len(filtered_comparisons)} comparisons "
                    f"below threshold {similarity_threshold}, filtering them out"
                )
                comparisons = filtered_comparisons

            # Calculate statistics (includes filtering statistics)
            stats = StatisticsCalculator.calculate_comparison_statistics(
                comparisons, similarity_threshold, summary_atomization, url_atomization
            )
            
            # Log filtering statistics
            logger.info(
                f"Similarity filtering complete: {stats['pairs_above_threshold']} pairs above threshold "
                f"({similarity_threshold}), {stats['pairs_below_threshold']} pairs filtered out "
                f"({stats['filtering_rate']*100:.1f}% filtering rate)"
            )

            consistency_result["claim_comparisons"] = comparisons
            consistency_result["statistics"] = stats

            # Format paragraph information
            consistency_result["url_paragraphs"] = [
                ClaimFormatter.format_paragraph_info(para) for para in url_atomization.paragraphs
            ]

            logger.info(
                f"Consistency analysis completed: {len(comparisons)} comparisons, "
                f"{stats['supported_count']} supported, {stats['contradicted_count']} contradicted"
            )
            consistency_result["processing_time"] = time.time() - start_time

        except Exception as e:
            logger.error(f"Summary vs URL consistency check failed: {e}", exc_info=True)
            consistency_result["error"] = str(e)

        return consistency_result

