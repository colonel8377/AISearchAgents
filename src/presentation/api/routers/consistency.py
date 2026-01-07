"""Consistency API routes - View layer only."""

from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException

from src.presentation.api.schemas import CheckConsistencyRequest, CompareClaimsRequest
from src.shared.utils import get_logger
from ..common import get_api_key, consistency_service

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/consistency", tags=["Consistency"])


@router.post("/compare-claims", response_model=dict)
async def compare_two_claims(
    request: CompareClaimsRequest,
    _api_key: str = Depends(get_api_key)
) -> Dict[str, Any]:
    """
    Compare two specific claims for consistency.

    This endpoint directly compares a summary claim against a URL claim without complex processing.
    Returns a simple, clear analysis of whether they are consistent.

    Args:
        request: CompareClaimsRequest with the two claims to compare
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Dict with consistency analysis: status, confidence, and concise reason
    """
    try:
        result = await consistency_service.compare_two_claims(
            summary_claim=request.summary_claim,
            url_claim=request.url_claim,
            url_content=request.url_content
        )
        return result
    except Exception as e:
        logger.error(f"Claim comparison failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Claim comparison failed: {str(e)}")


@router.post("/check-summary-url", response_model=dict)
async def check_summary_url_consistency(
    request: CheckConsistencyRequest,
    _api_key: str = Depends(get_api_key)
) -> Dict[str, Any]:
    """
    Check consistency between website summary and full HTML content with optimized claim matching.

    This endpoint verifies if a website's summary/abstract accurately reflects its full content:

    1. Extracts the main content from the URL
    2. Breaks down both summary and content into atomic claims
    3. Uses embedding similarity to find related claims efficiently
    4. Performs detailed LLM analysis only on highly similar claim pairs
    5. Identifies conflicts, missing information, and overall consistency score

    Performance optimization:
    - Embedding similarity reduces LLM calls by 60-80%
    - Similarity threshold controls accuracy vs. token usage tradeoff
    - Batch processing for better efficiency

    Use case: Verify if website summaries/abstracts are truthful representations of the content.

    Args:
        request: CheckConsistencyRequest with website summary, source URL, and optimization settings
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Dict with detailed consistency analysis including score, conflicts, claim mappings, and performance metrics
    """
    try:
        result = await consistency_service.check_summary_url_consistency(
            summary=request.summary,
            url=request.url,
            enable_deep_analysis=request.enable_deep_analysis,
            similarity_threshold=request.similarity_threshold,
            use_cot_atomization=request.use_cot_atomization,
            use_cot_audit=request.use_cot_audit
        )
        return result
    except Exception as e:
        logger.error(f"Summary-URL consistency check failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Consistency check failed: {str(e)}")
