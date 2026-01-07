"""Content API routes - View layer only."""

from fastapi import APIRouter, Depends, HTTPException

from src.shared.utils import get_logger
from ..common import get_api_key, split_text_into_paragraphs, content_service
from ..schemas import (
    ExtractContentRequest, ContentExtractionResponse, AtomizeClaimsRequest,
    ClaimAtomizationResponse, AtomicClaimResponse, ParagraphClaimsResponse
)

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/content", tags=["Content"])


@router.post("/extract", response_model=ContentExtractionResponse)
async def extract_content(
    request: ExtractContentRequest,
    _api_key: str = Depends(get_api_key)
) -> ContentExtractionResponse:
    """
    Extract academic content from URL, HTML, or text.

    This endpoint extracts the main title and body content from various sources,
    focusing on academic/informational content while excluding navigation,
    advertisements, and other non-content elements.

    Args:
        request: Content extraction request with URL, HTML, or text
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        ContentExtractionResponse with extracted title and main body
    """
    try:
        result_dict = await content_service.extract_content(
            url=request.url,
            text=request.text,
            title=request.title,
            use_llm=request.use_llm,
            use_cot=request.use_cot,
            custom_few_shots=request.custom_few_shots
        )

        # Add paragraph information if main_body exists
        if result_dict.get("main_body"):
            result_dict["paragraphs"] = split_text_into_paragraphs(result_dict["main_body"])

        return ContentExtractionResponse(**result_dict)

    except Exception as e:
        logger.error(f"Failed to extract content: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to extract content: {str(e)}")


@router.post("/atomize", response_model=ClaimAtomizationResponse)
async def atomize_claims(
    request: AtomizeClaimsRequest,
    _api_key: str = Depends(get_api_key)
) -> ClaimAtomizationResponse:
    """
    Decompose text into atomic claims.

    This endpoint breaks down provided text into independent, verifiable atomic claims,
    each containing only one factual point.

    Args:
        request: Claim atomization request with text and options
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        ClaimAtomizationResponse with atomic claims
    """
    try:
        result_dict = await content_service.atomize_claims(
            text=request.text,
            use_cot=request.use_cot,
            split_into_paragraphs=request.split_into_paragraphs,
            use_few_shots=request.use_few_shots
        )

        # Convert to response format
        atomic_claims = [
            AtomicClaimResponse(**claim.__dict__) for claim in result_dict["atomic_claims"]
        ]

        # Convert paragraphs to response format
        paragraphs = []
        for para in result_dict["paragraphs"]:
            para_claims = [
                AtomicClaimResponse(**claim.__dict__) for claim in para.atomic_claims
            ]
            paragraphs.append(ParagraphClaimsResponse(
                paragraph_index=para.paragraph_index,
                paragraph_text=para.paragraph_text,
                atomic_claims=para_claims
            ))

        return ClaimAtomizationResponse(
            atomic_claims=atomic_claims,
            paragraphs=paragraphs,
            original_text=result_dict["original_text"],
            metadata=result_dict["metadata"]
        )

    except Exception as e:
        logger.error(f"Failed to atomize claims: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to atomize claims: {str(e)}")
