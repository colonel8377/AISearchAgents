"""Content API routes."""

from fastapi import APIRouter, Depends, HTTPException

from ..common import get_api_key, split_text_into_paragraphs
from ..schemas import (
    ExtractContentRequest, ContentExtractionResponse, AtomizeClaimsRequest,
    ClaimAtomizationResponse, AtomicClaimResponse, ParagraphClaimsResponse
)
from ...config.settings import settings
from ...utils.logger import get_logger
from .opinion import compare_claims

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/content", tags=["Content"])

@router.post("/extract", response_model=ContentExtractionResponse)
async def extract_content(
    request: ExtractContentRequest,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):

    """

    Extract academic content from URL, HTML, or text.

    This endpoint extracts the main title and body content from various sources,

    focusing on academic/informational content while excluding navigation,

    advertisements, and other non-content elements. Creates a new agent instance

    for each request (stateless operation).

    When compare_claims=True and summary is provided, performs rigorous claim-level comparison

    to evaluate whether URL content agrees or disagrees with summary claims. The comparison:

    1. Atomizes claims from both summary and URL content

    2. For each summary claim, searches URL claims to find agreement/disagreement

    3. Classifies relationships: agree (URL agrees with summary), disagree (URL disagrees), or missing (no relevant claim)

    4. Returns detailed comparison results with similarity scores, reasoning, and agreement statistics

    Args:

        request: Content extraction request with URL, HTML, or text

        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:

        ContentExtractionResponse with extracted title and main body, and optional claim comparison

    """

    try:

        # Create a new agent instance for each request (stateless)

        from ...agents.content_extractor.agent import ContentExtractorAgent

        agent = ContentExtractorAgent(

            model_name=settings.openai_model,

            api_key=settings.openai_api_key,

            api_base=settings.openai_api_base,

            temperature=settings.agent_temperature

        )

        if request.url:

            result = agent.extract_from_url(request.url, use_llm=request.use_llm, use_cot=request.use_cot, custom_few_shots=request.custom_few_shots)

        elif request.html:

            result = agent.extract_from_html(request.html, use_llm=request.use_llm, use_cot=request.use_cot, custom_few_shots=request.custom_few_shots)

        else:  # request.text

            result = agent.extract_from_text(request.text, title=request.title, use_llm=request.use_llm, use_cot=request.use_cot, custom_few_shots=request.custom_few_shots)

        # Perform claim comparison if requested

        claim_comparison = None

        if request.compare_claims and request.summary and result.main_body:

            logger.info("Performing claim-level comparison between summary and URL content")

            # Create claim atomizer agents

            from ...agents.claim_atomizer.agent import ClaimAtomizerAgent

            claim_atomizer = ClaimAtomizerAgent(

                model_name=settings.openai_model,

                api_key=settings.openai_api_key,

                api_base=settings.openai_api_base,

                temperature=settings.agent_temperature

            )

            # Atomize summary claims

            summary_atomization = claim_atomizer.atomize_text(

                text=request.summary,

                use_cot=request.use_cot

            )

            # Atomize URL content claims (split into paragraphs)

            url_atomization = claim_atomizer.atomize_text(

                text=result.main_body,

                use_cot=request.use_cot,

                split_into_paragraphs=True

            )

            # Compare claims

            claim_comparison = await compare_claims(

                summary_claims=summary_atomization.atomic_claims,

                url_claims=url_atomization.atomic_claims,

                use_cot=request.use_cot,

                use_few_shots=request.use_few_shots

            )

            logger.info(f"Claim comparison complete: {len(claim_comparison.comparisons)} comparisons")

        response_dict = result.__dict__

        # Add paragraph information if main_body exists

        if result.main_body:

            response_dict["paragraphs"] = split_text_into_paragraphs(result.main_body)

        response_dict["claim_comparison"] = claim_comparison

        return ContentExtractionResponse(**response_dict)

    except Exception as e:

        logger.error(f"Failed to extract content: {e}", exc_info=True)

        raise HTTPException(status_code=500, detail=f"Failed to extract content: {str(e)}")

# Claim Atomizer Endpoints

@router.post("/atomize", response_model=ClaimAtomizationResponse)
async def atomize_claims(
    request: AtomizeClaimsRequest,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)

):

    """

    Decompose text into atomic claims.

    This endpoint breaks down provided text into independent, verifiable atomic claims,

    each containing only one factual point. Creates a new agent instance for each

    request (stateless operation).

    Args:

        request: Claim atomization request with text and options

        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:

        ClaimAtomizationResponse with atomic claims

    """

    try:

        # Create a new agent instance for each request (stateless)

        from ...agents.claim_atomizer.agent import ClaimAtomizerAgent

        agent = ClaimAtomizerAgent(

            model_name=settings.openai_model,

            api_key=settings.openai_api_key,

            api_base=settings.openai_api_base,

            temperature=settings.agent_temperature

        )

        result = agent.atomize_text(

            text=request.text,

            use_cot=request.use_cot,

            custom_few_shots=request.custom_few_shots,

            split_into_paragraphs=request.split_into_paragraphs

        )

        # Convert to response format

        atomic_claims = [

            AtomicClaimResponse(**claim.__dict__) for claim in result.atomic_claims

        ]

        # Convert paragraphs to response format

        paragraphs = []

        for para in result.paragraphs:

            para_claims = [

                AtomicClaimResponse(**claim.__dict__) for claim in para.atomic_claims

            ]

            paragraphs.append(ParagraphClaimsResponse(

                paragraph_index=para.paragraph_index,

                paragraph_text=para.paragraph_text,

                atomic_claims=para_claims

            ))

        return ClaimAtomizationResponse(

            atomic_claims=atomic_claims,  # Keep for backward compatibility

            paragraphs=paragraphs,        # New: claims grouped by paragraphs

            original_text=result.original_text,

            metadata=result.metadata

        )

    except Exception as e:

        logger.error(f"Failed to atomize claims: {e}", exc_info=True)

        raise HTTPException(status_code=500, detail=f"Failed to atomize claims: {str(e)}")


