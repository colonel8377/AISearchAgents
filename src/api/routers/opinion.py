"""Opinion API routes."""

from typing import List

from fastapi import APIRouter, Depends, HTTPException

from ..common import get_api_key, split_text_into_paragraphs
from ..schemas import (
    ExtractAndCleanRequest, ExtractAndCleanResponse, ExtractOpinionsRequest,
    ExtractOpinionsResponse, AnalyzeUrlRequest, BiasScoreRequest, BiasScoreResponse,
    BiasDistributionResponse, AtomicOpinionResponse, MBFCMetadataResponse,
    ClaimComparisonResponse, AtomicClaimResponse
)
from ...agents.web_opinion_extractor import WebOpinionAnalyzer, LogicMode
from ...agents.claim_atomizer.agent import AtomicClaim
from ...config.settings import settings
from ...utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/opinion", tags=["Opinion"])

@router.post("/extract-clean", response_model=ExtractAndCleanResponse)
async def extract_and_clean_from_url(
    request: ExtractAndCleanRequest,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):

    """

    Combined API: Extract HTML from URL and clean to text in one step.

    This endpoint combines HTML extraction and cleaning into a single API call,

    saving tokens by avoiding the need to pass large HTML content between calls.

    It fetches HTML from the URL and directly returns cleaned text using BeautifulSoup.

    If a proxy is needed, configure it using the OPENAI_PROXY setting or 

    HTTP_PROXY/HTTPS_PROXY environment variables.

    Args:

        request: ExtractAndCleanRequest with URL

        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:

        ExtractAndCleanResponse with cleaned text and title or error

    """

    logger.info(f"Extracting and cleaning HTML from URL: {request.url}")

    try:

        # Reuse OpenAI proxy settings if configured

        proxy = settings.openai_proxy or None

        analyzer = WebOpinionAnalyzer(

            execution_mode=settings.default_execution_mode,

            proxy=proxy

        )

        # Step 1: Extract HTML from URL

        html = analyzer.extract_html(request.url)

        if html is None:

            return ExtractAndCleanResponse(

                url=request.url,

                error="fetch_failed",

                error_message="Failed to fetch HTML from URL"

            )

        # Step 2: Clean HTML to extract text

        text, title = analyzer.clean_html(html)

        if text is None:

            return ExtractAndCleanResponse(

                url=request.url,

                error="cleaning_failed",

                error_message="Failed to clean HTML content"

            )

        # Split text into paragraphs

        paragraphs = split_text_into_paragraphs(text)

        return ExtractAndCleanResponse(

            url=request.url,

            text=text,

            paragraphs=paragraphs,

            title=title,

            text_length=len(text)

        )

    except Exception as e:

        logger.error(f"Failed to extract and clean from {request.url}: {e}", exc_info=True)

        return ExtractAndCleanResponse(

            url=request.url,

            error="extraction_failed",

            error_message=str(e)

        )

@router.post("/extract-opinions", response_model=ExtractOpinionsResponse)
async def extract_atomic_opinions(
    request: ExtractOpinionsRequest,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)

):

    """

    Extract atomic opinions from URL or text content.

    This endpoint can analyze content in two ways:

    1. Provide a URL: The endpoint will fetch HTML, extract text, and analyze it

    2. Provide text + title: The endpoint will analyze the provided text directly

    Each atomic opinion includes:

    - Text of the opinion

    - Opinion type (fact or opinion)

    - Bias probability distribution (left, right, neutral)

    - Optional reasoning (if CoT mode is enabled)

    Args:

        request: ExtractOpinionsRequest with either URL or text+title

        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:

        ExtractOpinionsResponse with extracted opinions and bias scores

    """

    try:

        execution_mode = request.execution_mode or settings.default_execution_mode

        analyzer = WebOpinionAnalyzer(

            execution_mode=execution_mode,

            proxy=settings.openai_proxy if settings.openai_proxy else None

        )

        # Handle URL input: fetch and extract content

        if request.url:

            logger.info(f"Extracting opinions from URL: {request.url}")

            # Step 1: Fetch HTML

            html = analyzer.extract_html(request.url)

            if html is None:

                return ExtractOpinionsResponse(

                    url=request.url,

                    title=None,

                    atomic_opinions=[],

                    facts=[],

                    opinions=[],

                    text_length=0,

                    truncated=False,

                    error="fetch_failed",

                    error_message="Failed to fetch HTML from URL"

                )

            # Step 2: Clean HTML to extract text and title

            text, title = analyzer.clean_html(html)

            if text is None:

                return ExtractOpinionsResponse(

                    url=request.url,

                    title=None,

                    atomic_opinions=[],

                    facts=[],

                    opinions=[],

                    text_length=0,

                    truncated=False,

                    error="cleaning_failed",

                    error_message="Failed to clean HTML content"

                )

            # Step 3: Analyze the extracted text

            result = analyzer.analyze_text(text, url=request.url, title=title, use_llm=request.use_llm, use_cot=request.use_cot, custom_few_shots=request.custom_few_shots)

        else:

            # Handle direct text input

            logger.info(f"Extracting opinions from text ({len(request.text)} chars)")

            result = analyzer.analyze_text(request.text, url=None, title=request.title, use_llm=request.use_llm, use_cot=request.use_cot, custom_few_shots=request.custom_few_shots)

        # Check for errors

        if result.extraction_metadata and "error" in result.extraction_metadata:

            return ExtractOpinionsResponse(

                url=request.url if request.url else None,

                title=request.title if not request.url else result.title,

                atomic_opinions=[],

                facts=[],

                opinions=[],

                text_length=len(request.text) if request.text else 0,

                truncated=False,

                error=result.extraction_metadata["error"],

                error_message=result.extraction_metadata.get("error_message", "Unknown error")

            )

        # Convert opinions to response format

        atomic_opinions = [_convert_atomic_opinion(op) for op in result.atomic_opinions]

        facts = [_convert_atomic_opinion(op) for op in result.facts]

        opinions = [_convert_atomic_opinion(op) for op in result.opinions]

        overall_bias = None

        if result.overall_bias_distribution:

            overall_bias = _convert_bias_distribution(result.overall_bias_distribution)

        return ExtractOpinionsResponse(

            url=result.url,

            title=result.title,

            atomic_opinions=atomic_opinions,

            facts=facts,

            opinions=opinions,

            overall_bias_distribution=overall_bias,

            text_length=result.text_length,

            truncated=result.truncated

        )

    except ValueError as e:

        # Handle validation errors (e.g., missing URL or text)

        logger.error(f"Validation error: {e}", exc_info=True)

        return ExtractOpinionsResponse(

            url=request.url if request.url else None,

            title=request.title if request.title else None,

            atomic_opinions=[],

            facts=[],

            opinions=[],

            text_length=0,

            truncated=False,

            error="validation_failed",

            error_message=str(e)

        )

    except Exception as e:

        logger.error(f"Failed to extract opinions: {e}", exc_info=True)

        return ExtractOpinionsResponse(

            url=request.url if request.url else None,

            title=request.title if request.title else None,

            atomic_opinions=[],

            facts=[],

            opinions=[],

            text_length=0,

            truncated=False,

            error="extraction_failed",

            error_message=str(e)

        )

@router.post("/analyze", response_model=ExtractOpinionsResponse)
async def analyze_url_complete(
    request: AnalyzeUrlRequest,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)

):

    """

    Complete analysis pipeline: Extract and analyze opinions from URL using WebOpinionEngine.

    This endpoint uses the new WebOpinionEngine with configurable:

    - Logic modes: LOCAL_CHAIN, NO_CHAIN, PURE_ONLINE

    - MBFC prior: Optional database lookup for bias prior

    - Few-shot examples: User can enable/disable or provide custom examples

    - Caching: Results are cached by hash to avoid reprocessing

    Args:

        request: AnalyzeUrlRequest with URL, mode, use_mbfc, use_few_shots, and optional shots

        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:

        ExtractOpinionsResponse with all extracted opinions and bias scores, including MBFC metadata if available

    """

    logger.info(f"Analyzing URL: {request.url}, mode={request.mode}, use_mbfc={request.use_mbfc}, use_few_shots={request.use_few_shots}")

    try:

        # Parse mode

        try:

            mode = LogicMode(request.mode.upper())

        except ValueError:

            mode = LogicMode.LOCAL_CHAIN

            logger.warning(f"Invalid mode '{request.mode}', using LOCAL_CHAIN")

        mode_str = mode.value

        # Check cache first

        cache = get_cache()

        cached_result = cache.get(

            url=request.url,

            mode=mode_str,

            use_mbfc=request.use_mbfc,

            use_few_shots=request.use_few_shots,

            atomizer_shots=request.atomizer_shots,

            scorer_shots=request.scorer_shots

        )

        if cached_result:

            logger.info(f"Returning cached result for URL: {request.url}")

            result = cached_result

        else:

            # Initialize engine (db_path comes from settings)

            engine = WebOpinionEngine(

                proxy=settings.openai_proxy if settings.openai_proxy else None

            )

            # Run pipeline

            result = engine.run(

                url=request.url,

                mode=mode,

                use_mbfc=request.use_mbfc,

                use_few_shots=request.use_few_shots,

                atomizer_shots=request.atomizer_shots,

                scorer_shots=request.scorer_shots

            )

            # Cache the result

            cache.set(

                url=request.url,

                mode=mode_str,

                use_mbfc=request.use_mbfc,

                use_few_shots=request.use_few_shots,

                result=result,

                atomizer_shots=request.atomizer_shots,

                scorer_shots=request.scorer_shots

            )

        # Convert to response format

        # Note: For the /api/v1/web-opinion/analyze endpoint, we follow the

        # v1 API contract where `atomic_opinions` should only contain items

        # labeled as "opinion" (no "fact" entries). The `facts` list is the

        # canonical place for fact-type units.

        atomic_opinions = []

        facts = []

        opinions = []

        if result.get("atomic_units"):

            for unit in result["atomic_units"]:

                # Create a simple bias distribution for atomic units (neutral by default)

                bias_dist = BiasDistributionResponse(

                    left=0.33,

                    right=0.33,

                    neutral=0.34,

                    dominant_bias="neutral",

                    bias_score=0.0

                )

                opinion_resp = AtomicOpinionResponse(

                    text=unit["statement"],

                    opinion_type=unit["type"],

                    bias_probabilities=bias_dist,

                    original_sentence=unit.get("original_sentence"),

                    confidence=unit.get("confidence"),

                    reasoning=unit.get("reasoning")

                )

                if unit["type"] == "fact":

                    facts.append(opinion_resp)

                else:

                    # Only opinions are included in atomic_opinions for this endpoint

                    opinions.append(opinion_resp)

                    atomic_opinions.append(opinion_resp)

        # Get overall bias from result

        overall_bias = None

        if result.get("bias_analysis") and result["bias_analysis"].get("distribution"):

            dist = result["bias_analysis"]["distribution"]

            overall_bias = BiasDistributionResponse(

                left=dist["left"],

                right=dist["right"],

                neutral=dist["neutral"],

                dominant_bias=result["bias_analysis"].get("dominant_bias", "neutral"),

                bias_score=dist["left"] * -1.0 + dist["right"] * 1.0

            )

        # Get MBFC metadata if available and enabled

        mbfc_metadata = None

        if request.use_mbfc and result.get("metadata"):

            metadata = result["metadata"]

            mbfc_metadata = MBFCMetadataResponse(

                source_name=metadata.get("source_name"),

                match_type=metadata.get("match_type"),

                bias_rating=metadata.get("bias_rating"),

                factual_reporting=metadata.get("factual_reporting"),

                raw_db_row=metadata.get("raw_db_row")

            )

        # Get MBFC influence note from bias analysis

        mbfc_influence_note = None

        if result.get("bias_analysis"):

            mbfc_influence_note = result["bias_analysis"].get("mbfc_influence_note")

        return ExtractOpinionsResponse(

            url=result["url"],

            title=result["article"].get("title"),

            atomic_opinions=atomic_opinions,

            facts=facts,

            opinions=opinions,

            overall_bias_distribution=overall_bias,

            mbfc_metadata=mbfc_metadata,

            mbfc_influence_note=mbfc_influence_note,

            text_length=result["article"].get("text_length", 0),

            truncated=False

        )

    except Exception as e:

        logger.error(f"Failed to analyze URL {request.url}: {e}", exc_info=True)

        return ExtractOpinionsResponse(

            url=request.url,

            title=None,

            atomic_opinions=[],

            facts=[],

            opinions=[],

            overall_bias_distribution=None,

            mbfc_metadata=None,

            text_length=0,

            truncated=False,

            error="analysis_failed",

            error_message=str(e)

        )


@router.post("/bias-score", response_model=BiasScoreResponse)
async def get_overall_bias_score(
    request: BiasScoreRequest,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)

):

    """

    Get overall bias score from URL.

    This endpoint provides a simplified API that returns only the overall

    bias distribution for a URL, without detailed opinion breakdowns.

    Uses WebOpinionEngine with MBFC support and caching to save tokens.

    Args:

        request: BiasScoreRequest with URL, mode, use_mbfc, and use_few_shots

        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:

        BiasScoreResponse with overall bias score and MBFC metadata if available

    """

    logger.info(f"Getting bias score for URL: {request.url}, mode={request.mode}, use_mbfc={request.use_mbfc}")

    try:

        # Parse mode

        try:

            mode = LogicMode(request.mode.upper())

        except ValueError:

            mode = LogicMode.LOCAL_CHAIN

            logger.warning(f"Invalid mode '{request.mode}', using LOCAL_CHAIN")

        mode_str = mode.value

        # Check cache first

        cache = get_cache()

        cached_result = cache.get(

            url=request.url,

            mode=mode_str,

            use_mbfc=request.use_mbfc,

            use_few_shots=request.use_few_shots,

            atomizer_shots=None,

            scorer_shots=None

        )

        if cached_result:

            logger.info(f"Returning cached result for URL: {request.url}")

            result = cached_result

        else:

            # Initialize engine (db_path comes from settings)

            engine = WebOpinionEngine(

                proxy=settings.openai_proxy if settings.openai_proxy else None

            )

            # Run pipeline

            result = engine.run(

                url=request.url,

                mode=mode,

                use_mbfc=request.use_mbfc,

                use_few_shots=request.use_few_shots

            )

            # Cache the result

            cache.set(

                url=request.url,

                mode=mode_str,

                use_mbfc=request.use_mbfc,

                use_few_shots=request.use_few_shots,

                result=result,

                atomizer_shots=None,

                scorer_shots=None

            )

        # Get overall bias from result

        overall_bias = None

        if result.get("bias_analysis") and result["bias_analysis"].get("distribution"):

            dist = result["bias_analysis"]["distribution"]

            overall_bias = BiasDistributionResponse(

                left=dist["left"],

                right=dist["right"],

                neutral=dist["neutral"],

                dominant_bias=result["bias_analysis"].get("dominant_bias", "neutral"),

                bias_score=dist["left"] * -1.0 + dist["right"] * 1.0

            )

        # Get MBFC metadata if available and enabled

        mbfc_metadata = None

        if request.use_mbfc and result.get("metadata"):

            metadata = result["metadata"]

            mbfc_metadata = MBFCMetadataResponse(

                source_name=metadata.get("source_name"),

                match_type=metadata.get("match_type"),

                bias_rating=metadata.get("bias_rating"),

                factual_reporting=metadata.get("factual_reporting"),

                raw_db_row=metadata.get("raw_db_row")

            )

        # Get MBFC influence note from bias analysis

        mbfc_influence_note = None

        if result.get("bias_analysis"):

            mbfc_influence_note = result["bias_analysis"].get("mbfc_influence_note")

        # Count opinions and facts

        opinions_count = 0

        facts_count = 0

        if result.get("atomic_units"):

            for unit in result["atomic_units"]:

                if unit.get("type") == "opinion":

                    opinions_count += 1

                elif unit.get("type") == "fact":

                    facts_count += 1

        return BiasScoreResponse(

            url=result["url"],

            overall_bias_distribution=overall_bias,

            mbfc_metadata=mbfc_metadata,

            mbfc_influence_note=mbfc_influence_note,

            opinions_count=opinions_count,

            facts_count=facts_count

        )

    except Exception as e:

        logger.error(f"Failed to get bias score for {request.url}: {e}", exc_info=True)

        return BiasScoreResponse(

            url=request.url,

            overall_bias_distribution=None,

            mbfc_metadata=None,

            opinions_count=0,

            facts_count=0,

            error="analysis_failed",

            error_message=str(e)

        )

# ===========================

# Academic Content Analysis Endpoints

# ===========================

# Content Extractor Endpoints

async def compare_claims(
    summary_claims: List[AtomicClaim],
    url_claims: List[AtomicClaim],

    use_cot: bool = False

) -> ClaimComparisonResponse:

    """

    Compare claims from summary with URL content to determine support.

    For each claim in the summary, this function checks whether it is supported

    by claims in the URL content. This evaluates how accurately the summary

    represents the actual content.

    Args:

        summary_claims: List of atomic claims extracted from summary

        url_claims: List of atomic claims extracted from URL content

        use_cot: Whether to use Chain of Thought reasoning

    Returns:

        ClaimComparisonResponse with comparison results showing which summary

        claims are supported by URL content

    """

    from langchain_openai import ChatOpenAI

    from langchain_core.messages import SystemMessage, HumanMessage

    from ...utils.llm_client import llm_manager

    import json

    logger.info(f"Comparing {len(summary_claims)} summary claims against {len(url_claims)} URL claims")

    # Create LLM client for claim matching

    http_client = llm_manager.get_http_client(proxy=settings.openai_proxy)

    llm = ChatOpenAI(

        model_name=settings.openai_model,

        api_key=settings.openai_api_key,

        base_url=settings.openai_api_base,

        temperature=0.1,  # Lower temperature for more consistent matching

        max_retries=settings.openai_max_retries,

        timeout=settings.openai_timeout,

        http_client=http_client

    )

    # Prepare claims for LLM with clear indexing

    summary_claims_list = [f"CLAIM_{i}: {claim.text}" for i, claim in enumerate(summary_claims)]

    url_claims_list = [f"CLAIM_{i}: {claim.text}" for i, claim in enumerate(url_claims)]

    summary_claims_text = "\n".join(summary_claims_list)

    url_claims_text = "\n".join(url_claims_list)

    system_prompt = """You are a rigorous academic content analyst specializing in claim verification and comparison. Your task is to evaluate whether URL content agrees or disagrees with claims from a summary, considering multiple aspects.

For each summary claim, you must conduct a thorough analysis:

1. SEARCH: Find the most relevant URL claim(s) that relate to the summary claim

2. EVALUATE MULTIPLE ASPECTS: Analyze agreement/disagreement across different dimensions:

   - Factual accuracy: Do the facts match?

   - Semantic meaning: Do they express the same meaning?

   - Tone/emphasis: Are they consistent in emphasis or tone?

   - Context: Do they align in context?

   - Specificity: Are details consistent?

3. CLASSIFY: Determine the overall relationship as one of:

   - "agree": The URL claim AGREES with or supports the summary claim (consensus across aspects)

   - "disagree": The URL claim DISAGREES with or contradicts the summary claim (conflict in key aspects)

   - "missing": No relevant claim found in URL (cannot determine agreement/disagreement)

CRITICAL REQUIREMENTS FOR ACADEMIC RIGOR:

- Be precise: "agree" requires substantial alignment across multiple aspects, not just partial similarity

- Be thorough: "disagree" can occur in different aspects - identify which aspects conflict

- Provide detailed reasoning: Explain exactly which aspects agree/disagree and why

- Consider nuance: A claim can agree in some aspects but disagree in others - determine the overall relationship

- Be strict: "missing" means no claim in URL is relevant enough to evaluate agreement/disagreement

- Each summary claim should match to AT MOST ONE URL claim (the best/most relevant match)

- Similarity score (0.0 to 1.0): Only for "agree" or "disagree", indicating overall semantic similarity

Return a JSON array where each object contains:

- "summary_claim_id": The ID/index of the summary claim (0-based integer)

- "summary_claim_text": The exact text of the summary claim

- "url_claim_id": The ID/index of the best matching URL claim (null if "missing")

- "url_claim_text": The exact text of the matching URL claim (null if "missing")

- "relationship": One of "agree", "disagree", or "missing"

- "similarity_score": A float between 0.0 and 1.0 (null if relationship is "missing")

- "reasoning": REQUIRED - Detailed explanation including:

  * Which aspects were analyzed (factual accuracy, semantic meaning, tone, context, specificity)

  * Which aspects show agreement (if any)

  * Which aspects show disagreement (if any)

  * Why the overall relationship was determined to be agree/disagree/missing

  * What specific evidence from the claims supports this conclusion

  * Any nuances or partial agreements/disagreements that were considered"""

    if use_cot:

        system_prompt += "\n\nUse Chain of Thought reasoning: For each summary claim, first analyze all URL claims to find potential matches, then evaluate semantic similarity, determine the relationship type, and explain your reasoning step by step."

    user_message = f"""Evaluate whether the following summary claims are supported by the URL content claims.

SUMMARY CLAIMS (to be evaluated):

{summary_claims_text}

URL CONTENT CLAIMS (to search for support):

{url_claims_text}

For each summary claim, determine if it is supported by the URL content. Return a JSON array of comparison results, one entry per summary claim."""

    messages = [

        SystemMessage(content=system_prompt),

        HumanMessage(content=user_message)

    ]

    try:

        response = llm.invoke(messages)

        response_text = response.content

        # Parse JSON from response (handle markdown code blocks if present)

        response_text = response_text.strip()

        if response_text.startswith("```json"):

            response_text = response_text[7:]

        if response_text.startswith("```"):

            response_text = response_text[3:]

        if response_text.endswith("```"):

            response_text = response_text[:-3]

        response_text = response_text.strip()

        comparisons_data = json.loads(response_text)

        # Convert to ClaimComparisonResult objects

        comparison_results = []

        for comp_data in comparisons_data:

            # Validate and normalize relationship type

            rel = comp_data.get("relationship", "missing").lower()

            # Handle legacy terms for backward compatibility

            if rel == "consistent":

                rel = "agree"

            elif rel == "contradictory":

                rel = "disagree"

            if rel not in ["agree", "disagree", "missing"]:

                logger.warning(f"Invalid relationship type '{rel}', defaulting to 'missing'")

                rel = "missing"

            # Ensure reasoning is provided (required for academic rigor)

            reasoning = comp_data.get("reasoning", "")

            if not reasoning or len(reasoning.strip()) == 0:

                reasoning = f"Relationship determined to be '{rel}' but no detailed reasoning provided."

                logger.warning(f"Missing reasoning for claim {comp_data.get('summary_claim_id', 'unknown')}")

            comparison_results.append(ClaimComparisonResult(

                summary_claim_id=str(comp_data.get("summary_claim_id", "")),

                summary_claim_text=comp_data.get("summary_claim_text", ""),

                url_claim_id=str(comp_data.get("url_claim_id")) if comp_data.get("url_claim_id") is not None else None,

                url_claim_text=comp_data.get("url_claim_text"),

                relationship=rel,

                similarity_score=comp_data.get("similarity_score"),

                reasoning=reasoning

            ))

        # Ensure we have results for all summary claims

        if len(comparison_results) != len(summary_claims):

            logger.warning(f"Number of comparison results ({len(comparison_results)}) does not match number of summary claims ({len(summary_claims)})")

            # Fill in missing comparisons

            existing_ids = {int(comp.summary_claim_id) for comp in comparison_results if comp.summary_claim_id.isdigit()}

            for i, claim in enumerate(summary_claims):

                if i not in existing_ids:

                    comparison_results.append(ClaimComparisonResult(

                        summary_claim_id=str(i),

                        summary_claim_text=claim.text,

                        relationship="missing",

                        reasoning="Comparison result was not returned by LLM"

                    ))

        # Convert claims to response format

        summary_claims_resp = [AtomicClaimResponse(**claim.__dict__) for claim in summary_claims]

        url_claims_resp = [AtomicClaimResponse(**claim.__dict__) for claim in url_claims]

        # Calculate comprehensive statistics with detailed categorization

        relationship_counts = {}

        agree_claims = []

        disagree_claims = []

        missing_claims = []

        total_similarity = 0.0

        similarity_count = 0

        for comp in comparison_results:

            rel = comp.relationship

            relationship_counts[rel] = relationship_counts.get(rel, 0) + 1

            # Categorize claims for detailed statistics

            claim_info = {

                "summary_claim_id": comp.summary_claim_id,

                "summary_claim_text": comp.summary_claim_text,

                "url_claim_id": comp.url_claim_id,

                "url_claim_text": comp.url_claim_text,

                "similarity_score": comp.similarity_score,

                "reasoning": comp.reasoning

            }

            if rel == "agree":

                agree_claims.append(claim_info)

            elif rel == "disagree":

                disagree_claims.append(claim_info)

            else:  # missing

                missing_claims.append(claim_info)

            if comp.similarity_score is not None:

                total_similarity += comp.similarity_score

                similarity_count += 1

        total_summary = len(summary_claims)

        agree_count = relationship_counts.get("agree", 0)

        disagree_count = relationship_counts.get("disagree", 0)

        missing_count = relationship_counts.get("missing", 0)

        # Calculate agreement/disagreement metrics

        agree_rate = agree_count / total_summary if total_summary > 0 else 0.0

        disagree_rate = disagree_count / total_summary if total_summary > 0 else 0.0

        missing_rate = missing_count / total_summary if total_summary > 0 else 0.0

        avg_similarity = total_similarity / similarity_count if similarity_count > 0 else None

        # Comprehensive statistics with categorized lists for academic analysis

        statistics = {

            # Overall counts

            "total_summary_claims": total_summary,

            "total_url_claims": len(url_claims),

            # Counts by relationship type

            "agree_count": agree_count,

            "disagree_count": disagree_count,

            "missing_count": missing_count,

            # Rates (percentages)

            "agree_rate": round(agree_rate, 4),

            "disagree_rate": round(disagree_rate, 4),

            "missing_rate": round(missing_rate, 4),

            # Similarity metrics

            "average_similarity_score": round(avg_similarity, 4) if avg_similarity is not None else None,

            "similarity_count": similarity_count,  # Number of claims with similarity scores

            # Detailed categorized lists for academic analysis

            "agree_claims": agree_claims,  # List of claims where URL agrees with summary (with reasoning)

            "disagree_claims": disagree_claims,  # List of claims where URL disagrees with summary (with reasoning)

            "missing_claims": missing_claims  # List of claims with no relevant URL claim (with reasoning)

        }

        return ClaimComparisonResponse(

            summary_claims=summary_claims_resp,

            url_claims=url_claims_resp,

            comparisons=comparison_results,

            statistics=statistics

        )

    except Exception as e:

        logger.error(f"Failed to compare claims: {e}", exc_info=True)

        # Return a fallback response with all claims marked as missing

        summary_claims_resp = [AtomicClaimResponse(**claim.__dict__) for claim in summary_claims]

        url_claims_resp = [AtomicClaimResponse(**claim.__dict__) for claim in url_claims]

        comparison_results = [

            ClaimComparisonResult(

                summary_claim_id=str(i),

                summary_claim_text=claim.text,

                relationship="missing",

                reasoning=f"Error during comparison: {str(e)}"

            )

            for i, claim in enumerate(summary_claims)

        ]

        return ClaimComparisonResponse(

            summary_claims=summary_claims_resp,

            url_claims=url_claims_resp,

            comparisons=comparison_results,

            statistics={"error": str(e)}

        )
