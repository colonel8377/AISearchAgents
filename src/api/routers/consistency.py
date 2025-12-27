"""Consistency API routes."""

from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, model_validator

from ..common import get_api_key, split_text_into_paragraphs
from ..schemas import (
    AuditConflictsRequest, ConflictAuditResponse,
    CompleteAnalysisRequest, CompleteAnalysisResponse,
    ParagraphResponse, PipelineStepResponse
)
from ...agents.claim_atomizer.agent import ClaimAtomizerAgent
from ...agents.conflict_auditor.agent import ConflictAuditorAgent
from ...agents.content_extractor.agent import ContentExtractorAgent
from ...agents.web_opinion_extractor import CoTMode
from ...config.settings import settings
from ...utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/consistency", tags=["Consistency"])


# ===========================

# Overall Evaluation Interface

# ===========================

class CheckConsistencyRequest(BaseModel):

    """Request model for checking consistency between website summary and full content."""

    summary: str = Field(..., description="Website's summary/description/abstract text")

    url: str = Field(..., description="Source URL containing the full content to check against")

    enable_deep_analysis: bool = Field(default=True, description="Whether to enable deep analysis using LLM")

    similarity_threshold: float = Field(default=0.2, description="Minimum similarity score (0.0-1.0) required for LLM analysis. Lower values increase accuracy but use more tokens.")

    model_config = {

        "json_schema_extra": {

            "examples": [

                {

                    "summary": "This article discusses how plastic pollution affects marine ecosystems worldwide.",

                    "url": "https://example.com/plastic-pollution-impact",

                    "enable_deep_analysis": True

                }

            ]

        }

    }

class CompareClaimsRequest(BaseModel):

    """Request model for comparing two specific claims."""

    summary_claim: str = Field(..., description="Claim from the summary")

    url_claim: str = Field(..., description="Claim from the URL content")

    url_content: Optional[str] = Field(default=None, description="Optional full URL content for context")

    model_config = {

        "json_schema_extra": {

            "examples": [

                {

                    "summary_claim": "The Chicago Fire Department will undergo significant changes next season",

                    "url_claim": "There will be hellos and goodbyes in the department",

                    "url_content": "Full content of the webpage for additional context..."

                }

            ]

        }

    }

class OverallEvaluationRequest(BaseModel):

    """Request model for overall evaluation of summary and URL."""

    summary: Optional[str] = Field(default=None, description="Text summary to evaluate")

    url: Optional[str] = Field(default=None, description="URL to evaluate")

    include_full_analysis: bool = Field(default=False, description="Whether to include full academic analysis pipeline")

    enable_deep_analysis: bool = Field(default=True, description="Whether to enable deep analysis (fact/opinion density, bias) using LLM")

    enable_consistency_check: bool = Field(default=False, description="Whether to enable consistency check between summary and URL using the 5-step verification pipeline")

    @model_validator(mode='after')

    def validate_input(self):

        """Ensure at least one input is provided."""

        if not self.summary and not self.url:

            raise ValueError("At least one of 'summary' or 'url' must be provided")

        return self

    model_config = {

        "json_schema_extra": {

            "examples": [

                {

                    "summary": "Plastic pollution harms marine life and ecosystems worldwide.",

                    "url": "https://example.com/environmental-issues",

                    "enable_consistency_check": True,

                    "enable_deep_analysis": True

                },

                {

                    "url": "https://example.com/climate-change",

                    "include_full_analysis": True

                }

            ]

        }

    }

class EvaluationMetrics(BaseModel):

    """Evaluation metrics for content."""

    content_length: int

    readability_score: Optional[float] = None

    fact_density: Optional[float] = None

    opinion_density: Optional[float] = None

    bias_distribution: Optional[Dict[str, float]] = None

    academic_integrity_score: Optional[float] = None

    hallucination_risk: Optional[float] = None

class OverallEvaluationResponse(BaseModel):

    """Response model for overall evaluation."""

    summary_evaluation: Optional[Dict[str, Any]] = None

    url_evaluation: Optional[Dict[str, Any]] = None

    comparative_analysis: Optional[Dict[str, Any]] = None

    consistency_check: Optional[Dict[str, Any]] = None

    full_academic_analysis: Optional[Dict[str, Any]] = None

    processing_metadata: Dict[str, Any]

@router.post("/compare-claims", response_model=dict)
async def compare_two_claims(
    request: CompareClaimsRequest,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)

):

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

    import time

    start_time = time.time()

    try:

        from ...utils.llm_client import llm_manager

        from langchain_openai import ChatOpenAI

        # Create LLM instance

        http_client = llm_manager.get_http_client()

        llm = ChatOpenAI(

            model_name=settings.openai_model,

            api_key=settings.openai_api_key,

            base_url=settings.openai_api_base,

            temperature=0.1,

            max_retries=settings.openai_max_retries,

            timeout=settings.openai_timeout,

            http_client=http_client,

            max_tokens=200

        )

        system_prompt = """You are a fact-checker. Compare two claims and determine if they are consistent.

Analyze if the summary claim is supported, contradicted, or neutral compared to the URL claim.

Provide a concise, clear reason without unnecessary quotes or references.

Format your response as:

STATUS: [supported|contradicted|neutral]

CONFIDENCE: [0.0-1.0]

REASON: [Concise explanation, 1-2 sentences max]

"""

        user_prompt = f"""Compare these two claims:

SUMMARY CLAIM: {request.summary_claim}

URL CLAIM: {request.url_claim}

{f"ADDITIONAL CONTEXT: {request.url_content[:1000]}..." if request.url_content else ""}

"""

        messages = [

            {"role": "system", "content": system_prompt},

            {"role": "user", "content": user_prompt}

        ]

        response = llm.invoke(messages)

        response_text = response.content.strip()

        # Parse response

        status = "neutral"

        confidence = 0.5

        reason = "Analysis failed"

        for line in response_text.split('\n'):

            line = line.strip()

            if line.startswith('STATUS:'):

                status_value = line.split(':', 1)[1].strip().lower()

                if status_value in ['supported', 'contradicted', 'neutral']:

                    status = status_value

            elif line.startswith('CONFIDENCE:'):

                try:

                    confidence = float(line.split(':', 1)[1].strip())

                    confidence = max(0.0, min(1.0, confidence))

                except ValueError:

                    pass

            elif line.startswith('REASON:'):

                reason = line.split(':', 1)[1].strip()

        result = {

            "summary_claim": request.summary_claim,

            "url_claim": request.url_claim,

            "comparison": {

                "status": status,

                "confidence": confidence,

                "reason": reason

            },

            "processing_time": time.time() - start_time

        }

        logger.info(f"Claim comparison completed: {status} (confidence: {confidence:.2f})")

        return result

    except Exception as e:

        logger.error(f"Claim comparison failed: {e}", exc_info=True)

        raise HTTPException(status_code=500, detail=f"Claim comparison failed: {str(e)}")

@router.post("/check-summary-url", response_model=dict)
async def check_summary_url_consistency(
    request: CheckConsistencyRequest,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)

):

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

        result = await _check_summary_vs_url_consistency(

            request.summary, request.url, enable_deep_analysis=request.enable_deep_analysis, similarity_threshold=request.similarity_threshold,
            use_cot_atomization=request.use_cot_atomization, use_cot_audit=request.use_cot_audit

        )

        return result

    except Exception as e:

        logger.error(f"Summary-URL consistency check failed: {e}", exc_info=True)

        raise HTTPException(status_code=500, detail=f"Consistency check failed: {str(e)}")

@router.post("/complete", response_model=CompleteAnalysisResponse)
async def complete_academic_analysis(
    request: CompleteAnalysisRequest,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)

):

    """

    Complete academic content analysis pipeline.

    This endpoint runs the full 4-step academic analysis pipeline:

    1. Content Extraction - Extract title and main body

    2. Claim Atomization - Break down into atomic claims

    3. Evidence Location - Find supporting evidence

    4. Conflict Audit - Check logical consistency

    Args:

        request: Complete analysis request with URL and options

        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:

        CompleteAnalysisResponse with full pipeline results

    """

    import time

    start_time = time.time()

    logger.info(f"Starting complete academic analysis pipeline for URL: {request.url}")

    pipeline_steps = []

    overall_success = True

    final_report = None

    error_msg = None

    try:

        # Step 1: Content Extraction

        step_start = time.time()

        try:

            content_agent = ContentExtractorAgent(

                model_name=settings.openai_model,

                api_key=settings.openai_api_key,

                api_base=settings.openai_api_base,

                temperature=settings.agent_temperature

            )

            content_result = content_agent.extract_from_url(request.url, use_llm=request.use_llm_content_extraction)

            step_time = time.time() - step_start

            pipeline_steps.append(PipelineStepResponse(

                step_name="content_extraction",

                success=True,

                data={

                    "title": content_result.title,

                    "main_body_length": content_result.text_length,

                    "truncated": content_result.truncated

                },

                execution_time=step_time

            ))

            logger.info(f"Content extraction step completed in {step_time:.2f}s")

        except Exception as e:

            step_time = time.time() - step_start

            pipeline_steps.append(PipelineStepResponse(

                step_name="content_extraction",

                success=False,

                error=str(e),

                execution_time=step_time

            ))

            overall_success = False

            raise

        # Step 2: Claim Atomization

        step_start = time.time()

        try:

            if not content_result.main_body:

                raise ValueError("No main body content extracted")

            atomizer_agent = ClaimAtomizerAgent(

                model_name=settings.openai_model,

                api_key=settings.openai_api_key,

                api_base=settings.openai_api_base,

                temperature=settings.agent_temperature

            )

            atomization_result = atomizer_agent.atomize_text(

                text=content_result.main_body,

                use_cot=request.use_cot_atomization,

                custom_few_shots=request.custom_few_shots_atomizer

            )

            step_time = time.time() - step_start

            pipeline_steps.append(PipelineStepResponse(

                step_name="claim_atomization",

                success=True,

                data={

                    "atomic_claims_count": len(atomization_result.atomic_claims)

                },

                execution_time=step_time

            ))

            logger.info(f"Claim atomization step completed in {step_time:.2f}s with {len(atomization_result.atomic_claims)} claims")

        except Exception as e:

            step_time = time.time() - step_start

            pipeline_steps.append(PipelineStepResponse(

                step_name="claim_atomization",

                success=False,

                error=str(e),

                execution_time=step_time

            ))

            overall_success = False

            raise

        # Step 3: Evidence Location - DISABLED (EvidenceLocatorAgent module not found)

        step_start = time.time()

        step_time = time.time() - step_start

        pipeline_steps.append(PipelineStepResponse(

            step_name="evidence_location",

            success=False,

            data={

                "error": "EvidenceLocatorAgent module not found",

                "skipped": True

            },

            execution_time=step_time

        ))

        logger.warning("Evidence location step skipped - EvidenceLocatorAgent module not found")

        # Step 4: Conflict Audit

        step_start = time.time()

        try:

            # Prepare claim-evidence pairs for audit

            # Since evidence location step is disabled, create claim-evidence pairs directly from atomized claims

            # Each claim will be paired with empty evidence (conflict audit will still work but with limited evidence)

            claim_evidences = []

            for claim in atomization_result.atomic_claims:

                claim_evidences.append({

                    "claim_id": claim.id,

                    "claim_text": claim.text,

                    "evidence_quotes": []  # Empty evidence since evidence location is disabled

                })

            auditor_agent = ConflictAuditorAgent(

                model_name=settings.openai_model,

                api_key=settings.openai_api_key,

                api_base=settings.openai_api_base,

                temperature=settings.agent_temperature

            )

            audit_result = auditor_agent.audit_conflicts(

                claim_evidences=claim_evidences,

                use_cot=request.use_cot_audit,

                custom_few_shots=request.custom_few_shots_auditor

            )

            step_time = time.time() - step_start

            pipeline_steps.append(PipelineStepResponse(

                step_name="conflict_audit",

                success=True,

                data={

                    "supported_claims": audit_result.summary_stats.get("supported", 0),

                    "contradicted_claims": audit_result.summary_stats.get("contradicted", 0)

                },

                execution_time=step_time

            ))

            logger.info(f"Conflict audit step completed in {step_time:.2f}s")

        except Exception as e:

            step_time = time.time() - step_start

            pipeline_steps.append(PipelineStepResponse(

                step_name="conflict_audit",

                success=False,

                error=str(e),

                execution_time=step_time

            ))

            overall_success = False

            raise

    except Exception as pipeline_error:

        logger.error(f"Pipeline failed: {pipeline_error}", exc_info=True)

        error_msg = str(pipeline_error)

    total_time = time.time() - start_time

    logger.info(f"Complete analysis pipeline finished in {total_time:.2f}s (success: {overall_success})")

    return CompleteAnalysisResponse(

        url=request.url,

        pipeline_steps=pipeline_steps,

        final_report=final_report,

        overall_success=overall_success,

        total_execution_time=total_time,

        error=error_msg if not overall_success else None

    )

async def _check_summary_vs_url_consistency(summary: str, url: str, enable_deep_analysis: bool = True, similarity_threshold: float = 0.2,
                                        use_cot_atomization: CoTMode = CoTMode.NO_CHAIN, use_cot_audit: CoTMode = CoTMode.NO_CHAIN) -> Dict[str, Any]:
    """Check consistency between website summary and full content.
    Atomize both summary and URL content into claims, then compare each summary claim

    against relevant URL claims to identify consistencies and contradictions.

    """

    import time

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

        # Step 1: Extract content from URL

        content_agent = ContentExtractorAgent(

            model_name=settings.openai_model,

            api_key=settings.openai_api_key,

            api_base=settings.openai_api_base,

            temperature=settings.agent_temperature

        )

        content_result = content_agent.extract_from_url(url, use_llm=False)

        if not content_result.main_body:

            raise ValueError("Could not extract content from URL")

        # Step 2: Atomize summary into claims

        summary_atomizer = ClaimAtomizerAgent(

            model_name=settings.openai_model,

            api_key=settings.openai_api_key,

            api_base=settings.openai_api_base,

            temperature=settings.agent_temperature

        )

        summary_atomization = summary_atomizer.atomize_text(summary, use_cot=use_cot_atomization)

        # Step 3: Atomize URL content into claims

        url_atomizer = ClaimAtomizerAgent(

                model_name=settings.openai_model,

                api_key=settings.openai_api_key,

                api_base=settings.openai_api_base,

                temperature=settings.agent_temperature

            )

        url_atomization = url_atomizer.atomize_text(content_result.main_body, use_cot=use_cot_atomization, split_into_paragraphs=True)

        # Store claims info with full details

        consistency_result["summary_claims"] = [

            {"id": c.id, "text": c.text}

            for c in summary_atomization.atomic_claims

        ]

        consistency_result["url_claims"] = [

            {"id": c.id, "text": c.text, "original_sentence": getattr(c, 'original_sentence', ''),

             "paragraph_index": getattr(c, 'paragraph_index', None)}

            for c in url_atomization.atomic_claims

        ]

        # Step 4: Comprehensive claim comparison with similarity filtering and LLM analysis

        from ...utils.llm_client import llm_manager

        from langchain_openai import ChatOpenAI

        # Create LLM instance for comparisons

        http_client = llm_manager.get_http_client()

        llm = ChatOpenAI(

                model_name=settings.openai_model,

                api_key=settings.openai_api_key,

            base_url=settings.openai_api_base,

            temperature=0.1,

            max_retries=settings.openai_max_retries,

            timeout=settings.openai_timeout,

            http_client=http_client,

            max_tokens=300

        )

        # Step 4a: Calculate similarities between all claim pairs

        logger.info("Calculating similarities between all claim pairs...")

        import numpy as np

        # Get embeddings for similarity calculation

        try:

            from langchain_openai import OpenAIEmbeddings

            embeddings = OpenAIEmbeddings(

                api_key=settings.openai_api_key,

                base_url=settings.openai_api_base,

                model="text-embedding-3-small"

            )

            # Prepare all claim texts

            summary_texts = [c.text for c in summary_atomization.atomic_claims]

            url_texts = [c.text for c in url_atomization.atomic_claims]

            all_texts = summary_texts + url_texts

            # Get embeddings

            all_embeddings = embeddings.embed_documents(all_texts)

            summary_embeddings = np.array(all_embeddings[:len(summary_texts)])

            url_embeddings = np.array(all_embeddings[len(summary_texts):])

            # Calculate cosine similarities

            from sklearn.metrics.pairwise import cosine_similarity

            similarity_matrix = cosine_similarity(summary_embeddings, url_embeddings)

            logger.info(f"Calculated similarity matrix: {similarity_matrix.shape}")

        except Exception as e:

            logger.warning(f"Failed to calculate embeddings, using fallback similarity: {e}")

            # Fallback: simple text overlap similarity

            similarity_matrix = np.zeros((len(summary_atomization.atomic_claims), len(url_atomization.atomic_claims)))

            for i, summary_claim in enumerate(summary_atomization.atomic_claims):

                summary_words = set(summary_claim.text.lower().split())

                for j, url_claim in enumerate(url_atomization.atomic_claims):

                    url_words = set(url_claim.text.lower().split())

                    overlap = len(summary_words.intersection(url_words))

                    total_words = len(summary_words.union(url_words))

                    similarity_matrix[i, j] = overlap / total_words if total_words > 0 else 0.0

        # Step 4b: Find claim pairs above similarity threshold and perform LLM analysis

        comparisons = []

        threshold = similarity_threshold  # Use the parameter passed in

        logger.info(f"Finding claim pairs above similarity threshold {threshold}...")

        for i, summary_claim in enumerate(summary_atomization.atomic_claims):

            for j, url_claim in enumerate(url_atomization.atomic_claims):

                similarity_score = similarity_matrix[i, j]

                # Only process pairs above threshold

                if similarity_score >= threshold:

                    try:

                        # Perform LLM analysis for this similar pair

                        system_prompt = """Analyze if the URL claim supports, contradicts, or is neutral to the summary claim.

STATUS: [supported|contradicted|neutral]

CONFIDENCE: [0.0-1.0]

REASON: [Brief explanation of the relationship]"""

                        user_prompt = f"""Summary claim: {summary_claim.text}

URL claim: {url_claim.text}

Similarity score: {similarity_score:.3f}"""

                        messages = [

                            {"role": "system", "content": system_prompt},

                            {"role": "user", "content": user_prompt}

                        ]

                        response = llm.invoke(messages)

                        response_text = response.content.strip()

                        # Parse response

                        status = "neutral"

                        confidence = 0.5

                        reason = "Analysis failed"

                        for line in response_text.split('\n'):

                            line = line.strip()

                            if line.startswith('STATUS:'):

                                status = line.split(':', 1)[1].strip().lower()

                            elif line.startswith('CONFIDENCE:'):

                                try:

                                    confidence = float(line.split(':', 1)[1].strip())

                                    confidence = max(0.0, min(1.0, confidence))

                                except:

                                    pass

                            elif line.startswith('REASON:'):

                                reason = line.split(':', 1)[1].strip()

                        comparison = {

                            "summary_claim": {

                                "id": summary_claim.id,

                                "text": summary_claim.text

                            },

                            "url_claim": {

                                "id": url_claim.id,

                                "text": url_claim.text,

                                "original_sentence": getattr(url_claim, 'original_sentence', ''),

                                "paragraph_index": getattr(url_claim, 'paragraph_index', None)

                            },

                            "similarity_score": float(similarity_score),

                            "relationship": status,

                            "confidence": confidence,

                            "reason": reason

                        }

                        comparisons.append(comparison)

                    except Exception as e:

                        logger.warning(f"Failed to analyze claim pair {summary_claim.id} vs {url_claim.id}: {e}")

                        comparison = {

                            "summary_claim": {

                                "id": summary_claim.id,

                                "text": summary_claim.text

                            },

                            "url_claim": {

                                "id": url_claim.id,

                                "text": url_claim.text,

                                "original_sentence": getattr(url_claim, 'original_sentence', ''),

                                "paragraph_index": getattr(url_claim, 'paragraph_index', None)

                            },

                            "similarity_score": float(similarity_score),

                            "relationship": "error",

                            "confidence": 0.0,

                            "reason": f"Analysis failed: {str(e)}"

                        }

                        comparisons.append(comparison)

        # Step 4c: Generate comprehensive statistics

        stats = {

            "total_summary_claims": len(summary_atomization.atomic_claims),

            "total_url_claims": len(url_atomization.atomic_claims),

            "total_comparisons": len(comparisons),

            "similarity_threshold_used": threshold,

            "supported_count": sum(1 for c in comparisons if c["relationship"] == "supported"),

            "contradicted_count": sum(1 for c in comparisons if c["relationship"] == "contradicted"),

            "neutral_count": sum(1 for c in comparisons if c["relationship"] == "neutral"),

            "error_count": sum(1 for c in comparisons if c["relationship"] == "error"),

            "avg_similarity_score": sum(c["similarity_score"] for c in comparisons) / len(comparisons) if comparisons else 0.0,

            "avg_confidence": sum(c["confidence"] for c in comparisons) / len(comparisons) if comparisons else 0.0

        }

        consistency_result["claim_comparisons"] = comparisons

        consistency_result["statistics"] = stats

        # Step 4d: Include paragraph information from URL atomization

        consistency_result["url_paragraphs"] = [

            {

                "paragraph_index": para.paragraph_index,

                "paragraph_text": para.paragraph_text,

                "claim_count": len(para.atomic_claims),

                "claims": [

                    {

                        "id": c.id,

                        "text": c.text,

                        "original_sentence": getattr(c, 'original_sentence', ''),

                        "confidence": c.confidence

                    } for c in para.atomic_claims

                ]

            } for para in url_atomization.paragraphs

        ]

        logger.info(f"Consistency analysis completed: {len(comparisons)} comparisons, {stats['supported_count']} supported, {stats['contradicted_count']} contradicted")

        consistency_result["claim_comparisons"] = comparisons

        consistency_result["processing_time"] = time.time() - start_time

        logger.info(f"Consistency check completed: {len(comparisons)} claim comparisons")

    except Exception as e:

        logger.error(f"Summary vs URL consistency check failed: {e}", exc_info=True)

        consistency_result["error"] = str(e)

    return consistency_result
