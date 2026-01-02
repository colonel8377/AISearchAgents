"""Quality API routes."""

from typing import Dict, Any, Optional, List

from fastapi import APIRouter, Depends
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.application.agents.claim_atomizer.agent import ClaimAtomizerAgent
from src.application.agents.content_extractor.agent import ContentExtractorAgent
from src.application.agents.web_opinion_extractor import WebOpinionAnalyzer
from src.shared.config.settings import settings
from src.shared.llm.llm_manager import llm_manager
from src.shared.utils import get_logger
from .consistency import complete_academic_analysis
from ..common import get_api_key
from ..schemas import (
    OverallEvaluationRequest, OverallEvaluationResponse,
    CompleteAnalysisRequest, EvaluationMetrics
)

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/quality", tags=["Quality"])

@router.post("/overall", response_model=OverallEvaluationResponse)
async def overall_evaluation(
    request: OverallEvaluationRequest,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)

):

    """

    Overall evaluation interface for summary and URL analysis with statistical results.

    This endpoint provides comprehensive evaluation of content including:

    - Content quality metrics

    - Academic integrity assessment

    - Bias analysis

    - Comparative analysis between summary and source

    - Optional full academic analysis pipeline

    Args:

        request: OverallEvaluationRequest with summary, URL, and options

        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:

        OverallEvaluationResponse with detailed evaluation and statistics

    """

    import time

    start_time = time.time()

    logger.info(f"Starting overall evaluation: summary={bool(request.summary)}, url={bool(request.url)}, full_analysis={request.include_full_analysis}")

    evaluation_result = {

        "summary_evaluation": None,

        "url_evaluation": None,

        "comparative_analysis": None,

        "full_academic_analysis": None,

        "processing_metadata": {}

    }

    try:

        # Evaluate summary if provided

        if request.summary:

            summary_eval = await _evaluate_content(request.summary, "summary", enable_deep_analysis=request.enable_deep_analysis)

            evaluation_result["summary_evaluation"] = summary_eval

        # Evaluate URL if provided

        if request.url:

            url_eval = await _evaluate_url(request.url, enable_deep_analysis=request.enable_deep_analysis)

            evaluation_result["url_evaluation"] = url_eval

        # Comparative analysis if both are provided

        if request.summary and request.url:

            if request.enable_consistency_check:

                # Run consistency check between summary and URL analysis

                consistency_result = await _check_summary_vs_url_consistency(

                    request.summary, request.url, enable_deep_analysis=request.enable_deep_analysis

                )

                evaluation_result["consistency_check"] = consistency_result

            else:

                # Run basic comparative analysis

                comparative = await _comparative_analysis(request.summary, request.url, enable_deep_analysis=request.enable_deep_analysis)

                evaluation_result["comparative_analysis"] = comparative

        # Full academic analysis if requested

        if request.include_full_analysis and request.url:

            # Run the complete academic analysis pipeline

            analysis_request = CompleteAnalysisRequest(

                url=request.url,

                use_llm_content_extraction=False,

                use_cot_atomization=False,

                use_cot_audit=False,

            )

            analysis_result = await complete_academic_analysis(analysis_request,  _api_key)

            evaluation_result["full_academic_analysis"] = {

                "overall_success": analysis_result.overall_success,

                "final_report": analysis_result.final_report.__dict__ if analysis_result.final_report else None,

                "pipeline_steps": [step.__dict__ for step in analysis_result.pipeline_steps],

                "total_execution_time": analysis_result.total_execution_time

            }

    except Exception as e:

        logger.error(f"Overall evaluation failed: {e}", exc_info=True)

        evaluation_result["processing_metadata"]["error"] = str(e)

    # Add processing metadata

    total_time = time.time() - start_time

    evaluation_result["processing_metadata"].update({

        "total_processing_time": total_time,

        "evaluation_timestamp": time.time(),

        "inputs_provided": {

            "summary": bool(request.summary),

            "url": bool(request.url),

            "full_analysis": request.include_full_analysis,

            "consistency_check": request.enable_consistency_check

        }

    })

    logger.info(f"Content evaluation completed in {total_time:.2f}s")

    return OverallEvaluationResponse(**evaluation_result)

async def _evaluate_content(content: str, content_type: str, enable_deep_analysis: bool = True) -> Dict[str, Any]:
    """Evaluate content quality and provide metrics."""
    try:

        # Basic content metrics

        content_length = len(content)

        sentences = content.split('.')

        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0

        # Simple readability score (approximate)

        readability_score = max(0, min(100, 206.835 - 1.015 * avg_sentence_length - 84.6 * (content.count(' ') / content_length)))

        # Initialize metrics with None values

        fact_density = None

        opinion_density = None

        bias_distribution = None

        academic_integrity_score = None

        hallucination_risk = None

        # Deep analysis using WebOpinionAnalyzer if enabled

        if enable_deep_analysis and content_length > 50:  # Only analyze if content is substantial

            try:
                analyzer = WebOpinionAnalyzer()

                # Analyze text to extract facts and opinions

                result = analyzer.analyze_text(content, use_llm=True, use_cot=False)

                if result and not (result.extraction_metadata and "error" in result.extraction_metadata):

                    total_items = len(result.atomic_opinions)

                    facts_count = len(result.facts)

                    opinions_count = len(result.opinions)

                    if total_items > 0:

                        # Calculate densities (ratio of facts/opinions to total atomic items)

                        fact_density = facts_count / total_items

                        opinion_density = opinions_count / total_items

                    # Get bias distribution if available

                    if result.overall_bias_distribution:

                        bias_distribution = {

                            "left": result.overall_bias_distribution.left,

                            "right": result.overall_bias_distribution.right,

                            "neutral": result.overall_bias_distribution.neutral,

                            "dominant_bias": result.overall_bias_distribution.dominant_bias,

                            "bias_score": result.overall_bias_distribution.bias_score

                        }

                    elif result.opinions:

                        # Calculate bias distribution from opinions if overall not available

                        total_left = sum(op.bias_probabilities.left for op in result.opinions)

                        total_right = sum(op.bias_probabilities.right for op in result.opinions)

                        total_neutral = sum(op.bias_probabilities.neutral for op in result.opinions)

                        n = len(result.opinions)

                        if n > 0:

                            # Determine dominant bias

                            if total_left > total_right and total_left > total_neutral:

                                dominant_bias = "left"

                            elif total_right > total_neutral:

                                dominant_bias = "right"

                            else:

                                dominant_bias = "neutral"

                            bias_distribution = {

                                "left": total_left / n,

                                "right": total_right / n,

                                "neutral": total_neutral / n,

                                "dominant_bias": dominant_bias,

                                "bias_score": abs(total_left - total_right) / n if n > 0 else 0.0

                            }

                    # Academic integrity score: higher when more facts, lower when more opinions

                    if total_items > 0:

                        academic_integrity_score = facts_count / total_items * 100

                    # Hallucination risk: inverse of fact density (lower fact density = higher risk)

                    if fact_density is not None:

                        hallucination_risk = (1.0 - fact_density) * 100

            except Exception as e:

                logger.warning(f"Deep analysis failed for {content_type}: {e}")

                # Continue with None values if analysis fails

        return {

            "content_type": content_type,

            "metrics": EvaluationMetrics(

                content_length=content_length,

                readability_score=readability_score,

                fact_density=fact_density,

                opinion_density=opinion_density,

                bias_distribution=bias_distribution,

                academic_integrity_score=academic_integrity_score,

                hallucination_risk=hallucination_risk

            ).__dict__,

            "basic_stats": {

                "sentence_count": len(sentences),

                "word_count": len(content.split()),

                "avg_sentence_length": avg_sentence_length

            }

        }

    except Exception as e:

        logger.error(f"Content evaluation failed: {e}")

        return {"error": str(e)}

async def _evaluate_url(url: str, enable_deep_analysis: bool = True) -> Dict[str, Any]:
    """Evaluate URL content quality."""
    try:

        # Extract content from URL

        content_agent = ContentExtractorAgent(

            model_name=settings.openai_model,

            api_key=settings.openai_api_key,

            api_base=settings.openai_api_base,

            temperature=settings.agent_temperature

        )

        content_result = content_agent.extract_from_url(url, use_llm=False)

        if content_result.main_body:

            content_eval = await _evaluate_content(content_result.main_body, "url_content", enable_deep_analysis=enable_deep_analysis)

            content_eval.update({

                "url": url,

                "title": content_result.title,

                "extraction_metadata": content_result.extraction_metadata

            })

            return content_eval

        else:

            return {"error": "Failed to extract content from URL"}

    except Exception as e:

        logger.error(f"URL evaluation failed: {e}")

        return {"error": str(e)}

async def _check_summary_vs_url_consistency(summary: str, url: str, enable_deep_analysis: bool = True, similarity_threshold: float = 0.2) -> Dict[str, Any]:
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

        summary_atomization = summary_atomizer.atomize_text(summary, use_cot=True)

        # Step 3: Atomize URL content into claims

        url_atomizer = ClaimAtomizerAgent(

                model_name=settings.openai_model,

                api_key=settings.openai_api_key,

                api_base=settings.openai_api_base,

                temperature=settings.agent_temperature

            )

        url_atomization = url_atomizer.atomize_text(content_result.main_body, use_cot=True, split_into_paragraphs=True)

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

async def _comparative_analysis(summary: str, url: str, enable_deep_analysis: bool = True) -> Dict[str, Any]:
    """Compare summary against source URL."""
    try:

        # Get both evaluations

        summary_eval = await _evaluate_content(summary, "summary", enable_deep_analysis=enable_deep_analysis)

        url_eval = await _evaluate_url(url, enable_deep_analysis=enable_deep_analysis)

        # Simple comparative metrics

        content_length = url_eval.get("metrics", {}).get("content_length", 1)

        compression_ratio = len(summary) / content_length if content_length > 0 else 0

        # Get titles for comparison

        summary_title = summary_eval.get("title")

        url_title = url_eval.get("title")

        title_match = None

        if summary_title and url_title:

            # Simple title matching (case-insensitive, normalized)

            summary_title_norm = summary_title.lower().strip()

            url_title_norm = url_title.lower().strip()

            title_match = summary_title_norm == url_title_norm

        return {

            "compression_ratio": compression_ratio,

            "summary_density": summary_eval.get("basic_stats", {}).get("word_count", 0) / max(summary_eval.get("basic_stats", {}).get("sentence_count", 1), 1),

            "source_density": url_eval.get("basic_stats", {}).get("word_count", 0) / max(url_eval.get("basic_stats", {}).get("sentence_count", 1), 1),

            "consistency_check": {

                "summary_length": len(summary),

                "source_length": url_eval.get("metrics", {}).get("content_length", 0),

                "title_match": title_match

            }

        }

    except Exception as e:

        logger.error(f"Comparative analysis failed: {e}")

        return {"error": str(e)}

# ===========================

# Complete Academic Analysis Pipeline Endpoint

# ===========================


class PipelineStepResponse(BaseModel):

    """Response model for a pipeline step."""

    step_name: str

    success: bool

    data: Optional[Dict[str, Any]] = None

    error: Optional[str] = None

    execution_time: Optional[float] = None

class CompleteAnalysisResponse(BaseModel):

    """Response model for complete academic analysis."""

    url: str

    pipeline_steps: List[PipelineStepResponse]

    final_report: Optional[Dict[str, Any]] = None

    overall_success: bool

    total_execution_time: Optional[float] = None

    error: Optional[str] = None
