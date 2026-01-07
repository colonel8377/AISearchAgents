"""Demographic API routes."""

from fastapi import APIRouter, Depends, HTTPException

from src.application.agents.demographic_evaluator import DemographicEvaluatorAgent
from src.shared.utils import get_logger
from ..common import get_api_key
from ..schemas import (
    EvaluateSentencesRequest, EvaluateSentencesResponse, JudgmentResponse
)

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/agent/demographic-evaluator", tags=["Demographic"])

@router.post("/evaluate", response_model=EvaluateSentencesResponse)
async def evaluate_sentences(
    request: EvaluateSentencesRequest,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):

    """

    Evaluate sentences from a demographic perspective.

    This is a stateless operation that evaluates sentences without requiring an agent instance.

    Args:

        request: Evaluation request with demographic profile and sentences

        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:

        Evaluation results with judgments for each sentence

    """

    try:

        # Create a temporary agent instance for stateless evaluation
        agent = DemographicEvaluatorAgent()
        result = await agent.evaluate_sentences(
            demography_json=request.demography_json,
            sentences=request.sentences,
            use_cot=request.use_cot,
            use_few_shots=request.use_few_shots,
            custom_few_shots=request.custom_few_shots,
            per_message=request.per_message,
            is_binary_agreement=request.is_binary_agreement
        )

        # Convert to response model with explicit validation
        judgments = []
        for i, judgment in enumerate(result["judgments"]):
            try:
                # Ensure all fields are properly typed before creating response model
                agree_value = judgment.get("agree", 0.0)
                
                # Preserve type based on is_binary_agreement
                # If binary mode, keep as int; otherwise keep as float
                if request.is_binary_agreement:
                    agree_value = int(agree_value) if isinstance(agree_value, (int, float)) else 0
                else:
                    agree_value = float(agree_value) if isinstance(agree_value, (int, float)) else 0.0
                
                validated_judgment = {
                    "index": int(judgment.get("index", i)),
                    "sentence": str(judgment.get("sentence", "")),
                    "agree": agree_value,
                    "reason": str(judgment.get("reason", ""))
                }
                judgments.append(JudgmentResponse(**validated_judgment))
            except (ValueError, TypeError, KeyError) as e:
                logger.error(f"Failed to validate judgment {i}: {e}, judgment data: {judgment}")
                raise ValueError(f"Invalid judgment data at index {i}: {e}")

        return EvaluateSentencesResponse(judgments=judgments)

    except ValueError as e:

        logger.warning(f"Evaluate_sentences validation error: {str(e)}")

        raise HTTPException(status_code=400, detail=str(e))

    except HTTPException:

        raise

    except Exception as e:

        logger.error(f"Failed to evaluate sentences: {e}", exc_info=True)

        raise HTTPException(status_code=500, detail=f"Failed to evaluate sentences: {str(e)}")

