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

        result = agent.evaluate_sentences(

            demography_json=request.demography_json,

            sentences=request.sentences,

            use_cot=request.use_cot,

            use_few_shots=request.use_few_shots,

            custom_few_shots=request.custom_few_shots

        )

        # Convert to response model

        judgments = [

            JudgmentResponse(**judgment) for judgment in result["judgments"]

        ]

        return EvaluateSentencesResponse(judgments=judgments)

    except ValueError as e:

        logger.warning(f"Evaluate_sentences validation error: {str(e)}")

        raise HTTPException(status_code=400, detail=str(e))

    except HTTPException:

        raise

    except Exception as e:

        logger.error(f"Failed to evaluate sentences: {e}", exc_info=True)

        raise HTTPException(status_code=500, detail=f"Failed to evaluate sentences: {str(e)}")

