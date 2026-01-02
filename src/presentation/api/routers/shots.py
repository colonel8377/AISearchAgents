"""Few-Shot Configuration API routes.

This module provides centralized management for few-shot examples configuration
across all agents. Few-shot examples are used to guide LLM behavior by providing
example inputs and outputs.

Each agent type may have different few-shot format requirements:
- String format: Most agents use simple string few-shot examples
- Dict format: Some agents like nudge-collapse use dict with turn keys
- Nested format: Web opinion extractor uses nested dict with atomizer_shots and scorer_shots
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from src.application.agents.bot_creator.agent import BotCreatorAgent
from src.application.agents.claim_atomizer.agent import ClaimAtomizerAgent
from src.application.agents.conflict_auditor.agent import ConflictAuditorAgent
from src.application.agents.content_extractor.agent import ContentExtractorAgent
from src.application.agents.demographic_evaluator.agent import DemographicEvaluatorAgent
from src.application.agents.nudge_collapse.agent import NudgeCollapseAgent
from src.application.agents.privacy_detector.agent import PrivacyDetectorAgent
from src.application.agents.summarizer.agent import SummarizerAgent
from src.application.agents.web_opinion_extractor import WebOpinionEngine
from src.shared.utils import get_logger
from ..common import get_api_key

from ..schemas import SetCustomFewShotsRequest, ShotsResponse


logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/shots", tags=["Few-Shot Configuration"])


# ============================================================================
# Summarizer Agent Few-Shot Endpoints
# ============================================================================

@router.get("/summarizer")
async def get_summarizer_shots(
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Get few-shot examples for Summarizer agent.

    Returns the effective few-shot examples (custom if set, otherwise system default).

    Args:
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Dictionary with agent_type, few_shots, and is_custom flag
    """
    try:

        few_shots = SummarizerAgent.get_effective_few_shots()
        custom_set = SummarizerAgent.get_custom_few_shots() is not None

        return {
            "agent_type": "summarizer",
            "few_shots": few_shots,
            "is_custom": custom_set
        }
    except Exception as e:
        logger.error(f"Failed to get few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get few shots: {str(e)}")


@router.post("/summarizer/custom")
async def set_summarizer_custom_shots(
    request: SetCustomFewShotsRequest,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Set custom few-shot examples for Summarizer agent.

    Args:
        request: Custom few-shot examples request
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Success message
    """
    try:
        # Validate the format for summarizer (should be string)
        if request.custom_few_shots is not None and not isinstance(request.custom_few_shots, str):
            raise HTTPException(
                status_code=400,
                detail="Custom few shots for summarizer must be a string"
            )

        SummarizerAgent.set_custom_few_shots(request.custom_few_shots)

        return {
            "message": "Custom few-shot examples set successfully for Summarizer agent",
            "agent_type": "summarizer"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set custom few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to set custom few shots: {str(e)}")


@router.get("/summarizer/custom")
async def get_summarizer_custom_shots(
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Get currently set custom few-shot examples for Summarizer agent.

    Args:
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Currently set custom few-shot examples or null if not set
    """
    try:
        custom_shots = SummarizerAgent.get_custom_few_shots()

        return {
            "agent_type": "summarizer",
            "custom_few_shots": custom_shots
        }
    except Exception as e:
        logger.error(f"Failed to get custom few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get custom few shots: {str(e)}")


@router.delete("/summarizer/custom")
async def reset_summarizer_custom_shots(
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Reset custom few-shot examples for Summarizer agent (revert to defaults).

    Args:
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Success message
    """
    try:
        SummarizerAgent.set_custom_few_shots(None)

        return {
            "message": "Custom few-shot examples reset successfully for Summarizer agent",
            "agent_type": "summarizer"
        }
    except Exception as e:
        logger.error(f"Failed to reset custom few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reset custom few shots: {str(e)}")


# ============================================================================
# Demographic Evaluator Agent Few-Shot Endpoints
# ============================================================================

@router.get("/demographic-evaluator")
async def get_demographic_evaluator_shots(
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Get few-shot examples for Demographic Evaluator agent.

    Returns the effective few-shot examples (custom if set, otherwise system default).

    Args:
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Dictionary with agent_type, few_shots, and is_custom flag
    """
    try:
        few_shots = DemographicEvaluatorAgent.get_effective_few_shots()
        custom_set = DemographicEvaluatorAgent.get_custom_few_shots() is not None

        return {
            "agent_type": "demographic_evaluator",
            "few_shots": few_shots,
            "is_custom": custom_set
        }
    except Exception as e:
        logger.error(f"Failed to get few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get few shots: {str(e)}")


@router.post("/demographic-evaluator/custom")
async def set_demographic_evaluator_custom_shots(
    request: SetCustomFewShotsRequest,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Set custom few-shot examples for Demographic Evaluator agent.

    Args:
        request: Custom few-shot examples request
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Success message
    """
    try:
        # Validate the format for demographic-evaluator (should be string)
        if request.custom_few_shots is not None and not isinstance(request.custom_few_shots, str):
            raise HTTPException(
                status_code=400,
                detail="Custom few shots for demographic-evaluator must be a string"
            )

        DemographicEvaluatorAgent.set_custom_few_shots(request.custom_few_shots)

        return {
            "message": "Custom few-shot examples set successfully for Demographic Evaluator agent",
            "agent_type": "demographic_evaluator"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set custom few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to set custom few shots: {str(e)}")


@router.get("/demographic-evaluator/custom")
async def get_demographic_evaluator_custom_shots(
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Get currently set custom few-shot examples for Demographic Evaluator agent.

    Args:
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Currently set custom few-shot examples or null if not set
    """
    try:
        custom_shots = DemographicEvaluatorAgent.get_custom_few_shots()

        return {
            "agent_type": "demographic_evaluator",
            "custom_few_shots": custom_shots
        }
    except Exception as e:
        logger.error(f"Failed to get custom few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get custom few shots: {str(e)}")


@router.delete("/demographic-evaluator/custom")
async def reset_demographic_evaluator_custom_shots(
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Reset custom few-shot examples for Demographic Evaluator agent (revert to defaults).

    Args:
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Success message
    """
    try:
        DemographicEvaluatorAgent.set_custom_few_shots(None)

        return {
            "message": "Custom few-shot examples reset successfully for Demographic Evaluator agent",
            "agent_type": "demographic_evaluator"
        }
    except Exception as e:
        logger.error(f"Failed to reset custom few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reset custom few shots: {str(e)}")


# ============================================================================
# Nudge Collapse Agent Few-Shot Endpoints
# ============================================================================

@router.get("/nudge-collapse")
async def get_nudge_collapse_shots(
    turn: Optional[int] = None,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Get few-shot examples for Nudge-Collapse agent.

    Returns the effective few-shot examples (custom if set, otherwise system default).

    Args:
        turn: Optional turn number (0-3). If provided, returns few shots for that turn only.
              If None, returns all turns as a dictionary.
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Few-shot examples for the specified turn or all turns
    """
    try:
        few_shots = NudgeCollapseAgent.get_effective_few_shots(turn=turn)
        custom_set = NudgeCollapseAgent.get_custom_few_shots() is not None

        return {
            "agent_type": "nudge_collapse",
            "turn": turn,
            "few_shots": few_shots,
            "is_custom": custom_set
        }
    except Exception as e:
        logger.error(f"Failed to get few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get few shots: {str(e)}")


@router.post("/nudge-collapse/custom")
async def set_nudge_collapse_custom_shots(
    request: SetCustomFewShotsRequest,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Set custom few-shot examples for Nudge-Collapse agent.

    Args:
        request: Custom few-shot examples request (must be dict with keys 'turn_0', 'turn_1', 'turn_2', 'turn_3')
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Success message
    """
    try:
        # Validate the format for nudge-collapse (should be dict with turn keys)
        if request.custom_few_shots is not None:
            if not isinstance(request.custom_few_shots, dict):
                raise HTTPException(
                    status_code=400,
                    detail="Custom few shots for nudge-collapse must be a dictionary with keys 'turn_0', 'turn_1', 'turn_2', 'turn_3'"
                )

            required_keys = {f'turn_{i}' for i in range(4)}
            if not all(key in request.custom_few_shots for key in required_keys):
                raise HTTPException(
                    status_code=400,
                    detail=f"Custom few shots must contain all turn keys: {required_keys}"
                )

        NudgeCollapseAgent.set_custom_few_shots(request.custom_few_shots)

        return {
            "message": "Custom few-shot examples set successfully for Nudge-Collapse agent",
            "agent_type": "nudge_collapse"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set custom few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to set custom few shots: {str(e)}")


@router.get("/nudge-collapse/custom")
async def get_nudge_collapse_custom_shots(
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Get currently set custom few-shot examples for Nudge-Collapse agent.

    Args:
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Currently set custom few-shot examples or null if not set
    """
    try:
        custom_shots = NudgeCollapseAgent.get_custom_few_shots()

        return {
            "agent_type": "nudge_collapse",
            "custom_few_shots": custom_shots
        }
    except Exception as e:
        logger.error(f"Failed to get custom few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get custom few shots: {str(e)}")


@router.delete("/nudge-collapse/custom")
async def reset_nudge_collapse_custom_shots(
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Reset custom few-shot examples for Nudge-Collapse agent (revert to defaults).

    Args:
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Success message
    """
    try:
        NudgeCollapseAgent.set_custom_few_shots(None)

        return {
            "message": "Custom few-shot examples reset successfully for Nudge-Collapse agent",
            "agent_type": "nudge_collapse"
        }
    except Exception as e:
        logger.error(f"Failed to reset custom few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reset custom few shots: {str(e)}")


# ============================================================================
# Bot Creator Agent Few-Shot Endpoints
# ============================================================================

@router.get("/bot-creator")
async def get_bot_creator_shots(
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Get few-shot examples for Bot Creator agent.

    Returns the effective few-shot examples (custom if set, otherwise system default).

    Args:
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Dictionary with agent_type, few_shots, and is_custom flag
    """
    try:
        few_shots = BotCreatorAgent.get_effective_few_shots()
        custom_set = BotCreatorAgent.get_custom_few_shots() is not None

        return {
            "agent_type": "bot_creator",
            "few_shots": few_shots,
            "is_custom": custom_set
        }
    except Exception as e:
        logger.error(f"Failed to get few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get few shots: {str(e)}")


@router.post("/bot-creator/custom")
async def set_bot_creator_custom_shots(
    request: SetCustomFewShotsRequest,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Set custom few-shot examples for Bot Creator agent.

    Args:
        request: Custom few-shot examples request
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Success message
    """
    try:
        # Validate the format for bot-creator (should be string)
        if request.custom_few_shots is not None and not isinstance(request.custom_few_shots, str):
            raise HTTPException(
                status_code=400,
                detail="Custom few shots for bot-creator must be a string"
            )

        BotCreatorAgent.set_custom_few_shots(request.custom_few_shots)

        return {
            "message": "Custom few-shot examples set successfully for Bot Creator agent",
            "agent_type": "bot_creator"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set custom few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to set custom few shots: {str(e)}")


@router.get("/bot-creator/custom")
async def get_bot_creator_custom_shots(
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Get currently set custom few-shot examples for Bot Creator agent.

    Args:
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Currently set custom few-shot examples or null if not set
    """
    try:
        custom_shots = BotCreatorAgent.get_custom_few_shots()

        return {
            "agent_type": "bot_creator",
            "custom_few_shots": custom_shots
        }
    except Exception as e:
        logger.error(f"Failed to get custom few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get custom few shots: {str(e)}")


@router.delete("/bot-creator/custom")
async def reset_bot_creator_custom_shots(
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Reset custom few-shot examples for Bot Creator agent (revert to defaults).

    Args:
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Success message
    """
    try:
        BotCreatorAgent.set_custom_few_shots(None)

        return {
            "message": "Custom few-shot examples reset successfully for Bot Creator agent",
            "agent_type": "bot_creator"
        }
    except Exception as e:
        logger.error(f"Failed to reset custom few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reset custom few shots: {str(e)}")


# ============================================================================
# Content Extractor Agent Few-Shot Endpoints
# ============================================================================

@router.get("/content-extractor")
async def get_content_extractor_shots(
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Get few-shot examples for Content Extractor agent.

    Returns the effective few-shot examples (custom if set, otherwise system default).

    Args:
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Dictionary with agent_type, few_shots, and is_custom flag
    """
    try:
        few_shots = ContentExtractorAgent.get_effective_few_shots()
        custom_set = ContentExtractorAgent.get_custom_few_shots() is not None

        return {
            "agent_type": "content_extractor",
            "few_shots": few_shots,
            "is_custom": custom_set
        }
    except Exception as e:
        logger.error(f"Failed to get few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get few shots: {str(e)}")


@router.post("/content-extractor/custom")
async def set_content_extractor_custom_shots(
    request: SetCustomFewShotsRequest,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Set custom few-shot examples for Content Extractor agent.

    Args:
        request: Custom few-shot examples request
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Success message
    """
    try:
        # Validate the format for content-extractor (should be string)
        if request.custom_few_shots is not None and not isinstance(request.custom_few_shots, str):
            raise HTTPException(
                status_code=400,
                detail="Custom few shots for content-extractor must be a string"
            )

        ContentExtractorAgent.set_custom_few_shots(request.custom_few_shots)

        return {
            "message": "Custom few-shot examples set successfully for Content Extractor agent",
            "agent_type": "content_extractor"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set custom few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to set custom few shots: {str(e)}")


@router.get("/content-extractor/custom")
async def get_content_extractor_custom_shots(
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Get currently set custom few-shot examples for Content Extractor agent.

    Args:
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Currently set custom few-shot examples or null if not set
    """
    try:
        custom_shots = ContentExtractorAgent.get_custom_few_shots()

        return {
            "agent_type": "content_extractor",
            "custom_few_shots": custom_shots
        }
    except Exception as e:
        logger.error(f"Failed to get custom few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get custom few shots: {str(e)}")


@router.delete("/content-extractor/custom")
async def reset_content_extractor_custom_shots(
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Reset custom few-shot examples for Content Extractor agent (revert to defaults).

    Args:
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Success message
    """
    try:
        ContentExtractorAgent.set_custom_few_shots(None)

        return {
            "message": "Custom few-shot examples reset successfully for Content Extractor agent",
            "agent_type": "content_extractor"
        }
    except Exception as e:
        logger.error(f"Failed to reset custom few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reset custom few shots: {str(e)}")


# ============================================================================
# Claim Atomizer Agent Few-Shot Endpoints
# ============================================================================

@router.get("/claim-atomizer")
async def get_claim_atomizer_shots(
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Get few-shot examples for Claim Atomizer agent.

    Returns the effective few-shot examples (custom if set, otherwise system default).

    Args:
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Dictionary with agent_type, few_shots, and is_custom flag
    """
    try:
        few_shots = ClaimAtomizerAgent.get_effective_few_shots()
        custom_set = ClaimAtomizerAgent.get_custom_few_shots() is not None

        return {
            "agent_type": "claim_atomizer",
            "few_shots": few_shots,
            "is_custom": custom_set
        }
    except Exception as e:
        logger.error(f"Failed to get few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get few shots: {str(e)}")


@router.post("/claim-atomizer/custom")
async def set_claim_atomizer_custom_shots(
    request: SetCustomFewShotsRequest,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Set custom few-shot examples for Claim Atomizer agent.

    Args:
        request: Custom few-shot examples request
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Success message
    """
    try:
        # Validate the format for claim-atomizer (should be string)
        if request.custom_few_shots is not None and not isinstance(request.custom_few_shots, str):
            raise HTTPException(
                status_code=400,
                detail="Custom few shots for claim-atomizer must be a string"
            )

        ClaimAtomizerAgent.set_custom_few_shots(request.custom_few_shots)

        return {
            "message": "Custom few-shot examples set successfully for Claim Atomizer agent",
            "agent_type": "claim_atomizer"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set custom few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to set custom few shots: {str(e)}")


@router.get("/claim-atomizer/custom")
async def get_claim_atomizer_custom_shots(
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Get currently set custom few-shot examples for Claim Atomizer agent.

    Args:
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Currently set custom few-shot examples or null if not set
    """
    try:
        custom_shots = ClaimAtomizerAgent.get_custom_few_shots()

        return {
            "agent_type": "claim_atomizer",
            "custom_few_shots": custom_shots
        }
    except Exception as e:
        logger.error(f"Failed to get custom few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get custom few shots: {str(e)}")


@router.delete("/claim-atomizer/custom")
async def reset_claim_atomizer_custom_shots(
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Reset custom few-shot examples for Claim Atomizer agent (revert to defaults).

    Args:
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Success message
    """
    try:
        ClaimAtomizerAgent.set_custom_few_shots(None)

        return {
            "message": "Custom few-shot examples reset successfully for Claim Atomizer agent",
            "agent_type": "claim_atomizer"
        }
    except Exception as e:
        logger.error(f"Failed to reset custom few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reset custom few shots: {str(e)}")


# ============================================================================
# Privacy Detector Agent Few-Shot Endpoints
# ============================================================================

@router.get("/privacy-detector")
async def get_privacy_detector_shots(
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Get few-shot examples for Privacy Detector agent.

    Returns the effective few-shot examples (custom if set, otherwise system default).

    Args:
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Dictionary with agent_type, few_shots, and is_custom flag
    """
    try:
        few_shots = PrivacyDetectorAgent.get_effective_few_shots()
        custom_set = PrivacyDetectorAgent.get_custom_few_shots() is not None

        return {
            "agent_type": "privacy_detector",
            "few_shots": few_shots,
            "is_custom": custom_set
        }
    except Exception as e:
        logger.error(f"Failed to get few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get few shots: {str(e)}")


@router.post("/privacy-detector/custom")
async def set_privacy_detector_custom_shots(
    request: SetCustomFewShotsRequest,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Set custom few-shot examples for Privacy Detector agent.

    Args:
        request: Custom few-shot examples request
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Success message
    """
    try:
        # Validate the format for privacy-detector (should be string)
        if request.custom_few_shots is not None and not isinstance(request.custom_few_shots, str):
            raise HTTPException(
                status_code=400,
                detail="Custom few shots for privacy-detector must be a string"
            )

        PrivacyDetectorAgent.set_custom_few_shots(request.custom_few_shots)

        return {
            "message": "Custom few-shot examples set successfully for Privacy Detector agent",
            "agent_type": "privacy_detector"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set custom few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to set custom few shots: {str(e)}")


@router.get("/privacy-detector/custom")
async def get_privacy_detector_custom_shots(
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Get currently set custom few-shot examples for Privacy Detector agent.

    Args:
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Currently set custom few-shot examples or null if not set
    """
    try:
        custom_shots = PrivacyDetectorAgent.get_custom_few_shots()

        return {
            "agent_type": "privacy_detector",
            "custom_few_shots": custom_shots
        }
    except Exception as e:
        logger.error(f"Failed to get custom few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get custom few shots: {str(e)}")


@router.delete("/privacy-detector/custom")
async def reset_privacy_detector_custom_shots(
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Reset custom few-shot examples for Privacy Detector agent (revert to defaults).

    Args:
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Success message
    """
    try:
        PrivacyDetectorAgent.set_custom_few_shots(None)

        return {
            "message": "Custom few-shot examples reset successfully for Privacy Detector agent",
            "agent_type": "privacy_detector"
        }
    except Exception as e:
        logger.error(f"Failed to reset custom few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reset custom few shots: {str(e)}")


# ============================================================================
# Conflict Auditor Agent Few-Shot Endpoints
# ============================================================================

@router.get("/conflict-auditor")
async def get_conflict_auditor_shots(
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Get few-shot examples for Conflict Auditor agent.

    Returns the effective few-shot examples (custom if set, otherwise system default).

    Args:
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Dictionary with agent_type, few_shots, and is_custom flag
    """
    try:
        few_shots = ConflictAuditorAgent.get_effective_few_shots()
        custom_set = ConflictAuditorAgent.get_custom_few_shots() is not None

        return {
            "agent_type": "conflict_auditor",
            "few_shots": few_shots,
            "is_custom": custom_set
        }
    except Exception as e:
        logger.error(f"Failed to get few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get few shots: {str(e)}")


@router.post("/conflict-auditor/custom")
async def set_conflict_auditor_custom_shots(
    request: SetCustomFewShotsRequest,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Set custom few-shot examples for Conflict Auditor agent.

    Args:
        request: Custom few-shot examples request
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Success message
    """
    try:
        # Validate the format for conflict-auditor (should be string)
        if request.custom_few_shots is not None and not isinstance(request.custom_few_shots, str):
            raise HTTPException(
                status_code=400,
                detail="Custom few shots for conflict-auditor must be a string"
            )

        ConflictAuditorAgent.set_custom_few_shots(request.custom_few_shots)

        return {
            "message": "Custom few-shot examples set successfully for Conflict Auditor agent",
            "agent_type": "conflict_auditor"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set custom few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to set custom few shots: {str(e)}")


@router.get("/conflict-auditor/custom")
async def get_conflict_auditor_custom_shots(
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Get currently set custom few-shot examples for Conflict Auditor agent.

    Args:
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Currently set custom few-shot examples or null if not set
    """
    try:
        custom_shots = ConflictAuditorAgent.get_custom_few_shots()

        return {
            "agent_type": "conflict_auditor",
            "custom_few_shots": custom_shots
        }
    except Exception as e:
        logger.error(f"Failed to get custom few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get custom few shots: {str(e)}")


@router.delete("/conflict-auditor/custom")
async def reset_conflict_auditor_custom_shots(
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Reset custom few-shot examples for Conflict Auditor agent (revert to defaults).

    Args:
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Success message
    """
    try:
        ConflictAuditorAgent.set_custom_few_shots(None)

        return {
            "message": "Custom few-shot examples reset successfully for Conflict Auditor agent",
            "agent_type": "conflict_auditor"
        }
    except Exception as e:
        logger.error(f"Failed to reset custom few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reset custom few shots: {str(e)}")


# ============================================================================
# Evidence Locator Agent Few-Shot Endpoints
# ============================================================================

@router.get("/evidence-locator")
async def get_evidence_locator_shots(
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Get few-shot examples for Evidence Locator agent.

    Args:
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Few-shot examples
    """
    try:
        # Evidence locator uses ContentExtractorAgent internally
        few_shots = ContentExtractorAgent.get_effective_few_shots()
        custom_set = ContentExtractorAgent.get_custom_few_shots() is not None

        return {
            "agent_type": "evidence_locator",
            "few_shots": few_shots,
            "is_custom": custom_set
        }
    except Exception as e:
        logger.error(f"Failed to get few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get few shots: {str(e)}")


# ============================================================================
# Web Opinion Extractor Few-Shot Endpoints
# ============================================================================

@router.get("/web-opinion-extractor", response_model=ShotsResponse)
async def get_web_opinion_extractor_shots(
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Get few-shot examples for Web Opinion Extractor.

    Returns the effective few-shot examples (custom if set, otherwise system default).
    These are used for atomization and bias scoring.

    Args:
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        ShotsResponse with atomizer_shots and scorer_shots
    """
    logger.info("Getting few-shot examples")
    try:
        engine = WebOpinionEngine()

        atomizer_shots = engine.get_effective_atomizer_shots()
        scorer_shots = engine.get_effective_scorer_shots()

        return ShotsResponse(
            atomizer_shots=atomizer_shots,
            scorer_shots=scorer_shots
        )
    except Exception as e:
        logger.error(f"Failed to get shots: {e}", exc_info=True)
        return ShotsResponse(
            atomizer_shots=[],
            scorer_shots=[]
        )


@router.post("/web-opinion-extractor/custom")
async def set_web_opinion_extractor_custom_shots(
    request: SetCustomFewShotsRequest,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Set custom few-shot examples for Web Opinion Extractor.

    Args:
        request: Custom few-shot examples request with atomizer_shots and scorer_shots
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Success message
    """
    try:
        # Validate the format for web opinion extractor (should be dict with atomizer_shots and scorer_shots)
        if request.custom_few_shots is not None:
            if not isinstance(request.custom_few_shots, dict):
                raise HTTPException(
                    status_code=400,
                    detail="Custom few shots for web opinion extractor must be a dictionary with 'atomizer_shots' and 'scorer_shots' keys"
                )

            if 'atomizer_shots' not in request.custom_few_shots or 'scorer_shots' not in request.custom_few_shots:
                raise HTTPException(
                    status_code=400,
                    detail="Custom few shots must contain 'atomizer_shots' and 'scorer_shots' keys"
                )

            if not isinstance(request.custom_few_shots['atomizer_shots'], list) or not isinstance(request.custom_few_shots['scorer_shots'], list):
                raise HTTPException(
                    status_code=400,
                    detail="'atomizer_shots' and 'scorer_shots' must be lists"
                )

        if request.custom_few_shots is not None:
            WebOpinionEngine.set_custom_atomizer_shots(request.custom_few_shots['atomizer_shots'])
            WebOpinionEngine.set_custom_scorer_shots(request.custom_few_shots['scorer_shots'])
        else:
            WebOpinionEngine.set_custom_atomizer_shots(None)
            WebOpinionEngine.set_custom_scorer_shots(None)

        return {
            "message": "Custom few-shot examples set successfully for Web Opinion Extractor",
            "agent_type": "web_opinion_extractor"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set custom few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to set custom few shots: {str(e)}")


@router.get("/web-opinion-extractor/custom")
async def get_web_opinion_extractor_custom_shots(
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Get currently set custom few-shot examples for Web Opinion Extractor.

    Args:
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Currently set custom few-shot examples or null if not set
    """
    try:
        custom_atomizer = WebOpinionEngine.get_custom_atomizer_shots()
        custom_scorer = WebOpinionEngine.get_custom_scorer_shots()

        return {
            "agent_type": "web_opinion_extractor",
            "custom_few_shots": {
                "atomizer_shots": custom_atomizer,
                "scorer_shots": custom_scorer
            } if custom_atomizer is not None and custom_scorer is not None else None
        }
    except Exception as e:
        logger.error(f"Failed to get custom few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get custom few shots: {str(e)}")


@router.delete("/web-opinion-extractor/custom")
async def reset_web_opinion_extractor_custom_shots(
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):
    """
    Reset custom few-shot examples for Web Opinion Extractor (revert to defaults).

    Args:
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Success message
    """
    try:
        WebOpinionEngine.set_custom_atomizer_shots(None)
        WebOpinionEngine.set_custom_scorer_shots(None)

        return {
            "message": "Custom few-shot examples reset successfully for Web Opinion Extractor",
            "agent_type": "web_opinion_extractor"
        }
    except Exception as e:
        logger.error(f"Failed to reset custom few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reset custom few shots: {str(e)}")

