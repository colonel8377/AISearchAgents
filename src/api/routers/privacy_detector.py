"""Privacy Detector API routes."""

from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from ..common import get_api_key
from ..schemas import (
    DetectPrivacyRequest, PrivacyDetectionResponse
)
from ...agents.privacy_detector.agent import PrivacyDetectorAgent
from ...utils.logger import get_logger
from ...storage import get_database

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/privacy-detector", tags=["Privacy Detector"])

@router.post("/detect", response_model=PrivacyDetectionResponse)
async def detect_privacy_leaks(
    request: DetectPrivacyRequest,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)
):

    """

    Detect privacy leaks in conversation records.

    This is a stateless operation that analyzes conversation records for privacy violations

    without requiring an agent instance.

    Args:

        request: Privacy detection request with conversation records

        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:

        Privacy detection results with leak analysis

    """

    try:

        # Create a temporary agent instance for stateless detection

        agent = PrivacyDetectorAgent()

        result = await agent.detect_privacy_leaks(

            conversation_records=request.conversation_records,

            execution_mode=request.execution_mode,

            use_few_shots=request.use_few_shots

        )

        # Convert to response model

        return PrivacyDetectionResponse(**result)

    except ValueError as e:

        logger.warning(f"Detect privacy leaks validation error: {str(e)}")

        raise HTTPException(status_code=400, detail=str(e))

    except HTTPException:

        raise

    except Exception as e:

        logger.error(f"Failed to detect privacy leaks: {e}", exc_info=True)

        raise HTTPException(status_code=500, detail=f"Failed to detect privacy leaks: {str(e)}")


@router.get("/results/{detection_id}", response_model=PrivacyDetectionResponse)
async def get_privacy_detection_result(
    detection_id: str,
    _api_key: str = Depends(get_api_key)
):
    """
    Get a specific privacy detection result by ID.

    Args:
        detection_id: Unique identifier of the detection result

    Returns:
        Privacy detection result
    """
    try:
        database = get_database()
        result = database.load_privacy_detection_result(detection_id)

        if not result:
            raise HTTPException(status_code=404, detail=f"Privacy detection result with ID '{detection_id}' not found")

        # Merge detection_result with metadata
        detection_result = result["detection_result"]
        detection_result["detection_id"] = result["detection_id"]

        return PrivacyDetectionResponse(**detection_result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get privacy detection result {detection_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get privacy detection result: {str(e)}")


@router.get("/results", response_model=List[PrivacyDetectionResponse])
async def get_privacy_detection_results(
    limit: int = Query(50, description="Maximum number of results to return", ge=1, le=1000),
    offset: int = Query(0, description="Number of results to skip", ge=0),
    _api_key: str = Depends(get_api_key)
):
    """
    Get a list of privacy detection results.

    Args:
        limit: Maximum number of results to return (1-1000)
        offset: Number of results to skip

    Returns:
        List of privacy detection results
    """
    try:
        database = get_database()
        results = database.load_all_privacy_detection_results(limit=limit, offset=offset)

        response_results = []
        for result in results:
            # Merge detection_result with metadata
            detection_result = result["detection_result"]
            detection_result["detection_id"] = result["detection_id"]
            response_results.append(PrivacyDetectionResponse(**detection_result))

        return response_results

    except Exception as e:
        logger.error(f"Failed to get privacy detection results: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get privacy detection results: {str(e)}")


@router.get("/stats")
async def get_privacy_detection_stats(
    _api_key: str = Depends(get_api_key)
):
    """
    Get statistics about privacy detection results.

    Returns:
        Statistics about privacy detections
    """
    try:
        database = get_database()
        stats = database.get_privacy_detection_stats()

        return stats

    except Exception as e:
        logger.error(f"Failed to get privacy detection stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get privacy detection stats: {str(e)}")


@router.delete("/results/{detection_id}")
async def delete_privacy_detection_result(
    detection_id: str,
    _api_key: str = Depends(get_api_key)
):
    """
    Delete a privacy detection result by ID.

    Args:
        detection_id: Unique identifier of the detection result

    Returns:
        Success message
    """
    try:
        database = get_database()
        success = database.delete_privacy_detection_result(detection_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"Privacy detection result with ID '{detection_id}' not found or could not be deleted")

        return {"message": f"Privacy detection result '{detection_id}' deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete privacy detection result {detection_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete privacy detection result: {str(e)}")
