"""Privacy Detector API routes."""

from typing import List, Optional, Dict, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form
from pydantic import BaseModel, Field, field_validator

from src.application.services.privacy_detector_service import PrivacyDetectorService
from src.shared.utils import get_logger
from ..common import get_api_key
from ..schemas import (
    DetectPrivacyRequest, PrivacyDetectionResponse, PrivacyLeakItem
)

logger = get_logger(__name__)

router = APIRouter(prefix="/privacy", tags=["Privacy Detector"])

# Initialize service instance (singleton pattern)
_privacy_detector_service = None

def get_privacy_detector_service() -> PrivacyDetectorService:
    """Get or create privacy detector service instance."""
    global _privacy_detector_service
    if _privacy_detector_service is None:
        _privacy_detector_service = PrivacyDetectorService()
    return _privacy_detector_service


class MaskPrivacyRequest(BaseModel):
    """Request model for privacy masking."""
    conversation_records: List[Dict[str, str]] = Field(
        ..., 
        description="List of user messages with 'user' key containing the message content"
    )
    
    @field_validator('conversation_records')
    @classmethod
    def validate_conversation_records(cls, v):
        """Validate that each record contains 'user' key."""
        for i, record in enumerate(v):
            if not isinstance(record, dict):
                raise ValueError(f"Record {i} must be a dictionary")
            if 'user' not in record:
                raise ValueError(f"Record {i} must contain 'user' key")
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "conversation_records": [
                        {"user": "My phone is 13812340000 and email is john@example.com"},
                        {"user": "Call me at 555-123-4567 if you need anything"}
                    ]
                }
            ]
        }
    }


class MaskedMessage(BaseModel):
    """Model for a single masked message."""
    original_text: str = Field(..., description="Original message text")
    masked_text: str = Field(..., description="Masked message text")
    entities_detected: int = Field(..., description="Number of entities detected and masked")


class MaskPrivacyResponse(BaseModel):
    """Response model for privacy masking."""
    masked_messages: List[MaskedMessage] = Field(..., description="List of masked messages")
    total_entities_detected: int = Field(..., description="Total number of entities detected across all messages")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "masked_messages": [
                        {
                            "original_text": "My phone is 13812340000 and email is john@example.com",
                            "masked_text": "My phone is 138****0000 and email is j***@example.com",
                            "entities_detected": 2
                        },
                        {
                            "original_text": "Call me at 555-123-4567 if you need anything",
                            "masked_text": "Call me at 555-****-4567 if you need anything",
                            "entities_detected": 1
                        }
                    ],
                    "total_entities_detected": 3
                }
            ]
        }
    }


class DetectionIdsResponse(BaseModel):
    """Response model for listing detection IDs."""
    detection_ids: List[str] = Field(..., description="List of detection IDs")
    count: int = Field(..., description="Total number of detection IDs returned")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "detection_ids": [
                        "550e8400-e29b-41d4-a716-446655440000",
                        "550e8400-e29b-41d4-a716-446655440001",
                        "550e8400-e29b-41d4-a716-446655440002"
                    ],
                    "count": 3
                }
            ]
        }
    }


@router.post("/mask", response_model=MaskPrivacyResponse)
async def mask_privacy_entities(
    request: MaskPrivacyRequest,
    _api_key: str = Depends(get_api_key)
):
    """
    Mask privacy entities in conversation messages for UI display.
    
    This endpoint performs fast masking without LLM calls:
    - Detects privacy entities using HybridDetector
    - Masks entities using pure string manipulation (e.g., "138****0000")
    - NO external API calls, NO LLM processing
    
    Args:
        request: Mask request with conversation_records array
        _api_key: Authentication dependency
        
    Returns:
        List of masked messages with privacy entities replaced using asterisks/patterns
    """
    try:
        # Get service instance (business logic layer)
        service = get_privacy_detector_service()
        
        # Delegate business logic to service layer
        result = service.mask_privacy_entities(
            conversation_records=request.conversation_records
        )
        
        # Convert service result to response model (presentation layer)
        masked_messages = [
            MaskedMessage(**msg) for msg in result["masked_messages"]
        ]
        
        return MaskPrivacyResponse(
            masked_messages=masked_messages,
            total_entities_detected=result["total_entities_detected"]
        )
    except ValueError as e:
        logger.warning(f"Mask privacy entities validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to mask privacy entities: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to mask privacy entities: {str(e)}")


@router.post("/detect", response_model=PrivacyDetectionResponse)
async def detect_privacy_leaks(
    request: DetectPrivacyRequest,
    _api_key: str = Depends(get_api_key)
):
    """
    Detect privacy leaks in user messages.

    This endpoint analyzes conversation records for privacy leaks.
    For file analysis, use the /upload-file endpoint after creating a detection.

    Args:
        request: Privacy detection request with user messages
        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:
        Privacy detection results with leak analysis
    """
    try:
        # Get service instance (business logic layer)
        service = get_privacy_detector_service()
        
        # Delegate business logic to service layer
        # Service layer handles: business logic, persistence, enum conversion
        result = await service.detect_privacy_leaks(
            conversation_records=request.conversation_records,
            use_cot=request.use_cot,
            use_few_shots=request.use_few_shots,
            cot_mode=request.cot_mode,
            account_id=request.account_id
        )

        # Convert to response model (Service layer already handled all transformations)
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
        # Get service instance
        service = get_privacy_detector_service()
        
        # Delegate to service layer
        # Service layer handles: data loading, enum conversion
        detection_result = service.get_detection_result(detection_id)

        if not detection_result:
            raise HTTPException(
                status_code=404,
                detail=f"Privacy detection result with ID '{detection_id}' not found"
            )

        # Convert to response model (Service layer already handled all transformations)
        return PrivacyDetectionResponse(**detection_result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get privacy detection result {detection_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get privacy detection result: {str(e)}"
        )


@router.get("/results", response_model=DetectionIdsResponse)
async def get_privacy_detection_results(
    limit: int = Query(50, description="Maximum number of results to return", ge=1, le=1000, example=50),
    offset: int = Query(0, description="Number of results to skip", ge=0, example=0),
    _api_key: str = Depends(get_api_key)
):
    """
    Get a list of privacy detection result IDs.

    Args:
        limit: Maximum number of results to return (1-1000)
        offset: Number of results to skip

    Returns:
        List of detection IDs. Use /results/{detection_id} to get full details.
        
    Example Response:
        {
            "detection_ids": ["550e8400-e29b-41d4-a716-446655440000", "550e8400-e29b-41d4-a716-446655440001"],
            "count": 2
        }
    """
    try:
        # Get service instance
        service = get_privacy_detector_service()
        
        # Delegate to service layer
        result = service.list_detection_results(limit=limit, offset=offset)

        return DetectionIdsResponse(**result)

    except Exception as e:
        logger.error(f"Failed to get privacy detection results: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get privacy detection results: {str(e)}"
        )


@router.post("/upload-file")
async def upload_file_for_analysis(
    file: UploadFile = File(..., description="File to upload and analyze"),
    account_id: str = Form(..., description="User account identifier"),
    detection_id: str = Form(..., description="Detection ID from /detect endpoint"),
    message_index: int = Form(..., description="Message index in the conversation (0-based)"),
    _api_key: str = Depends(get_api_key)
):
    """
    Upload a file and bind it to a specific message in an existing detection.

    Workflow:
    1. Create detection via /detect endpoint
    2. Upload file via this endpoint, binding to a specific message
    3. File analysis is integrated into the detection result (view via /results/{detection_id})

    This endpoint requires:
    1. A detection created via /detect endpoint
    2. A valid message_index within that detection's conversation
    3. The file will be analyzed and integrated into the detection context

    Args:
        file: File to upload and analyze (PDF, DOCX, images)
        account_id: User account identifier (must match the detection)
        detection_id: Detection ID from /detect endpoint
        message_index: Index of the message in the conversation (0-based)

    Returns:
        Success confirmation with detection_id
    """
    try:
        # Read file bytes
        file_bytes = await file.read()
        
        # Get service instance
        service = get_privacy_detector_service()
        
        # Delegate business logic to service layer
        result = await service.upload_file_and_update_detection(
            detection_id=detection_id,
            account_id=account_id,
            file_bytes=file_bytes,
            filename=file.filename or "uploaded_file",
            message_index=message_index
        )
        
        return {
            "success": True,
            "detection_id": result["detection_id"],
            "message_index": result["message_index"],
            "filename": result["filename"],
            "message": f"File uploaded and analyzed. Detection updated. View results via /results/{detection_id}"
        }

    except ValueError as e:
        # Service layer validation errors
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        elif "does not belong" in str(e).lower():
            raise HTTPException(status_code=403, detail=str(e))
        else:
            raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload and analyze file: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload and analyze file: {str(e)}"
        )


@router.get("/stats")
async def get_privacy_detection_stats(
    account_id: Optional[str] = Query(None, description="Optional account ID to filter statistics by"),
    _api_key: str = Depends(get_api_key)
):
    """
    Get comprehensive statistics about privacy detection results.

    Args:
        account_id: Optional account ID to filter results by

    Returns:
        Comprehensive statistics about privacy detections
    """
    try:
        # Get service instance
        service = get_privacy_detector_service()
        
        # Delegate to service layer
        stats = service.get_detection_stats(account_id=account_id)

        return stats

    except Exception as e:
        logger.error(f"Failed to get privacy detection stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get privacy detection stats: {str(e)}"
        )


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
        # Get service instance
        service = get_privacy_detector_service()
        
        # Delegate to service layer
        success = service.delete_detection_result(detection_id)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Privacy detection result with ID '{detection_id}' not found or could not be deleted"
            )

        return {"message": f"Privacy detection result '{detection_id}' deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete privacy detection result {detection_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete privacy detection result: {str(e)}"
        )
