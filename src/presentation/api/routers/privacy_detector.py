"""Privacy Detector API routes.

This file acts as the View layer in MVC, handling HTTP requests and delegating to the Service layer.
Location: src/presentation/api/routers/privacy_detector.py
"""

from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form
from pydantic import BaseModel, Field, field_validator

# 引入 Service
from src.application.services.privacy_detector_service import PrivacyDetectorService
from src.shared.utils.logger import get_logger
from ..common import get_api_key
# 引入 API Schemas (确保 DetectPrivacyRequest 和 PrivacyDetectionResponse 在 schemas.py 中定义)
from ..schemas import (
    DetectPrivacyRequest, PrivacyDetectionResponse
)

logger = get_logger(__name__)

# --- 核心定义：router 对象 (必须存在，否则 main.py 报错) ---
router = APIRouter(prefix="/privacy", tags=["Privacy Detector"])

from ..common import privacy_detector_service


# --- Request/Response Models for Masking (Router specific) ---
# 这些模型如果不在 schemas.py 中，保留在这里定义是正确的

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
                        {"user": "My phone is 13812340000"},
                        {"user": "Email: test@example.com"}
                    ]
                }
            ]
        }
    }


class MaskedMessage(BaseModel):
    original_text: str
    masked_text: str
    entities_detected: int


class MaskPrivacyResponse(BaseModel):
    masked_messages: List[MaskedMessage]
    total_entities_detected: int


class DetectionIdsResponse(BaseModel):
    detection_ids: List[str]
    count: int


# --- API Endpoints ---

@router.post("/mask", response_model=MaskPrivacyResponse)
async def mask_privacy_entities(
    request: MaskPrivacyRequest,
    _api_key: str = Depends(get_api_key)
):
    """
    Mask privacy entities in conversation messages for UI display (Fast Mode).
    """
    try:
        # Get service instance (business logic layer)
        service = privacy_detector_service

        # Delegate business logic to service layer
        result = service.mask_privacy_entities(
            conversation_records=request.conversation_records
        )

        # Convert service result to response model
        masked_messages = [
            MaskedMessage(**msg) for msg in result["masked_messages"]
        ]

        return MaskPrivacyResponse(
            masked_messages=masked_messages,
            total_entities_detected=result["total_entities_detected"]
        )
    except ValueError as e:
        logger.warning(f"Mask privacy validation error: {str(e)}")
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
    Detect privacy leaks in user messages using LLM analysis.
    """
    try:
        service = privacy_detector_service

        # Delegate to service layer
        # Service ensures result is a dict matching PrivacyDetectionResponse structure
        # (including 'overall_score' and correct EnumResponse formats)
        result = await service.detect_privacy_leaks(
            conversation_records=request.conversation_records,
            use_cot=request.use_cot,
            use_few_shots=request.use_few_shots,
            cot_mode=request.cot_mode,
            account_id=request.account_id
        )

        # Validates and converts the dict to Pydantic model
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
    """
    try:
        service = privacy_detector_service

        detection_result = service.get_detection_result(detection_id)

        if not detection_result:
            raise HTTPException(
                status_code=404,
                detail=f"Privacy detection result with ID '{detection_id}' not found"
            )

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
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    _api_key: str = Depends(get_api_key)
):
    """
    Get a list of privacy detection result IDs.
    """
    try:
        service = privacy_detector_service
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
    file: UploadFile = File(...),
    account_id: str = Form(...),
    detection_id: str = Form(...),
    message_index: int = Form(...),
    _api_key: str = Depends(get_api_key)
):
    """
    Upload a file and bind it to a specific message in an existing detection.
    """
    try:
        file_bytes = await file.read()
        service = privacy_detector_service

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
            "message": f"File analyzed. Results updated."
        }

    except ValueError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        elif "does not belong" in str(e).lower():
            raise HTTPException(status_code=403, detail=str(e))
        else:
            raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_privacy_detection_stats(
    account_id: Optional[str] = Query(None),
    _api_key: str = Depends(get_api_key)
):
    """
    Get comprehensive statistics about privacy detection results.
    """
    try:
        service = privacy_detector_service
        return service.get_detection_stats(account_id=account_id)
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/results/{detection_id}")
async def delete_privacy_detection_result(
    detection_id: str,
    _api_key: str = Depends(get_api_key)
):
    """
    Delete a privacy detection result by ID.
    """
    try:
        service = privacy_detector_service
        success = service.delete_detection_result(detection_id)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Result '{detection_id}' not found"
            )

        return {"message": "Deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete result: {e}")
        raise HTTPException(status_code=500, detail=str(e))