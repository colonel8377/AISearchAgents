"""Privacy Detector Service - Handles privacy detection business logic.

This service layer acts as the Controller in MVC architecture:
- Orchestrates business logic
- Coordinates between Agent (Model) and API (View)
- Handles data persistence
- Manages transactions and error handling
- Transforms data between layers (enum conversion, etc.)
"""

import uuid
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

from src.application.agents.privacy_detector.agent import PrivacyDetectorAgent
from src.application.agents.privacy_detector.core.interfaces import PrivacyEntity
from src.infrastructure.storage.persistence import get_database
from src.shared.config.settings import settings, ExecutionMode
from src.shared.constant.enums import CoTMode, PrivacyType, PrivacySeverity
from src.shared.utils.logger import get_logger
from src.shared.utils.text_normalizer import TextNormalizer

logger = get_logger(__name__)


def _convert_privacy_type_to_enum_response(privacy_type: str) -> Dict[str, Any]:
    """
    Convert privacy type string to EnumResponse format.
    
    Returns dictionary compatible with EnumResponse Pydantic model.
    Uses the same logic as API layer's _convert_enum_to_response.
    """
    try:
        enum_value = PrivacyType(privacy_type)
        string_value = str(enum_value)
        
        # Get the numeric code from the enum class (definition order index)
        numeric_code = 0  # Default fallback
        if hasattr(PrivacyType, '__members__'):
            enum_members = list(PrivacyType)
            for i, member in enumerate(enum_members):
                if member.value == string_value:
                    numeric_code = i
                    break
        
        return {
            "value": string_value,
            "code": numeric_code
        }
    except (ValueError, AttributeError, TypeError):
        # Fallback for unknown types
        return {
            "value": str(privacy_type),
            "code": 0
        }


def _convert_privacy_severity_to_enum_response(severity: str) -> Dict[str, Any]:
    """
    Convert privacy severity string to EnumResponse format.
    
    Returns dictionary compatible with EnumResponse Pydantic model.
    Uses the same logic as API layer's _convert_enum_to_response.
    """
    try:
        enum_value = PrivacySeverity(severity)
        string_value = str(enum_value)
        
        # Get the numeric code from the enum class (definition order index)
        numeric_code = 0  # Default fallback
        if hasattr(PrivacySeverity, '__members__'):
            enum_members = list(PrivacySeverity)
            for i, member in enumerate(enum_members):
                if member.value == string_value:
                    numeric_code = i
                    break
        
        return {
            "value": string_value,
            "code": numeric_code
        }
    except (ValueError, AttributeError, TypeError):
        # Fallback for unknown severities
        return {
            "value": str(severity),
            "code": 0
        }


class PrivacyDetectorService:
    """
    Service for privacy detector operations.
    
    Single Responsibility: Handle privacy detection business logic.
    Follows MVC pattern - Controller layer for business logic.
    
    Responsibilities:
    - Business logic orchestration
    - Data persistence management
    - Request/response transformation
    - Error handling and validation
    """
    
    def __init__(self, agent: PrivacyDetectorAgent = None):
        """
        Initialize the privacy detector service.
        
        Args:
            agent: Optional PrivacyDetectorAgent instance. If None, creates a new one.
        """
        self.agent = agent or PrivacyDetectorAgent()
        self._database = get_database() if settings.enable_persistence else None
        
        # Load custom few-shots from database if persistence is enabled (business logic)
        if self._database:
            try:
                few_shots = self._database.load_custom_few_shots("privacy_detector")
                if few_shots:
                    PrivacyDetectorAgent._custom_few_shots_cache = few_shots
            except Exception as e:
                logger.warning(f"Failed to load custom few shots during initialization: {e}")
        
        logger.debug("PrivacyDetectorService initialized")
    
    def normalize_and_detect_entities(self, text: str) -> Tuple[str, List[PrivacyEntity]]:
        """
        Normalize text and detect privacy entities.
        
        This is a shared business logic method for text normalization and entity detection.
        
        Args:
            text: Original text to normalize and detect entities in
            
        Returns:
            Tuple of (normalized_text, entities) where:
            - normalized_text: Text after normalization (CJK-Latin spacing, full-width to half-width)
            - entities: List of detected PrivacyEntity objects
        """
        normalized_text = TextNormalizer.normalize(text)
        entities = self.agent.detector.detect(normalized_text)
        return normalized_text, entities
    
    def mask_privacy_entities(
        self,
        conversation_records: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Mask privacy entities in conversation messages for UI display.
        
        This method performs fast masking without LLM calls:
        - Detects privacy entities using HybridDetector
        - Masks entities using pure string manipulation (e.g., "138****0000")
        - NO external API calls, NO LLM processing
        
        Args:
            conversation_records: List of conversation records, each containing 'user' key
            
        Returns:
            Dictionary with:
            - masked_messages: List of masked messages with original_text, masked_text, entities_detected
            - total_entities_detected: Total number of entities detected across all messages
            
        Raises:
            ValueError: If conversation_records is invalid
        """
        if not conversation_records:
            return {
                "masked_messages": [],
                "total_entities_detected": 0
            }
        
        masked_messages = []
        total_entities = 0
        
        # Process each message in the conversation
        for record in conversation_records:
            if not isinstance(record, dict):
                raise ValueError(f"Invalid record format: {record}")
            
            original_text = record.get('user', '')
            
            if not original_text:
                # Empty message, skip masking
                masked_messages.append({
                    "original_text": original_text,
                    "masked_text": original_text,
                    "entities_detected": 0
                })
                continue
            
            # Normalize text and detect entities (business logic)
            normalized_text, entities = self.normalize_and_detect_entities(original_text)
            
            # Mask entities (pure string manipulation, no API calls) on normalized text
            # Note: We mask the normalized text to match the entity indices
            masked_text = self.agent.sanitizer.mask(normalized_text, entities)
            
            masked_messages.append({
                "original_text": original_text,  # Keep original for display
                "masked_text": masked_text,  # Return masked normalized text
                "entities_detected": len(entities)
            })
            
            total_entities += len(entities)
        
        return {
            "masked_messages": masked_messages,
            "total_entities_detected": total_entities
        }
    
    async def detect_privacy_leaks(
        self,
        conversation_records: List[Dict[str, str]],
        execution_mode: Optional[ExecutionMode] = None,
        use_cot: Optional[ExecutionMode] = None,
        use_few_shots: bool = True,
        cot_mode: Optional[CoTMode] = None,
        account_id: Optional[str] = None,
        file_input: Optional[Tuple[bytes, str]] = None
    ) -> Dict[str, Any]:
        """
        Detect privacy leaks in conversation records.
        
        Business logic orchestration:
        1. Prepare conversation records
        2. Call agent for detection
        3. Persist results
        4. Transform enums to API format
        5. Return formatted result
        
        Args:
            conversation_records: List of conversation records
            execution_mode: Execution mode
            use_cot: Legacy CoT mode
            use_few_shots: Whether to use few-shot examples
            cot_mode: CoT mode
            account_id: Account identifier
            file_input: Optional file input (bytes, filename)
            
        Returns:
            Detection result with detection_id and enum conversions
        """
        # Prepare conversation records (business logic)
        processed_records = []
        for record in conversation_records:
            if not record.get('user'):
                continue
            
            processed_record = {
                "user": record['user'],
                "role": "user",
                "account_id": account_id or record.get("account_id")
            }
            processed_records.append(processed_record)
        
        # Delegate core detection to agent (Model layer)
        result = await self.agent.detect_privacy_leaks(
            conversation_records=processed_records,
            execution_mode=execution_mode,
            use_cot=use_cot,
            use_few_shots=use_few_shots,
            cot_mode=cot_mode,
            account_id=account_id,
            file_input=file_input
        )
        
        # Persist result (business logic)
        detection_id = self._persist_detection_result(
            conversation_records=processed_records,
            detection_result=result,
            execution_mode=execution_mode or settings.default_execution_mode,
            use_few_shots=use_few_shots,
            account_id=account_id
        )
        result["detection_id"] = detection_id
        
        # Transform enums to API format (business logic - data transformation)
        result = self._transform_detection_result_for_api(result)
        
        return result
    
    async def upload_file_and_update_detection(
        self,
        detection_id: str,
        account_id: str,
        file_bytes: bytes,
        filename: str,
        message_index: int
    ) -> Dict[str, Any]:
        """
        Upload file and update existing detection.
        
        Business logic:
        1. Validate detection exists and belongs to account
        2. Validate message_index
        3. Re-run detection with file input
        4. Update persisted result
        
        Args:
            detection_id: Detection ID
            account_id: Account identifier
            file_bytes: File content
            filename: File name
            message_index: Message index in conversation
            
        Returns:
            Success result
            
        Raises:
            ValueError: If validation fails
        """
        # Load existing detection
        detection_result = self.get_detection_result(detection_id)
        if not detection_result:
            raise ValueError(f"Detection '{detection_id}' not found")
        
        # Validate account_id
        if detection_result.get("account_id") != account_id:
            raise ValueError(f"Detection '{detection_id}' does not belong to account '{account_id}'")
        
        # Validate message_index
        conversation_records = detection_result.get("conversation_records", [])
        if message_index < 0 or message_index >= len(conversation_records):
            raise ValueError(
                f"Invalid message_index {message_index}. "
                f"Detection has {len(conversation_records)} messages"
            )
        
        # Re-run detection with file input
        execution_mode = detection_result.get("execution_mode")
        result = await self.agent.detect_privacy_leaks(
            conversation_records=conversation_records,
            execution_mode=execution_mode,
            use_few_shots=True,
            account_id=account_id,
            file_input=(file_bytes, filename)
        )
        
        # Update persisted result
        self._persist_detection_result(
            conversation_records=conversation_records,
            detection_result=result,
            execution_mode=execution_mode,
            use_few_shots=True,
            account_id=account_id,
            detection_id=detection_id  # Update existing
        )
        
        return {
            "success": True,
            "detection_id": detection_id,
            "message_index": message_index,
            "filename": filename
        }
    
    def get_detection_result(self, detection_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detection result by ID.
        
        Args:
            detection_id: Detection ID
            
        Returns:
            Detection result dictionary with detection_result merged and enums transformed, or None if not found
        """
        if not self._database:
            return None
        
        try:
            result = self._database.load_privacy_detection_result(detection_id)
            if result:
                # Merge detection_result with metadata
                detection_result = result["detection_result"]
                detection_result["detection_id"] = result["detection_id"]
                # Also include conversation_records for file upload endpoint
                detection_result["conversation_records"] = result.get("conversation_records", [])
                detection_result["execution_mode"] = result.get("execution_mode")
                detection_result["account_id"] = result.get("account_id")
                
                # Transform enums to API format (business logic)
                detection_result = self._transform_detection_result_for_api(detection_result)
                
                return detection_result
            return None
        except Exception as e:
            logger.error(f"Failed to load detection result {detection_id}: {e}")
            return None
    
    def _transform_detection_result_for_api(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform detection result for API response.
        
        Converts enum strings to EnumResponse format (business logic transformation).
        This method handles all data transformation between Agent (Model) and API (View).
        
        Args:
            result: Raw detection result from agent
            
        Returns:
            Transformed result with enum conversions, ready for API response
        """
        # Convert privacy_leaks enums to EnumResponse format
        if result.get("privacy_leaks"):
            converted_leaks = []
            for leak in result["privacy_leaks"]:
                converted_leak = {
                    "privacy_type": _convert_privacy_type_to_enum_response(leak["privacy_type"]),
                    "severity": _convert_privacy_severity_to_enum_response(leak["severity"]),
                    "severity_level": leak.get("severity_level"),
                    "category": leak.get("category"),
                    "reasoning": leak.get("reasoning", ""),
                    "detected_items": leak.get("detected_items", []),
                    "confidence": leak.get("confidence", 0.5),
                    "region": leak.get("region")
                }
                converted_leaks.append(converted_leak)
            result["privacy_leaks"] = converted_leaks
        
        # Convert overall_severity to EnumResponse format
        if "overall_severity" in result and result["overall_severity"]:
            result["overall_severity"] = _convert_privacy_severity_to_enum_response(
                result["overall_severity"]
            )
        
        # Ensure overall_severity_level exists
        if "overall_severity_level" not in result:
            result["overall_severity_level"] = None
        
        return result
    
    def list_detection_results(
        self,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        List detection result IDs.
        
        Args:
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            Dictionary with detection_ids and count
        """
        if not self._database:
            return {"detection_ids": [], "count": 0}
        
        try:
            results = self._database.load_all_privacy_detection_results(limit=limit, offset=offset)
            detection_ids = [result["detection_id"] for result in results]
            return {"detection_ids": detection_ids, "count": len(detection_ids)}
        except Exception as e:
            logger.error(f"Failed to list detection results: {e}")
            return {"detection_ids": [], "count": 0}
    
    def get_detection_stats(self, account_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detection statistics.
        
        Args:
            account_id: Optional account ID to filter by
            
        Returns:
            Statistics dictionary
        """
        if not self._database:
            return {}
        
        try:
            return self._database.get_privacy_detection_stats(account_id=account_id)
        except Exception as e:
            logger.error(f"Failed to get detection stats: {e}")
            return {}
    
    def delete_detection_result(self, detection_id: str) -> bool:
        """
        Delete detection result.
        
        Args:
            detection_id: Detection ID
            
        Returns:
            True if deleted, False otherwise
        """
        if not self._database:
            return False
        
        try:
            return self._database.delete_privacy_detection_result(detection_id)
        except Exception as e:
            logger.error(f"Failed to delete detection result {detection_id}: {e}")
            return False
    
    def _persist_detection_result(
        self,
        conversation_records: List[Dict[str, str]],
        detection_result: Dict[str, Any],
        execution_mode: str,
        use_few_shots: bool,
        account_id: Optional[str] = None,
        detection_id: Optional[str] = None
    ) -> str:
        """
        Persist detection result to database.
        
        Args:
            conversation_records: Conversation records
            detection_result: Detection result
            execution_mode: Execution mode
            use_few_shots: Whether few-shots were used
            account_id: Account identifier
            detection_id: Optional detection ID (for updates)
            
        Returns:
            Detection ID
        """
        if not self._database:
            return detection_id or str(uuid.uuid4())
        
        try:
            if not detection_id:
                detection_id = str(uuid.uuid4())
            
            result_data = {
                "detection_id": detection_id,
                "conversation_records": conversation_records,
                "detection_result": detection_result,
                "execution_mode": execution_mode,
                "use_few_shots": use_few_shots,
                "conversation_length": detection_result.get("conversation_length"),
                "analyzed_at": detection_result.get("analyzed_at"),
                "agent_version": detection_result.get("agent_version"),
                "error": detection_result.get("error"),
                "account_id": account_id
            }
            
            self._database.save_privacy_detection_result(result_data)
            logger.debug(f"Privacy detection result persisted: {detection_id}")
            return detection_id
        except Exception as e:
            logger.error(f"Error persisting detection result: {e}")
            return detection_id or str(uuid.uuid4())

