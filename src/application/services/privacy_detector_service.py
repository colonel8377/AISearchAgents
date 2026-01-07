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

# 引用 Agent 和 Interfaces
from src.application.agents.privacy_detector.agent import PrivacyDetectorAgent
from src.application.agents.privacy_detector.core.interfaces import PrivacyEntity
from src.application.services.base_service import BaseService
from src.infrastructure.storage.persistence import get_database
from src.shared.config.settings import settings, ExecutionMode
from src.shared.constant.enums import AgentType, CoTMode, PrivacyType, PrivacySeverity
from src.shared.utils.logger import get_logger
from src.shared.utils.text_normalizer import TextNormalizer

logger = get_logger(__name__)


def _convert_privacy_type_to_enum_response(privacy_type: Any) -> Dict[str, Any]:
    """
    Convert privacy type to EnumResponse format.
    Robust handling for String, Enum, None, or Dict inputs.
    """
    # 1. Handle None
    if privacy_type is None:
        return {"value": "UNKNOWN", "code": 0}

    # 2. Handle Dict (Already formatted)
    if isinstance(privacy_type, dict) and "value" in privacy_type:
        return privacy_type

    try:
        # 3. Handle Enum Object
        if hasattr(privacy_type, 'value'):
            string_value = str(privacy_type.value)
        else:
            # 4. Handle String
            string_value = str(privacy_type)

        # Try to find numeric code
        numeric_code = 0
        if hasattr(PrivacyType, '__members__'):
            for i, member in enumerate(PrivacyType):
                # Compare against member.value (string)
                if str(member.value) == string_value:
                    numeric_code = i
                    break

        return {
            "value": string_value,
            "code": numeric_code
        }
    except Exception as e:
        logger.warning(f"Error converting privacy type {privacy_type}: {e}")
        return {"value": "UNKNOWN", "code": 0}


def _convert_privacy_severity_to_enum_response(severity: Any) -> Dict[str, Any]:
    """
    Convert privacy severity to EnumResponse format.
    Robust handling for String, Enum, None, or Dict inputs.
    """
    # 1. Handle None
    if severity is None:
        return {"value": "NONE", "code": 0}

    # 2. Handle Dict (Already formatted)
    if isinstance(severity, dict) and "value" in severity:
        return severity

    try:
        # 3. Handle Enum Object
        if hasattr(severity, 'value'):
            string_value = str(severity.value)
        else:
            # 4. Handle String
            string_value = str(severity)

        numeric_code = 0
        if hasattr(PrivacySeverity, '__members__'):
            for i, member in enumerate(PrivacySeverity):
                if str(member.value) == string_value:
                    numeric_code = i
                    break

        return {
            "value": string_value,
            "code": numeric_code
        }
    except Exception as e:
        logger.warning(f"Error converting severity {severity}: {e}")
        return {"value": "NONE", "code": 0}


class PrivacyDetectorService(BaseService):
    """
    Service for privacy detector operations.
    """

    DEFAULT_PRIVACY_DETECTOR_ID = "default_privacy_detector"

    def __init__(self, agent_manager=None):
        from ...application.agents.manager import AgentManager
        super().__init__(agent_manager or AgentManager())
        self._database = get_database() if settings.enable_persistence else None
        logger.debug("PrivacyDetectorService initialized")

    def _get_privacy_detector_agent(self) -> PrivacyDetectorAgent:
        """Get or create PrivacyDetectorAgent instance."""
        agent = self.agent_manager.get_agent(self.DEFAULT_PRIVACY_DETECTOR_ID)
        if agent is None:
            agent = PrivacyDetectorAgent()
            # Register agent using AgentManager's create_agent method
            # PrivacyDetectorAgent should be added to AgentFactory in the future
            self.agent_manager.create_agent(
                agent_instance=agent,
                agent_type=AgentType.PRIVACY_DETECTOR,
                agent_id=self.DEFAULT_PRIVACY_DETECTOR_ID
            )
            logger.info(f"PrivacyDetectorAgent registered with ID: {self.DEFAULT_PRIVACY_DETECTOR_ID}")
        return agent

    @property
    def agent(self) -> PrivacyDetectorAgent:
        """Get the privacy detector agent instance."""
        return self._get_privacy_detector_agent()

    def normalize_and_detect_entities(self, text: str) -> Tuple[str, List[PrivacyEntity]]:
        """Fast Algorithmic Mode for quick UI masking."""
        normalized_text = TextNormalizer.normalize(text)
        entities = self.agent.detector.detect(normalized_text)
        return normalized_text, entities

    def mask_privacy_entities(
        self,
        conversation_records: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Fast Mode: Mask privacy entities without LLM."""
        if not conversation_records:
            return {"masked_messages": [], "total_entities_detected": 0}

        masked_messages = []
        total_entities = 0

        for record in conversation_records:
            if not isinstance(record, dict):
                continue

            original_text = record.get('user', '')
            if not original_text:
                masked_messages.append({
                    "original_text": original_text,
                    "masked_text": original_text,
                    "entities_detected": 0
                })
                continue

            # 1. Normalize & Detect
            normalized_text, entities = self.normalize_and_detect_entities(original_text)

            # 2. Mask
            masked_text, _ = self.agent.sanitizer.sanitize(normalized_text, entities)

            masked_messages.append({
                "original_text": original_text,
                "masked_text": masked_text,
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
        Detect privacy leaks using the Enhanced Agent.
        """
        # 1. Prepare Input: Merge records context
        text_segments = []
        for record in conversation_records:
            content = record.get('user', '')
            if content:
                text_segments.append(content)

        full_text = "\n\n".join(text_segments)

        # Handle File Input
        if file_input:
            try:
                file_bytes, filename = file_input
                file_content = file_bytes.decode('utf-8', errors='ignore')
                full_text += f"\n\n--- File Content: {filename} ---\n{file_content}"
            except Exception as e:
                logger.warning(f"Failed to append file content: {e}")

        if not full_text.strip():
            return self._create_empty_result()

        # 2. Call Enhanced Agent
        agent_result = await self.agent.detect_and_mask(
            text=full_text,
            account_id=account_id
        )

        # 3. Transform Agent Output to Service Schema
        # FIX: Robustly map agent result, calculating overall_score if missing
        overall_score = agent_result.get("overall_score")
        if overall_score is None:
             overall_score = agent_result.get("risk_score", 0.0)

        final_result = {
            "privacy_detected": agent_result.get("privacy_detected", False),
            "masked_text": agent_result.get("masked_text", ""),
            "overall_severity": agent_result.get("overall_severity", "none"),
            "overall_score": overall_score,
            "risk_score": overall_score,
            "privacy_leaks": [],
            # Metadata
            "conversation_length": len(conversation_records),
            "messages_analyzed": len(conversation_records),
            "analyzed_at": datetime.utcnow().isoformat(),
            "agent_version": "3.2.0",
            "metadata_registry": agent_result.get("metadata_registry", {})
        }

        # Map agent 'leaks' to API 'privacy_leaks'
        # COMPATIBILITY: Check for both 'leaks' (new) and 'privacy_leaks' (old) keys in agent result
        agent_leaks = agent_result.get("leaks") or agent_result.get("privacy_leaks") or []

        for leak in agent_leaks:
            # COMPATIBILITY: Check both 'privacy_type' and 'type' keys
            p_type = leak.get("privacy_type") or leak.get("type") or "UNKNOWN"

            # COMPATIBILITY: Check both 'detected_items' and 'value'
            items = leak.get("detected_items", [])
            if not items and leak.get("value"):
                items = [leak.get("value")]

            final_result["privacy_leaks"].append({
                "privacy_type": p_type,
                "severity": leak.get("severity", "none"),
                "category": leak.get("category", "UNKNOWN"),
                "reasoning": leak.get("reasoning", ""),
                "detected_items": items,
                "confidence": leak.get("confidence", 0.0)
            })

        # 4. Persist Result
        detection_id = self._persist_detection_result(
            conversation_records=conversation_records,
            detection_result=final_result,
            execution_mode=execution_mode or settings.default_execution_mode,
            use_few_shots=use_few_shots,
            account_id=account_id
        )
        final_result["detection_id"] = detection_id

        # 5. Transform Enums for API Response
        return self._transform_detection_result_for_api(final_result)

    async def upload_file_and_update_detection(
        self,
        detection_id: str,
        account_id: str,
        file_bytes: bytes,
        filename: str,
        message_index: int
    ) -> Dict[str, Any]:
        """Re-run detection with uploaded file."""
        detection_result = self.get_detection_result(detection_id)
        if not detection_result:
            raise ValueError(f"Detection '{detection_id}' not found")

        if detection_result.get("account_id") != account_id:
            raise ValueError(f"Access denied for detection '{detection_id}'")

        conversation_records = detection_result.get("conversation_records", [])

        result = await self.detect_privacy_leaks(
            conversation_records=conversation_records,
            account_id=account_id,
            file_input=(file_bytes, filename)
        )

        return {
            "success": True,
            "detection_id": result["detection_id"],
            "message_index": message_index,
            "filename": filename
        }

    def get_detection_result(self, detection_id: str) -> Optional[Dict[str, Any]]:
        """Get detection result by ID."""
        if not self._database:
            return None

        try:
            result = self._database.load_privacy_detection_result(detection_id)
            if result:
                data = result["detection_result"]
                data["detection_id"] = result["detection_id"]
                data["conversation_records"] = result.get("conversation_records", [])
                data["account_id"] = result.get("account_id")
                return self._transform_detection_result_for_api(data)
            return None
        except Exception as e:
            logger.error(f"Failed to load detection result {detection_id}: {e}")
            return None

    def _transform_detection_result_for_api(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform internal result dict to API-compliant dict (EnumResponse).
        Ensures strict Pydantic compatibility.
        """
        if "privacy_leaks" not in result:
            result["privacy_leaks"] = []

        # Transform Leaks
        transformed_leaks = []
        for leak in result["privacy_leaks"]:
            # 1. Convert Type (Force valid EnumResponse)
            p_type = leak.get("privacy_type")
            p_type_converted = _convert_privacy_type_to_enum_response(p_type)

            # 2. Convert Severity (Force valid EnumResponse)
            sev = leak.get("severity")
            sev_converted = _convert_privacy_severity_to_enum_response(sev)

            new_leak = {
                "privacy_type": p_type_converted,
                "severity": sev_converted,
                "category": leak.get("category", "UNKNOWN"),
                "reasoning": leak.get("reasoning", ""),
                "detected_items": leak.get("detected_items", []),
                "confidence": leak.get("confidence", 0.0)
            }
            transformed_leaks.append(new_leak)

        result["privacy_leaks"] = transformed_leaks

        # Transform Overall Severity
        result["overall_severity"] = _convert_privacy_severity_to_enum_response(
            result.get("overall_severity")
        )

        # Force overall_score existence
        if "overall_score" not in result or result["overall_score"] is None:
            result["overall_score"] = result.get("risk_score", 0.0)

        return result

    def _create_empty_result(self) -> Dict[str, Any]:
        """Create a valid empty result structure."""
        return {
            "privacy_detected": False,
            "masked_text": "",
            "overall_severity": "none",
            "overall_score": 0.0,
            "risk_score": 0.0,
            "privacy_leaks": [],
            "detection_id": str(uuid.uuid4())
        }

    def _persist_detection_result(self, **kwargs) -> str:
        """Helper to save to DB."""
        if not self._database:
            return str(uuid.uuid4())
        try:
            detection_id = kwargs.get("detection_id") or str(uuid.uuid4())
            data = kwargs.copy()
            data["detection_id"] = detection_id
            data["created_at"] = datetime.utcnow()
            self._database.save_privacy_detection_result(data)
            return detection_id
        except Exception as e:
            logger.error(f"Persistence error: {e}")
            return str(uuid.uuid4())

    # --- Pass-through methods ---
    def list_detection_results(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        if not self._database: return {"detection_ids": [], "count": 0}
        results = self._database.load_all_privacy_detection_results(limit=limit, offset=offset)
        return {"detection_ids": [r["detection_id"] for r in results], "count": len(results)}

    def get_detection_stats(self, account_id: Optional[str] = None) -> Dict[str, Any]:
        if not self._database: return {}
        return self._database.get_privacy_detection_stats(account_id=account_id)

    def delete_detection_result(self, detection_id: str) -> bool:
        if not self._database: return False
        return self._database.delete_privacy_detection_result(detection_id)