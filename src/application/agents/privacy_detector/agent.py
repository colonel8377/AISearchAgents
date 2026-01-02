"""Privacy Detector Agent - Refactored for Performance & Robustness.

P-Guard 3.1 Architecture:
- Input: Raw Text (handled internally)
- Output: Structured Pydantic Objects
- Logic: Detect -> Mask -> LLM Analyze (with Metadata) -> Unmask/Validate
"""

import hashlib
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union, Set

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.infrastructure.storage.persistence import get_database
from src.shared.cache.decorator import cached
from src.shared.config.settings import settings, ExecutionMode
from src.shared.constant.enums import (
    CoTMode,
    PrivacyType,
    PrivacySeverity,
    PrivacyCategory
)
from src.shared.llm import llm_retry
from src.shared.llm.llm_manager import llm_manager
from src.shared.utils.logger import get_logger
from src.shared.utils.text_normalizer import TextNormalizer
from .core.interfaces import (
    IPrivacyDetector,
    ISanitizer,
    PrivacyEntity
)
from .core.utils import (
    map_privacy_type_to_category,
    map_severity_to_level
)
from .impl.detectors import HybridDetector
from .impl.file_parser import SmartFileParser
from .impl.sanitizer import ConsistentSanitizer
from .impl.verifier import PrivacyVerifier
from ...few_shots.privacy_detector.few_shots import PRIVACY_DETECTOR_FEW_SHOTS

logger = get_logger(__name__)

# --- 1. 定义结构化输出模型 (Pydantic) ---

class PrivacyLeakResult(BaseModel):
    """Structured model for a single privacy leak."""
    privacy_type: str = Field(description="The specific type of privacy leak (e.g., EMAIL, NAME, PASSWORD)")
    severity: str = Field(description="Severity: HIGH, MEDIUM, LOW")
    severity_level: str = Field(description="Level: L1, L2, L3, L4", default="L1")
    category: str = Field(description="Category: IDENTITY, FINANCIAL, MEDICAL, TECHNICAL", default="IDENTITY")
    reasoning: str = Field(description="Brief explanation of why this is a leak")
    detected_items: List[str] = Field(description="List of specific placeholders (e.g., <EMAIL_1>) or strings found")
    confidence: float = Field(description="Confidence score 0.0-1.0", default=0.5)

class PrivacyDetectionResponse(BaseModel):
    """Structured model for the overall detection response."""
    privacy_detected: bool = Field(description="Whether any privacy leaks were found")
    privacy_leaks: List[PrivacyLeakResult] = Field(default_factory=list)
    overall_severity: str = Field(description="Highest severity found", default="NONE")
    overall_severity_level: str = Field(description="Highest severity level (L1-L4)", default="none")
    overall_score: float = Field(description="Risk score 0.0-1.0", default=0.0)

# ---------------------------------------------

class PrivacyDetectorAgent:
    """
    Optimized Privacy Detector Agent (v3.1).
    Handles the full pipeline: Raw Text -> Detection -> Sanitization -> LLM Analysis.
    """

    _custom_few_shots_cache: Optional[str] = None

    def __init__(
        self,
        detector: Optional[IPrivacyDetector] = None,
        sanitizer: Optional[ISanitizer] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: Optional[float] = None,
        session_id: Optional[str] = None
    ):
        self.detector = detector or HybridDetector()
        self.sanitizer = sanitizer or ConsistentSanitizer(session_id=session_id or str(uuid.uuid4()))

        # Initialize LLM client
        http_client = llm_manager.get_http_client()
        base_llm = ChatOpenAI(
            model_name=model_name or settings.openai_model,
            api_key=api_key or settings.openai_api_key,
            base_url=api_base or settings.openai_api_base,
            temperature=temperature if temperature is not None else settings.agent_temperature,
            max_retries=settings.openai_max_retries,
            timeout=settings.openai_timeout,
            http_client=http_client
        )

        # Configure Structured Output (Robust Parsing)
        if hasattr(base_llm, "with_structured_output"):
            self.structured_llm = base_llm.with_structured_output(PrivacyDetectionResponse)
        else:
            self.structured_llm = base_llm # Fallback (less reliable)
            logger.warning("LLM does not support native structured output. Parsing might be fragile.")

        self.raw_llm = base_llm # Keep raw for file parsing/verification
        self.file_parser = SmartFileParser(self.raw_llm)
        self.verifier = PrivacyVerifier(self.raw_llm)

        logger.info("PrivacyDetectorAgent initialized (v3.1 Optimized)")

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
        Public entry point. Handles hashing and delegates to cached internal method.
        Input `conversation_records` should contain RAW text.
        """
        # Calculate hash for cache key optimization
        file_hash = None
        if file_input:
            file_bytes, _ = file_input
            file_hash = hashlib.sha256(file_bytes).hexdigest()

        return await self._detect_privacy_leaks_internal(
            conversation_records=conversation_records,
            execution_mode=execution_mode,
            use_cot=use_cot,
            use_few_shots=use_few_shots,
            cot_mode=cot_mode,
            account_id=account_id,
            file_hash=file_hash,
            file_input=file_input
        )

    @cached(ttl=3600)
    async def _detect_privacy_leaks_internal(
        self,
        conversation_records: List[Dict[str, str]],
        execution_mode: Optional[ExecutionMode] = None,
        use_cot: Optional[ExecutionMode] = None,
        use_few_shots: bool = True,
        cot_mode: Optional[CoTMode] = None,
        account_id: Optional[str] = None,
        file_hash: Optional[str] = None,
        file_input: Optional[Tuple[bytes, str]] = None
    ) -> Dict[str, Any]:
        """Internal logic with caching."""
        try:
            # 1. Pre-processing & Normalization
            # Combine all user messages into one context for detection
            formatted_text = self._format_conversation_records(conversation_records)
            formatted_text = TextNormalizer.normalize(formatted_text)

            effective_cot_mode = self._determine_cot_mode(cot_mode, use_cot, execution_mode)

            # 2. File Analysis (if exists)
            file_text = ""
            if file_input:
                try:
                    file_bytes, filename = file_input
                    file_text = await self.file_parser.analyze_file(file_bytes, filename)
                    file_text = TextNormalizer.normalize(file_text)
                except Exception as e:
                    logger.warning(f"File analysis failed: {e}")

            # 3. Detection (Raw Text -> Entities)
            # Detect entities in the normalized raw text
            entities = self.detector.detect(formatted_text)
            file_entities = self.detector.detect(file_text) if file_text else []
            all_entities = entities + file_entities

            # 4. LLM Verification (Optional)
            if getattr(settings, 'enable_llm_verification', False):
                combined_context = f"{formatted_text}\n\n--- File ---\n{file_text}" if file_text else formatted_text
                all_entities = await self.verifier.verify_entities(all_entities, combined_context)

            # 5. Sanitization (Raw Text + Entities -> Masked Text + Metadata)
            # This generates the text LLM will see (e.g., "My email is <EMAIL_1>")
            combined_text_raw = f"{formatted_text}\n\n--- File Content ---\n{file_text}" if file_text else formatted_text
            sanitized_text, metadata_registry = self.sanitizer.sanitize(combined_text_raw, all_entities)

            # 6. Prompt Engineering (Injecting Metadata)
            few_shots = self.get_effective_few_shots() if use_few_shots else ""
            system_prompt = self._get_system_prompt(effective_cot_mode, few_shots)
            human_prompt = self._get_human_prompt(
                effective_cot_mode,
                sanitized_text,
                metadata_registry,
                file_text is not None
            )

            # 7. LLM Analysis (Structured Output)
            detection_result_model = await self._call_llm_structured(system_prompt, human_prompt)

            # 8. Post-process & Validation (Algorithmically Optimized)
            final_result = self._post_process_result(
                detection_result_model,
                all_entities,
                metadata_registry
            )

            # 9. Add Metadata & Persist
            result_dict = final_result # Already a dict from post_process

            metadata = {
                "execution_mode": execution_mode or settings.default_execution_mode,
                "conversation_length": len(conversation_records),
                "messages_analyzed": len(conversation_records),
                "analyzed_at": self._get_timestamp(),
                "agent_version": "3.1.0"
            }
            result_dict.update(metadata)

            detection_id = self._persist_detection_result(
                conversation_records,
                result_dict,
                execution_mode or settings.default_execution_mode,
                use_few_shots,
                account_id
            )
            result_dict["detection_id"] = detection_id

            return result_dict

        except Exception as e:
            logger.error(f"Privacy detection critical failure: {e}", exc_info=True)
            return self._create_error_response(str(e), len(conversation_records), execution_mode)

    @llm_retry
    async def _call_llm_structured(self, system_prompt: str, user_prompt: str) -> PrivacyDetectionResponse:
        """Call LLM ensuring structured output."""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        if hasattr(self.structured_llm, "ainvoke"):
            return await self.structured_llm.ainvoke(messages)
        else:
            # Fallback for older models: Get text and parse manually
            resp = await self.raw_llm.ainvoke(messages)
            return self._parse_response_legacy(resp.content)

    def _post_process_result(
        self,
        result: Union[PrivacyDetectionResponse, Dict],
        entities: List[PrivacyEntity],
        metadata_registry: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Optimized validation: O(1) lookups using HashMaps.
        Maps LLM returned placeholders (e.g., <EMAIL_1>) back to original values or validated entities.
        """
        # Convert to dict
        result_dict = result.model_dump() if isinstance(result, BaseModel) else result.copy()

        # 1. Build Lookup Maps (Pre-computation)

        # Map: Sanitized Placeholder -> Original Data
        # e.g., {"<EMAIL_1>": {"original": "bob@abc.com", "type": "EMAIL"}}
        sanitized_lookup: Dict[str, Dict] = {}
        if metadata_registry:
            for original, meta in metadata_registry.items():
                s_val = meta.get("sanitized_value", "").strip()
                if s_val:
                    sanitized_lookup[s_val] = {
                        "original": original,
                        "type": meta.get("type"),
                        "category": meta.get("category")
                    }

        # Map: Normalized Original Text -> Entity Object (for hallucination check)
        # e.g., {"bob@abc.com": Entity(...)}
        entity_lookup: Dict[str, PrivacyEntity] = {}
        for ent in entities:
            if ent.text:
                norm_text = ent.text.lower().replace(" ", "")
                entity_lookup[norm_text] = ent

        validated_leaks = []

        # 2. Validate Leaks
        for leak in result_dict.get("privacy_leaks", []):
            detected_items = leak.get("detected_items", [])
            validated_items = []

            for item in detected_items:
                if not item: continue
                item_str = str(item).strip()

                # Check A: Item is a Placeholder (e.g., <EMAIL_1>) - Preferred Match
                if item_str in sanitized_lookup:
                    info = sanitized_lookup[item_str]
                    validated_items.append(info["original"]) # Restore original for reporting

                    # Fix type if LLM guessed wrong
                    if leak.get("privacy_type") == "UNKNOWN":
                        leak["privacy_type"] = info["type"]
                    continue

                # Check B: Item is the Original Text (e.g., "bob@abc.com")
                item_norm = item_str.lower().replace(" ", "")
                if item_norm in entity_lookup:
                    validated_items.append(entity_lookup[item_norm].text)
                    continue

                # Check C: Substring fallback (Use cautiously)
                # Only if text is long enough to avoid false positive on "the"
                if len(item_norm) > 4:
                    found = False
                    for s_val, info in sanitized_lookup.items():
                        if s_val in item_str: # e.g. "email is <EMAIL_1>"
                            validated_items.append(info["original"])
                            found = True
                            break
                    if found: continue

                # If we get here, the item was not found in our detection registry.
                # It might be a hallucination or a leak missed by regex but caught by LLM context.
                # Only keep if confidence is high.
                if leak.get("confidence", 0) > 0.8:
                     validated_items.append(item_str)

            if validated_items:
                leak["detected_items"] = list(set(validated_items)) # Deduplicate

                # Ensure Enums
                if not leak.get("severity_level"):
                    leak["severity_level"] = map_severity_to_level(
                        leak.get("severity"), leak.get("privacy_type")
                    ).value

                if not leak.get("category"):
                    leak["category"] = map_privacy_type_to_category(
                        leak.get("privacy_type")
                    ).value

                validated_leaks.append(leak)

        # 3. Finalize
        result_dict["privacy_leaks"] = validated_leaks
        result_dict["privacy_detected"] = len(validated_leaks) > 0

        # Recalculate Overall Severity Level
        if validated_leaks:
             level_order = {"L4": 4, "L3": 3, "L2": 2, "L1": 1, "none": 0}
             # Default to L1 if missing
             levels = [l.get("severity_level", "L1") for l in validated_leaks]
             max_lvl = max(levels, key=lambda x: level_order.get(x, 0))
             result_dict["overall_severity_level"] = max_lvl
        else:
             result_dict["overall_severity_level"] = "none"

        return result_dict

    def _format_conversation_records(self, records: List[Dict[str, str]]) -> str:
        return "\n".join(f"[{i+1}] User: {r.get('user', '')}" for i, r in enumerate(records))

    def _determine_cot_mode(
        self,
        cot_mode: Optional[CoTMode],
        use_cot: Optional[ExecutionMode],
        execution_mode: Optional[ExecutionMode]
    ) -> CoTMode:
        if cot_mode: return cot_mode
        val = str(use_cot or execution_mode or "")
        if "online" in val: return CoTMode.CHAIN_ONLINE
        if "no_chain" in val: return CoTMode.NO_CHAIN
        return CoTMode.CHAIN_LOCAL

    def _get_system_prompt(self, cot_mode: CoTMode, few_shots: str) -> str:
        """System prompt with explicit instruction on Placeholders."""
        base = (
            "You are a Privacy Security Expert. "
            "Analyze the provided text for privacy leaks. "
            "IMPORTANT: The text contains placeholders (e.g., <EMAIL_1>) representing sensitive entities. "
            "You MUST use the provided Metadata Registry to interpret these placeholders.\n"
            "Assess severity based on the type provided in metadata."
        )

        mode_instruction = {
            CoTMode.NO_CHAIN: "Return strict JSON.",
            CoTMode.CHAIN_LOCAL: "Think step-by-step about the context of each placeholder.",
            CoTMode.CHAIN_ONLINE: "Use NIST Privacy Framework. Grade L1 (Public) to L4 (Secrets)."
        }

        return f"{base}\n{mode_instruction.get(cot_mode)}\n\n{few_shots}"

    def _get_human_prompt(
        self,
        cot_mode: CoTMode,
        sanitized_text: str,
        metadata_registry: Dict[str, Dict[str, Any]],
        has_file: bool
    ) -> str:
        """User prompt that injects the Metadata Legend."""
        # Create a legend so LLM knows <EMAIL_1> = EMAIL
        legend_lines = []
        for k, v in list(metadata_registry.items())[:20]: # Limit to avoid context overflow
            legend_lines.append(f"- {v['sanitized_value']}: Type={v.get('type')}, Category={v.get('category')}")

        legend_text = "\n".join(legend_lines) if legend_lines else "(No specific entities pre-detected)"
        file_note = "\n[File content included]" if has_file else ""

        return (
            f"### Metadata Registry (Reference)\n{legend_text}\n\n"
            f"### User Conversation\n{sanitized_text}{file_note}\n\n"
            "Identify real privacy leaks. Ignore generic references. Return JSON."
        )

    def _parse_response_legacy(self, response: str) -> PrivacyDetectionResponse:
        """Fallback for models without native structured output."""
        try:
            # Simple wrapper to parse JSON manually if needed
            # (Reuse your old _parse_response logic here, adapted to return Pydantic model)
            import json, re
            if "```" in response:
                match = re.search(r'```\w*\s*(\{.*?\})\s*```', response, re.DOTALL)
                response = match.group(1).strip() if match else response
            data = json.loads(response)
            return PrivacyDetectionResponse(**data)
        except Exception as e:
            logger.error(f"Legacy parsing failed: {e}")
            return PrivacyDetectionResponse(privacy_detected=False)

    def _create_error_response(self, error_msg: str, msg_count: int, mode: Any) -> Dict[str, Any]:
        return {
            "error": error_msg,
            "privacy_detected": False,
            "privacy_leaks": [],
            "overall_severity": "NONE",
            "conversation_length": msg_count,
            "analyzed_at": self._get_timestamp(),
            "detection_id": str(uuid.uuid4())
        }

    def _persist_detection_result(self, records, result, mode, few_shots, acct_id):
        try:
            detection_id = str(uuid.uuid4())
            data = {
                "detection_id": detection_id,
                "conversation_records": records,
                "detection_result": result,
                "execution_mode": mode,
                "account_id": acct_id,
                "created_at": datetime.utcnow()
            }
            get_database().save_privacy_detection_result(data)
            return detection_id
        except Exception as e:
            logger.error(f"Persistence error: {e}")
            return str(uuid.uuid4())

    def _get_timestamp(self) -> str:
        return datetime.utcnow().isoformat()

    @classmethod
    def get_effective_few_shots(cls) -> str:
        custom = cls.get_custom_few_shots()
        return custom or PRIVACY_DETECTOR_FEW_SHOTS

    @classmethod
    def set_custom_few_shots(cls, custom_few_shots: Optional[str] = None) -> None:
        if settings.enable_persistence:
            get_database().save_custom_few_shots("privacy_detector", custom_few_shots)
        cls._custom_few_shots_cache = custom_few_shots

    @classmethod
    def get_custom_few_shots(cls) -> Optional[str]:
        if cls._custom_few_shots_cache: return cls._custom_few_shots_cache
        if settings.enable_persistence:
            cls._custom_few_shots_cache = get_database().load_custom_few_shots("privacy_detector")
        return cls._custom_few_shots_cache