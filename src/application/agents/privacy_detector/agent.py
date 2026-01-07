import uuid
from typing import List, Dict, Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.infrastructure.repositories import AgentProtocol
from src.shared.config.settings import settings
from src.shared.constant.enums import PrivacyType, PrivacySeverity
from src.shared.utils.logger import get_logger
from src.shared.utils.text_normalizer import TextNormalizer
from .core.interfaces import PrivacyEntity
from .impl.detectors import HybridDetector
from .impl.sanitizer import ConsistentSanitizer

logger = get_logger(__name__)


# --- 1. Pydantic 模型 (仅用于类型提示和内部解析，不再直接用于生成 Schema) ---

class DetectedLeak(BaseModel):
    """
    LLM 对单个隐私实体的分析结果。
    """
    original_value: str
    privacy_type: PrivacyType
    severity: PrivacySeverity
    is_real_leak: bool
    reasoning: str
    confidence: float


class PrivacyAnalysisResponse(BaseModel):
    """LLM 的整体返回结构"""
    leaks: List[DetectedLeak]


# ----------------------------------------------------

class PrivacyDetectorAgent(AgentProtocol):
    """
    Privacy Detector Agent (Refactored for Accuracy & Compatibility).
    """

    def __init__(
            self,
            model_name: str = settings.openai_model,
            api_key: Optional[str] = settings.openai_api_key,
            base_url: Optional[str] = settings.openai_api_base,
            temperature: float = 0.0  # 建议设为 0 以获得更稳定的 JSON
    ):
        self.detector = HybridDetector()
        self.sanitizer = ConsistentSanitizer()

        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature
        )

        # FIX: 手动构造完全展开的 Schema，避免 Pydantic 生成 $defs/$ref 导致部分模型报错 (Error 400)
        schema = self._get_expanded_schema()

        if hasattr(self.llm, "with_structured_output"):
            # 传入 Dict 而不是 Pydantic 类，LangChain 会直接使用该 Schema
            self.structured_llm = self.llm.with_structured_output(schema)
        else:
            raise RuntimeError("Current model configuration does not support structured output.")

    def _get_expanded_schema(self) -> Dict[str, Any]:
        """
        生成无 $defs/$ref 的扁平化 JSON Schema，适配 Gemini/Vertex AI 等严格后端。
        """
        return {
            "title": "PrivacyAnalysisResponse",
            "description": "Analysis result containing a list of detected privacy leaks.",
            "type": "object",
            "properties": {
                "leaks": {
                    "title": "Leaks",
                    "description": "List of confirmed or potential leaks found.",
                    "type": "array",
                    "items": {
                        "title": "DetectedLeak",
                        "type": "object",
                        "properties": {
                            "original_value": {
                                "title": "Original Value",
                                "description": "The exact substring found in the text that constitutes a leak.",
                                "type": "string"
                            },
                            "privacy_type": {
                                "title": "Privacy Type",
                                "description": "The specific category of the privacy data.",
                                "type": "string",
                                "enum": [t.value for t in PrivacyType]
                            },
                            "severity": {
                                "title": "Severity",
                                "description": "The severity level of the leak based on risk.",
                                "type": "string",
                                "enum": [s.value for s in PrivacySeverity]
                            },
                            "is_real_leak": {
                                "title": "Is Real Leak",
                                "description": "True if this is actual private data; False if it's example data, public info, or false positive.",
                                "type": "boolean"
                            },
                            "reasoning": {
                                "title": "Reasoning",
                                "description": "Brief explanation of why this is or isn't a leak (contextual analysis).",
                                "type": "string"
                            },
                            "confidence": {
                                "title": "Confidence",
                                "description": "Confidence score between 0.0 and 1.0.",
                                "type": "number"
                            }
                        },
                        "required": ["original_value", "privacy_type", "severity", "is_real_leak", "reasoning",
                                     "confidence"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["leaks"],
            "additionalProperties": False
        }

    async def detect_and_mask(
            self,
            text: str,
            account_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Public API: 检测隐私泄漏并返回脱敏后的文本。
        """
        try:
            # Step 1: 文本标准化
            normalized_text = TextNormalizer.normalize(text)

            # Step 2: 候选生成 (Algorithm Layer)
            candidates: List[PrivacyEntity] = self.detector.detect(normalized_text)

            # Step 3: LLM 裁决 (Cognitive Layer)
            llm_result_dict = await self._analyze_with_llm(normalized_text, candidates)

            # 由于 with_structured_output 传入 Dict 时返回的通常是 Dict，我们需要手动转为对象或直接使用 Dict
            # 这里统一按 Dict 处理
            if isinstance(llm_result_dict, PrivacyAnalysisResponse):
                llm_leaks = llm_result_dict.leaks
            elif isinstance(llm_result_dict, dict):
                llm_leaks = [DetectedLeak(**l) for l in llm_result_dict.get("leaks", [])]
            else:
                llm_leaks = []

            # 封装一个临时对象方便后续处理，或者修改 _merge_results 接受 list
            # 为了保持 _merge_results 签名不变，我们构造一个简单的 Namespace 或 Model
            llm_result_obj = PrivacyAnalysisResponse(leaks=llm_leaks)

            # Step 4: 结果融合
            confirmed_entities = self._merge_results(normalized_text, candidates, llm_result_obj)

            # Step 5: 执行脱敏
            masked_text, registry = self.sanitizer.sanitize(normalized_text, confirmed_entities)

            # Step 6: 构造返回
            return self._build_final_response(confirmed_entities, masked_text, registry)

        except Exception as e:
            logger.error(f"Detection critical failure: {e}", exc_info=True)
            # 降级模式：仅使用算法检测
            logger.info("Falling back to algorithmic detection only.")

            # 使用算法结果
            fallback_entities = self.detector.detect(text)
            masked_text, registry = self.sanitizer.sanitize(text, fallback_entities)

            # FIX: 确保返回结构完整，包含 overall_score 且 Enum 转为字符串
            return {
                "privacy_detected": len(fallback_entities) > 0,
                "masked_text": masked_text,
                "overall_severity": PrivacySeverity.LOW.value,  # 转为字符串值
                "overall_score": 0.0,  # 补充缺失字段
                "risk_score": 0.0,
                "leaks": [],  # 降级时不返回详细 leaks 以免误导，或者也可以转换算法结果
                "metadata_registry": registry,
                "error": str(e)
            }

    async def _analyze_with_llm(
            self,
            text: str,
            candidates: List[PrivacyEntity]
    ) -> Any:  # 返回类型可能是 Dict 或 Model，取决于 LangChain 版本
        """
        构建 Prompt 并调用 LLM。
        """
        candidates_context = "No algorithmic candidates found."
        if candidates:
            cand_list = [
                f"- '{c.text}' (Detected as: {c.entity_type}, Conf: {c.confidence:.2f})"
                for c in candidates
            ]
            candidates_context = "\n".join(cand_list)

        system_prompt = f"""You are an Expert Privacy Security Analyst. 
Your task is to analyze user text and algorithmic candidates to identify REAL privacy leaks.

## Allowed Output Values
You must strictly use the following Enums for classification:

**Privacy Types**:
{[t.value for t in PrivacyType]}

**Severity Levels**:
- {PrivacySeverity.CRITICAL.value}: Passwords, API Keys, Private Keys, SSN.
- {PrivacySeverity.HIGH.value}: Phone numbers, Real Names with Context.
- {PrivacySeverity.MEDIUM.value}: Names alone, Email addresses (if public).
- {PrivacySeverity.LOW.value}: Public URLs, generic usernames.
- {PrivacySeverity.NONE.value}: No risk.

## Analysis Rules
1. **Context is King**: 
   - 'password=123456' in a curl command -> `is_real_leak: False`.
   - 'My password is hunter2' -> `is_real_leak: True`.
2. **Sidecar Hints**: Use candidates as hints but verify them.
3. **Hallucination Check**: Only report text that explicitly appears in the user input.
"""

        user_prompt = f"""## Algorithmic Candidates (Hints)
{candidates_context}

## User Raw Text
{text}

## Task
Analyze the text. Return JSON."""

        return await self.structured_llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

    def _merge_results(
            self,
            text: str,
            candidates: List[PrivacyEntity],
            llm_result: PrivacyAnalysisResponse
    ) -> List[PrivacyEntity]:
        """
        融合逻辑：Algorithm (Recall) + LLM (Precision).
        """
        final_entities = []
        candidate_map = {c.text: c for c in candidates}

        for llm_leak in llm_result.leaks:
            if not llm_leak.is_real_leak:
                continue
            if llm_leak.original_value not in text:
                continue

            base_confidence = llm_leak.confidence
            matched_cand = candidate_map.get(llm_leak.original_value)

            if matched_cand:
                final_conf = min(0.99, matched_cand.confidence * 0.4 + base_confidence * 0.6)
                start, end = matched_cand.start, matched_cand.end
            else:
                final_conf = base_confidence
                start = text.find(llm_leak.original_value)
                end = start + len(llm_leak.original_value)

            entity = PrivacyEntity(
                text=llm_leak.original_value,
                entity_type=llm_leak.privacy_type.value,  # 已经是 Enum 值字符串
                start=start,
                end=end,
                confidence=final_conf,
                category="DETECTED",
                severity=llm_leak.severity.value,  # 已经是 Enum 值字符串
                metadata={
                    "reasoning": llm_leak.reasoning,
                    "enum_type": llm_leak.privacy_type,
                    "enum_severity": llm_leak.severity,
                    "source": "LLM_HYBRID"
                }
            )
            final_entities.append(entity)

        return final_entities

    def _build_final_response(
            self,
            entities: List[PrivacyEntity],
            masked_text: str,
            registry: Dict
    ) -> Dict[str, Any]:
        """
        构造最终输出字典。
        """
        severity_weight = {
            PrivacySeverity.CRITICAL: 4,
            PrivacySeverity.HIGH: 3,
            PrivacySeverity.MEDIUM: 2,
            PrivacySeverity.LOW: 1,
            PrivacySeverity.NONE: 0
        }

        overall_severity = PrivacySeverity.NONE
        max_weight = 0
        leaks_output = []

        total_conf = 0.0

        for ent in entities:
            # 安全获取 severity enum，如果是字符串则尝试转换
            sev_val = ent.severity
            if hasattr(sev_val, 'value'): sev_val = sev_val.value

            # 查找对应的 Enum 对象用于计算权重
            sev_enum = PrivacySeverity.LOW
            try:
                sev_enum = PrivacySeverity(sev_val)
            except:
                pass

            current_weight = severity_weight.get(sev_enum, 0)
            if current_weight > max_weight:
                max_weight = current_weight
                overall_severity = sev_enum

            total_conf += ent.confidence

            leaks_output.append({
                "privacy_type": ent.entity_type,
                "value": ent.text,
                "severity": ent.severity,
                "confidence": ent.confidence,
                "reasoning": ent.metadata.get("reasoning", "")
            })

        # 计算 overall_score
        avg_conf = total_conf / len(entities) if entities else 0.0
        max_weight_norm = max_weight / 4.0 if max_weight > 0 else 0.0
        risk_score = avg_conf * max_weight_norm

        return {
            "privacy_detected": len(entities) > 0,
            "masked_text": masked_text,
            "overall_severity": overall_severity.value,  # 返回字符串
            "overall_score": risk_score,  # 确保有这个字段
            "risk_score": risk_score,
            "leaks": leaks_output,
            "metadata_registry": registry
        }
    
    def reset(self) -> None:
        """Reset the agent to initial state."""
        logger.info("Resetting PrivacyDetectorAgent state")
        # PrivacyDetectorAgent is stateless, no state to reset
        logger.debug("Agent reset complete")