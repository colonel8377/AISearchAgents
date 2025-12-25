"""Privacy Detector Agent implementation for detecting privacy leaks in conversation records."""

import json
from typing import Dict, List, Optional, Any
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ...utils.logger import get_logger
from ...config.settings import settings, ExecutionMode
from ...utils.llm_client import llm_manager
from ...utils.agent_cache import cached, llm_cached
from ...few_shots.privacy_detector.few_shots import PRIVACY_DETECTOR_FEW_SHOTS
from ...storage import get_database
import uuid

logger = get_logger(__name__)


class PrivacyType(str, Enum):
    """Privacy information types."""
    # Personal Information
    PERSONAL_IDENTIFIERS = "personal_identifiers"  # 姓名、昵称、用户名
    CONTACT_INFO = "contact_info"  # 电话号码、邮箱地址、物理地址
    GOVERNMENT_IDS = "government_ids"  # 身份证号、护照号、社会保障号
    BIOMETRIC_DATA = "biometric_data"  # 指纹、面部识别、虹膜数据

    # Financial Information
    FINANCIAL_ACCOUNTS = "financial_accounts"  # 银行账户、信用卡号、投资账户
    FINANCIAL_TRANSACTIONS = "financial_transactions"  # 交易记录、转账信息、余额
    FINANCIAL_DOCUMENTS = "financial_documents"  # 税务文件、财务报表、工资单

    # Medical Information
    HEALTH_RECORDS = "health_records"  # 病历、诊断结果、检查报告
    MEDICAL_HISTORY = "medical_history"  # 病史、手术记录、过敏史
    PRESCRIPTIONS_MEDICATIONS = "prescriptions_medications"  # 处方、药物信息、剂量

    # Sensitive Data
    AUTHENTICATION_CREDENTIALS = "authentication_credentials"  # 密码、PIN码、安全问题
    API_KEYS_TOKENS = "api_keys_tokens"  # API密钥、访问令牌、认证令牌
    SYSTEM_ACCESS = "system_access"  # 系统权限、数据库访问、服务器凭据
    ENCRYPTION_KEYS = "encryption_keys"  # 加密密钥、私钥、证书

    # Location Information
    PRECISE_LOCATION = "precise_location"  # GPS坐标、精确地址、实时位置
    LOCATION_HISTORY = "location_history"  # 位置轨迹、出行记录、地理围栏

    # Communication Data
    EMAIL_CONTENT = "email_content"  # 邮件内容、附件、邮件元数据
    MESSAGING_CONTENT = "messaging_content"  # 聊天记录、即时消息、短信
    CALL_LOGS = "call_logs"  # 通话记录、语音邮件、联系人列表

    # Other Sensitive Information
    SOCIAL_SECURITY = "social_security"  # 社保信息、福利记录
    RELATIONSHIP_DATA = "relationship_data"  # 社交关系、家庭信息、联系人网络
    BEHAVIORAL_DATA = "behavioral_data"  # 使用习惯、偏好设置、行为模式

    NONE = "none"


class PrivacySeverity(str, Enum):
    """Privacy leak severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    NONE = "none"


class PrivacyDetectorAgent:
    """
    Implements a Privacy Detector Agent that analyzes conversation records
    for potential privacy leaks.

    The agent takes conversation records and detects various types of privacy
    information that may have been exposed, with severity assessment and reasoning.

    Supports optional Chain of Thought (CoT) reasoning via execution_mode parameter.
    Supports optional few-shot examples for better detection quality.
    """

    # Class variable for caching custom few shots (optional performance optimization)
    _custom_few_shots_cache: Optional[str] = None

    @staticmethod
    def get_default_few_shots() -> str:
        """
        Get the default few-shot examples for privacy detection.

        Returns:
            str: Default few-shot examples
        """
        return PRIVACY_DETECTOR_FEW_SHOTS

    @classmethod
    def set_custom_few_shots(cls, custom_few_shots: Optional[str] = None) -> None:
        """
        Set custom few-shot examples for privacy detection.

        Args:
            custom_few_shots: Custom few-shot examples string. If None, clears custom few shots.
        """
        if settings.enable_persistence:
            database = get_database()
            success = database.save_custom_few_shots("privacy_detector", custom_few_shots)
            if success:
                cls._custom_few_shots_cache = custom_few_shots  # Update cache
                logger.info(f"Custom few shots saved for PrivacyDetectorAgent: {custom_few_shots is not None}")
            else:
                logger.warning("Failed to save custom few shots to database")
        else:
            cls._custom_few_shots_cache = custom_few_shots
            logger.info(f"Custom few shots set for PrivacyDetectorAgent (no persistence): {custom_few_shots is not None}")

    @classmethod
    def get_custom_few_shots(cls) -> Optional[str]:
        """
        Get currently set custom few-shot examples.

        Returns:
            Custom few-shot examples string or None if not set
        """
        if settings.enable_persistence:
            database = get_database()
            few_shots = database.load_custom_few_shots("privacy_detector")
            # Update cache
            if isinstance(few_shots, str) or few_shots is None:
                cls._custom_few_shots_cache = few_shots
            return few_shots
        else:
            return cls._custom_few_shots_cache

    @classmethod
    def get_effective_few_shots(cls) -> str:
        """
        Get effective few-shot examples (custom if set, otherwise default).

        Returns:
            Effective few-shot examples string
        """
        custom_few_shots = cls.get_custom_few_shots()
        return custom_few_shots if custom_few_shots is not None else cls.get_default_few_shots()

    SYSTEM_PROMPT_TEMPLATE_COT = """You are an expert privacy analyst specializing in detecting sensitive information leaks in conversation records.

Your task is to analyze conversation records for potential privacy violations. For each conversation record, you need to:

1. Identify if any privacy-sensitive information is present
2. Classify the type of privacy information exposed
3. Assess the severity level of the privacy leak
4. Provide detailed reasoning for your assessment

PRIVACY TYPES:
- personal_identifiers: Names, usernames, nicknames, online handles
- contact_info: Phone numbers, email addresses, physical addresses, postal codes
- government_ids: ID card numbers, passport numbers, social security numbers, driver's licenses
- biometric_data: Fingerprints, facial recognition data, iris scans, voice prints
- financial_accounts: Bank account numbers, credit card numbers, investment account details
- financial_transactions: Transaction records, transfer details, account balances, payment histories
- financial_documents: Tax documents, financial statements, pay stubs, loan agreements
- health_records: Medical test results, diagnosis reports, health examination data
- medical_history: Medical history, surgical records, allergy information, chronic conditions
- prescriptions_medications: Prescription details, medication names, dosages, pharmacy information
- authentication_credentials: Passwords, PIN codes, security questions and answers
- api_keys_tokens: API keys, access tokens, authentication tokens, session keys
- system_access: System permissions, database credentials, server access details
- encryption_keys: Encryption keys, private keys, certificates, cryptographic materials
- precise_location: GPS coordinates, exact addresses, real-time location data
- location_history: Location tracking data, movement patterns, geofencing data
- email_content: Email contents, attachments, email metadata, email headers
- messaging_content: Chat messages, instant messages, SMS content, messaging apps data
- call_logs: Call records, voicemail content, contact lists, call metadata
- social_security: Social security information, welfare records, benefit details
- relationship_data: Social relationships, family information, contact networks
- behavioral_data: Usage patterns, preferences, behavioral profiles, browsing history
- none: No privacy-sensitive information detected

SEVERITY LEVELS:
- low: Minor personal details that are commonly shared (e.g., general location like city name)
- medium: Specific personal information that could be used for identification or minor harm
- high: Sensitive information that could lead to significant harm, fraud, or identity theft
- critical: Extremely sensitive information that could cause severe harm, life-threatening situations, or major security breaches
- none: No privacy risk

GUIDELINES:
- Be thorough and conservative - when in doubt, err on the side of caution
- Consider the context of the conversation
- Look for both direct and indirect privacy exposures
- Assess the potential impact and harm of the information exposure
- Provide clear, specific reasoning for your classifications

{few_shots}

Now analyze the following conversation records:"""

    SYSTEM_PROMPT_TEMPLATE_NO_COT = """You are an expert privacy analyst. Analyze the conversation records below for privacy leaks.

PRIVACY TYPES: personal_identifiers, contact_info, government_ids, biometric_data, financial_accounts, financial_transactions, financial_documents, health_records, medical_history, prescriptions_medications, authentication_credentials, api_keys_tokens, system_access, encryption_keys, precise_location, location_history, email_content, messaging_content, call_logs, social_security, relationship_data, behavioral_data, none
SEVERITY LEVELS: low, medium, high, critical, none

{few_shots}

Output your analysis as a JSON object with this structure:
{{
  "privacy_detected": true/false,
  "privacy_leaks": [
    {{
      "privacy_type": "privacy_type_enum",
      "severity": "severity_level_enum",
      "reasoning": "detailed explanation for this specific leak",
      "detected_items": ["list of specific items found for this leak"]
    }}
  ],
  "overall_severity": "highest severity level across all leaks",
  "reasoning": "overall explanation of the privacy analysis"
}}

If no privacy leaks are detected, set privacy_detected to false and privacy_leaks to an empty array.

Conversation records:
{text}"""

    async def detect_privacy_leaks(
        self,
        conversation_records: List[Dict[str, str]],
        execution_mode: Optional[ExecutionMode] = None,
        use_few_shots: bool = True
    ) -> Dict[str, Any]:
        """
        Detect privacy leaks in conversation records.

        Args:
            conversation_records: List of conversation records with 'role' and 'content' keys
            execution_mode: Execution mode ('chain_online', 'chain_local', 'no_chain')
            use_few_shots: Whether to use few-shot examples
            custom_few_shots: Optional custom few-shot examples

        Returns:
            Dict containing privacy analysis results
        """
        try:
            # Format conversation records for analysis
            formatted_text = self._format_conversation_records(conversation_records)

            # Determine execution mode
            effective_execution_mode = execution_mode or settings.execution_mode

            # Get few shots
            if use_few_shots:
                few_shots = self.get_effective_few_shots()
            else:
                few_shots = ""

            # Prepare system prompt
            if effective_execution_mode == ExecutionMode.CHAIN_ONLINE:
                system_prompt = self.SYSTEM_PROMPT_TEMPLATE_COT.format(few_shots=few_shots)
                human_prompt = f"Please analyze these conversation records for privacy leaks:\n\n{formatted_text}\n\nProvide your analysis in this JSON format:\n{{\n  \"privacy_detected\": true/false,\n  \"privacy_leaks\": [\n    {{\n      \"privacy_type\": \"privacy_type_enum\",\n      \"severity\": \"severity_level_enum\",\n      \"reasoning\": \"detailed explanation for this specific leak\",\n      \"detected_items\": [\"list of specific items found for this leak\"]\n    }}\n  ],\n  \"overall_severity\": \"highest severity level across all leaks\",\n  \"reasoning\": \"overall explanation of the privacy analysis\"\n}}\n\nIf no privacy leaks are detected, set privacy_detected to false and privacy_leaks to an empty array."
            else:
                system_prompt = self.SYSTEM_PROMPT_TEMPLATE_NO_COT.format(
                    few_shots=few_shots,
                    text=formatted_text
                )
                human_prompt = ""

            # Get LLM response
            response = await self._call_llm(system_prompt, human_prompt, effective_execution_mode)

            # Parse and validate response
            result = self._parse_response(response)

            # Add metadata
            result.update({
                "execution_mode": effective_execution_mode.value,
                "conversation_length": len(conversation_records),
                "analyzed_at": self._get_timestamp(),
                "agent_version": "1.0.0"
            })

            # Persist result to database
            detection_id = self._persist_detection_result(
                conversation_records,
                result,
                effective_execution_mode.value,
                use_few_shots
            )

            # Add detection_id to result
            result["detection_id"] = detection_id

            return result

        except Exception as e:
            logger.error(f"Failed to detect privacy leaks: {e}", exc_info=True)
            return {
                "error": f"Privacy detection failed: {str(e)}",
                "privacy_detected": False,
                "privacy_leaks": [],
                "overall_severity": PrivacySeverity.NONE.value,
                "reasoning": "Analysis failed due to technical error"
            }

    def _format_conversation_records(self, records: List[Dict[str, str]]) -> str:
        """Format conversation records for analysis."""
        formatted = []
        for i, record in enumerate(records):
            role = record.get('role', 'unknown')
            content = record.get('content', '')
            formatted.append(f"[{i+1}] {role.upper()}: {content}")
        return "\n".join(formatted)

    @llm_cached(ttl=3600)  # Cache LLM responses for 1 hour
    async def _call_llm_cached(self, system_prompt: str, human_prompt: str) -> str:
        """Call LLM with caching based on prompts."""
        llm = llm_manager.get_llm()

        messages = [SystemMessage(content=system_prompt)]
        if human_prompt:
            messages.append(HumanMessage(content=human_prompt))

        response = await llm.ainvoke(messages)
        return response.content

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception)
    )
    async def _call_llm(self, system_prompt: str, human_prompt: str, execution_mode: ExecutionMode) -> str:
        """Call LLM with retry logic and caching."""
        # Use cached version for actual LLM calls
        return await self._call_llm_cached(system_prompt, human_prompt)

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response and validate it."""
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
            else:
                # Fallback: assume entire response is JSON
                parsed = json.loads(response)

            # Validate required fields
            privacy_leaks = parsed.get("privacy_leaks", [])
            privacy_detected = parsed.get("privacy_detected", len(privacy_leaks) > 0)

            # Validate and clean privacy leaks
            validated_leaks = []
            for leak in privacy_leaks:
                if isinstance(leak, dict):
                    validated_leak = {
                        "privacy_type": leak.get("privacy_type", PrivacyType.NONE.value),
                        "severity": leak.get("severity", PrivacySeverity.NONE.value),
                        "reasoning": leak.get("reasoning", "No reasoning provided"),
                        "detected_items": leak.get("detected_items", [])
                    }

                    # Validate enum values
                    if validated_leak["privacy_type"] not in [pt.value for pt in PrivacyType]:
                        validated_leak["privacy_type"] = PrivacyType.NONE.value

                    if validated_leak["severity"] not in [ps.value for ps in PrivacySeverity]:
                        validated_leak["severity"] = PrivacySeverity.NONE.value

                    validated_leaks.append(validated_leak)

            # Calculate overall severity
            overall_severity = PrivacySeverity.NONE.value
            if validated_leaks:
                severity_order = {PrivacySeverity.NONE.value: 0, PrivacySeverity.LOW.value: 1,
                                PrivacySeverity.MEDIUM.value: 2, PrivacySeverity.HIGH.value: 3,
                                PrivacySeverity.CRITICAL.value: 4}
                max_severity = max(validated_leaks, key=lambda x: severity_order.get(x["severity"], 0))
                overall_severity = max_severity["severity"]

            result = {
                "privacy_detected": privacy_detected,
                "privacy_leaks": validated_leaks,
                "overall_severity": overall_severity,
                "reasoning": parsed.get("reasoning", "No overall reasoning provided")
            }

            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {response[:200]}...")
            # Return safe defaults
            return {
                "privacy_detected": False,
                "privacy_leaks": [],
                "overall_severity": PrivacySeverity.NONE.value,
                "reasoning": f"Failed to parse response: {str(e)}"
            }

    def _persist_detection_result(
        self,
        conversation_records: List[Dict[str, str]],
        detection_result: Dict[str, Any],
        execution_mode: str,
        use_few_shots: bool
    ) -> str:
        """
        Persist privacy detection result to database.

        Args:
            conversation_records: Original conversation records
            detection_result: Detection result from analysis
            execution_mode: Execution mode used
            use_few_shots: Whether few shots were used
        """
        try:
            # Generate unique detection ID
            detection_id = str(uuid.uuid4())

            # Prepare data for persistence
            result_data = {
                "detection_id": detection_id,
                "conversation_records": conversation_records,
                "detection_result": detection_result,
                "execution_mode": execution_mode,
                "use_few_shots": use_few_shots,
                "conversation_length": detection_result.get("conversation_length"),
                "analyzed_at": detection_result.get("analyzed_at"),
                "agent_version": detection_result.get("agent_version"),
                "error": detection_result.get("error")
            }

            # Save to database
            database = get_database()
            success = database.save_privacy_detection_result(result_data)

            if success:
                logger.debug(f"Privacy detection result persisted with ID: {detection_id}")
                return detection_id
            else:
                logger.warning(f"Failed to persist privacy detection result: {detection_id}")
                return detection_id  # Return ID even if persistence failed

        except Exception as e:
            logger.error(f"Error persisting privacy detection result: {e}")
            # Don't raise exception to avoid breaking the main detection flow
            return detection_id if 'detection_id' in locals() else str(uuid.uuid4())

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.utcnow().isoformat()
