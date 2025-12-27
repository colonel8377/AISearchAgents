"""Privacy Detector Agent implementation for detecting privacy leaks in user messages."""

import json
from typing import Dict, List, Optional, Any
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ...utils.logger import get_logger
from ...config.settings import settings, ExecutionMode
from ...agents.web_opinion_extractor import CoTMode
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
    Implements a Privacy Detector Agent that analyzes user messages
    for potential privacy leaks.

    The agent takes user messages and detects various types of privacy
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

    SYSTEM_PROMPT_TEMPLATE_COT = """You are an expert privacy analyst specializing in detecting sensitive information leaks in user messages.

Your task is to systematically analyze user messages for potential privacy violations using a structured approach:

## ANALYSIS FRAMEWORK:

### 1. INFORMATION IDENTIFICATION
Scan for any sensitive information across these categories:

**PERSONAL IDENTIFIERS:**
- Full names, usernames, nicknames, online handles
- Date of birth, age, gender
- Ethnic background, religion, political affiliation
- Professional titles, workplace information

**CONTACT INFORMATION:**
- Phone numbers (any format: (123)456-7890, 123-456-7890, +1-123-456-7890)
- Email addresses (name@domain.com)
- Physical addresses, postal codes, ZIP codes
- Social media profiles, handles (@username)

**GOVERNMENT IDENTIFIERS:**
- Social Security Numbers (XXX-XX-XXXX)
- Driver's License numbers, ID card numbers
- Passport numbers, visa numbers
- Tax ID numbers, voter ID

**FINANCIAL INFORMATION:**
- Bank account numbers, routing numbers (ABA)
- Credit/debit card numbers (partial or full)
- Investment account details, stock holdings
- Cryptocurrency wallets, addresses

**HEALTH & MEDICAL:**
- Medical conditions, diagnoses, test results
- Medications, dosages, prescriptions
- Doctor names, hospital visits, medical history
- Allergies, blood type, medical device info

**AUTHENTICATION CREDENTIALS:**
- Passwords, PIN codes, security questions
- API keys, access tokens, session tokens
- Two-factor authentication codes
- Recovery codes, backup codes

**LOCATION DATA:**
- GPS coordinates, exact addresses
- Real-time location, tracking data
- Home/work addresses, frequent locations
- Travel itineraries, check-in data

### 2. PATTERN-BASED DETECTION
Look for common exposure patterns:
- "My [sensitive_info] is [value]"
- "I live at [address]"
- "Contact me at [phone/email]"
- "My password is [password]"
- Sharing screenshots, photos, documents
- Mentioning third-party accounts/services

### 3. CONTEXT ANALYSIS
Consider:
- Message intent (accidental vs deliberate sharing)
- Relationship context (trusted vs public communication)
- Information necessity (required vs optional sharing)
- Potential harm scenarios

### 4. SEVERITY ASSESSMENT
Use this severity matrix:

**LOW:**
- Publicly available information
- General location (city, state)
- Common preferences (favorite color, etc.)
- Non-sensitive public records

**MEDIUM:**
- Specific contact methods
- Partial identifiers (first name only)
- Non-critical personal details
- Information useful for identification

**HIGH:**
- Complete identity information
- Financial account details
- Health records, medical history
- Authentication credentials
- Precise location data

**CRITICAL:**
- Government ID numbers + personal details
- Full financial account access
- Life-critical medical information
- System/network access credentials
- Compromised security infrastructure

### 5. DETECTION RULES
Apply these systematic checks:

**REGEX PATTERNS:**
- Phone: /\\b\\d{3}[-.]?\\d{3}[-.]?\\d{4}\\b/
- Email: /\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b/
- SSN: /\\b\\d{3}[-]?\\d{2}[-]?\\d{4}\\b/
- Credit Card: /\\b\\d{4}[- ]?\\d{4}[- ]?\\d{4}[- ]?\\d{4}\\b/

**KEYWORD DETECTION:**
- "password", "PIN", "SSN", "passport"
- "account number", "routing", "credit card"
- "address", "phone", "email", "location"
- "diagnosis", "medication", "doctor"

**CONTEXT ANALYSIS:**
- Multiple identifiers combined
- Sensitive information + intent to share
- Technical details + access credentials

{few_shots}

Now analyze ALL the user messages provided above using this systematic framework. You MUST examine every single message in the conversation, even if some messages appear to contain no sensitive information. For each privacy leak you detect, you MUST specify which message number it came from in the locations field."""

    SYSTEM_PROMPT_TEMPLATE_NO_COT = """You are an expert privacy analyst. Analyze ALL the user messages below for privacy leaks. You MUST examine every single message in the conversation.

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

User messages:
{text}"""

    def _preprocess_for_patterns(self, records: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Enhanced preprocessing to identify potential privacy patterns using comprehensive regex and keyword detection.
        Supports both English and Chinese patterns.

        Returns:
            Dict with pattern analysis results
        """
        import re

        pattern_indicators = {
            'phone_numbers': [],
            'emails': [],
            'ssn_patterns': [],
            'credit_cards': [],
            'addresses': [],
            'api_keys': [],
            'passwords': [],
            'coordinates': [],
            'id_numbers': [],  # Chinese ID cards, passports, etc.
            'bank_accounts': [],  # Bank account numbers
            'health_records': [],  # Medical information
            'biometric_data': [],  # Fingerprint, facial recognition mentions
            'social_accounts': [],  # Social media handles
            'ip_addresses': [],  # IP addresses
            'urls_with_tokens': [],  # URLs containing tokens
            'encryption_keys': [],  # Encryption keys and certificates
            'database_credentials': [],  # Database connection strings
            'oauth_tokens': [],  # OAuth tokens
            'session_ids': [],  # Session identifiers
            'mac_addresses': [],  # MAC addresses
            'imei_numbers': [],  # IMEI numbers
            'vin_numbers': []  # Vehicle identification numbers
        }

        # Comprehensive regex patterns for both English and Chinese
        patterns = {
            # Contact Information
            'phone': [
                r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # US phone
                r'\b\+?\d{1,4}?[-.\s]?\(?0?\d{1,3}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,4}\b',  # International
                r'\b1[3-9]\d{9}\b'  # Chinese mobile
            ],
            'email': [r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'],
            'social_accounts': [
                r'\B@([A-Za-z0-9_]+)',  # Twitter handles
                r'facebook\.com/[A-Za-z0-9_.]+',  # Facebook profiles
                r'instagram\.com/[A-Za-z0-9_.]+',  # Instagram profiles
                r'weibo\.com/[A-Za-z0-9_.]+',  # Weibo profiles (Chinese)
                r'zhihu\.com/people/[A-Za-z0-9_.]+'  # Zhihu profiles (Chinese)
            ],

            # Identification Numbers
            'ssn': [r'\b\d{3}[-]?\d{2}[-]?\d{4}\b'],  # US SSN
            'id_numbers': [
                r'\b\d{17}[\dXx]\b',  # Chinese ID card (18 digits)
                r'\b[A-Z]\d{7}[A-Z\d]\b',  # Hong Kong ID
                r'\b\d{9}\b',  # Various 9-digit IDs
                r'\b[A-Z]{1,2}\d{6}[A-Z\d]\b'  # UK-style IDs
            ],
            'credit_card': [r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'],
            'bank_accounts': [
                r'\b\d{8,17}\b',  # Various bank account lengths
                r'\b\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{3,4}\b'  # Account + routing
            ],

            # Technical Credentials
            'api_keys': [
                r'\b(sk-|pk_|AKIAI|AIza|xoxp-)[A-Za-z0-9_-]{10,}\b',  # Common API prefixes
                r'\b[A-Za-z0-9_-]{20,}\b(?=\s+(?:key|token|secret|auth))',  # Long random strings near keywords
                r'\beyJ[A-Za-z0-9_-]*\.eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*\b'  # JWT tokens
            ],
            'oauth_tokens': [
                r'\b(?:access_token|refresh_token|bearer)\s*[=:]\s*[A-Za-z0-9_-]{20,}\b',
                r'\b[A-Za-z0-9_-]{32,}\b(?=\s+(?:oauth|bearer|authorization))'
            ],
            'database_credentials': [
                r'mongodb://[^\s"\'<>]+',
                r'postgresql://[^\s"\'<>]+',
                r'mysql://[^\s"\'<>]+',
                r'redis://[^\s"\'<>]+'
            ],
            'encryption_keys': [
                r'\b-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----[^-]*-----END\s+(?:RSA\s+)?PRIVATE\s+KEY-----\b',
                r'\b-----BEGIN\s+(?:RSA\s+)?PUBLIC\s+KEY-----[^-]*-----END\s+(?:RSA\s+)?PUBLIC\s+KEY-----\b',
                r'\b(?:ssh-rsa|ssh-dss|ssh-ed25519|ecdsa-sha2-nistp256)\s+[A-Za-z0-9+/=]+\s'
            ],

            # Location and Network
            'coordinates': [
                r'\b\d{1,3}\.\d{4,},?\s*-?\d{1,3}\.\d{4,}\b',  # GPS coordinates
                r'\b\d{1,2}°\d{1,2}\'\d{1,2}"?\s*[NS],\s*\d{1,3}°\d{1,2}\'\d{1,2}"?\s*[EW]\b'  # DMS format
            ],
            'ip_addresses': [r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'],
            'mac_addresses': [r'\b[0-9A-Fa-f]{2}[:-]{0,1}[0-9A-Fa-f]{2}[:-]{0,1}[0-9A-Fa-f]{2}[:-]{0,1}[0-9A-Fa-f]{2}[:-]{0,1}[0-9A-Fa-f]{2}[:-]{0,1}[0-9A-Fa-f]{2}\b'],
            'imei_numbers': [r'\b\d{15}\b'],

            # Addresses (enhanced for both languages)
            'addresses': [
                r'\b\d+\s+[A-Za-z0-9\s,.-]+\b.{10,}',  # English addresses
                r'\b[\u4e00-\u9fff]+(?:省|市|区|县|镇|乡|村|路|街|巷|号|室?)\s*\d*[\u4e00-\u9fff\d\s,-]*\b'  # Chinese addresses
            ],

            # Vehicle and Device
            'vin_numbers': [r'\b[A-HJ-NPR-Z0-9]{17}\b'],  # Vehicle identification numbers

            # URLs with potential tokens
            'urls_with_tokens': [
                r'https?://[^\s"\'<>]+\?(?:[^&\s]*[?&])?(?:token|key|auth|access_token|session_id)=[^\s"\'<>&]+',
                r'https?://[^\s"\'<>]+/[A-Za-z0-9_-]{20,}'  # URLs ending with long tokens
            ],

            # Session and temporary identifiers
            'session_ids': [
                r'\bsession[_-]?id\s*[=:]\s*[A-Za-z0-9_-]{10,}\b',
                r'\bPHPSESSID\s*[=:]\s*[A-Za-z0-9_-]+\b',
                r'\bJSESSIONID\s*[=:]\s*[A-Za-z0-9_-]+\b'
            ]
        }

        # Enhanced keywords that might indicate sensitive content (English + Chinese)
        sensitive_keywords = [
            # English
            'password', 'ssn', 'social security', 'passport', 'license', 'driver license',
            'account number', 'routing', 'credit card', 'bank', 'bank account',
            'diagnosis', 'medication', 'doctor', 'hospital', 'medical record',
            'api key', 'token', 'secret', 'private key', 'access token', 'bearer token',
            'encryption key', 'certificate', 'ssl key', 'rsa key',
            'fingerprint', 'biometric', 'facial recognition', 'iris scan',
            'ip address', 'mac address', 'imei', 'vin', 'vehicle id',
            'oauth', 'jwt', 'session id', 'cookie',
            'database', 'mongodb', 'mysql', 'postgresql', 'redis',
            'coordinates', 'gps', 'location', 'latitude', 'longitude',

            # Chinese
            '密码', '口令', '秘钥', '密钥', '私钥', '公钥',
            '身份证', '护照', '驾照', '身份证号', '护照号',
            '银行卡', '信用卡', '银行账号', '账号', '账户',
            '医保', '社保', '医疗记录', '诊断', '药物', '处方',
            '医生', '医院', '病历', '体检报告',
            '电话', '手机号', '邮箱', '地址', '住址',
            '位置', '坐标', 'GPS', '经度', '纬度',
            '指纹', '人脸识别', '虹膜', '生物识别',
            'IP地址', 'MAC地址', 'IMEI', 'VIN',
            'API密钥', '访问令牌', '认证令牌', '会话ID',
            '数据库', 'MongoDB', 'MySQL', 'PostgreSQL', 'Redis',
            '加密', '证书', 'SSL证书', 'RSA密钥'
        ]

        # Context patterns that indicate sensitive information
        context_patterns = [
            r'\bmy\s+(?:password|ssn|account|card|key|token|address|phone|email)\s+is\b',
            r'\b我的\s*(?:密码|身份证|账号|卡|密钥|令牌|地址|电话|邮箱)\s*是\b',
            r'\bhere\s+is\s+my\s+(?:password|ssn|account|card|key|token)\b',
            r'\b这是\s*我的\s*(?:密码|身份证|账号|卡|密钥|令牌)\b',
            r'\bdon\'t\s+share\s+(?:this|my)\b.*\b(?:password|key|token)\b',
            r'\b不要分享\b.*\b(?:密码|密钥|令牌)\b',
            r'\bconfidential\b.*\b(?:information|data)\b',
            r'\b机密\b.*\b(?:信息|数据)\b'
        ]

        logger.debug(f"Preprocessing {len(records)} conversation records for pattern analysis")

        for i, record in enumerate(records):
            original_content = record.get('user', '')
            content = original_content.lower()
            logger.debug(f"Processing message {i+1}/{len(records)}: {len(original_content)} chars")

            # Apply regex patterns
            for pattern_category, pattern_list in patterns.items():
                for pattern in pattern_list:
                    try:
                        matches = re.findall(pattern, original_content, re.IGNORECASE | re.UNICODE)
                        if matches:
                            # Clean and deduplicate matches
                            cleaned_matches = []
                            for match in matches:
                                if isinstance(match, tuple):
                                    match = ''.join(match)  # Handle capture groups
                                match = match.strip()
                                if match and len(match) > 3 and match not in cleaned_matches:
                                    cleaned_matches.append(match)

                            if pattern_category in pattern_indicators:
                                pattern_indicators[pattern_category].extend(cleaned_matches)
                    except re.error as e:
                        logger.warning(f"Regex pattern error for {pattern_category}: {e}")
                        continue

            # Check for sensitive keywords with context
            for keyword in sensitive_keywords:
                if keyword.lower() in content:
                    # Add context around the keyword
                    keyword_pos = content.find(keyword.lower())
                    start = max(0, keyword_pos - 30)
                    end = min(len(original_content), keyword_pos + len(keyword) + 30)
                    context = original_content[start:end]

                    pattern_indicators.setdefault('keyword_matches', []).append({
                        'keyword': keyword,
                        'context': context,
                        'position': keyword_pos
                    })

            # Check for context patterns
            for context_pattern in context_patterns:
                try:
                    if re.search(context_pattern, content, re.IGNORECASE | re.UNICODE):
                        pattern_indicators.setdefault('context_matches', []).append({
                            'pattern': context_pattern,
                            'matched_text': original_content
                        })
                except re.error as e:
                    logger.warning(f"Context pattern error: {e}")
                    continue

        # Remove duplicates and filter out obviously invalid matches
        for category in pattern_indicators:
            if category != 'keyword_matches' and category != 'context_matches':
                items = pattern_indicators[category]
                # Basic validation for different categories
                if category == 'phone_numbers':
                    # Validate phone-like patterns (basic check)
                    valid_items = [item for item in items if len(re.sub(r'\D', '', item)) >= 7]
                    pattern_indicators[category] = list(set(valid_items))
                elif category == 'emails':
                    # Basic email validation
                    valid_items = [item for item in items if '@' in item and '.' in item.split('@')[1]]
                    pattern_indicators[category] = list(set(valid_items))
                elif category == 'ip_addresses':
                    # Validate IP addresses
                    valid_items = []
                    for item in items:
                        parts = item.split('.')
                        if len(parts) == 4 and all(part.isdigit() and 0 <= int(part) <= 255 for part in parts):
                            valid_items.append(item)
                    pattern_indicators[category] = list(set(valid_items))
                else:
                    pattern_indicators[category] = list(set(items))

        return pattern_indicators

    def _post_process_with_patterns(self, result: Dict[str, Any], pattern_analysis: Dict[str, Any], conversation_records: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Post-process LLM results with pattern analysis for validation and enhancement.

        Args:
            result: LLM analysis result
            pattern_analysis: Pattern detection results
            conversation_records: Original conversation records for location validation

        Returns:
            Enhanced result with pattern validation
        """
        # Enhanced pattern-based detection with LLM collaboration
        pattern_score = self._calculate_overall_pattern_score(pattern_analysis)

        # If LLM detected no leaks but patterns suggest otherwise, add pattern-based detection
        if not result.get("privacy_detected", False):
            significant_patterns = []
            high_risk_categories = [
                'api_keys', 'passwords', 'ssn_patterns', 'credit_cards',
                'encryption_keys', 'database_credentials', 'oauth_tokens',
                'id_numbers', 'biometric_data'
            ]

            for category, items in pattern_analysis.items():
                if items and len(items) > 0:
                    if category in high_risk_categories:
                        significant_patterns.append((category, len(items), 'high'))
                    elif category in ['phone_numbers', 'emails', 'bank_accounts', 'health_records']:
                        if len(items) >= 2:  # Multiple instances
                            significant_patterns.append((category, len(items), 'medium'))
                    elif len(items) >= 3:  # Many instances of less critical patterns
                        significant_patterns.append((category, len(items), 'low'))

            if significant_patterns:
                result["privacy_detected"] = True

                # Determine severity based on most severe pattern
                max_severity = max((severity for _, _, severity in significant_patterns),
                                 key=lambda x: {'high': 3, 'medium': 2, 'low': 1}[x])

                severity_map = {'high': 'high', 'medium': 'medium', 'low': 'low'}
                # Don't set overall_severity here - it will be calculated later in _validate_result_structure

                # Create detailed pattern-based leaks
                for category, count, severity in significant_patterns:
                    leak_items = pattern_analysis[category][:5]  # Limit to first 5 items
                    privacy_type = self._map_pattern_to_privacy_type(category)
                    # Clean and format the detected items
                    cleaned_items = self._clean_and_format_detected_items(leak_items, privacy_type)

                    # Only create leak if we have actual detected items after cleaning
                    if cleaned_items:
                        result["privacy_leaks"].append({
                            "privacy_type": privacy_type,
                            "severity": severity_map[severity],
                            "reasoning": f"Pattern detection identified {len(cleaned_items)} potential {category.replace('_', ' ')} instances. This may indicate privacy-sensitive information that requires manual verification.",
                            "detected_items": cleaned_items,
                            "confidence": self._calculate_pattern_confidence(category, count, pattern_score)
                        })

        # Adjust LLM-detected leaks based on pattern corroboration
        elif result.get("privacy_leaks"):
            for leak in result["privacy_leaks"]:
                llm_privacy_type = leak.get("privacy_type", "")
                pattern_corroboration = self._check_pattern_corroboration(llm_privacy_type, pattern_analysis)

                # Adjust confidence based on pattern corroboration
                original_confidence = leak.get("confidence", 0.5)
                if pattern_corroboration > 0:
                    # Increase confidence when patterns support LLM findings
                    leak["confidence"] = min(0.95, original_confidence + (pattern_corroboration * 0.2))
                    leak["pattern_corroboration"] = pattern_corroboration
                elif pattern_score < 0.3:
                    # When overall pattern score is low, rely more on LLM but reduce confidence slightly
                    leak["confidence"] = max(0.3, original_confidence * 0.9)
                    leak["llm_dependent"] = True

        # Add confidence scoring based on pattern corroboration and location validation
        if result.get("privacy_leaks"):
            for leak in result["privacy_leaks"]:
                # First calculate pattern-based confidence
                pattern_confidence = self._calculate_confidence(leak, pattern_analysis)

                # Then validate locations against original messages
                leak = self._validate_leak_locations(leak, conversation_records)

                # Combine confidences: location validation takes precedence
                location_confidence_raw = leak.get("confidence", 0.5)
                if isinstance(location_confidence_raw, str):
                    if location_confidence_raw == "invalid":
                        location_confidence = 0.1  # Very low confidence for invalid validation
                    else:
                        location_confidence = 0.5  # Default for other string values
                else:
                    location_confidence = float(location_confidence_raw)

                if location_confidence <= 0.1:
                    leak["confidence"] = 0.1  # Very low confidence due to validation failure
                else:
                    # Weighted combination of pattern and location confidence
                    combined_confidence = (pattern_confidence * 0.4) + (location_confidence * 0.6)
                    leak["confidence"] = max(0.1, min(0.95, combined_confidence))

        return result

    def _calculate_confidence(self, leak: Dict[str, Any], pattern_analysis: Dict[str, Any]) -> float:
        """
        Calculate confidence level for a detected leak based on pattern corroboration.

        Args:
            leak: Individual leak detection
            pattern_analysis: Pattern detection results

        Returns:
            Confidence score between 0.0 and 1.0
        """
        privacy_type = leak.get("privacy_type", "")
        detected_items = leak.get("detected_items", [])

        # Pattern corroboration mapping
        corroboration_map = {
            "contact_info": ["phone_numbers", "emails", "addresses"],
            "government_ids": ["ssn_patterns"],
            "financial_accounts": ["credit_cards"],
            "authentication_credentials": ["passwords", "api_keys"],
            "precise_location": ["coordinates", "addresses"]
        }

        expected_patterns = corroboration_map.get(privacy_type, [])
        matched_patterns = sum(1 for pattern in expected_patterns if pattern_analysis.get(pattern))

        # Base confidence from pattern matching
        pattern_confidence = min(0.9, matched_patterns * 0.3 + 0.4)  # 0.4 to 0.9

        # Adjust based on detected items count
        item_multiplier = min(1.0, len(detected_items) * 0.2)  # More items = higher confidence
        confidence = pattern_confidence * item_multiplier

        # Ensure reasonable bounds
        return max(0.1, min(0.95, confidence))

    def _validate_leak_locations(self, leak: Dict[str, Any], conversation_records: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Validate that detected items actually exist in the original messages.

        Args:
            leak: Individual leak detection result
            conversation_records: Original conversation records

        Returns:
            Validated leak with confidence score
        """
        # Validate input parameters
        if not conversation_records or not isinstance(conversation_records, list):
            leak["confidence"] = "invalid"
            leak["validation_error"] = "Invalid conversation records provided"
            return leak

        detected_items = leak.get("detected_items", [])
        locations = leak.get("locations", [])

        validated_items = []
        validated_locations = []

        # Validate each detected item exists in original messages
        for item in detected_items:
            # Skip pattern category items (they are added by our pattern detection logic)
            if item.startswith("Pattern category:"):
                validated_items.append(item)
                validated_locations.append({
                    "message_index": -1,  # Special marker for pattern-detected items
                    "exact_text": item,
                    "start_position": -1,
                    "end_position": -1
                })
                continue

            item_found = False
            item_locations = []

            # Check each message for this item
            for msg_idx, record in enumerate(conversation_records):
                message_text = record.get('user', '')

                # Find all occurrences of this item in the message
                start_pos = 0
                while True:
                    pos = message_text.find(item, start_pos)
                    if pos == -1:
                        break

                    item_found = True
                    end_pos = pos + len(item)

                    item_locations.append({
                        "message_index": msg_idx,
                        "exact_text": item,
                        "start_position": pos,
                        "end_position": end_pos
                    })

                    start_pos = pos + 1  # Continue searching for more occurrences

            if item_found:
                validated_items.append(item)
                validated_locations.extend(item_locations)
            else:
                logger.warning(f"Detected item '{item}' not found in original messages - potential LLM hallucination")

        # If no items were validated, mark as invalid
        if not validated_items:
            leak["confidence"] = "invalid"
            leak["validation_error"] = "No detected items found in original messages"
            return leak

        # Update leak with validated data
        leak["detected_items"] = validated_items
        leak["locations"] = validated_locations

        # Calculate final confidence based on validation
        total_validated_items = len(validated_items)
        total_original_items = len(detected_items)

        if total_validated_items == total_original_items:
            # All items validated successfully
            base_confidence = 0.8 if total_validated_items >= 2 else 0.6
            leak["confidence"] = min(0.95, base_confidence)
        elif total_validated_items > 0:
            # Some items validated
            validation_ratio = total_validated_items / total_original_items
            leak["confidence"] = max(0.3, min(0.7, validation_ratio * 0.7))
            leak["validation_note"] = f"{total_validated_items}/{total_original_items} items validated"
        else:
            # No items validated - very low confidence
            leak["confidence"] = 0.1
            leak["validation_error"] = "No detected items validated against original messages"

        return leak

    def _validate_result_structure(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and ensure the result structure meets API requirements.

        Args:
            result: Analysis result to validate

        Returns:
            Validated and corrected result
        """
        # Ensure required fields exist
        required_fields = {
            "privacy_detected": False,
            "privacy_leaks": [],
            "overall_severity": "none",
            "overall_score": 0.0
        }

        for field, default in required_fields.items():
            if field not in result:
                result[field] = default

        # Validate privacy_leaks structure
        if not isinstance(result["privacy_leaks"], list):
            result["privacy_leaks"] = []

        # Clean up leak entries
        validated_leaks = []
        for leak in result["privacy_leaks"]:
            if isinstance(leak, dict):
                validated_leak = {
                    "privacy_type": leak.get("privacy_type", "unknown"),
                    "severity": leak.get("severity", "low"),
                    "reasoning": leak.get("reasoning", "Detected potential privacy concern"),
                    "detected_items": leak.get("detected_items", [])
                }
                # Add confidence if not present
                if "confidence" not in validated_leak:
                    validated_leak["confidence"] = 0.5
                # Add locations if present
                if "locations" in leak:
                    validated_leak["locations"] = leak["locations"]
                # Add validation metadata if present
                if "validation_error" in leak:
                    validated_leak["validation_error"] = leak["validation_error"]
                if "validation_note" in leak:
                    validated_leak["validation_note"] = leak["validation_note"]
                validated_leaks.append(validated_leak)

        result["privacy_leaks"] = validated_leaks

        # Always recalculate overall severity based on all leaks
        if validated_leaks:
            severity_levels = {"low": 1, "medium": 2, "high": 3, "critical": 4}
            max_severity = max(
                (leak["severity"] for leak in validated_leaks if leak["severity"] in severity_levels),
                key=lambda s: severity_levels.get(s, 0),
                default="low"
            )
            result["overall_severity"] = max_severity

            # Calculate overall score as average confidence
            total_confidence = sum(float(leak.get("confidence", 0.5)) for leak in validated_leaks)
            result["overall_score"] = round(total_confidence / len(validated_leaks), 3)
        else:
            result["overall_severity"] = "none"
            result["overall_score"] = 0.0

        return result

    async def detect_privacy_leaks(
        self,
        conversation_records: List[Dict[str, str]],
        execution_mode: Optional[ExecutionMode] = None,
        use_cot: Optional[ExecutionMode] = None,
        use_few_shots: bool = True
    ) -> Dict[str, Any]:
        """
        Detect privacy leaks in user messages using systematic pattern analysis and LLM reasoning.

        Args:
            conversation_records: List of user messages with 'user' key containing the message content
            execution_mode: Execution mode ('chain_online', 'chain_local', 'no_chain') - deprecated, use use_cot instead
            use_cot: Chain of Thought mode ('chain_online', 'chain_local', 'no_chain')
            use_few_shots: Whether to use few-shot examples

        Returns:
            Dict containing privacy analysis results
        """
        try:
            # Preprocess for pattern detection
            pattern_analysis = self._preprocess_for_patterns(conversation_records)

            # Format conversation records for analysis
            formatted_text = self._format_conversation_records(conversation_records)

            # Add pattern insights to the prompt if patterns detected
            pattern_summary = ""
            if any(pattern_analysis.values()):
                pattern_summary = "\n\nPATTERN DETECTION RESULTS:\n"
                for category, items in pattern_analysis.items():
                    if items:
                        pattern_summary += f"- {category.replace('_', ' ').title()}: {len(items)} potential matches\n"
                pattern_summary += "\nConsider these patterns in your analysis.\n\n"

            # Determine execution mode (support both execution_mode and use_cot parameters)
            if use_cot is not None:
                effective_execution_mode = use_cot.value if isinstance(use_cot, CoTMode) else str(use_cot)
            else:
                effective_execution_mode = execution_mode or settings.default_execution_mode

            # Get few shots
            if use_few_shots:
                few_shots = self.get_effective_few_shots()
            else:
                few_shots = ""

            # Prepare system prompt
            if effective_execution_mode == "chain_online":
                system_prompt = self.SYSTEM_PROMPT_TEMPLATE_COT.format(few_shots=few_shots)
                human_prompt = f"Please analyze ALL these user messages for privacy leaks. You MUST examine every single message in the conversation, even if some messages contain no sensitive information:{pattern_summary}\n\n{formatted_text}\n\nSTRICT REQUIREMENTS - FOLLOW EXACTLY:\n\nFor detected_items array, you MUST ONLY include extremely concise privacy entities in this exact format:\n- \"SSN: 123-45-6789\"\n- \"Email: user@domain.com\"\n- \"Phone: (555)123-4567\"\n- \"Credit Card: ************1111\"\n- \"Address: 123 Main St\"\n\nDO NOT include:\n- Full sentences or paragraphs\n- Long text excerpts\n- Explanatory text\n- Any content longer than 50 characters per item\n- Multiple sentences\n\nEach detected_items entry must be a single, concise entity identifier.\n\nProvide your analysis in this JSON format:\n{{\n  \"privacy_detected\": true/false,\n  \"privacy_leaks\": [\n    {{\n      \"privacy_type\": \"privacy_type_enum\",\n      \"severity\": \"severity_level_enum\",\n      \"reasoning\": \"detailed explanation for this specific leak\",\n      \"detected_items\": [\"SSN: 123-45-6789\"],\n      \"locations\": [\n        {{\n          \"message_index\": 0,\n          \"exact_text\": \"exact text from original message\",\n          \"start_position\": 0,\n          \"end_position\": 50\n        }}\n      ]\n    }}\n  ],\n  \"overall_severity\": \"highest severity level across all leaks\",\n  \"reasoning\": \"overall explanation of the privacy analysis\"\n}}\n\nIf no privacy leaks are detected, set privacy_detected to false and privacy_leaks to an empty array."
            else:
                system_prompt = self.SYSTEM_PROMPT_TEMPLATE_NO_COT.format(
                    few_shots=few_shots,
                    text=formatted_text
                )
                human_prompt = f"IMPORTANT: Analyze ALL user messages above for privacy leaks. You MUST examine every single message in the conversation.\n\nSTRICT REQUIREMENTS - FOLLOW EXACTLY:\n\nFor detected_items array, you MUST ONLY include extremely concise privacy entities in this exact format:\n- \"SSN: 123-45-6789\"\n- \"Email: user@domain.com\"\n- \"Phone: (555)123-4567\"\n- \"Credit Card: ************1111\"\n- \"Address: 123 Main St\"\n\nDO NOT include:\n- Full sentences or paragraphs\n- Long text excerpts\n- Explanatory text\n- Any content longer than 50 characters per item\n- Multiple sentences\n\nEach detected_items entry must be a single, concise entity identifier.\n\nAnalyze the above content for privacy leaks and provide your response in JSON format with locations for each leak."

            # Get LLM response
            response = await self._call_llm(system_prompt, human_prompt, effective_execution_mode)

            # Parse and validate response
            result = self._parse_response(response)

            # Post-process with pattern validation
            result = self._post_process_with_patterns(result, pattern_analysis, conversation_records)

            # Add metadata
            result.update({
                "execution_mode": effective_execution_mode,
                "conversation_length": len(conversation_records),
                "messages_analyzed": len(conversation_records),  # Confirm all messages were analyzed
                "analyzed_at": self._get_timestamp(),
                "agent_version": "1.0.0",
                "pattern_analysis": pattern_analysis  # Include pattern analysis for transparency
            })

            # Extract account_id from conversation_records if present
            account_id = None
            if conversation_records and len(conversation_records) > 0:
                # Check if there's an account_id in the first record or as a separate field
                first_record = conversation_records[0]
                if isinstance(first_record, dict) and 'account_id' in first_record:
                    account_id = first_record['account_id']

            # Persist result to database
            detection_id = self._persist_detection_result(
                conversation_records,
                result,
                effective_execution_mode,
                use_few_shots,
                account_id
            )

            # Add detection_id to result
            result["detection_id"] = detection_id

            # Validate final result structure
            result = self._validate_result_structure(result)

            # Final validation: ensure all messages were analyzed
            if result.get("messages_analyzed") != len(conversation_records):
                logger.warning(f"Message count mismatch: analyzed {result.get('messages_analyzed')} but received {len(conversation_records)} messages")
                result["messages_analyzed"] = len(conversation_records)  # Correct the count

            return result

        except Exception as e:
            logger.error(f"Failed to detect privacy leaks: {e}", exc_info=True)

            # Generate unique detection ID for error case
            import uuid
            detection_id = str(uuid.uuid4())

            return {
                "error": f"Privacy detection failed: {str(e)}",
                "privacy_detected": False,
                "privacy_leaks": [],
                "overall_severity": PrivacySeverity.NONE.value,
                "overall_score": 0.0,
                "execution_mode": execution_mode if execution_mode else settings.default_execution_mode,
                "conversation_length": len(conversation_records),
                "analyzed_at": self._get_timestamp(),
                "agent_version": "1.0.0",
                "detection_id": detection_id
            }

    def _format_conversation_records(self, records: List[Dict[str, str]]) -> str:
        """Format user messages for analysis."""
        logger.debug(f"Formatting {len(records)} conversation records for LLM analysis")
        formatted = []
        for i, record in enumerate(records):
            user_content = record.get('user', '')
            formatted.append(f"[{i+1}] User: {user_content}")
            logger.debug(f"Message {i+1}: {len(user_content)} characters")
        result = "\n".join(formatted)
        logger.debug(f"Formatted text length: {len(result)} characters")
        return result

    @llm_cached(ttl=3600)  # Cache LLM responses for 1 hour
    async def _call_llm_cached(self, system_prompt: str, human_prompt: str) -> str:
        """Call LLM with caching based on prompts."""
        llm = llm_manager.get_llm_client()

        messages = [SystemMessage(content=system_prompt)]
        if human_prompt:
            messages.append(HumanMessage(content=human_prompt))
        else:
            # Ensure we have at least one user message for APIs that require it
            messages.append(HumanMessage(content="Please analyze the above system prompt and provide a response."))

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
            # Clean the response
            response = response.strip()

            # Handle markdown code blocks
            if "```" in response:
                # Look for JSON within code blocks
                import re

                # Pattern to find JSON within ```json or ``` blocks
                json_pattern = r'```\w*\s*(\{.*?\})\s*```'
                match = re.search(json_pattern, response, re.DOTALL)
                if match:
                    json_str = match.group(1).strip()
                    parsed = json.loads(json_str)
                else:
                    # Fallback: look for any JSON object within backticks
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        # Make sure we're not inside a code block marker
                        before_json = response[:json_start]
                        after_json = response[json_end:]
                        if not ('```' in before_json and '```' in after_json):
                            json_str = response[json_start:json_end]
                            parsed = json.loads(json_str)
                        else:
                            # Extract from within code block
                            code_block_match = re.search(r'```\w*\s*(\{.*?\})\s*```', response, re.DOTALL)
                            if code_block_match:
                                parsed = json.loads(code_block_match.group(1).strip())
                            else:
                                parsed = json.loads(response)
                    else:
                        parsed = json.loads(response)
            else:
                # No code blocks, try direct JSON parsing
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    parsed = json.loads(json_str)
                else:
                    # Fallback: assume entire response is JSON
                    parsed = json.loads(response)

            # Validate that parsed is a dictionary
            if not isinstance(parsed, dict):
                raise json.JSONDecodeError("Parsed response is not a dictionary", response, 0)

            # Validate required fields
            privacy_leaks = parsed.get("privacy_leaks", [])
            privacy_detected = parsed.get("privacy_detected", len(privacy_leaks) > 0)

            # Validate and clean privacy leaks
            validated_leaks = []
            for leak in privacy_leaks:
                if isinstance(leak, dict):
                    # Validate and clean detected_items
                    detected_items = leak.get("detected_items", [])
                    if not isinstance(detected_items, list):
                        detected_items = []
                    else:
                        # Ensure all items are strings and filter out empty/invalid items
                        detected_items = [str(item).strip() for item in detected_items if item and str(item).strip()]

                        # Clean and format detected items to be concise entities
                        privacy_type = leak.get("privacy_type", PrivacyType.NONE.value)
                        detected_items = self._clean_and_format_detected_items(detected_items, privacy_type)

                    validated_leak = {
                        "privacy_type": leak.get("privacy_type", PrivacyType.NONE.value),
                        "severity": leak.get("severity", PrivacySeverity.NONE.value),
                        "reasoning": leak.get("reasoning", "No reasoning provided"),
                        "detected_items": detected_items
                    }

                    # Validate enum values
                    if validated_leak["privacy_type"] not in [pt.value for pt in PrivacyType]:
                        validated_leak["privacy_type"] = PrivacyType.NONE.value

                    if validated_leak["severity"] not in [ps.value for ps in PrivacySeverity]:
                        validated_leak["severity"] = PrivacySeverity.NONE.value

                    # Only add leak if it has detected items
                    if validated_leak.get("detected_items"):
                        validated_leaks.append(validated_leak)

            # Don't calculate overall_severity here - it will be calculated later in _validate_result_structure
            result = {
                "privacy_detected": privacy_detected,
                "privacy_leaks": validated_leaks
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
        use_few_shots: bool,
        account_id: Optional[str] = None
    ) -> str:
        """
        Persist privacy detection result to database.

        Args:
            conversation_records: Original user messages
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
                "error": detection_result.get("error"),
                "account_id": account_id
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

    def _calculate_overall_pattern_score(self, pattern_analysis: Dict[str, Any]) -> float:
        """
        Calculate an overall score for pattern detection strength.

        Args:
            pattern_analysis: Pattern detection results

        Returns:
            Score between 0.0 and 1.0 indicating pattern detection strength
        """
        total_items = 0
        weighted_score = 0.0

        # Define weights for different pattern categories
        weights = {
            'api_keys': 1.0, 'passwords': 1.0, 'encryption_keys': 1.0,
            'database_credentials': 1.0, 'oauth_tokens': 1.0,
            'ssn_patterns': 0.9, 'credit_cards': 0.9, 'id_numbers': 0.9,
            'biometric_data': 0.8, 'health_records': 0.8,
            'phone_numbers': 0.6, 'emails': 0.6, 'bank_accounts': 0.7,
            'coordinates': 0.5, 'ip_addresses': 0.4, 'mac_addresses': 0.4,
            'addresses': 0.5, 'social_accounts': 0.4,
            'urls_with_tokens': 0.7, 'session_ids': 0.6,
            'imei_numbers': 0.5, 'vin_numbers': 0.4
        }

        for category, items in pattern_analysis.items():
            if category in ['keyword_matches', 'context_matches']:
                # Special handling for keyword and context matches
                if items:
                    count = len(items)
                    total_items += count
                    weighted_score += count * 0.3  # Keywords are less definitive
            elif items:
                count = len(items)
                weight = weights.get(category, 0.1)
                total_items += count
                weighted_score += count * weight

        if total_items == 0:
            return 0.0

        # Normalize to 0-1 scale, with diminishing returns for very high counts
        raw_score = weighted_score / (total_items + weighted_score * 0.5)
        return min(1.0, raw_score)

    def _map_pattern_to_privacy_type(self, pattern_category: str) -> str:
        """
        Map pattern detection categories to privacy types.

        Args:
            pattern_category: Pattern category name

        Returns:
            Corresponding privacy type string
        """
        mapping = {
            'phone_numbers': 'contact_info',
            'emails': 'contact_info',
            'social_accounts': 'contact_info',
            'ssn_patterns': 'government_ids',
            'id_numbers': 'government_ids',
            'credit_cards': 'financial_accounts',
            'bank_accounts': 'financial_accounts',
            'api_keys': 'api_keys_tokens',
            'oauth_tokens': 'api_keys_tokens',
            'session_ids': 'api_keys_tokens',
            'database_credentials': 'system_access',
            'encryption_keys': 'encryption_keys',
            'coordinates': 'precise_location',
            'ip_addresses': 'precise_location',
            'mac_addresses': 'precise_location',
            'addresses': 'contact_info',
            'biometric_data': 'biometric_data',
            'health_records': 'health_records',
            'urls_with_tokens': 'api_keys_tokens',
            'imei_numbers': 'biometric_data',
            'vin_numbers': 'personal_identifiers'
        }
        return mapping.get(pattern_category, 'personal_identifiers')

    def _calculate_pattern_confidence(self, category: str, count: int, overall_pattern_score: float) -> float:
        """
        Calculate confidence score for pattern-based detections.

        Args:
            category: Pattern category
            count: Number of matches
            overall_pattern_score: Overall pattern detection score

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence by category
        base_confidence = {
            'api_keys': 0.8, 'passwords': 0.8, 'encryption_keys': 0.8,
            'database_credentials': 0.8, 'oauth_tokens': 0.8,
            'ssn_patterns': 0.7, 'credit_cards': 0.7, 'id_numbers': 0.7,
            'biometric_data': 0.6, 'health_records': 0.6,
            'phone_numbers': 0.5, 'emails': 0.5, 'bank_accounts': 0.6,
            'coordinates': 0.5, 'ip_addresses': 0.4, 'mac_addresses': 0.4,
            'addresses': 0.4, 'social_accounts': 0.4,
            'urls_with_tokens': 0.6, 'session_ids': 0.5,
            'imei_numbers': 0.5, 'vin_numbers': 0.4
        }.get(category, 0.3)

        # Adjust by count (more matches = higher confidence)
        count_multiplier = min(1.5, 1.0 + (count - 1) * 0.1)

        # Adjust by overall pattern strength
        pattern_multiplier = 0.8 + (overall_pattern_score * 0.4)

        confidence = base_confidence * count_multiplier * pattern_multiplier
        return max(0.1, min(0.9, confidence))  # Conservative upper bound for pattern-only detection

    def _clean_and_format_detected_items(self, detected_items: List[str], privacy_type: str) -> List[str]:
        """
        Clean and format detected items to ensure they are concise entity representations.

        Args:
            detected_items: Raw detected items from LLM
            privacy_type: Type of privacy information

        Returns:
            Cleaned and formatted entity list
        """
        if not detected_items:
            return []

        cleaned_items = []

        for item in detected_items:
            if not item or not isinstance(item, str):
                continue

            item = item.strip()

            # Skip if item is too long (likely a sentence rather than an entity)
            if len(item) > 100:  # More aggressive filtering for long text
                continue

            # Skip if item contains multiple sentences or long text
            if item.count('.') > 1 or len(item.split()) > 8:  # Very strict filtering
                continue

            # Skip items that look like full paragraphs or long text
            if '\n' in item or item.count(' ') > 50:
                continue

            # Skip items that don't look like entities (contain common paragraph words)
            paragraph_indicators = [' the ', ' and ', ' or ', ' but ', ' however ', ' therefore ', ' moreover ']
            if any(indicator in item.lower() for indicator in paragraph_indicators) and len(item.split()) > 8:
                continue

            # Additional strict filtering for items that don't match expected entity format
            # Should be in format like "Type: value" or just the value
            if ':' in item:
                # Check if it's a proper entity format
                parts = item.split(':', 1)
                if len(parts) == 2:
                    entity_type = parts[0].strip()
                    entity_value = parts[1].strip()
                    # Valid entity types
                    valid_types = ['SSN', 'Email', 'Phone', 'Credit Card', 'Bank Account', 'API Key', 'Address']
                    if entity_type in valid_types and len(entity_value) > 0 and len(entity_value) <= 100:
                        pass  # Valid format
                    else:
                        continue  # Invalid format
                else:
                    continue  # Malformed
            else:
                # If no colon, it could be various types of entities
                # Allow reasonable length items that might be addresses or other entities
                if len(item) > 100:  # More lenient for non-colon format items
                    continue

            # Format based on privacy type and content
            formatted_item = self._format_entity_item(item, privacy_type)
            if formatted_item:
                cleaned_items.append(formatted_item)

        # Remove duplicates while preserving order
        seen = set()
        unique_items = []
        for item in cleaned_items:
            if item not in seen:
                seen.add(item)
                unique_items.append(item)

        return unique_items[:10]  # Limit to 10 items max

    def _format_entity_item(self, item: str, privacy_type: str) -> Optional[str]:
        """
        Format a single detected item into a clean entity representation.

        Args:
            item: Raw detected item
            privacy_type: Privacy type for context

        Returns:
            Formatted entity string or None if invalid
        """
        # Common patterns to clean up
        item = item.strip()

        # If already in good format (like "SSN: 123-45-6789"), keep it
        if ':' in item and len(item.split(':', 1)[0].strip()) <= 20:
            return item

        # Format based on privacy type first, then content patterns
        import re

        # Type-specific formatting
        if privacy_type == 'contact_info':
            # Phone numbers
            if re.match(r'[\d\s\-\(\)\+\.]{7,20}', item):
                cleaned = re.sub(r'[^\d\+\-\(\)\.\s]', '', item).strip()
                if cleaned and len(cleaned) >= 7:
                    return f"Phone: {cleaned}"

            # Email addresses
            if '@' in item and '.' in item.split('@')[1]:
                return f"Email: {item}"

            # Addresses (for contact_info type)
            if len(item) > 5 and any(keyword in item.lower() for keyword in ['street', 'avenue', 'road', 'drive', 'lane', 'boulevard', 'place', 'court']):
                # This looks like an address
                if len(item) <= 100:
                    return f"Address: {item}"

        elif privacy_type == 'financial_accounts':
            # Credit cards (16 digits)
            if re.match(r'\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}', item):
                cleaned = re.sub(r'\D', '', item)
                if len(cleaned) == 16:
                    masked = '*' * 12 + cleaned[-4:]
                    return f"Credit Card: {masked}"

            # Bank accounts (8-17 digits, not credit card)
            elif re.match(r'\d{8,17}', item):
                cleaned = re.sub(r'\D', '', item)
                if len(cleaned) >= 8:
                    masked = '*' * (len(cleaned) - 4) + cleaned[-4:]
                    return f"Bank Account: {masked}"

        elif privacy_type == 'government_ids':
            # SSN patterns
            if re.match(r'\d{3}[-]?\d{2}[-]?\d{4}', item):
                return f"SSN: {item}"

        elif privacy_type in ['api_keys_tokens', 'authentication_credentials']:
            # API keys and tokens
            if len(item) > 15 and any(keyword in item.lower() for keyword in ['key', 'token', 'secret', 'bearer']):
                if len(item) > 20:
                    masked = item[:12] + '*' * (len(item) - 12)
                    return f"API Key: {masked}"
                else:
                    return f"API Key: {item}"

        # Generic patterns for other types
        # Email addresses (if not caught above)
        if '@' in item and '.' in item.split('@')[1]:
            return f"Email: {item}"

        # Phone numbers (if not caught above)
        if re.match(r'[\d\s\-\(\)\+\.]{7,15}', item):
            cleaned = re.sub(r'[^\d\+\-\(\)\.\s]', '', item).strip()
            if cleaned and len(cleaned) >= 7:
                return f"Phone: {cleaned}"

        # API keys (generic fallback)
        if len(item) > 20 and re.match(r'[A-Za-z0-9_\-]{20,}', item):
            masked = item[:10] + '*' * (len(item) - 10)
            return f"API Key: {masked}"

        # For other cases, if it's reasonably short, keep as is
        if len(item) <= 50 and not any(char in item for char in ['\n', '\r']):
            return item

        return None

    def _check_pattern_corroboration(self, privacy_type: str, pattern_analysis: Dict[str, Any]) -> float:
        """
        Check how well patterns corroborate a given privacy type.

        Args:
            privacy_type: Privacy type from LLM analysis
            pattern_analysis: Pattern detection results

        Returns:
            Corroboration score between 0.0 and 1.0
        """
        # Define which patterns support which privacy types
        corroboration_map = {
            "contact_info": ["phone_numbers", "emails", "addresses", "social_accounts"],
            "government_ids": ["ssn_patterns", "id_numbers"],
            "financial_accounts": ["credit_cards", "bank_accounts"],
            "authentication_credentials": ["passwords", "api_keys", "oauth_tokens"],
            "api_keys_tokens": ["api_keys", "oauth_tokens", "urls_with_tokens", "session_ids"],
            "system_access": ["database_credentials", "encryption_keys"],
            "encryption_keys": ["encryption_keys"],
            "precise_location": ["coordinates", "ip_addresses", "addresses"],
            "location_history": ["coordinates", "ip_addresses"],
            "biometric_data": ["biometric_data", "imei_numbers"],
            "health_records": ["health_records"],
            "personal_identifiers": ["id_numbers", "vin_numbers", "social_accounts"]
        }

        expected_patterns = corroboration_map.get(privacy_type, [])
        total_expected = len(expected_patterns)
        if total_expected == 0:
            return 0.0

        matches = sum(1 for pattern in expected_patterns if pattern_analysis.get(pattern))
        return matches / total_expected
