"""File parsing utilities optimized for privacy detection with local/cloud/hybrid modes."""

import abc
import hashlib
import logging
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from langchain_core.messages import SystemMessage, HumanMessage

from src.infrastructure.storage.persistence import get_database
from src.shared.constant.enums import ParseMode, FileType, FileContentCategory, PrivacyType
from src.shared.llm.llm_manager import llm_manager
from src.shared.parsers.parser_config import ParserConfig

logger = logging.getLogger(__name__)

# --- Constants & Patterns ---

# Regex Patterns
PATTERNS = {
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "phone": [
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',      # US phone
        r'\b\d{3}[-]?\d{4}[-]?\d{4}\b',        # Chinese mobile
        r'\+\d{1,3}[-]?\d{3}[-]?\d{3}[-]?\d{4}\b',  # International
    ],
    "ssn": r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',
    "credit_card": r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
    "id_card": r'\b\d{17}[\dXx]\b',            # Chinese ID
    "passport": [
        r'\b[A-Z]{1,2}\d{6,9}\b',              # Common passport
        r'\b\d{9}\b',                          # US passport
    ],
    "drivers_license": r'\b[A-Z]{1,2}\d{6,8}\b',
    "bank_account": [
        r'\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b',   # IBAN
        r'\b\d{8,17}\b',                       # Account numbers
    ],
    "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
    "url": r'\bhttps?://[^\s]+\b',
    "date_of_birth": [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY
        r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',    # YYYY/MM/DD
    ],
    "tax_id": [
        r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',        # US SSN/Tax ID
        r'\b\d{2}[-]?\d{7}\b',                 # EIN
    ],
}

# Keywords for Context Analysis
KEYWORDS = {
    "address": [
        "street", "avenue", "road", "boulevard", "lane", "drive", "way",
        "city", "state", "province", "zip", "postal", "address",
        "街道", "大道", "路", "巷", "城市", "州", "省", "邮编", "地址"
    ],
    "medical_info": [
        "diagnosis", "treatment", "medication", "prescription", "doctor",
        "hospital", "clinic", "medical", "health", "insurance",
        "诊断", "治疗", "药物", "处方", "医生", "医院", "诊所", "医疗", "健康", "保险"
    ],
    "financial_info": [
        "salary", "income", "wage", "bank", "account", "credit", "debit",
        "transaction", "balance", "payment", "finance", "financial",
        "薪水", "收入", "工资", "银行", "账户", "信用", "借记", "交易", "余额", "支付", "金融"
    ],
    "digital_credential": [
        "password", "passcode", "pin", "security", "token", "api", "key", "secret",
        "login", "username", "credential", "authentication",
        "密码", "验证码", "pin码", "安全", "令牌", "api", "密钥", "秘钥", "登录", "用户名"
    ],
    "biometric": [
        "biometric", "biometrics", "fingerprint", "fingerprints",
        "facial recognition", "face id", "iris scan", "retina scan",
        "voice recognition", "dna", "genetic",
        "生物识别", "指纹", "指纹识别", "面部识别", "虹膜扫描",
        "视网膜扫描", "语音识别", "dna", "基因"
    ],
    "financial_transaction": [
        "transaction", "transfer", "payment", "deposit", "withdrawal",
        "balance", "statement", "invoice", "receipt", "wire transfer",
        "交易", "转账", "支付", "存款", "取款", "余额", "对账单",
        "发票", "收据", "电汇"
    ],
    "financial_document": [
        "tax return", "w2", "1099", "financial statement", "audit report",
        "payroll", "salary slip", "bonus", "compensation",
        "纳税申报", "w2表", "1099表", "财务报表", "审计报告",
        "工资单", "薪资单", "奖金", "补偿"
    ],
    "medical_history": [
        "medical history", "surgical history", "allergy", "allergies",
        "chronic condition", "chronic disease", "family history",
        "previous surgery", "past treatment",
        "病史", "手术史", "过敏", "过敏史", "慢性病", "家族史",
        "既往手术", "既往治疗"
    ],
    "prescription": [
        "prescription", "dosage", "milligrams", "mg", "tablets", "capsules",
        "pills", "injection", "intravenous", "oral medication",
        "处方", "剂量", "毫克", "mg", "片剂", "胶囊", "药丸", "注射",
        "静脉注射", "口服药"
    ],
    "encryption_key": [
        "private key", "public key", "ssl certificate", "tls certificate",
        "rsa key", "aes key", "encryption key", "decryption key",
        "私钥", "公钥", "ssl证书", "tls证书", "rsa密钥", "aes密钥",
        "加密密钥", "解密密钥"
    ],
    "precise_location": [
        "coordinates", "latitude", "longitude", "gps location",
        "real-time location", "current position", "exact address",
        "坐标", "纬度", "经度", "gps位置", "实时位置", "当前位置",
        "精确地址"
    ],
    "location_history": [
        "location history", "tracking data", "movement pattern",
        "geofence", "check-in", "route", "path", "trajectory",
        "位置历史", "跟踪数据", "移动模式", "地理围栏", "签到",
        "路线", "路径", "轨迹"
    ],
    "email_content": [
        "email content", "email attachment", "email metadata",
        "email header", "email body", "sent emails", "received emails",
        "邮件内容", "邮件附件", "邮件元数据", "邮件头", "邮件正文",
        "发送邮件", "接收邮件"
    ],
    "messaging": [
        "chat log", "instant message", "text message", "sms",
        "conversation", "message history", "chat record",
        "聊天记录", "即时消息", "短信", "会话", "消息历史", "聊天记录"
    ],
    "call_log": [
        "call log", "call history", "phone call", "voicemail",
        "contact list", "phone book", "dialed numbers",
        "通话记录", "通话历史", "电话", "语音邮件", "联系人列表",
        "电话簿", "拨打号码"
    ],
    "social_security": [
        "social security", "social benefits", "welfare", "unemployment",
        "disability benefits", "retirement benefits",
        "社会保障", "社会福利", "失业救济", "残疾福利", "退休福利"
    ],
    "relationship": [
        "relationship", "family member", "contact network",
        "social persistence", "personal relationship",
        "关系", "家庭成员", "联系人网络", "社交关系", "个人关系"
    ],
    "behavioral": [
        "behavior pattern", "usage habit", "preference", "setting",
        "browsing history", "search history", "app usage",
        "行为模式", "使用习惯", "偏好", "设置", "浏览历史",
        "搜索历史", "应用使用"
    ]
}

# Privacy Type Mapping
PRIVACY_TYPE_MAPPING = {
    # Contact Info
    "email": PrivacyType.CONTACT_INFO.value,
    "phone": PrivacyType.CONTACT_INFO.value,
    "address": PrivacyType.CONTACT_INFO.value,
    # Government IDs
    "ssn": PrivacyType.GOVERNMENT_IDS.value,
    "id_card": PrivacyType.GOVERNMENT_IDS.value,
    "passport": PrivacyType.GOVERNMENT_IDS.value,
    "drivers_license": PrivacyType.GOVERNMENT_IDS.value,
    "tax_id": PrivacyType.GOVERNMENT_IDS.value,
    # Financial
    "credit_card": PrivacyType.FINANCIAL_ACCOUNTS.value,
    "bank_account": PrivacyType.FINANCIAL_ACCOUNTS.value,
    "financial_info": PrivacyType.FINANCIAL_ACCOUNTS.value,
    "financial_transaction": PrivacyType.FINANCIAL_TRANSACTIONS.value,
    "financial_document": PrivacyType.FINANCIAL_DOCUMENTS.value,
    # Authentication
    "digital_credential": PrivacyType.AUTHENTICATION_CREDENTIALS.value,
    "password": PrivacyType.AUTHENTICATION_CREDENTIALS.value,
    "api_key": PrivacyType.API_KEYS_TOKENS.value,
    "token": PrivacyType.API_KEYS_TOKENS.value,
    # Location
    "ip_address": PrivacyType.PRECISE_LOCATION.value,
    "url": PrivacyType.PRECISE_LOCATION.value,
    "coordinates": PrivacyType.PRECISE_LOCATION.value,
    "precise_location": PrivacyType.PRECISE_LOCATION.value,
    "location_history": PrivacyType.LOCATION_HISTORY.value,
    # Medical
    "medical_info": PrivacyType.HEALTH_RECORDS.value,
    "medical_history": PrivacyType.MEDICAL_HISTORY.value,
    "prescription": PrivacyType.PRESCRIPTIONS_MEDICATIONS.value,
    # Personal
    "date_of_birth": PrivacyType.PERSONAL_IDENTIFIERS.value,
    "name": PrivacyType.PERSONAL_IDENTIFIERS.value,
    # Biometric
    "biometric": PrivacyType.BIOMETRIC_DATA.value,
    "fingerprint": PrivacyType.BIOMETRIC_DATA.value,
    # System Access
    "persistence": PrivacyType.SYSTEM_ACCESS.value,
    "encryption_key": PrivacyType.ENCRYPTION_KEYS.value,
    # Communication
    "email_content": PrivacyType.EMAIL_CONTENT.value,
    "messaging": PrivacyType.MESSAGING_CONTENT.value,
    "call_log": PrivacyType.CALL_LOGS.value,
    # Other
    "social_security": PrivacyType.SOCIAL_SECURITY.value,
    "relationship": PrivacyType.RELATIONSHIP_DATA.value,
    "behavioral": PrivacyType.BEHAVIORAL_DATA.value,
}


def map_legacy_indicator_to_privacy_type(indicator: str) -> str:
    """
    Map legacy privacy indicators to PrivacyType enum values.

    Args:
        indicator: Legacy indicator string (e.g., "email", "phone")

    Returns:
        PrivacyType enum value string
    """
    return PRIVACY_TYPE_MAPPING.get(indicator, PrivacyType.NONE.value)


def identify_privacy_indicators(text: str) -> List[str]:
    """
    Identify specific types of privacy information in text.

    Enhanced to detect comprehensive privacy indicators including:
    - Contact information (email, phone, address)
    - Financial information (credit cards, bank accounts)
    - Government IDs (SSN, passport, driver's license)
    - Digital credentials (passwords, API keys)
    - Medical information
    - Location data
    """
    indicators = []
    text_lower = text.lower()

    # Email addresses
    if re.search(PATTERNS["email"], text):
        indicators.append("email")

    # Phone numbers (various formats)
    if any(re.search(pattern, text) for pattern in PATTERNS["phone"]):
        indicators.append("phone")

    # Social Security Numbers
    if re.search(PATTERNS["ssn"], text):
        indicators.append("ssn")

    # Credit/Debit card numbers
    if re.search(PATTERNS["credit_card"], text):
        indicators.append("credit_card")

    # Chinese ID cards
    if re.search(PATTERNS["id_card"], text):
        indicators.append("id_card")

    # Passport numbers (various formats)
    if any(re.search(pattern, text) for pattern in PATTERNS["passport"]):
        indicators.append("passport")

    # Driver's license numbers (state-specific patterns)
    if re.search(PATTERNS["drivers_license"], text) and any(word in text_lower for word in ["license", "driver", "驾照"]):
        indicators.append("drivers_license")

    # Bank account/IBAN numbers
    if any(re.search(pattern, text) for pattern in PATTERNS["bank_account"]):
        indicators.append("bank_account")

    # IP addresses
    if re.search(PATTERNS["ip_address"], text):
        indicators.append("ip_address")

    # URLs/Websites
    if re.search(PATTERNS["url"], text):
        indicators.append("url")

    # Addresses (enhanced detection)
    if any(keyword in text_lower for keyword in KEYWORDS["address"]):
        indicators.append("address")

    # Dates of birth
    if any(re.search(pattern, text) for pattern in PATTERNS["date_of_birth"]):
        indicators.append("date_of_birth")

    # Tax IDs
    if any(re.search(pattern, text) for pattern in PATTERNS["tax_id"]):
        indicators.append("tax_id")

    # Medical information keywords
    if any(keyword in text_lower for keyword in KEYWORDS["medical_info"]):
        indicators.append("medical_info")

    # Financial keywords
    if any(keyword in text_lower for keyword in KEYWORDS["financial_info"]):
        indicators.append("financial_info")

    # Digital credentials
    if any(keyword in text_lower for keyword in KEYWORDS["digital_credential"]):
        indicators.append("digital_credential")

    # Biometric data
    if any(keyword in text_lower for keyword in KEYWORDS["biometric"]):
        indicators.append("biometric")

    # Financial transactions
    if any(keyword in text_lower for keyword in KEYWORDS["financial_transaction"]):
        indicators.append("financial_transaction")

    # Financial documents
    if any(keyword in text_lower for keyword in KEYWORDS["financial_document"]):
        indicators.append("financial_document")

    # Medical history
    if any(keyword in text_lower for keyword in KEYWORDS["medical_history"]):
        indicators.append("medical_history")

    # Prescriptions and medications
    if any(keyword in text_lower for keyword in KEYWORDS["prescription"]):
        indicators.append("prescription")

    # Encryption keys
    if any(keyword in text_lower for keyword in KEYWORDS["encryption_key"]):
        indicators.append("encryption_key")

    # Precise location
    if any(keyword in text_lower for keyword in KEYWORDS["precise_location"]):
        indicators.append("precise_location")

    # Location history
    if any(keyword in text_lower for keyword in KEYWORDS["location_history"]):
        indicators.append("location_history")

    # Email content
    if any(keyword in text_lower for keyword in KEYWORDS["email_content"]):
        indicators.append("email_content")

    # Messaging content
    if any(keyword in text_lower for keyword in KEYWORDS["messaging"]):
        indicators.append("messaging")

    # Call logs
    if any(keyword in text_lower for keyword in KEYWORDS["call_log"]):
        indicators.append("call_log")

    # Social security
    if any(keyword in text_lower for keyword in KEYWORDS["social_security"]):
        indicators.append("social_security")

    # Relationship data
    if any(keyword in text_lower for keyword in KEYWORDS["relationship"]):
        indicators.append("relationship")

    # Behavioral data
    if any(keyword in text_lower for keyword in KEYWORDS["behavioral"]):
        indicators.append("behavioral")

    # Remove duplicates and map to PrivacyType values
    unique_indicators = list(set(indicators))
    privacy_types = [map_legacy_indicator_to_privacy_type(indicator) for indicator in unique_indicators]
    return list(set(privacy_types))


@dataclass
class PrivacyTextSegment:
    """A segment of text that may contain privacy information."""
    text: str
    confidence: float  # 0.0 to 1.0
    location: Dict[str, Any]  # page, coordinates, etc.
    source_type: str  # "ocr", "pdf_text", "table_cell", etc.
    privacy_indicators: List[str]  # ["email", "phone", "address", etc.]
    message_index: Optional[str] = None  # Unique message identifier
    region: Optional[str] = None  # Spatial region: "HEADER", "BODY", "FOOTER", "SIDEBAR" (P-Guard Framework)


@dataclass
class ParseResult:
    """Result of parsing a file for privacy detection."""
    segments: List[PrivacyTextSegment]
    metadata: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None
    parse_time: float = 0.0
    parse_mode: ParseMode = ParseMode.HYBRID  # Default to hybrid mode
    message_index: Optional[str] = None  # Unique message identifier


class FileParserError(Exception):
    """Base exception for file parsing errors."""
    pass


class UnsupportedFileTypeError(FileParserError):
    """Raised when file type is not supported."""
    pass


class ParseModeNotAvailableError(FileParserError):
    """Raised when requested parse mode is not available."""
    pass


class FileParser(abc.ABC):
    """
    Abstract base class for file parsers optimized for privacy detection.

    Focuses on extracting text segments that may contain privacy information
    rather than complete document parsing.
    """

    def __init__(self, config: ParserConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    @abc.abstractmethod
    def supported_file_types(self) -> List[FileType]:
        """Return list of file types this parser can handle."""
        pass

    @property
    @abc.abstractmethod
    def supported_modes(self) -> List[ParseMode]:
        """Return list of parsing modes this parser supports."""
        pass

    @abc.abstractmethod
    async def parse_local(self, file_path: Path, file_type: FileType) -> ParseResult:
        """Parse file using local processing."""
        pass

    @abc.abstractmethod
    async def parse_cloud(self, file_path: Path, file_type: FileType) -> ParseResult:
        """Parse file using cloud services."""
        pass

    async def parse(self, file_path: Path, file_type: FileType, mode: ParseMode = ParseMode.LOCAL) -> ParseResult:
        """
        Parse file with specified mode, with fallback logic.

        Args:
            file_path: Path to the file
            file_type: Type of the file
            mode: Preferred parsing mode

        Returns:
            ParseResult with privacy-relevant text segments
        """
        start_time = time.time()

        try:
            # Validate file first
            self.validate_file(file_path)

            # Try requested mode first
            if mode in self.supported_modes:
                if mode == ParseMode.LOCAL:
                    result = await self.parse_local(file_path, file_type)
                elif mode == ParseMode.CLOUD:
                    result = await self.parse_cloud(file_path, file_type)
                else:  # HYBRID
                    result = await self._parse_hybrid(file_path, file_type)
            else:
                raise ParseModeNotAvailableError(f"Mode {mode.value} not supported by {self.__class__.__name__}")

            result.parse_time = time.time() - start_time
            result.parse_mode = mode
            return result

        except Exception as e:
            self.logger.error(f"Failed to parse {file_path} with mode {mode.value}: {e}")
            return ParseResult(
                segments=[],
                metadata=self.get_file_info(file_path),
                success=False,
                error_message=str(e),
                parse_time=time.time() - start_time,
                parse_mode=mode
            )

    async def _parse_hybrid(self, file_path: Path, file_type: FileType) -> ParseResult:
        """Hybrid parsing: try cloud first, fallback to local."""
        try:
            # Try cloud first
            result = await self.parse_cloud(file_path, file_type)
            if result.success:
                return result
        except Exception as e:
            self.logger.warning(f"Cloud parsing failed, falling back to local: {e}")

        # Fallback to local
        return await self.parse_local(file_path, file_type)

    def validate_file(self, file_path: Path) -> None:
        """Validate that the file can be processed."""
        if not file_path.exists():
            raise FileParserError(f"File does not exist: {file_path}")

        file_size = file_path.stat().st_size
        if file_size > self.config.max_file_size:
            raise FileParserError(f"File too large: {file_size} bytes (max: {self.config.max_file_size})")

        if file_size == 0:
            raise FileParserError("File is empty")

        if self.config.supported_extensions:
            if file_path.suffix.lower().lstrip('.') not in self.config.supported_extensions:
                raise UnsupportedFileTypeError(f"File extension not supported: {file_path.suffix}")

    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get basic file information."""
        stat = file_path.stat()
        return {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_size": stat.st_size,
            "file_extension": file_path.suffix.lower(),
            "modified_time": stat.st_mtime,
        }

    def filter_privacy_segments(self, all_segments: List[PrivacyTextSegment]) -> List[PrivacyTextSegment]:
        """
        Filter segments that are likely to contain privacy information.

        This is a basic keyword-based filter. More sophisticated filtering
        can be implemented in subclasses.
        """
        filtered = []

        for segment in all_segments:
            # Check if segment contains privacy keywords
            text_lower = segment.text.lower()
            has_privacy_keywords = any(keyword.lower() in text_lower for keyword in self.config.privacy_keywords)

            # Check for patterns (emails, phones, etc.)
            has_patterns = self._contains_privacy_patterns(segment.text)

            if has_privacy_keywords or has_patterns:
                segment.privacy_indicators = identify_privacy_indicators(segment.text)
                filtered.append(segment)

        return filtered

    def _contains_privacy_patterns(self, text: str) -> bool:
        """
        Check if text contains common privacy-related patterns.

        Enhanced to detect various formats of sensitive information.
        """
        patterns = [
            # Email addresses
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',

            # Phone numbers (various formats)
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',      # US phone: 123-456-7890
            r'\b\d{3}[-]?\d{4}[-]?\d{4}\b',        # Chinese mobile: 138-0013-8000
            r'\+\d{1,3}[-]?\d{3}[-]?\d{3}[-]?\d{4}\b',  # International: +1-123-456-7890

            # Social Security Numbers
            r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',        # SSN: 123-45-6789

            # Credit/Debit card numbers
            r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',  # 1234-5678-9012-3456

            # Chinese ID cards
            r'\b\d{17}[\dXx]\b',                    # 18-digit Chinese ID

            # Passport numbers (various formats)
            r'\b[A-Z]{1,2}\d{6,9}\b',              # Common passport format
            r'\b\d{9}\b',                          # US passport

            # Driver's license (simplified)
            r'\b[A-Z]{1,2}\d{6,8}\b',              # State DL format

            # Bank account/IBAN
            r'\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b',  # IBAN format

            # IP addresses
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',

            # URLs
            r'\bhttps?://[^\s]+\b',

            # Dates (potential DOB)
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',    # YYYY/MM/DD

            # Tax IDs / EIN
            r'\b\d{2}[-]?\d{7}\b',                  # EIN format
        ]

        return any(re.search(pattern, text) for pattern in patterns)

    async def parse_local(self, file_path: Path, file_type: FileType) -> ParseResult:
        """Parse file using local processing."""
        pass

    async def parse_cloud(self, file_path: Path, file_type: FileType) -> ParseResult:
        """Parse file using cloud services."""
        pass

    async def _parse_hybrid(self, file_path: Path, file_type: FileType) -> ParseResult:
        """Hybrid parsing: try cloud first, fallback to local."""
        try:
            # Try cloud first
            result = await self.parse_cloud(file_path, file_type)
            if result.success:
                return result
        except Exception as e:
            self.logger.warning(f"Cloud parsing failed, falling back to local: {e}")

        # Fallback to local
        return await self.parse_local(file_path, file_type)


class SmartModeSelector:
    """
    Intelligent parsing mode selector based on file type and user context.

    Selection strategy:
    - TEXT_BASED files (Word, Excel, CSV, TXT): LOCAL priority (fast and sufficient)
    - MIXED_CONTENT files (PDF): HYBRID mode (cloud primary, local fallback)
    - IMAGE_ONLY files: HYBRID mode (better OCR with cloud services)
    - User context analysis for additional optimization
    """

    @staticmethod
    def get_file_category(file_type: FileType) -> FileContentCategory:
        """Determine file content category based on file type."""
        text_based_types = [
            FileType.WORD_DOCX, FileType.WORD_DOC,
            FileType.EXCEL_XLSX, FileType.EXCEL_XLS,
            FileType.TEXT_TXT, FileType.TEXT_CSV
        ]

        mixed_content_types = [FileType.PDF]

        image_only_types = [
            FileType.IMAGE_JPG, FileType.IMAGE_JPEG,
            FileType.IMAGE_PNG, FileType.IMAGE_BMP, FileType.IMAGE_TIFF
        ]

        if file_type in text_based_types:
            return FileContentCategory.TEXT_BASED
        elif file_type in mixed_content_types:
            return FileContentCategory.MIXED_CONTENT
        elif file_type in image_only_types:
            return FileContentCategory.IMAGE_ONLY
        else:
            return FileContentCategory.UNKNOWN

    @staticmethod
    def analyze_user_context(user_message: str) -> Dict[str, Any]:
        """
        Analyze user message to understand context and adjust parsing strategy.

        Args:
            user_message: User's message text

        Returns:
            Dict with context analysis results
        """
        context_info = {
            "urgency": "normal",  # low, normal, high
            "complexity": "simple",  # simple, complex
            "privacy_sensitivity": "normal",  # low, normal, high
            "content_type_hints": [],  # hints about content type
            "quality_requirements": "standard"  # basic, standard, high
        }

        if not user_message:
            return context_info

        message_lower = user_message.lower()

        # Analyze urgency
        urgent_keywords = ["紧急", "urgent", "快", "quick", "立即", "immediate", "asap"]
        if any(keyword in message_lower for keyword in urgent_keywords):
            context_info["urgency"] = "high"

        # Analyze complexity
        complex_keywords = ["复杂", "complex", "详细", "detailed", "分析", "analysis",
                          "表格", "table", "图表", "chart", "公式", "formula"]
        if any(keyword in message_lower for keyword in complex_keywords):
            context_info["complexity"] = "complex"

        # Analyze privacy sensitivity
        privacy_keywords = ["隐私", "privacy", "敏感", "sensitive", "机密", "confidential",
                          "个人", "personal", "身份证", "id", "护照", "passport"]
        if any(keyword in message_lower for keyword in privacy_keywords):
            context_info["privacy_sensitivity"] = "high"

        # Content type hints
        if "图片" in message_lower or "image" in message_lower:
            context_info["content_type_hints"].append("image")
        if "文档" in message_lower or "document" in message_lower:
            context_info["content_type_hints"].append("document")
        if "表格" in message_lower or "excel" in message_lower or "spreadsheet" in message_lower:
            context_info["content_type_hints"].append("spreadsheet")

        # Quality requirements
        quality_keywords = ["高质量", "high quality", "精确", "accurate", "清晰", "clear"]
        if any(keyword in message_lower for keyword in quality_keywords):
            context_info["quality_requirements"] = "high"

        return context_info

    @staticmethod
    def select_optimal_mode(
        file_type: FileType,
        user_message: Optional[str] = None,
        force_mode: Optional[ParseMode] = None
    ) -> Tuple[ParseMode, Dict[str, Any]]:
        """
        Select the optimal parsing mode based on file type and user context.

        Args:
            file_type: Type of the file
            user_message: User's message context (optional)
            force_mode: Force specific mode if provided

        Returns:
            Tuple of (selected_mode, selection_reason_dict)
        """
        # If mode is forced, use it
        if force_mode:
            return force_mode, {"reason": "forced_mode", "forced_mode": force_mode.value}

        # Get file category
        category = SmartModeSelector.get_file_category(file_type)

        # Analyze user context if provided
        context_info = {}
        if user_message:
            context_info = SmartModeSelector.analyze_user_context(user_message)

        # Default selections based on category
        mode_selections = {
            FileContentCategory.TEXT_BASED: ParseMode.LOCAL,
            FileContentCategory.MIXED_CONTENT: ParseMode.HYBRID,
            FileContentCategory.IMAGE_ONLY: ParseMode.HYBRID,
            FileContentCategory.UNKNOWN: ParseMode.HYBRID
        }

        selected_mode = mode_selections.get(category, ParseMode.HYBRID)

        # Apply context-based adjustments
        if context_info:
            # High urgency: prefer faster modes
            if context_info["urgency"] == "high" and category == FileContentCategory.TEXT_BASED:
                selected_mode = ParseMode.LOCAL  # Ensure local for speed

            # High quality requirements: prefer hybrid for better accuracy
            if context_info["quality_requirements"] == "high":
                if category in [FileContentCategory.MIXED_CONTENT, FileContentCategory.IMAGE_ONLY]:
                    selected_mode = ParseMode.HYBRID

            # High privacy sensitivity: prefer local mode
            if context_info["privacy_sensitivity"] == "high":
                if category == FileContentCategory.TEXT_BASED:
                    selected_mode = ParseMode.LOCAL

        # Prepare reasoning information
        reasoning = {
            "file_category": category.value,
            "base_selection": mode_selections.get(category, ParseMode.HYBRID).value,
            "context_analysis": context_info,
            "final_mode": selected_mode.value,
            "selection_strategy": "intelligent_adaptive"
        }

        return selected_mode, reasoning


class ParserRegistry:
    """
    Registry for managing file parsers with mode support.

    Maintains a registry of parser classes and provides factory methods
    for creating appropriate parsers based on file type and parsing mode.
    """

    _parsers: Dict[FileType, Dict[ParseMode, type]] = {}

    @classmethod
    def register_parser(cls, file_type: FileType, parse_mode: ParseMode, parser_class: type):
        """Register a parser class for a specific file type and parse mode."""
        if file_type not in cls._parsers:
            cls._parsers[file_type] = {}

        cls._parsers[file_type][parse_mode] = parser_class

        # Also register for HYBRID mode if both LOCAL and CLOUD are available
        if parse_mode in [ParseMode.LOCAL, ParseMode.CLOUD]:
            other_mode = ParseMode.CLOUD if parse_mode == ParseMode.LOCAL else ParseMode.LOCAL
            if other_mode in cls._parsers[file_type] and ParseMode.HYBRID not in cls._parsers[file_type]:
                # Register the same class for HYBRID mode
                cls._parsers[file_type][ParseMode.HYBRID] = parser_class

        logger.info(f"Registered parser {parser_class.__name__} for {file_type.value} in {parse_mode.value} mode")

    @classmethod
    def get_parser(cls, file_type: FileType, parse_mode: ParseMode, config: ParserConfig) -> FileParser:
        """Get parser instance for the given file type and parse mode."""
        if file_type not in cls._parsers or parse_mode not in cls._parsers[file_type]:
            raise UnsupportedFileTypeError(f"No parser registered for {file_type.value} in {parse_mode.value} mode")

        parser_class = cls._parsers[file_type][parse_mode]
        return parser_class(config)

    @classmethod
    def get_supported_file_types(cls) -> List[FileType]:
        """Get list of all supported file types."""
        return list(cls._parsers.keys())

    @classmethod
    def get_supported_modes(cls, file_type: FileType) -> List[ParseMode]:
        """Get supported parse modes for a file type."""
        return list(cls._parsers.get(file_type, {}).keys())

    @classmethod
    def detect_file_type(cls, filename: str, content_type: Optional[str] = None) -> FileType:
        """
        Detect file type from filename and/or content type.

        Args:
            filename: Name of the file
            content_type: MIME content type (optional)

        Returns:
            Detected FileType
        """
        # Extract extension
        file_path = Path(filename)
        extension = file_path.suffix.lower().lstrip('.')

        # Map extensions to file types
        extension_map = {
            'pdf': FileType.PDF,
            'docx': FileType.WORD_DOCX,
            'doc': FileType.WORD_DOC,
            'jpg': FileType.IMAGE_JPG,
            'jpeg': FileType.IMAGE_JPEG,
            'png': FileType.IMAGE_PNG,
            'bmp': FileType.IMAGE_BMP,
            'tiff': FileType.IMAGE_TIFF,
            'xlsx': FileType.EXCEL_XLSX,
            'xls': FileType.EXCEL_XLS,
            'txt': FileType.TEXT_TXT,
            'csv': FileType.TEXT_CSV,
        }

        file_type = extension_map.get(extension, FileType.UNKNOWN)

        # Cross-validate with content type if provided
        if content_type and file_type != FileType.UNKNOWN:
            content_type_map = {
                'application/pdf': FileType.PDF,
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document': FileType.WORD_DOCX,
                'application/msword': FileType.WORD_DOC,
                'image/jpeg': FileType.IMAGE_JPEG,
                'image/png': FileType.IMAGE_PNG,
                'image/bmp': FileType.IMAGE_BMP,
                'image/tiff': FileType.IMAGE_TIFF,
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': FileType.EXCEL_XLSX,
                'application/vnd.ms-excel': FileType.EXCEL_XLS,
                'text/plain': FileType.TEXT_TXT,
                'text/csv': FileType.TEXT_CSV,
            }

            expected_type = content_type_map.get(content_type)
            if expected_type and expected_type != file_type:
                logger.warning(f"Content type {content_type} doesn't match extension {extension}. Using extension-based type.")

        return file_type


class PrivacyDetectionExtractor:
    """
    Main class for extracting privacy-relevant content from uploaded files.

    Optimized for privacy detection rather than general content extraction.
    Supports local/cloud parsing modes with intelligent fallback.
    """

    def __init__(self, config: Optional[ParserConfig] = None):
        self.config = config or ParserConfig()
        self.registry = ParserRegistry()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Performance tracking
        self.stats = {
            "total_files_processed": 0,
            "successful_parses": 0,
            "failed_parses": 0,
            "parse_times": [],
            "mode_usage": {mode.value: 0 for mode in ParseMode}
        }

    async def extract_privacy_segments(self,
                                     upload_file: Any,
                                     preferred_mode: ParseMode = ParseMode.HYBRID,
                                     message_index: Optional[str] = None,
                                     user_message_context: Optional[str] = None,
                                     enable_smart_selection: bool = True) -> ParseResult:
        """
        Extract privacy-relevant text segments from an uploaded file.

        Args:
            upload_file: FastAPI UploadFile object or similar
            preferred_mode: Preferred parsing mode (LOCAL, CLOUD, or HYBRID)
            message_index: Unique identifier for the message containing this file
            user_message_context: User's message context for intelligent mode selection
            enable_smart_selection: Whether to enable intelligent mode selection

        Returns:
            ParseResult with privacy-relevant text segments
        """
        temp_file_path = None

        try:
            # Detect file type
            file_type = ParserRegistry.detect_file_type(
                upload_file.filename,
                getattr(upload_file, 'content_type', None)
            )

            if file_type == FileType.UNKNOWN:
                raise UnsupportedFileTypeError(f"Unsupported file type: {upload_file.filename}")

            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{upload_file.filename}") as temp_file:
                temp_file_path = Path(temp_file.name)
                content = await upload_file.read()
                temp_file.write(content)

            # Intelligent mode selection
            actual_mode = preferred_mode
            mode_reasoning = {"reason": "user_specified", "specified_mode": preferred_mode.value}

            if enable_smart_selection:
                actual_mode, mode_reasoning = SmartModeSelector.select_optimal_mode(
                    file_type=file_type,
                    user_message=user_message_context,
                    force_mode=preferred_mode if preferred_mode != ParseMode.HYBRID else None
                )
                self.logger.info(f"Smart mode selection for {file_type.value}: {actual_mode.value} ({mode_reasoning})")

            # Get appropriate parser for the selected mode
            try:
                parser = self.registry.get_parser(file_type, actual_mode, self.config)
            except UnsupportedFileTypeError:
                # Try fallback modes
                available_modes = self.registry.get_supported_modes(file_type)
                if available_modes:
                    fallback_mode = available_modes[0]  # Use first available mode
                    self.logger.warning(f"Selected mode {actual_mode.value} not available for {file_type.value}, using {fallback_mode.value}")
                    parser = self.registry.get_parser(file_type, fallback_mode, self.config)
                    actual_mode = fallback_mode
                    mode_reasoning["fallback_used"] = True
                    mode_reasoning["fallback_mode"] = fallback_mode.value
                else:
                    raise UnsupportedFileTypeError(f"No parsers available for file type: {file_type.value}")

            # Parse the file
            result = await parser.parse(temp_file_path, file_type, actual_mode)

            # Add file metadata
            result.metadata.update({
                "original_filename": upload_file.filename,
                "detected_file_type": file_type.value,
                "content_type": getattr(upload_file, 'content_type', 'unknown'),
                "preferred_mode": preferred_mode.value,
                "actual_mode": result.parse_mode.value,
                "smart_selection_enabled": enable_smart_selection,
                "mode_selection_reasoning": mode_reasoning,
            })

            # Set message index for tracking
            if message_index:
                result.message_index = message_index
                # Also set message_index on all segments
                for segment in result.segments:
                    segment.message_index = message_index

            # Update statistics
            self._update_stats(result.success, result.parse_time, result.parse_mode)

            return result

        except Exception as e:
            self.logger.error(f"Failed to extract privacy segments from {upload_file.filename}: {e}")

            # Update failure stats
            self._update_stats(False, 0.0, preferred_mode)

            return ParseResult(
                segments=[],
                metadata={
                    "original_filename": upload_file.filename,
                    "error": str(e),
                    "preferred_mode": preferred_mode.value,
                },
                success=False,
                error_message=str(e),
                parse_mode=preferred_mode
            )

        finally:
            # Clean up temporary file
            if temp_file_path and temp_file_path.exists():
                try:
                    temp_file_path.unlink()
                except Exception as e:
                    self.logger.warning(f"Failed to clean up temporary file {temp_file_path}: {e}")

    async def extract_multiple_files(self,
                                   upload_files: List[Any],
                                   preferred_mode: ParseMode = ParseMode.HYBRID,
                                   user_message_context: Optional[str] = None,
                                   enable_smart_selection: bool = True) -> List[ParseResult]:
        """
        Extract privacy segments from multiple uploaded files.

        Args:
            upload_files: List of FastAPI UploadFile objects
            preferred_mode: Preferred parsing mode
            user_message_context: User's message context for intelligent mode selection
            enable_smart_selection: Whether to enable intelligent mode selection

        Returns:
            List of ParseResult objects
        """
        results = []
        for upload_file in upload_files:
            result = await self.extract_privacy_segments(
                upload_file=upload_file,
                preferred_mode=preferred_mode,
                user_message_context=user_message_context,
                enable_smart_selection=enable_smart_selection
            )
            results.append(result)

        return results

    async def extract_privacy_segments_with_ai_enhancement(
        self,
        upload_file: Any,
        user_message_context: str = "",
        preferred_mode: ParseMode = ParseMode.HYBRID,
        message_index: Optional[str] = None,
        enable_smart_selection: bool = True
    ) -> ParseResult:
        """
        Extract privacy segments with AI-enhanced analysis for token optimization.

        For text-based files (PDF, Word, Excel), this method:
        1. Performs local parsing and basic privacy filtering
        2. Uses AI to enhance privacy detection with user context
        3. Reduces token usage by only sending relevant segments to AI

        For images, this method performs multimodal analysis combining OCR with user context.

        Args:
            upload_file: FastAPI UploadFile object or similar
            user_message_context: User's message context for better privacy detection
            preferred_mode: Preferred parsing mode
            message_index: Unique message identifier
            enable_smart_selection: Whether to enable intelligent mode selection

        Returns:
            ParseResult with AI-enhanced privacy segments
        """
        # Detect file type first
        file_type = ParserRegistry.detect_file_type(
            upload_file.filename,
            getattr(upload_file, 'content_type', None)
        )

        if file_type == FileType.UNKNOWN:
            raise UnsupportedFileTypeError(f"Unsupported file type: {upload_file.filename}")

        # Save to temporary file
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{upload_file.filename}") as temp_file:
                temp_file_path = Path(temp_file.name)
                content = await upload_file.read()
                temp_file.write(content)

            # Route to appropriate processing method based on file type
            if file_type in [FileType.PDF, FileType.WORD_DOCX, FileType.WORD_DOC, FileType.EXCEL_XLSX, FileType.EXCEL_XLS]:
                result = await self._process_text_file_with_ai(
                    temp_file_path, file_type, upload_file.filename,
                    user_message_context, preferred_mode, message_index, enable_smart_selection
                )
            elif file_type in [FileType.IMAGE_JPG, FileType.IMAGE_JPEG, FileType.IMAGE_PNG, FileType.IMAGE_BMP, FileType.IMAGE_TIFF]:
                result = await self._process_image_file_with_multimodal(
                    temp_file_path, file_type, upload_file.filename,
                    user_message_context, preferred_mode, message_index, enable_smart_selection
                )
            else:
                # Fallback to regular processing for other file types
                result = await self.extract_privacy_segments(
                    upload_file, preferred_mode, message_index,
                    user_message_context, enable_smart_selection
                )

            # Add additional metadata
            result.metadata.update({
                "ai_enhanced": True,
                "user_context_provided": bool(user_message_context),
                "processing_method": "token_optimized"
            })

            return result

        finally:
            # Clean up temporary file
            if temp_file_path and temp_file_path.exists():
                try:
                    temp_file_path.unlink()
                except Exception as e:
                    self.logger.warning(f"Failed to clean up temporary file {temp_file_path}: {e}")

    async def _process_text_file_with_ai(
        self,
        file_path: Path,
        file_type: FileType,
        original_filename: str,
        user_message_context: str,
        preferred_mode: ParseMode,
        message_index: Optional[str],
        enable_smart_selection: bool = True
    ) -> ParseResult:
        """
        Process text-based files with local parsing + AI enhancement for token optimization.
        """
        try:
            # Step 1: Intelligent mode selection for initial parsing
            initial_mode, _ = SmartModeSelector.select_optimal_mode(
                file_type=file_type,
                user_message=user_message_context,
                force_mode=ParseMode.LOCAL if not enable_smart_selection else None
            )

            # Use LOCAL mode for initial parsing to ensure basic filtering works
            initial_mode = ParseMode.LOCAL

            parser = self.registry.get_parser(file_type, initial_mode, self.config)
            local_result = await parser.parse(file_path, file_type, initial_mode)

            if not local_result.success or not local_result.segments:
                return local_result

            # Step 2: Filter segments that are most likely to contain privacy info
            # This reduces the amount of text sent to AI
            high_confidence_segments = [
                seg for seg in local_result.segments
                if seg.confidence > 0.8 and (
                    seg.privacy_indicators or
                    self._contains_privacy_keywords(seg.text)
                )
            ]

            # If no high-confidence segments, return original result
            if not high_confidence_segments:
                return local_result

            # Step 3: Use AI to enhance privacy detection with context
            enhanced_segments = await self._enhance_privacy_detection_with_ai(
                high_confidence_segments,
                user_message_context,
                file_type.value
            )

            # Step 4: Create enhanced result
            enhanced_result = ParseResult(
                segments=enhanced_segments,
                metadata={
                    **local_result.metadata,
                    "original_segments_count": len(local_result.segments),
                    "high_confidence_segments": len(high_confidence_segments),
                    "ai_enhanced_segments": len(enhanced_segments),
                    "token_optimization": "text_local_ai_enhance"
                },
                success=True,
                parse_mode=ParseMode.HYBRID,  # Indicate hybrid processing
                message_index=message_index
            )

            return enhanced_result

        except Exception as e:
            self.logger.error(f"AI-enhanced text processing failed for {original_filename}: {e}")
            # Fallback to regular processing
            return await self.extract_privacy_segments(
                type('MockFile', (), {'filename': original_filename, 'read': lambda: open(file_path, 'rb').read()})(),
                preferred_mode,
                message_index
            )

    async def _process_image_file_with_multimodal(
        self,
        file_path: Path,
        file_type: FileType,
        original_filename: str,
        user_message_context: str,
        preferred_mode: ParseMode,
        message_index: Optional[str],
        enable_smart_selection: bool = True
    ) -> ParseResult:
        """
        Process image files with multimodal analysis combining OCR and user context.
        """
        try:
            # Step 1: Perform OCR
            parser = self.registry.get_parser(file_type, ParseMode.LOCAL, self.config)
            ocr_result = await parser.parse(file_path, file_type, ParseMode.LOCAL)

            if not ocr_result.success:
                return ocr_result

            # Step 2: Use multimodal AI analysis
            enhanced_segments = await self._perform_multimodal_privacy_analysis(
                ocr_result.segments,
                file_path,
                user_message_context
            )

            # Step 3: Create enhanced result
            enhanced_result = ParseResult(
                segments=enhanced_segments,
                metadata={
                    **ocr_result.metadata,
                    "multimodal_analysis": True,
                    "user_context_integrated": bool(user_message_context),
                    "token_optimization": "image_multimodal"
                },
                success=True,
                parse_mode=ParseMode.HYBRID,
                message_index=message_index
            )

            return enhanced_result

        except Exception as e:
            self.logger.error(f"Multimodal image processing failed for {original_filename}: {e}")
            # Fallback to regular processing
            return await self.extract_privacy_segments(
                type('MockFile', (), {'filename': original_filename, 'read': lambda: open(file_path, 'rb').read()})(),
                preferred_mode,
                message_index
            )

    def _contains_privacy_keywords(self, text: str) -> bool:
        """Check if text contains basic privacy keywords."""
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in self.config.privacy_keywords)

    async def _enhance_privacy_detection_with_ai(
        self,
        segments: List[PrivacyTextSegment],
        user_context: str,
        file_type: str
    ) -> List[PrivacyTextSegment]:
        """
        Use AI to enhance privacy detection with user context.
        This is a placeholder for actual AI integration.
        """
        # TODO: Integrate with actual LLM for privacy detection enhancement
        # For now, return segments with enhanced confidence and indicators

        enhanced_segments = []
        for segment in segments:
            # Simulate AI enhancement
            enhanced_segment = PrivacyTextSegment(
                text=segment.text,
                confidence=min(segment.confidence + 0.1, 1.0),  # Boost confidence
                location=segment.location,
                source_type=segment.source_type,
                privacy_indicators=identify_privacy_indicators(segment.text),
                message_index=segment.message_index
            )
            enhanced_segments.append(enhanced_segment)

        return enhanced_segments

    async def _perform_multimodal_privacy_analysis(
        self,
        ocr_segments: List[PrivacyTextSegment],
        image_path: Path,
        user_context: str
    ) -> List[PrivacyTextSegment]:
        """
        Perform multimodal analysis combining OCR text with user context using LLM.
        
        This method uses LLM to enhance privacy detection by:
        1. Analyzing OCR text segments in context of user message
        2. Identifying privacy-sensitive information that OCR might have missed
        3. Improving confidence scores based on contextual understanding
        
        Args:
            ocr_segments: List of text segments extracted via OCR
            image_path: Path to the image file
            user_context: User's message context for better understanding
            
        Returns:
            Enhanced privacy text segments with improved detection
        """
        if not ocr_segments:
            return []
        
        try:
            # Combine OCR text for LLM analysis
            ocr_text = "\n".join([seg.text for seg in ocr_segments if seg.text.strip()])
            
            if not ocr_text.strip():
                # No OCR text, return original segments
                return ocr_segments
            
            system_prompt = """You are a privacy detection expert analyzing OCR text extracted from images.

Your task is to:
1. Identify privacy-sensitive information in the OCR text
2. Consider the user's message context to understand what they're sharing
3. Extract specific privacy entities (names, IDs, addresses, etc.)
4. Assess confidence levels for each detection

Return a JSON array of privacy-sensitive text segments with their privacy indicators."""
            
            user_prompt = f"""User's message context: {user_context if user_context else 'No specific context provided'}

OCR text extracted from image:
{ocr_text}

Analyze this OCR text and identify all privacy-sensitive segments. Return a JSON array where each item contains:
- "text": The privacy-sensitive text segment
- "privacy_indicators": List of privacy types (e.g., ["email", "phone", "id_card"])
- "confidence": Confidence score (0.0-1.0)

Return only the JSON array, no additional text."""
            
            llm = llm_manager.get_llm_client()
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await llm.ainvoke(messages)
            response_text = response.content.strip()
            
            # Parse LLM response
            import json
            import re
            
            # Extract JSON array from response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                llm_analysis = json.loads(json_match.group(0))
            else:
                # Fallback: try parsing entire response
                llm_analysis = json.loads(response_text)
            
            # Map LLM analysis back to segments
            enhanced_segments = []
            llm_segments_dict = {}
            
            # Create mapping from text to LLM analysis
            for item in llm_analysis:
                text_key = item.get("text", "").strip().lower()
                if text_key:
                    llm_segments_dict[text_key] = item
            
            # Enhance OCR segments with LLM insights
            for ocr_seg in ocr_segments:
                ocr_text_lower = ocr_seg.text.strip().lower()
                
                # Try to find matching LLM analysis
                llm_match = None
                for key, analysis in llm_segments_dict.items():
                    if key in ocr_text_lower or ocr_text_lower in key:
                        llm_match = analysis
                        break
                
                if llm_match:
                    # Use LLM-enhanced indicators and confidence
                    privacy_indicators = llm_match.get("privacy_indicators", [])
                    llm_confidence = float(llm_match.get("confidence", 0.5))
                    
                    # Combine OCR confidence with LLM confidence
                    combined_confidence = min(0.95, (ocr_seg.confidence * 0.4) + (llm_confidence * 0.6))
                    
                    enhanced_segment = PrivacyTextSegment(
                        text=ocr_seg.text,
                        confidence=combined_confidence,
                        location=ocr_seg.location,
                        source_type="multimodal_ocr_llm",
                        privacy_indicators=privacy_indicators or identify_privacy_indicators(ocr_seg.text),
                        message_index=ocr_seg.message_index
                    )
                else:
                    # No LLM match, use original segment with context boost
                    context_boost = 0.05 if user_context else 0.0
                    enhanced_segment = PrivacyTextSegment(
                        text=ocr_seg.text,
                        confidence=min(ocr_seg.confidence + context_boost, 1.0),
                        location=ocr_seg.location,
                        source_type="multimodal_ocr",
                        privacy_indicators=identify_privacy_indicators(ocr_seg.text),
                        message_index=ocr_seg.message_index
                    )
                
                enhanced_segments.append(enhanced_segment)
            
            return enhanced_segments
            
        except Exception as e:
            self.logger.warning(f"Multimodal LLM analysis failed, using OCR-only results: {e}")
            # Fallback to OCR-only with context boost
            enhanced_segments = []
            context_boost = 0.05 if user_context else 0.0
            
            for segment in ocr_segments:
                enhanced_segment = PrivacyTextSegment(
                    text=segment.text,
                    confidence=min(segment.confidence + context_boost, 1.0),
                    location=segment.location,
                    source_type="multimodal_ocr",
                    privacy_indicators=identify_privacy_indicators(segment.text),
                    message_index=segment.message_index
                )
                enhanced_segments.append(enhanced_segment)
            
            return enhanced_segments

    async def extract_privacy_segments_with_message_context(
        self,
        upload_file: Any,
        account_id: str,
        detection_id: str,
        message_index: int,
        preferred_mode: ParseMode = ParseMode.HYBRID
    ) -> ParseResult:
        """
        Extract privacy segments from a file with message context tracking.

        This method automatically generates a unique message index for tracking
        file attachments in privacy detection conversations.

        Args:
            upload_file: FastAPI UploadFile object or similar
            account_id: User account identifier
            detection_id: Privacy detection session ID
            message_index: Index of the message in the conversation
            preferred_mode: Preferred parsing mode

        Returns:
            ParseResult with privacy segments and unique message tracking
        """
        # Generate unique message index
        unique_message_index = MessageIndexGenerator.generate_message_index(
            account_id, detection_id, message_index
        )

        # Extract privacy segments with the unique index
        result = await self.extract_privacy_segments(
            upload_file=upload_file,
            preferred_mode=preferred_mode,
            message_index=unique_message_index
        )

        # Add additional context to metadata
        result.metadata.update({
            "account_id": account_id,
            "detection_id": detection_id,
            "original_message_index": message_index,
            "unique_message_index": unique_message_index,
        })

        return result

    async def process_conversation_files(
        self,
        conversation_records: List[Dict[str, Any]],
        account_id: str,
        detection_id: str
    ) -> Dict[str, ParseResult]:
        """
        Process all files attached to messages in a conversation.

        This method scans conversation records for file attachments and
        processes them with unique message indices.

        Args:
            conversation_records: List of conversation records (like privacy detection input)
            account_id: User account identifier
            detection_id: Privacy detection session ID

        Returns:
            Dict mapping message indices to ParseResult objects
        """
        file_results = {}

        for message_idx, record in enumerate(conversation_records):
            # Check if this message has file attachments
            # Note: This assumes files are stored in the record or accessible via some mechanism
            # The exact implementation would depend on how files are attached to messages

            # For now, this is a placeholder for future file attachment processing
            # In a real implementation, you'd check for file attachments in the record
            # and process them using extract_privacy_segments_with_message_context

            if "files" in record and record["files"]:
                # Process each file attached to this message
                for file_idx, file_info in enumerate(record["files"]):
                    try:
                        # This would need to be adapted based on actual file storage mechanism
                        # For now, it's a conceptual implementation
                        result = await self.extract_privacy_segments_with_message_context(
                            upload_file=file_info,  # This would be the actual file object
                            account_id=account_id,
                            detection_id=detection_id,
                            message_index=message_idx
                        )
                        file_results[f"{message_idx}_{file_idx}"] = result

                    except Exception as e:
                        self.logger.error(f"Failed to process file {file_idx} in message {message_idx}: {e}")
                        continue

        return file_results

    def _update_stats(self, success: bool, parse_time: float, parse_mode: ParseMode):
        """Update internal statistics."""
        self.stats["total_files_processed"] += 1

        if success:
            self.stats["successful_parses"] += 1
        else:
            self.stats["failed_parses"] += 1

        if parse_time > 0:
            self.stats["parse_times"].append(parse_time)

        self.stats["mode_usage"][parse_mode.value] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.stats.copy()

        # Calculate averages
        if stats["parse_times"]:
            stats["avg_parse_time"] = sum(stats["parse_times"]) / len(stats["parse_times"])
            stats["max_parse_time"] = max(stats["parse_times"])
            stats["min_parse_time"] = min(stats["parse_times"])
        else:
            stats["avg_parse_time"] = 0.0
            stats["max_parse_time"] = 0.0
            stats["min_parse_time"] = 0.0

        # Calculate success rate
        if stats["total_files_processed"] > 0:
            stats["success_rate"] = stats["successful_parses"] / stats["total_files_processed"]
        else:
            stats["success_rate"] = 0.0

        return stats

    def reset_stats(self):
        """Reset performance statistics."""
        self.stats = {
            "total_files_processed": 0,
            "successful_parses": 0,
            "failed_parses": 0,
            "parse_times": [],
            "mode_usage": {mode.value: 0 for mode in ParseMode}
        }


class PrivacyLeakDetector:
    """
    Unified Privacy Leak Detector that coordinates file parsing and conversation analysis.

    This class orchestrates the complete privacy leak detection workflow:
    1. Parse files for privacy-sensitive content
    2. Analyze conversation messages for privacy leaks
    3. Correlate findings across files and messages
    4. Generate comprehensive privacy leak reports

    Single Responsibility: Coordinate privacy detection across multiple sources
    """

    def __init__(self, config: Optional[ParserConfig] = None):
        self.config = config or ParserConfig()
        self.privacy_extractor = PrivacyDetectionExtractor(self.config)
        self.logger = logging.getLogger(self.__class__.__name__)

    async def detect_privacy_leaks_comprehensive(
        self,
        conversation_records: List[Dict[str, Any]],
        account_id: str,
        detection_session_id: str
    ) -> Dict[str, Any]:
        """
        Comprehensive privacy leak detection across files and conversation.

        Args:
            conversation_records: List of conversation records, each containing:
                - 'user': message content
                - 'files': optional list of file objects
                - 'message_index': optional message index
            account_id: User account identifier
            detection_session_id: Unique session identifier

        Returns:
            Comprehensive privacy leak detection report
        """
        start_time = time.time()

        # Step 1: Process all files in conversation with enhanced context
        file_results = await self._process_conversation_files(
            conversation_records, account_id, detection_session_id
        )

        # Step 2: Extract text content from messages (without files for conversation analysis)
        message_only_records = [
            {k: v for k, v in record.items() if k != 'files'}
            for record in conversation_records
        ]

        # Step 3: Analyze conversation messages for privacy leaks
        conversation_analysis = await self._analyze_conversation_messages(
            message_only_records, account_id, detection_session_id
        )

        # Step 4: Correlate findings across files and messages
        correlation_results = self._correlate_privacy_findings(
            file_results, conversation_analysis, conversation_records
        )

        # Step 5: Generate comprehensive report
        report = self._generate_comprehensive_report(
            file_results, conversation_analysis, correlation_results,
            account_id, detection_session_id, time.time() - start_time
        )

        return report

    async def _process_conversation_files(
        self,
        conversation_records: List[Dict[str, Any]],
        account_id: str,
        detection_session_id: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process all files attached to conversation messages with precise message association.

        Returns a dict where keys are file identifiers and values contain both
        the parse result and the associated message context.

        Single Responsibility: Handle file processing with message context
        """
        file_results = {}

        for message_idx, record in enumerate(conversation_records):
            files = record.get('files', [])
            if not files:
                continue

            # Extract message context for better file analysis
            message_content = record.get('user', '')
            message_context = {
                'message_index': message_idx,
                'message_content': message_content,
                'message_timestamp': record.get('timestamp'),
                'conversation_context': self._extract_conversation_context(
                    conversation_records, message_idx
                )
            }

            for file_idx, file_obj in enumerate(files):
                file_identifier = self._generate_file_identifier(file_obj, account_id)

                try:
                    # Check cache first
                    cached_result = await self._check_file_cache(
                        file_identifier, account_id, message_context
                    )

                    if cached_result:
                        result = cached_result['parse_result']
                        self.logger.info(
                            f"Cache hit for file {file_identifier} in message {message_idx}"
                        )
                    else:
                        # Process file with enhanced context
                        result = await self._process_file_with_context(
                            file_obj, account_id, detection_session_id,
                            message_idx, message_context
                        )

                        # Cache the result
                        await self._cache_file_result(
                            file_identifier, result, account_id, message_context
                        )

                    result_key = f"message_{message_idx}_file_{file_idx}"
                    file_results[result_key] = {
                        'parse_result': result,
                        'message_context': message_context,
                        'file_identifier': file_identifier,
                        'file_index': file_idx
                    }

                    self.logger.info(
                        f"Processed file {file_identifier} in message {message_idx}: "
                        f"{result.success}, {len(result.segments)} segments"
                    )

                except Exception as e:
                    self.logger.error(
                        f"Failed to process file {file_identifier} in message {message_idx}: {e}"
                    )
                    # Create error result
                    error_result = ParseResult(
                        segments=[],
                        metadata={
                            "error": str(e),
                            "message_index": message_idx,
                            "file_index": file_idx,
                            "file_identifier": file_identifier,
                            "original_filename": getattr(file_obj, 'filename', 'unknown')
                        },
                        success=False,
                        error_message=str(e)
                    )
                    file_results[f"message_{message_idx}_file_{file_idx}"] = {
                        'parse_result': error_result,
                        'message_context': message_context,
                        'file_identifier': file_identifier,
                        'file_index': file_idx
                    }

        return file_results

    def _generate_file_identifier(self, file_obj: Any, account_id: str) -> str:
        """
        Generate a unique identifier for a file based on its content and metadata.

        This ensures that identical files uploaded by the same user get the same identifier.
        """
        import hashlib

        # Get file metadata
        filename = getattr(file_obj, 'filename', 'unknown')
        content_type = getattr(file_obj, 'content_type', 'unknown')
        file_size = getattr(file_obj, 'size', 0)

        # Create a hash of file metadata for identification
        # In a real implementation, you'd hash the file content as well
        identifier_data = f"{account_id}:{filename}:{content_type}:{file_size}"
        return hashlib.sha256(identifier_data.encode()).hexdigest()[:16]

    def _extract_conversation_context(
        self,
        conversation_records: List[Dict[str, Any]],
        current_message_idx: int
    ) -> Dict[str, Any]:
        """
        Extract conversation context around a message to provide better file analysis context.
        """
        context = {
            'previous_messages': [],
            'subsequent_messages': [],
            'conversation_length': len(conversation_records)
        }

        # Get previous 2 messages
        start_idx = max(0, current_message_idx - 2)
        context['previous_messages'] = [
            {
                'index': i,
                'content': record.get('user', ''),
                'has_files': bool(record.get('files'))
            }
            for i, record in enumerate(conversation_records[start_idx:current_message_idx])
        ]

        # Get next 2 messages
        end_idx = min(len(conversation_records), current_message_idx + 3)
        context['subsequent_messages'] = [
            {
                'index': i,
                'content': record.get('user', ''),
                'has_files': bool(record.get('files'))
            }
            for i, record in enumerate(conversation_records[current_message_idx + 1:end_idx])
        ]

        return context

    async def _check_file_cache(
        self,
        file_identifier: str,
        account_id: str,
        message_context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Check if file has been processed before with similar context.

        Returns cached result if available, None otherwise.
        """
        try:

            database = get_database()

            # Check for cached file analysis
            cached_data = database.load_cached_file_analysis(file_identifier, account_id)

            if cached_data:
                # Validate cache is still relevant (basic check)
                cached_context = cached_data.get('message_context', {})
                current_message = message_context.get('message_content', '')

                # If message content is very similar, use cache
                if self._is_context_similar(cached_context, message_context):
                    return cached_data

            return None

        except Exception as e:
            self.logger.warning(f"Cache check failed for file {file_identifier}: {e}")
            return None

    def _is_context_similar(self, cached_context: Dict, current_context: Dict) -> bool:
        """
        Check if cached context is similar enough to current context to reuse results.
        """
        # Simple similarity check - can be enhanced with more sophisticated logic
        cached_message = cached_context.get('message_content', '')
        current_message = current_context.get('message_content', '')

        # If messages are identical or very similar, use cache
        if cached_message == current_message:
            return True

        # Check if key privacy-related keywords are present in both
        privacy_keywords = ['privacy', 'personal', 'confidential', 'private', 'sensitive']
        cached_has_keywords = any(kw in cached_message.lower() for kw in privacy_keywords)
        current_has_keywords = any(kw in current_message.lower() for kw in privacy_keywords)

        return cached_has_keywords == current_has_keywords

    async def _process_file_with_context(
        self,
        file_obj: Any,
        account_id: str,
        detection_id: str,
        message_index: int,
        message_context: Dict[str, Any]
    ) -> ParseResult:
        """
        Process a file with enhanced message context for better privacy detection.
        
        Args:
            file_obj: File object to process
            account_id: User account identifier
            detection_id: Detection ID (unique identifier for the detection)
            message_index: Index of the message in the conversation
            message_context: Message context dictionary
            
        Returns:
            ParseResult with privacy segments
        """
        # Add detection_id to message_context for tracking
        enhanced_message_context = dict(message_context)
        enhanced_message_context['detection_id'] = detection_id
        
        # Use the existing message context extraction method
        result = await self.privacy_extractor.extract_privacy_segments_with_message_context(
            upload_file=file_obj,
            account_id=account_id,
            detection_id=detection_id,
            message_index=message_index
        )

        # Enhance result with message context
        if result.metadata:
            result.metadata.update({
                'message_context': enhanced_message_context,
                'context_enhanced': True,
                'detection_id': detection_id
            })

        return result

    async def _cache_file_result(
        self,
        file_identifier: str,
        parse_result: ParseResult,
        account_id: str,
        message_context: Dict[str, Any]
    ):
        """
        Cache file analysis result for future reuse.
        """
        try:

            database = get_database()

            cache_data = {
                'file_identifier': file_identifier,
                'account_id': account_id,
                'parse_result': {
                    'segments': [segment.__dict__ for segment in parse_result.segments],
                    'metadata': parse_result.metadata,
                    'success': parse_result.success,
                    'error_message': parse_result.error_message,
                    'parse_time': parse_result.parse_time,
                    'parse_mode': parse_result.parse_mode.value if parse_result.parse_mode else None,
                    'message_index': parse_result.message_index
                },
                'message_context': message_context,
                'cached_at': self._get_timestamp(),
                'cache_version': '1.0'
            }

            database.save_cached_file_analysis(cache_data)

        except Exception as e:
            self.logger.warning(f"Failed to cache file result for {file_identifier}: {e}")
            # Don't raise exception - caching failure shouldn't break the main flow

    def _get_timestamp(self) -> str:
        """Get current timestamp for caching."""
        from datetime import datetime
        return datetime.utcnow().isoformat()

    async def _analyze_conversation_messages(
        self,
        message_records: List[Dict[str, Any]],
        account_id: str,
        detection_session_id: str
    ) -> Dict[str, Any]:
        """
        Analyze conversation messages for privacy leaks using PrivacyDetectorAgent.

        Single Responsibility: Handle conversation message analysis
        """
        try:
            from ...application.agents.privacy_detector.agent import PrivacyDetectorAgent

            # Create privacy detector instance
            privacy_detector = PrivacyDetectorAgent()

            # Add account_id to records for tracking
            enhanced_records = []
            for record in message_records:
                enhanced_record = dict(record)
                enhanced_record['account_id'] = account_id
                enhanced_records.append(enhanced_record)

            # Perform privacy detection
            conversation_analysis = await privacy_detector.detect_privacy_leaks(
                conversation_records=enhanced_records,
                use_cot=True,  # Use chain of thought for better analysis
                use_few_shots=True
            )

            # Add session tracking
            conversation_analysis['detection_session_id'] = detection_session_id
            conversation_analysis['account_id'] = account_id

            return conversation_analysis

        except Exception as e:
            self.logger.error(f"Conversation privacy analysis failed: {e}")
            return {
                "error": str(e),
                "privacy_detected": False,
                "privacy_leaks": [],
                "overall_severity": "none",
                "overall_score": 0.0,
                "detection_session_id": detection_session_id,
                "account_id": account_id
            }

    def _correlate_privacy_findings(
        self,
        file_results: Dict[str, Dict[str, Any]],
        conversation_analysis: Dict[str, Any],
        original_records: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Correlate privacy findings across files and conversation messages with enhanced context.

        Single Responsibility: Find relationships between file and message privacy leaks
        """
        correlation_results = {
            "cross_references": [],
            "severity_escalation": [],
            "data_consistency": [],
            "risk_patterns": [],
            "message_file_associations": []
        }

        # Extract all privacy types found in files with message context
        file_privacy_data = []
        for result_key, file_data in file_results.items():
            result = file_data['parse_result']
            message_context = file_data['message_context']

            if result.success and result.segments:
                file_info = {
                    'privacy_types': set(),
                    'message_index': message_context['message_index'],
                    'message_content': message_context['message_content'],
                    'file_identifier': file_data['file_identifier']
                }

                for segment in result.segments:
                    if segment.privacy_indicators:
                        file_info['privacy_types'].update(segment.privacy_indicators)

                file_privacy_data.append(file_info)

        # Extract all privacy types found in conversation with message indices
        conversation_privacy_data = []
        if conversation_analysis.get("privacy_leaks"):
            for leak in conversation_analysis["privacy_leaks"]:
                leak_locations = leak.get("locations", [])
                if leak_locations:
                    for location in leak_locations:
                        message_idx = location.get("message_index")
                        if message_idx is not None:
                            conversation_privacy_data.append({
                                'privacy_type': leak.get("privacy_type", ""),
                                'message_index': message_idx,
                                'severity': leak.get("severity", "low")
                            })

        # Analyze message-specific correlations
        for file_info in file_privacy_data:
            message_idx = file_info['message_index']
            message_content = file_info['message_content']

            # Find conversation leaks in the same message
            message_conversation_leaks = [
                leak for leak in conversation_privacy_data
                if leak['message_index'] == message_idx
            ]

            # Check for cross-references within the same message
            message_privacy_types = {leak['privacy_type'] for leak in message_conversation_leaks}
            common_types = file_info['privacy_types'].intersection(message_privacy_types)

            if common_types:
                correlation_results["message_file_associations"].append({
                    "message_index": message_idx,
                    "message_content": message_content[:100] + "..." if len(message_content) > 100 else message_content,
                    "common_privacy_types": list(common_types),
                    "file_identifier": file_info['file_identifier'],
                    "risk_level": "critical",  # Same message, same privacy type = very high risk
                    "correlation_type": "same_message_cross_reference"
                })

            # Check for complementary privacy information
            complementary_types = file_info['privacy_types'] - message_privacy_types
            if complementary_types and message_conversation_leaks:
                correlation_results["message_file_associations"].append({
                    "message_index": message_idx,
                    "message_content": message_content[:100] + "..." if len(message_content) > 100 else message_content,
                    "complementary_privacy_types": list(complementary_types),
                    "conversation_privacy_types": list(message_privacy_types),
                    "file_identifier": file_info['file_identifier'],
                    "risk_level": "high",
                    "correlation_type": "complementary_information"
                })

        # Global cross-references (same privacy type across different messages)
        all_file_types = set()
        for file_info in file_privacy_data:
            all_file_types.update(file_info['privacy_types'])

        all_conversation_types = {leak['privacy_type'] for leak in conversation_privacy_data}

        common_global_types = all_file_types.intersection(all_conversation_types)
        if common_global_types:
            correlation_results["cross_references"].extend([
                {
                    "privacy_type": privacy_type,
                    "sources": ["files", "conversation"],
                    "risk_level": "high",
                    "correlation_type": "global_cross_reference"
                }
                for privacy_type in common_global_types
            ])

        # Enhanced severity escalation analysis
        file_severity = self._calculate_file_severity_from_enhanced_results(file_results)
        conversation_severity = conversation_analysis.get("overall_severity", "none")

        severity_levels = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        combined_severity_score = max(
            severity_levels.get(file_severity, 0),
            severity_levels.get(conversation_severity, 0)
        )

        # Check for message-specific severity escalation
        message_specific_escalations = []
        for file_info in file_privacy_data:
            message_idx = file_info['message_index']
            message_conversation_leaks = [
                leak for leak in conversation_privacy_data
                if leak['message_index'] == message_idx
            ]

            if message_conversation_leaks:
                file_severity_score = self._calculate_file_severity_score(file_info['privacy_types'])
                max_conversation_severity = max(
                    severity_levels.get(leak['severity'], 1)
                    for leak in message_conversation_leaks
                )

                if file_severity_score > max_conversation_severity:
                    message_specific_escalations.append({
                        "message_index": message_idx,
                        "file_severity": list(severity_levels.keys())[file_severity_score],
                        "conversation_severity": list(severity_levels.keys())[max_conversation_severity],
                        "escalation_factor": file_severity_score - max_conversation_severity,
                        "file_identifier": file_info['file_identifier']
                    })

        if message_specific_escalations:
            correlation_results["severity_escalation"].extend(message_specific_escalations)

        # Overall severity escalation
        if combined_severity_score > severity_levels.get(conversation_severity, 0):
            correlation_results["severity_escalation"].append({
                "reason": "file_content_increases_overall_severity",
                "original_severity": conversation_severity,
                "escalated_severity": list(severity_levels.keys())[combined_severity_score],
                "trigger": "sensitive_files_attached",
                "correlation_type": "overall_escalation"
            })

        return correlation_results

    def _calculate_file_severity_from_enhanced_results(self, file_results: Dict[str, Dict[str, Any]]) -> str:
        """Calculate overall severity from enhanced file results."""
        if not file_results:
            return "none"

        severity_scores = []
        for file_data in file_results.values():
            result = file_data['parse_result']
            if result.success and result.segments:
                privacy_types = set()
                for segment in result.segments:
                    if segment.privacy_indicators:
                        privacy_types.update(segment.privacy_indicators)

                severity_score = self._calculate_file_severity_score(privacy_types)
                severity_scores.append(severity_score)

        if not severity_scores:
            return "none"

        avg_severity = sum(severity_scores) / len(severity_scores)
        if avg_severity >= 3.5:
            return "critical"
        elif avg_severity >= 2.5:
            return "high"
        elif avg_severity >= 1.5:
            return "medium"
        elif avg_severity >= 0.5:
            return "low"
        else:
            return "none"

    def _calculate_file_severity_score(self, privacy_types: set) -> int:
        """Calculate severity score for a set of privacy types."""
        high_risk_types = {
            'government_ids', 'financial_accounts', 'authentication_credentials',
            'api_keys_tokens', 'system_access', 'encryption_keys', 'biometric_data'
        }
        medium_risk_types = {
            'financial_transactions', 'health_records', 'precise_location',
            'email_content', 'messaging_content', 'call_logs'
        }

        if any(pt in high_risk_types for pt in privacy_types):
            return 3  # high
        elif any(pt in medium_risk_types for pt in privacy_types):
            return 2  # medium
        elif privacy_types:
            return 1  # low
        else:
            return 0  # none

    def _calculate_file_severity(self, file_results: Dict[str, ParseResult]) -> str:
        """Calculate overall severity from file analysis results."""
        if not file_results:
            return "none"

        severity_scores = []
        for result in file_results.values():
            if result.success and result.segments:
                # Simple severity calculation based on number and types of segments
                segment_count = len(result.segments)
                if segment_count > 10:
                    severity_scores.append(3)  # high
                elif segment_count > 5:
                    severity_scores.append(2)  # medium
                elif segment_count > 0:
                    severity_scores.append(1)  # low

        if not severity_scores:
            return "none"

        avg_severity = sum(severity_scores) / len(severity_scores)
        if avg_severity >= 2.5:
            return "high"
        elif avg_severity >= 1.5:
            return "medium"
        elif avg_severity >= 0.5:
            return "low"
        else:
            return "none"

    def _generate_comprehensive_report(
        self,
        file_results: Dict[str, Dict[str, Any]],
        conversation_analysis: Dict[str, Any],
        correlation_results: Dict[str, Any],
        account_id: str,
        detection_session_id: str,
        processing_time: float
    ) -> Dict[str, Any]:
        """
        Generate comprehensive privacy leak detection report with enhanced file-message associations.

        Single Responsibility: Compile final report from all analysis components
        """
        # Calculate overall metrics from enhanced file results
        total_files = len(file_results)
        successful_files = sum(
            1 for file_data in file_results.values()
            if file_data['parse_result'].success
        )
        total_file_segments = sum(
            len(file_data['parse_result'].segments)
            for file_data in file_results.values()
            if file_data['parse_result'].success
        )

        # Track cache hits
        cache_hits = sum(
            1 for file_data in file_results.values()
            if file_data['parse_result'].metadata.get('cached', False)
        )

        # Calculate overall severity
        file_severity = self._calculate_file_severity_from_enhanced_results(file_results)
        conversation_severity = conversation_analysis.get("overall_severity", "none")

        severity_levels = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        overall_severity_score = max(
            severity_levels.get(file_severity, 0),
            severity_levels.get(conversation_severity, 0)
        )
        overall_severity = list(severity_levels.keys())[overall_severity_score]

        # Calculate overall confidence score
        file_confidence = sum(
            file_data['parse_result'].metadata.get("confidence", 0.5)
            for file_data in file_results.values()
        ) / max(len(file_results), 1)
        conversation_confidence = conversation_analysis.get("overall_score", 0.0)

        overall_confidence = (file_confidence + conversation_confidence) / 2 if file_results else conversation_confidence

        # Generate enhanced file results summary
        file_results_summary = {}
        message_file_mappings = {}

        for result_key, file_data in file_results.items():
            result = file_data['parse_result']
            message_context = file_data['message_context']
            file_identifier = file_data['file_identifier']

            file_results_summary[result_key] = {
                "success": result.success,
                "segments_count": len(result.segments),
                "file_identifier": file_identifier,
                "message_index": message_context['message_index'],
                "cached": result.metadata.get('cached', False),
                "metadata": result.metadata,
                "error": result.error_message if not result.success else None
            }

            # Build message-to-files mapping
            msg_idx = message_context['message_index']
            if msg_idx not in message_file_mappings:
                message_file_mappings[msg_idx] = []
            message_file_mappings[msg_idx].append({
                'file_identifier': file_identifier,
                'result_key': result_key,
                'success': result.success,
                'segments_count': len(result.segments)
            })

        return {
            "detection_session_id": detection_session_id,
            "account_id": account_id,
            "timestamp": time.time(),
            "processing_time_seconds": round(processing_time, 2),

            # Enhanced file analysis results
            "file_analysis": {
                "total_files_processed": total_files,
                "successful_files": successful_files,
                "failed_files": total_files - successful_files,
                "cached_files": cache_hits,
                "cache_hit_rate": round(cache_hits / max(total_files, 1), 2),
                "total_privacy_segments": total_file_segments,
                "file_severity": file_severity,
                "file_results": file_results_summary,
                "message_file_mappings": message_file_mappings
            },

            # Conversation analysis results
            "conversation_analysis": conversation_analysis,

            # Enhanced correlation results
            "correlation_analysis": correlation_results,

            # Overall assessment
            "overall_assessment": {
                "privacy_leaks_detected": (
                    bool(file_results) or conversation_analysis.get("privacy_detected", False)
                ),
                "overall_severity": overall_severity,
                "overall_confidence": round(overall_confidence, 3),
                "risk_level": self._calculate_risk_level(overall_severity_score, correlation_results),
                "recommendations": self._generate_recommendations(
                    overall_severity, correlation_results
                )
            },

            "data_sources": {
                "conversation_messages": len(conversation_analysis.get("conversation_records", [])),
                "attached_files": total_files,
                "unique_file_identifiers": len(set(
                    file_data['file_identifier'] for file_data in file_results.values()
                )),
                "messages_with_files": len(message_file_mappings),
                "message_indices": list(message_file_mappings.keys())
            },

            "performance_metrics": {
                "cache_utilization": round(cache_hits / max(total_files, 1), 2),
                "processing_efficiency": "high" if cache_hits > total_files * 0.5 else "medium" if cache_hits > 0 else "low",
                "message_file_associations": len(correlation_results.get("message_file_associations", []))
            }
        }

    def _calculate_risk_level(self, severity_score: int, correlation_results: Dict[str, Any]) -> str:
        """Calculate overall risk level based on severity and correlations."""
        base_risk = severity_score

        # Increase risk if cross-references found
        if correlation_results.get("cross_references"):
            base_risk += 1

        # Increase risk if severity escalation occurred
        if correlation_results.get("severity_escalation"):
            base_risk += 0.5

        if base_risk >= 4:
            return "critical"
        elif base_risk >= 3:
            return "high"
        elif base_risk >= 2:
            return "medium"
        elif base_risk >= 1:
            return "low"
        else:
            return "none"

    def _generate_recommendations(
        self,
        overall_severity: str,
        correlation_results: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations based on findings."""
        recommendations = []

        if overall_severity in ["high", "critical"]:
            recommendations.extend([
                "立即审查并删除包含敏感信息的文件",
                "修改密码和安全凭据",
                "联系相关机构监控账户活动",
                "启用双因素认证"
            ])

        if overall_severity in ["medium", "high", "critical"]:
            recommendations.extend([
                "避免在聊天中提及个人信息",
                "使用加密的文件传输",
                "定期审查共享的内容"
            ])

        if correlation_results.get("cross_references"):
            recommendations.append(
                "注意：相同类型的敏感信息同时出现在文件和聊天中，建议全面检查信息泄露范围"
            )

        if overall_severity == "low":
            recommendations.append("保持警惕，继续良好的隐私保护习惯")

        if overall_severity == "none":
            recommendations.append("未检测到明显隐私泄露风险，继续保持谨慎")

        return recommendations


class MessageIndexGenerator:
    """
    Generates unique identifiers for user messages that accompany files.

    Format: {account_id}:{detection_id}:{message_index}
    Hash: SHA256 hash for consistent length
    """

    @staticmethod
    def generate_message_index(account_id: str, detection_id: str, message_index: int) -> str:
        """
        Generate a unique message index for tracking file attachments.

        Args:
            account_id: User account identifier
            detection_id: Privacy detection session ID
            message_index: Index of the message in the conversation

        Returns:
            Unique message index string
        """
        # Create composite key
        composite_key = f"{account_id}:{detection_id}:{message_index}"

        # Generate SHA256 hash for consistent length and uniqueness
        hash_obj = hashlib.sha256(composite_key.encode('utf-8'))
        return hash_obj.hexdigest()[:16]  # Use first 16 chars for readability

    @staticmethod
    def parse_message_index(message_index: str) -> Optional[Tuple[str, str, int]]:
        """
        Parse a message index back to its components (for debugging).

        Note: This is one-way parsing since we use hashing.
        This method exists for logging/debugging purposes only.

        Args:
            message_index: The hashed message index

        Returns:
            None (since it's hashed, we can't reverse it)
        """
        # Since we use SHA256 hashing, we cannot reverse the index
        # This method exists for API consistency and future extensibility
        return None

    @staticmethod
    def generate_conversation_indices(
        account_id: str,
        detection_id: str,
        message_count: int
    ) -> List[str]:
        """
        Generate unique indices for all messages in a conversation.

        Args:
            account_id: User account identifier
            detection_id: Privacy detection session ID
            message_count: Total number of messages in conversation

        Returns:
            List of unique message indices
        """
        return [
            MessageIndexGenerator.generate_message_index(account_id, detection_id, i)
            for i in range(message_count)
        ]
