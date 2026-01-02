"""
Common enumerations used across the application.
"""
from enum import Enum

class VectorStoreType(str, Enum):
    """Supported vector store types."""
    REDIS = "redis"
    POSTGRES = "postgres"
    CHROMA = "chroma"


class EmbeddingProviderType(str, Enum):
    """Supported embedding provider types."""
    OPENAI = "openai"
    QWEN = "qwen"
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"


class AgentType(str, Enum):
    """Supported agent types."""
    NUDGE_COLLAPSE = "nudge_collapse"
    SUMMARIZER = "summarizer"
    BOT_CREATOR = "bot_creator"
    DEMOGRAPHIC_EVALUATOR = "demographic_evaluator"
    CONTENT_EXTRACTOR = "content_extractor"
    CLAIM_ATOMIZER = "claim_atomizer"
    CONFLICT_AUDITOR = "conflict_auditor"
    SYNTHESIS_AGGREGATOR = "synthesis_aggregator"
    PRIVACY_DETECTOR = "privacy_detector"


# =============================================================================
# Agent Logic Enums
# =============================================================================

class CoTMode(str, Enum):
    """
    Chain of Thought (CoT) reasoning modes for LLM agents.
    
    This enum is shared across all agents in the platform to ensure consistency.

    - CHAIN_ONLINE: LLM handles task decomposition and chaining (more tokens, better reasoning)
    - CHAIN_LOCAL: System handles task decomposition, LLM executes individual steps (balanced)
    - NO_CHAIN: Direct prompt without CoT reasoning (fewer tokens, simpler)
    """
    CHAIN_ONLINE = "chain_online"
    CHAIN_LOCAL = "chain_local"
    NO_CHAIN = "no_chain"

    @classmethod
    def from_code_or_value(cls, value):
        """Convert code (int) or value (str) to CoTMode enum."""
        if isinstance(value, int):
            # Map numeric codes to enum values
            code_map = {0: cls.CHAIN_ONLINE, 1: cls.CHAIN_LOCAL, 2: cls.NO_CHAIN}
            if value in code_map:
                return code_map[value]
            raise ValueError(f"Invalid CoT mode code: {value}. Must be 0, 1, or 2.")
        elif isinstance(value, str):
            # Try direct enum value
            try:
                return cls(value.lower())
            except ValueError:
                # Try code as string
                try:
                    code = int(value)
                    return cls.from_code_or_value(code)
                except ValueError:
                    raise ValueError(
                        f"Invalid CoT mode: {value}. "
                        f"Must be 'chain_online', 'chain_local', 'no_chain', or code 0, 1, 2."
                    )
        else:
            raise ValueError(f"CoT mode must be string or int, got {type(value)}")


class LogicMode(str, Enum):
    """
    Logic mode for the web opinion analysis pipeline.

    - LOCAL_CHAIN: Extract -> Atomize -> Score (full pipeline with atomization)
    - NO_CHAIN: Extract -> Score (skip atomization, score full text)
    - PURE_ONLINE: End-to-end LLM analysis without intermediate steps
    """
    LOCAL_CHAIN = "LOCAL_CHAIN"
    NO_CHAIN = "NO_CHAIN"
    PURE_ONLINE = "PURE_ONLINE"


class ConflictType(str, Enum):
    """Types of conflicts between claims and evidence."""
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NEUTRAL_MISSING = "neutral_missing"
    NUMERICAL_DISCREPANCY = "numerical_discrepancy"
    TEMPORAL_ERROR = "temporal_error"
    DIRECTIONAL_CONTRADICTION = "directional_contradiction"
    SCOPE_DISTORTION = "scope_distortion"


# =============================================================================
# Privacy Detection Enums
# =============================================================================

class PrivacyCategory(str, Enum):
    """Academic taxonomy for privacy information categories."""
    IDENTITY = "identity"
    FINANCIAL = "financial"
    MEDICAL = "medical"
    TECHNICAL = "technical"
    NONE = "none"


class SeverityLevel(str, Enum):
    """Risk grading levels (L1-L4) based on NIST/ISO standards."""
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"
    NONE = "none"


class PrivacyType(str, Enum):
    """Privacy information types."""
    PERSONAL_IDENTIFIERS = "personal_identifiers"
    CONTACT_INFO = "contact_info"
    GOVERNMENT_IDS = "government_ids"
    BIOMETRIC_DATA = "biometric_data"
    FINANCIAL_ACCOUNTS = "financial_accounts"
    FINANCIAL_TRANSACTIONS = "financial_transactions"
    FINANCIAL_DOCUMENTS = "financial_documents"
    HEALTH_RECORDS = "health_records"
    MEDICAL_HISTORY = "medical_history"
    PRESCRIPTIONS_MEDICATIONS = "prescriptions_medications"
    AUTHENTICATION_CREDENTIALS = "authentication_credentials"
    API_KEYS_TOKENS = "api_keys_tokens"
    SYSTEM_ACCESS = "system_access"
    ENCRYPTION_KEYS = "encryption_keys"
    PRECISE_LOCATION = "precise_location"
    LOCATION_HISTORY = "location_history"
    EMAIL_CONTENT = "email_content"
    MESSAGING_CONTENT = "messaging_content"
    CALL_LOGS = "call_logs"
    SOCIAL_SECURITY = "social_security"
    RELATIONSHIP_DATA = "relationship_data"
    BEHAVIORAL_DATA = "behavioral_data"
    NONE = "none"


class PrivacySeverity(str, Enum):
    """Privacy leak severity levels (legacy)."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    NONE = "none"

class ParseMode(str, Enum):
    """Parsing mode enumeration."""
    LOCAL = "local"      # Local OCR and parsing only
    CLOUD = "cloud"      # Cloud multimodal parsing only
    HYBRID = "hybrid"    # Hybrid: cloud primary, local fallback (default)


class FileType(str, Enum):
    """Supported file types for parsing."""
    PDF = "pdf"
    WORD_DOCX = "docx"
    WORD_DOC = "doc"
    IMAGE_JPG = "jpg"
    IMAGE_JPEG = "jpeg"
    IMAGE_PNG = "png"
    IMAGE_BMP = "bmp"
    IMAGE_TIFF = "tiff"
    EXCEL_XLSX = "xlsx"
    EXCEL_XLS = "xls"
    TEXT_TXT = "txt"
    TEXT_CSV = "csv"
    IMAGE = "image"  # Generic image type (for privacy detector compatibility)
    UNKNOWN = "unknown"


class FileContentCategory(str, Enum):
    """File content categories for intelligent mode selection."""
    TEXT_BASED = "text_based"        # 纯文本：Word, Excel, CSV, TXT
    MIXED_CONTENT = "mixed_content"  # 混合内容：PDF
    IMAGE_ONLY = "image_only"        # 纯图片：JPG, PNG, BMP, TIFF
    UNKNOWN = "unknown"


class StorageType(Enum):
    """Storage type classifications."""
    AGENT_STATE = "agent_state"          # Agent configuration and state
    CONVERSATION = "conversation"        # Chat conversations
    USER_PROFILE = "user_profile"        # User profile data
    EMBEDDINGS = "embeddings"           # Vector embeddings
    METADATA = "metadata"               # General metadata
    CACHE = "cache"                     # Temporary cache data


class StoragePriority(Enum):
    """Storage priority levels."""
    LOW = "low"           # Can be lost, fast access
    MEDIUM = "medium"     # Important but can be recreated
    HIGH = "high"         # Critical data, must persist
    CRITICAL = "critical" # Must be persisted immediately