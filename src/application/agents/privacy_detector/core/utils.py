"""
Utility functions for privacy detector.
"""

import re
from typing import List, Optional, Dict, Any
from src.shared.constant.enums import (
    PrivacyCategory,
    SeverityLevel,
    PrivacyType,
)

# Allow list patterns to filter false positives
def build_allow_list() -> List[re.Pattern]:
    """
    Build allow list patterns to filter false positives.

    Filters out:
    - Local IPs (127.0.0.1, localhost, etc.)
    - Common ports (8080, 3000, etc.)
    - Version numbers (v1.0, v1.0.0, etc.)
    """
    return [
        # Local IPs
        re.compile(r'^127\.0\.0\.1$', re.IGNORECASE),
        re.compile(r'^localhost$', re.IGNORECASE),
        re.compile(r'^0\.0\.0\.0$', re.IGNORECASE),
        # Common ports (standalone numbers)
        re.compile(r'^(8080|3000|8000|5000|4000|9000)$'),
        # Version numbers (v1.0, v1.0.0, v2.1.3, etc.)
        re.compile(r'^v\d+\.\d+(\.\d+)?$', re.IGNORECASE),
    ]

def is_in_allow_list(text: str, entity_type: str, patterns: List[re.Pattern]) -> bool:
    """
    Check if detected entity matches allow list patterns (false positive filter).
    """
    # Only filter specific entity types that are prone to false positives
    filterable_types = {"IP_ADDRESS", "PHONE_NUMBER", "PHONE_NUMBER_CN"}
    if entity_type not in filterable_types:
        return False

    # Check against allow list patterns
    for pattern in patterns:
        if pattern.match(text.strip()):
            return True

    return False

def has_chinese_text(text: str) -> bool:
    """Check if text contains Chinese characters."""
    return any('\u4e00' <= char <= '\u9fff' for char in text)

def severity_to_int(severity: str) -> int:
    """Helper for comparing severity levels."""
    mapping = {"L1": 1, "L2": 2, "L3": 3, "L4": 4}
    return mapping.get(severity, 0)

def map_privacy_type_to_category(privacy_type: str) -> PrivacyCategory:
    """Map PrivacyType to PrivacyCategory."""
    identity_types = {
        PrivacyType.PERSONAL_IDENTIFIERS.value,
        PrivacyType.CONTACT_INFO.value,
        PrivacyType.GOVERNMENT_IDS.value,
        PrivacyType.BIOMETRIC_DATA.value,
    }

    financial_types = {
        PrivacyType.FINANCIAL_ACCOUNTS.value,
        PrivacyType.FINANCIAL_TRANSACTIONS.value,
        PrivacyType.FINANCIAL_DOCUMENTS.value,
    }

    medical_types = {
        PrivacyType.HEALTH_RECORDS.value,
        PrivacyType.MEDICAL_HISTORY.value,
        PrivacyType.PRESCRIPTIONS_MEDICATIONS.value,
    }

    technical_types = {
        PrivacyType.AUTHENTICATION_CREDENTIALS.value,
        PrivacyType.API_KEYS_TOKENS.value,
        PrivacyType.SYSTEM_ACCESS.value,
        PrivacyType.ENCRYPTION_KEYS.value,
    }

    if privacy_type in identity_types:
        return PrivacyCategory.IDENTITY
    elif privacy_type in financial_types:
        return PrivacyCategory.FINANCIAL
    elif privacy_type in medical_types:
        return PrivacyCategory.MEDICAL
    elif privacy_type in technical_types:
        return PrivacyCategory.TECHNICAL
    else:
        return PrivacyCategory.NONE

def map_severity_to_level(severity: str, privacy_type: str, region: Optional[str] = None) -> SeverityLevel:
    """Map severity to L1-L4 level."""
    # Rule-based overrides
    if region == "HEADER" and privacy_type == PrivacyType.CONTACT_INFO.value:
        return SeverityLevel.L1

    critical_types = {
        PrivacyType.API_KEYS_TOKENS.value,
        PrivacyType.AUTHENTICATION_CREDENTIALS.value,
        PrivacyType.ENCRYPTION_KEYS.value,
        PrivacyType.SYSTEM_ACCESS.value,
    }

    high_risk_types = {
        PrivacyType.FINANCIAL_ACCOUNTS.value,
        PrivacyType.GOVERNMENT_IDS.value,
    }

    sensitive_types = {
        PrivacyType.HEALTH_RECORDS.value,
        PrivacyType.MEDICAL_HISTORY.value,
        PrivacyType.PRESCRIPTIONS_MEDICATIONS.value,
    }

    if privacy_type in critical_types:
        return SeverityLevel.L4

    if privacy_type in high_risk_types:
        if severity in ["critical", "high"]:
            return SeverityLevel.L4
        else:
            return SeverityLevel.L3

    if privacy_type in sensitive_types:
        return SeverityLevel.L3

    severity_mapping = {
        "low": SeverityLevel.L1,
        "medium": SeverityLevel.L2,
        "high": SeverityLevel.L3,
        "critical": SeverityLevel.L4,
        "none": SeverityLevel.NONE,
    }

    return severity_mapping.get(severity.lower(), SeverityLevel.L2)

