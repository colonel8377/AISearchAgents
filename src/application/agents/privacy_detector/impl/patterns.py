"""
Enhanced pattern matching library for privacy detection.

Supports:
- Chinese-specific formats (ID cards, bank cards, passports)
- Format validation (checksum, Luhn algorithm)
- High-entropy string detection (API keys)
- Context-aware validation
"""

import re
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from .....shared.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PatternConfig:
    """Pattern configuration for enhanced matching."""
    pattern: str                              # Regular expression
    score: float                              # Base confidence score
    severity: str = "L3"                      # Severity level L1-L4
    category: str = "identity"                # Category
    requires_context: bool = False            # Whether context validation is required
    context_keywords: List[str] = field(default_factory=list)  # Context keywords
    validator: Optional[str] = None           # Validator function name


# Enhanced pattern definitions
ENHANCED_PATTERNS: Dict[str, PatternConfig] = {
    # ========== Chinese-specific patterns ==========
    "CN_PHONE": PatternConfig(
        pattern=r"(?<![0-9])1[3-9]\d{9}(?![0-9])",
        score=0.9,
        severity="L3",
        category="identity"
    ),
    "CN_ID_CARD": PatternConfig(
        pattern=r"(?<![0-9])[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?![0-9])",
        score=0.95,
        severity="L4",
        category="identity",
        validator="verify_cn_id_checksum"
    ),
    "CN_BANK_CARD": PatternConfig(
        pattern=r"(?<![0-9])(?:62|60|65|68|69|58|52|53|55|40|41|42|43|44|45|46|47|48|49)\d{14,17}(?![0-9])",
        score=0.85,
        severity="L4",
        category="financial",
        requires_context=True,
        context_keywords=["银行", "卡号", "账号", "bank", "card", "account"],
        validator="luhn_check"
    ),
    "CN_PASSPORT": PatternConfig(
        pattern=r"(?<![A-Za-z0-9])(?:E|G|D|S|P|H)\d{8}(?![0-9])",
        score=0.85,
        severity="L4",
        category="identity",
        requires_context=True,
        context_keywords=["护照", "passport", "出境", "入境"]
    ),
    "CN_DRIVER_LICENSE": PatternConfig(
        pattern=r"(?<![0-9A-Za-z])[0-9]{12}(?![0-9A-Za-z])",
        score=0.8,
        severity="L3",
        category="identity",
        requires_context=True,
        context_keywords=["驾驶证", "驾照", "driver", "license"]
    ),
    
    # ========== Credential patterns ==========
    "JWT_TOKEN": PatternConfig(
        pattern=r"eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+",
        score=0.95,
        severity="L4",
        category="technical"
    ),
    "AWS_ACCESS_KEY": PatternConfig(
        pattern=r"(?<![A-Z0-9])AKIA[0-9A-Z]{16}(?![A-Z0-9])",
        score=0.98,
        severity="L4",
        category="technical"
    ),
    "GITHUB_TOKEN": PatternConfig(
        pattern=r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}",
        score=0.98,
        severity="L4",
        category="technical"
    ),
    "OPENAI_KEY": PatternConfig(
        pattern=r"sk-[A-Za-z0-9]{48}",
        score=0.98,
        severity="L4",
        category="technical"
    ),
    "STRIPE_KEY": PatternConfig(
        pattern=r"(?:sk|pk)_(?:test|live)_[A-Za-z0-9]{24,51}",
        score=0.98,
        severity="L4",
        category="technical"
    ),
    
    # ========== Cryptocurrency ==========
    "BTC_ADDRESS": PatternConfig(
        pattern=r"(?<![A-Za-z0-9])(?:bc1|[13])[a-zA-HJ-NP-Z0-9]{25,39}(?![A-Za-z0-9])",
        score=0.9,
        severity="L4",
        category="financial"
    ),
    "ETH_ADDRESS": PatternConfig(
        pattern=r"(?<![A-Fa-f0-9x])0x[a-fA-F0-9]{40}(?![A-Fa-f0-9])",
        score=0.9,
        severity="L4",
        category="financial"
    ),
}


class EnhancedPatternMatcher:
    """Enhanced pattern matcher with validation and context awareness."""
    
    def __init__(self):
        self.compiled_patterns: Dict[str, re.Pattern] = {}
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        """Pre-compile all regular expressions."""
        for name, config in ENHANCED_PATTERNS.items():
            try:
                self.compiled_patterns[name] = re.compile(config.pattern)
            except re.error as e:
                logger.error(f"Failed to compile pattern {name}: {e}")
    
    def match_all(self, text: str) -> List[Tuple[str, str, int, int, float, str, str]]:
        """
        Match all patterns in text.
        
        Returns:
            List of (pattern_name, matched_text, start, end, confidence, severity, category)
        """
        matches = []
        
        for name, config in ENHANCED_PATTERNS.items():
            if name not in self.compiled_patterns:
                continue
            
            pattern = self.compiled_patterns[name]
            for match in pattern.finditer(text):
                matched_text = match.group(0)
                start, end = match.span()
                
                # Base confidence from pattern config
                confidence = config.score
                
                # Context validation
                if config.requires_context:
                    context_score = self.check_context(
                        text, start, end, config.context_keywords
                    )
                    if context_score < 0.3:
                        # Low context score, reduce confidence
                        confidence *= 0.7
                
                # Format validation
                if config.validator:
                    validator_func = getattr(self, config.validator, None)
                    if validator_func:
                        if not validator_func(matched_text):
                            # Validation failed, reduce confidence significantly
                            confidence *= 0.5
                
                matches.append((
                    name,
                    matched_text,
                    start,
                    end,
                    confidence,
                    config.severity,
                    config.category
                ))
        
        return matches
    
    def check_context(
        self,
        text: str,
        start: int,
        end: int,
        keywords: List[str],
        window: int = 50
    ) -> float:
        """
        Check context keywords around the match.
        
        Returns:
            Confidence adjustment value (0.0-1.0)
        """
        if not keywords:
            return 0.5  # Neutral if no keywords
        
        # Extract context window
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        context = text[context_start:context_end].lower()
        
        # Check if any keyword appears in context
        found_keywords = sum(1 for kw in keywords if kw.lower() in context)
        
        if found_keywords > 0:
            return min(1.0, 0.5 + (found_keywords / len(keywords)) * 0.5)
        
        return 0.3  # Low confidence if no keywords found
    
    def calculate_entropy(self, text: str) -> float:
        """
        Calculate Shannon entropy of text.
        
        Higher entropy indicates more randomness (likely to be a secret/API key).
        
        Args:
            text: Input text
            
        Returns:
            Entropy value (bits per character)
        """
        if not text:
            return 0.0
        
        # Count character frequencies
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate entropy
        length = len(text)
        entropy = 0.0
        for count in char_counts.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def detect_high_entropy_secrets(
        self,
        text: str,
        min_length: int = 20,
        entropy_threshold: float = 4.5
    ) -> List[Tuple[str, int, int, float]]:
        """
        Detect high-entropy strings (likely API keys or secrets).
        
        Rules:
        1. Length > min_length
        2. Contains both digits and letters
        3. Entropy > entropy_threshold
        4. Context contains key/token/secret keywords (increases confidence)
        
        Args:
            text: Input text
            min_length: Minimum length to consider
            entropy_threshold: Minimum entropy threshold
            
        Returns:
            List of (secret_text, start, end, confidence)
        """
        secrets = []
        
        # Pattern to match potential secrets (alphanumeric strings)
        # Look for sequences of alphanumeric characters
        pattern = re.compile(r'[A-Za-z0-9_-]{' + str(min_length) + r',}')
        
        for match in pattern.finditer(text):
            matched_text = match.group(0)
            start, end = match.span()
            
            # Must contain both letters and digits
            has_letters = bool(re.search(r'[A-Za-z]', matched_text))
            has_digits = bool(re.search(r'[0-9]', matched_text))
            
            if not (has_letters and has_digits):
                continue
            
            # Calculate entropy
            entropy = self.calculate_entropy(matched_text)
            
            if entropy < entropy_threshold:
                continue
            
            # Check context for credential keywords
            context_start = max(0, start - 30)
            context_end = min(len(text), end + 30)
            context = text[context_start:context_end].lower()
            
            credential_keywords = [
                "key", "token", "secret", "api", "password", "credential",
                "access", "auth", "apikey", "apikey=", "token=", "secret="
            ]
            
            has_credential_context = any(kw in context for kw in credential_keywords)
            
            # Base confidence from entropy
            confidence = min(0.95, 0.7 + (entropy - entropy_threshold) * 0.1)
            
            # Boost confidence if context suggests credential
            if has_credential_context:
                confidence = min(0.98, confidence + 0.15)
            
            secrets.append((matched_text, start, end, confidence))
        
        return secrets
    
    # ========== Validation functions ==========
    
    @staticmethod
    def verify_cn_id_checksum(id_number: str) -> bool:
        """
        Verify Chinese ID card checksum digit.
        
        Algorithm:
        1. First 17 digits multiplied by weights [7,9,10,5,8,4,2,1,6,3,7,9,10,5,8,4,2]
        2. Sum modulo 11
        3. Lookup checksum digit from "10X98765432"
        
        Args:
            id_number: 18-digit ID card number
            
        Returns:
            True if checksum is valid
        """
        if len(id_number) != 18:
            return False
        
        # Extract first 17 digits
        try:
            digits = [int(id_number[i]) for i in range(17)]
        except ValueError:
            return False
        
        # Weights for each position
        weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
        
        # Calculate weighted sum
        weighted_sum = sum(digits[i] * weights[i] for i in range(17))
        
        # Get checksum digit
        checksum_index = weighted_sum % 11
        checksum_table = "10X98765432"
        expected_checksum = checksum_table[checksum_index]
        
        # Compare with actual checksum (case-insensitive)
        actual_checksum = id_number[17].upper()
        
        return actual_checksum == expected_checksum
    
    @staticmethod
    def luhn_check(card_number: str) -> bool:
        """
        Validate card number using Luhn algorithm.
        
        Args:
            card_number: Card number (digits only)
            
        Returns:
            True if Luhn check passes
        """
        # Remove non-digits
        digits = re.sub(r'\D', '', card_number)
        
        if len(digits) < 13:
            return False
        
        # Reverse the digits
        reversed_digits = digits[::-1]
        
        # Apply Luhn algorithm
        total = 0
        for i, digit in enumerate(reversed_digits):
            n = int(digit)
            if i % 2 == 1:
                # Double every second digit
                n *= 2
                if n > 9:
                    n -= 9
            total += n
        
        # Check if divisible by 10
        return total % 10 == 0





