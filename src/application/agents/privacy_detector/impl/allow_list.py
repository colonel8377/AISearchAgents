"""
Enhanced Allow List for filtering false positives.

Filters out:
- Example/test data patterns
- Documentation/placeholder data
- Sequential/repeated digits
- Common test values
"""

import re
from typing import Set, Dict, List, Optional

from .....shared.utils.logger import get_logger

logger = get_logger(__name__)


# ========== Example data patterns ==========
EXAMPLE_PATTERNS: Dict[str, List[str]] = {
    "PHONE": [
        "555-",                    # US fictional number prefix
        "123-456-7890",
        "000-000-0000",
        "10086", "10010", "10000", # Chinese carrier service numbers
        "12345678901",             # Sequential digits
        "13800138000",             # China Mobile test number
        "11111111111",
        "00000000000",
    ],
    "EMAIL": [
        "example.com",
        "test.com",
        "sample.com",
        "localhost",
        "your-email.com",
        "@xxx.xxx",
        "test@test.com",
        "example@example.com",
        "user@example.com",
        "admin@test.com",
    ],
    "SSN": [
        "123-45-6789",
        "000-00-0000",
        "111-11-1111",
        "123456789",
    ],
    "CREDIT_CARD": [
        "4111111111111111",        # Visa test card
        "5500000000000004",        # MasterCard test card
        "4242424242424242",        # Stripe test card
        "4000000000000002",        # Generic test card
    ],
    "CN_ID_CARD": [
        "110101199001011234",      # Common example
        "000000000000000000",
        "111111111111111111",
    ],
    "CN_BANK_CARD": [
        "6222020000000000000",     # Common test pattern
        "0000000000000000000",
    ],
}


# ========== Documentation/code context indicators ==========
EXAMPLE_CONTEXT_PATTERNS: List[str] = [
    r"格式[如:：]",
    r"例如[:：]",
    r"比如[:：]",
    r"示例[:：]",
    r"for example",
    r"e\.g\.",
    r"such as",
    r"placeholder",
    r"sample",
    r"demo",
    r"test",
    r"mock",
    r"fake",
    r"dummy",
    r"example",
    r"格式说明",
    r"示例数据",
    r"测试数据",
]


class EnhancedAllowList:
    """Enhanced allow list checker for filtering false positives."""
    
    def __init__(self):
        self._compile_patterns()
        self._build_example_sets()
    
    def _compile_patterns(self) -> None:
        """Pre-compile regular expressions."""
        try:
            self.context_regex = re.compile(
                "|".join(EXAMPLE_CONTEXT_PATTERNS),
                re.IGNORECASE
            )
        except re.error as e:
            logger.error(f"Failed to compile context patterns: {e}")
            self.context_regex = re.compile("")
    
    def _build_example_sets(self) -> None:
        """Build sets for fast lookup."""
        self.example_sets: Dict[str, Set[str]] = {}
        for entity_type, patterns in EXAMPLE_PATTERNS.items():
            self.example_sets[entity_type] = set(patterns)
    
    def should_filter(
        self,
        text: str,
        entity_type: str,
        full_context: str = "",
        position: int = -1
    ) -> bool:
        """
        Determine if a detection result should be filtered.
        
        Check order:
        1. Example data pattern matching
        2. Documentation/code context detection
        3. Sequential digits detection
        4. Repeated digits detection
        
        Args:
            text: Detected text
            entity_type: Entity type
            full_context: Full context text
            position: Position of entity in context
            
        Returns:
            True if should be filtered (is example data or false positive)
        """
        # Normalize text for comparison
        normalized_text = text.strip().lower()
        
        # 1. Check example patterns
        if self.is_example_pattern(normalized_text, entity_type):
            return True
        
        # 2. Check context
        if position >= 0 and full_context:
            if self.is_in_example_context(full_context, position):
                return True
        
        # 3. Check sequential digits
        if self.is_sequential_digits(text):
            return True
        
        # 4. Check repeated digits
        if self.is_repeated_digits(text):
            return True
        
        return False
    
    def is_example_pattern(self, text: str, entity_type: str) -> bool:
        """
        Check if text matches example data patterns.
        
        Args:
            text: Normalized text (lowercase)
            entity_type: Entity type
            
        Returns:
            True if matches example pattern
        """
        # Check exact match
        if entity_type in self.example_sets:
            if text in self.example_sets[entity_type]:
                return True
            
            # Check substring match (for patterns like "555-")
            for pattern in self.example_sets[entity_type]:
                if pattern in text or text in pattern:
                    return True
        
        # Check common patterns across types
        if text.startswith("test@") or text.startswith("example@"):
            return True
        
        if text.startswith("555-") and entity_type in ["PHONE", "PHONE_NUMBER", "PHONE_NUMBER_CN"]:
            return True
        
        return False
    
    def is_in_example_context(
        self,
        full_context: str,
        position: int,
        window: int = 50
    ) -> bool:
        """
        Check if entity is in example/documentation context.
        
        Args:
            full_context: Full context text
            position: Position of entity
            window: Context window size
            
        Returns:
            True if in example context
        """
        # Extract context window
        context_start = max(0, position - window)
        context_end = min(len(full_context), position + window)
        context_window = full_context[context_start:context_end]
        
        # Check for example context patterns
        if self.context_regex.search(context_window):
            return True
        
        return False
    
    def is_sequential_digits(self, text: str) -> bool:
        """
        Check if text is sequential digits.
        
        Examples:
        - "123456789" -> True (increasing)
        - "987654321" -> True (decreasing)
        - "138123456" -> False
        
        Args:
            text: Text to check
            
        Returns:
            True if sequential digits
        """
        # Extract digits only
        digits = re.sub(r'\D', '', text)
        
        if len(digits) < 6:  # Too short to be meaningful
            return False
        
        # Check if all digits are sequential (increasing or decreasing)
        is_increasing = True
        is_decreasing = True
        
        for i in range(len(digits) - 1):
            curr = int(digits[i])
            next_d = int(digits[i + 1])
            
            if next_d != curr + 1:
                is_increasing = False
            if next_d != curr - 1:
                is_decreasing = False
            
            if not is_increasing and not is_decreasing:
                break
        
        return is_increasing or is_decreasing
    
    def is_repeated_digits(self, text: str, threshold: int = 4) -> bool:
        """
        Check if text contains repeated digits.
        
        Examples:
        - "1111111111" -> True
        - "0000000000" -> True
        - "13812345678" -> False
        
        Args:
            text: Text to check
            threshold: Minimum consecutive repeated digits
            
        Returns:
            True if contains repeated digits
        """
        # Extract digits only
        digits = re.sub(r'\D', '', text)
        
        if len(digits) < threshold:
            return False
        
        # Check for consecutive repeated digits
        pattern = re.compile(r'(\d)\1{' + str(threshold - 1) + r',}')
        return bool(pattern.search(digits))
    
    def is_test_number(self, text: str, entity_type: str) -> bool:
        """
        Check if text is a known test number.
        
        Args:
            text: Text to check
            entity_type: Entity type
            
        Returns:
            True if is test number
        """
        normalized = text.strip().lower()
        
        # Chinese carrier service numbers
        if entity_type in ["PHONE", "PHONE_NUMBER", "PHONE_NUMBER_CN"]:
            test_numbers = {"10086", "10010", "10000", "10001"}
            if normalized in test_numbers:
                return True
        
        return False




