"""
Sanitization implementation with session-based consistency.
Optimized for both LLM context retention (Sanitize) and UI security (Masking).
"""

from typing import List, Dict, Any, Tuple, Optional

from ..core.interfaces import ISanitizer, PrivacyEntity
from .....shared.utils.logger import get_logger


# Lazy load Faker to improve startup time
try:
    from faker import Faker
    FAKER_AVAILABLE = True
except ImportError:
    FAKER_AVAILABLE = False
    Faker = None

logger = get_logger(__name__)


class ConsistentSanitizer(ISanitizer):
    """
    Consistent sanitizer that maintains session-based consistency for LLM inputs
    and provides stateless high-performance masking for UI outputs.

    Features:
    - Session Consistency: "Alice" -> "Bob" (always same replacement in session)
    - Context Retention: Uses Faker to generate semantic-aware replacements
    - UI Masking: "Alice" -> "A***" (stateless, fast, secure)
    """

    def __init__(self, session_id: str = "default"):
        """
        Initialize sanitizer with session ID for consistency.

        Args:
            session_id: Session identifier for consistency mapping
        """
        self.session_id = session_id
        self.consistency_map: Dict[str, str] = {}  # original -> synthetic
        self._fake: Optional[Any] = None # Lazy initialization

    @property
    def fake(self) -> Optional[Any]:
        """Lazy loader for Faker instance."""
        if self._fake is None and FAKER_AVAILABLE:
            try:
                self._fake = Faker()
                logger.debug("Faker initialized for synthetic data generation")
            except Exception as e:
                logger.warning(f"Failed to initialize Faker: {e}")
        return self._fake

    def sanitize(
        self,
        text: str,
        entities: List[PrivacyEntity]
    ) -> Tuple[str, Dict[str, Dict[str, Any]]]:
        """
        Sanitize text for LLM consumption (Semantic Replacement).
        Replaces PII with realistic synthetic data to preserve context for the model.

        Args:
            text: Original text containing privacy entities
            entities: List of detected privacy entities

        Returns:
            Tuple of (sanitized_text, registry)
        """
        sanitized_text = text
        logger.info(f"Sanitized text: {sanitized_text}")
        registry = {}

        if not entities:
            return sanitized_text, registry

        # Sort entities by start position (reverse order) to avoid index shifts
        sorted_entities = sorted(entities, key=lambda e: e.start, reverse=True)

        for entity in sorted_entities:
            original_value = entity.text

            # Check consistency map first (Session Persistence)
            if original_value in self.consistency_map:
                synthetic_value = self.consistency_map[original_value]
            else:
                # Generate new synthetic value
                synthetic_value = self._generate_synthetic_value(entity)
                self.consistency_map[original_value] = synthetic_value

            sanitized_text = (
                sanitized_text[:entity.start] +
                synthetic_value +
                sanitized_text[entity.end:]
            )

            # Store in registry
            # Key format: TYPE_HASH to avoid storing full PII in keys if possible
            registry_key = f"{entity.entity_type}_{hash(original_value)}"
            registry[registry_key] = {
                "type": entity.entity_type,
                "original_value": original_value,
                "synthetic_value": synthetic_value,
                "category": entity.category or "none",
                "start": entity.start,
                "end": entity.end,
                "confidence": entity.confidence
            }

        logger.debug(f"ConsistentSanitizer sanitized {len(registry)} entities")
        return sanitized_text, registry

    def mask(
        self,
        text: str,
        entities: List[PrivacyEntity]
    ) -> str:
        """
        Mask privacy entities for UI Display (Physical Redaction).
        Optimized for speed and readability (e.g. "138****0000").

        Args:
            text: Original text
            entities: List of detected privacy entities

        Returns:
            Masked text
        """
        if not entities:
            return text

        # Use a list of characters for mutable string operations (faster than concat)
        masked_chars = list(text)

        # Sort by start position (reverse) isn't strictly necessary for list replacement
        # if we trust the entity indices, but good for safety against overlaps
        sorted_entities = sorted(entities, key=lambda e: e.start, reverse=True)

        for entity in sorted_entities:
            # Safety check for boundaries
            if entity.start < 0 or entity.end > len(text):
                continue

            original_span = text[entity.start:entity.end]
            masked_span = self._generate_mask_pattern(original_span, entity.entity_type)

            # Replace the slice in the character list
            masked_chars[entity.start:entity.end] = list(masked_span)

        return "".join(masked_chars)

    def _generate_synthetic_value(self, entity: PrivacyEntity) -> str:
        """Generate realistic synthetic data using Faker."""
        entity_type = entity.entity_type.upper()
        label_lower = entity_type.lower()

        if not self.fake:
            return f"<{entity_type}>"

        # Priority 1: Check semantic labels (GLiNER style)
        if 'phone' in label_lower: return self.fake.phone_number()
        if 'email' in label_lower: return self.fake.email()
        if 'name' in label_lower or 'person' in label_lower: return self.fake.name()
        if 'address' in label_lower or 'location' in label_lower: return self.fake.address()
        if 'ip' in label_lower: return self.fake.ipv4()
        if 'credit' in label_lower or 'card' in label_lower: return self.fake.credit_card_number()

        # Priority 2: Check standard types (Presidio style)
        generators = {
            "PHONE_NUMBER": self.fake.phone_number,
            "EMAIL_ADDRESS": self.fake.email,
            "PERSON": self.fake.name,
            "CREDIT_CARD": self.fake.credit_card_number,
            "LOCATION": self.fake.address,
            "IP_ADDRESS": self.fake.ipv4,
            "US_SSN": self.fake.ssn,
            "SSN": self.fake.ssn,
            "IBAN": self.fake.iban,
            "URL": self.fake.url,
        }

        gen_func = generators.get(entity_type)
        if gen_func:
            return gen_func()

        # Fallback: Generic Placeholder
        return f"<{entity_type}>"

    def _generate_mask_pattern(self, text: str, entity_type: str) -> str:
        """
        Generate mask string (Stateless, Pure Logic).
        Strategies tailored by entity type for best UX.
        """
        length = len(text)
        if length == 0: return ""

        etype = entity_type.upper()

        # Strategy 1: Email (j***@example.com)
        if "EMAIL" in etype:
            if "@" in text:
                local, domain = text.split("@", 1)
                if len(local) > 2:
                    return f"{local[0]}***@{domain}"
                return f"***@{domain}"
            return "*" * length

        # Strategy 2: Phone (138****0000)
        if "PHONE" in etype:
            if length > 7:
                return f"{text[:3]}****{text[-4:]}"
            if length > 4:
                return f"{text[:2]}**{text[-2:]}"
            return "*" * length

        # Strategy 3: ID Card / SSN (**************1234)
        if "ID" in etype or "SSN" in etype:
            if length > 4:
                return "*" * (length - 4) + text[-4:]
            return "*" * length

        # Strategy 4: Name (A***)
        if "PERSON" in etype or "NAME" in etype:
            if length > 1:
                return f"{text[0]}***"
            return "*"

        # Default: Full Mask
        return "*" * length

    def reset_session(self, new_session_id: str = None):
        """Reset consistency map for new session."""
        if new_session_id:
            self.session_id = new_session_id
        self.consistency_map.clear()
        logger.debug(f"ConsistentSanitizer reset for session: {self.session_id}")