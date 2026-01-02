from typing import Optional
from dataclasses import dataclass

@dataclass
class PrivacyEntity:
    """Represents a detected privacy entity."""
    text: str
    entity_type: str
    start: int
    end: int
    confidence: float
    category: Optional[str] = None
    region: Optional[str] = None
    source: Optional[str] = None  # "presidio" or "gliner" to track detection source
    severity: Optional[str] = None  # Severity level (e.g., "low", "medium", "high", "critical")