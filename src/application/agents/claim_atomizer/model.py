from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class AtomicClaim:
    """An atomic claim extracted from text."""
    id: str
    text: str
    original_sentence: str
    confidence: float
    paragraph_index: Optional[int] = None
    paragraph_text: Optional[str] = None


@dataclass
class ParagraphClaims:
    """Claims from a single paragraph."""
    paragraph_index: int
    paragraph_text: str
    atomic_claims: List[AtomicClaim]


@dataclass
class ClaimAtomizationResult:
    """Result of claim atomization."""
    atomic_claims: List[AtomicClaim]
    paragraphs: List[ParagraphClaims]
    original_text: str
    execution_mode: str
    metadata: Optional[Dict[str, Any]] = None