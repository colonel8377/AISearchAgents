"""Web Opinion Extractor Agent module."""

from .agent import WebOpinionExtractor
from .models import AtomicOpinion, OpinionExtractionResult
from .exceptions import WebExtractionError, NetworkError, ContentExtractionError

__all__ = [
    "WebOpinionExtractor",
    "AtomicOpinion",
    "OpinionExtractionResult",
    "WebExtractionError",
    "NetworkError",
    "ContentExtractionError",
]
