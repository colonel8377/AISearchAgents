"""Web Opinion Extractor Agent module."""

from .agent import WebOpinionExtractor
from .models import AtomicOpinion, OpinionExtractionResult, BiasDistribution
from .exceptions import WebExtractionError, NetworkError, ContentExtractionError

__all__ = [
    "WebOpinionExtractor",
    "AtomicOpinion",
    "OpinionExtractionResult",
    "BiasDistribution",
    "WebExtractionError",
    "NetworkError",
    "ContentExtractionError",
]
