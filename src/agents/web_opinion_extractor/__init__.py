"""Web Opinion Extractor Agent module."""

from .agent import WebOpinionExtractor, WebOpinionAnalyzer
from .models import AtomicOpinion, OpinionExtractionResult, BiasDistribution
from .exceptions import WebExtractionError, NetworkError, ContentExtractionError

__all__ = [
    "WebOpinionExtractor",
    "WebOpinionAnalyzer",
    "AtomicOpinion",
    "OpinionExtractionResult",
    "BiasDistribution",
    "WebExtractionError",
    "NetworkError",
    "ContentExtractionError",
]
