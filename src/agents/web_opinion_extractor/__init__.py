"""Web Opinion Extractor Agent module."""

from .agent import WebOpinionExtractor, WebOpinionAnalyzer
from .engine import WebOpinionEngine
from .models import (
    AtomicOpinion,
    OpinionExtractionResult,
    BiasDistribution,
    ArticleContent,
    SourceMetadata,
    AtomicUnit,
    BiasResult
)
from .exceptions import WebExtractionError, NetworkError, ContentExtractionError

__all__ = [
    "WebOpinionExtractor",
    "WebOpinionAnalyzer",
    "WebOpinionEngine",
    "AtomicOpinion",
    "OpinionExtractionResult",
    "BiasDistribution",
    "ArticleContent",
    "SourceMetadata",
    "AtomicUnit",
    "BiasResult",
    "WebExtractionError",
    "NetworkError",
    "ContentExtractionError",
]
