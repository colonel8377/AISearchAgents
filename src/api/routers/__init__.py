"""API routers package."""

from .agents import router as agents_router
from .bot import router as bot_router
from .consistency import router as consistency_router
from .content import router as content_router
from .debate import router as debate_router
from .demographic import router as demographic_router
from .nudge_collapse import router as nudge_collapse_router
from .opinion import router as opinion_router
from .privacy_detector import router as privacy_detector_router
from .quality import router as quality_router
from .shots import router as shots_router
from .summarizer import router as summarizer_router
from .system import router as system_router

__all__ = [
    "agents_router",
    "bot_router",
    "consistency_router",
    "content_router",
    "debate_router",
    "demographic_router",
    "nudge_collapse_router",
    "opinion_router",
    "privacy_detector_router",
    "quality_router",
    "shots_router",
    "summarizer_router",
    "system_router"
]


