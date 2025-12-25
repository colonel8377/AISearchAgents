"""FastAPI application for AI Search Agents Platform - Refactored Version."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..config.settings import settings
from ..utils.logger import configure_app_logging, get_logger
from ..storage import get_storage

# Configure logging
configure_app_logging(
    log_level=settings.log_level,
    log_file=settings.log_file,
    enable_debug=settings.enable_debug
)
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Search Agents Platform",
    description="Platform for experimenting with AI search agents with multi-agent support and authentication",
    version="2.0.0"
)

# Add CORS middleware if needed
if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Import and include routers
from .routers import (
    agents,
    bot,
    consistency,
    content,
    debate,
    demographic,
    nudge_collapse,
    opinion,
    privacy_detector,
    quality,
    shots,
    summarizer,
    system
)

# Register all routers
# Note: system.default_router (root endpoint) is included first to handle root path
app.include_router(system.default_router)  # Root/default endpoints (merged with System in API docs)
app.include_router(system.router)  # System management endpoints
app.include_router(agents.router)
app.include_router(nudge_collapse.router)
app.include_router(demographic.router)
app.include_router(summarizer.router)
app.include_router(bot.router)
app.include_router(debate.router)
app.include_router(opinion.router)
app.include_router(content.router)
app.include_router(consistency.router)
app.include_router(quality.router)
app.include_router(privacy_detector.router)  # Privacy Detection endpoints
app.include_router(shots.router)  # Few-Shot Configuration endpoints


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    if settings.enable_persistence:
        storage = get_storage()
        # Only initialize connection, don't load data (data is loaded on-demand from database)
        logger.info("Storage database initialized")
    else:
        logger.info("Persistence disabled, using in-memory storage only")


