"""FastAPI application for AI Search Agents Platform - Refactored Version."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from ...infrastructure.storage.persistence import get_storage
from ...shared.config.settings import settings
from ...shared.utils.logger import configure_app_logging, get_logger

# Configure logging
configure_app_logging(
    log_level=settings.log_level,
    log_file=settings.log_file,
    enable_debug=settings.enable_debug
)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize application services on startup and cleanup on shutdown."""
    # Startup logic
    if settings.enable_persistence:
        get_storage()
        # Only initialize persistence, don't load data (data is loaded on-demand from persistence)
        logger.info("Storage persistence initialized")
    else:
        logger.info("Persistence disabled, using in-memory storage only")

    yield

    # Shutdown logic can be added here if needed
    logger.info("Application shutdown")


# Create FastAPI app
app = FastAPI(
    title="AI Search Agents Platform",
    description="Platform for experimenting with AI search agents with multi-agent support and authentication",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware if needed
if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,  # type: ignore
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

