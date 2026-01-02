"""Main entry point for running the AI Search Agents Platform server."""

import os
import sys

import uvicorn

from src.shared.utils.logger import get_logger

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# IMPORTANT: Set TOKENIZERS_PARALLELISM before importing any transformers/tokenizers libraries
# This must be done before any fork (e.g., uvicorn workers) to avoid deadlocks
from shared.config.settings import settings

logger = get_logger(__name__)
# Set tokenizers parallelism from settings (must be before any imports that use tokenizers)
os.environ["TOKENIZERS_PARALLELISM"] = settings.tokenizers_parallelism

def main():
    """Run the uvicorn server with the FastAPI application."""
    # Preload models before starting server (downloads to cache, avoids runtime downloads)
    try:
        from shared.utils.model_preloader import preload_all_models
        preload_all_models()
    except Exception as e:
        # Don't fail startup if preloading fails - models will download on first use
        import sys
        logger.warning(f"Model preloading failed: {e}")
        logger.info("Models will be downloaded on first use instead.")
    
    uvicorn.run(
        "src.presentation.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
