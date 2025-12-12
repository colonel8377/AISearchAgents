"""Main entry point for running the AI Search Agents Platform server."""

import sys
import os
import uvicorn

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import settings


def main():
    """Run the uvicorn server with the FastAPI application."""
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
