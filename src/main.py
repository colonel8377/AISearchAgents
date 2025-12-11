"""Main entry point for running the AI Search Agents Platform server."""

import uvicorn
from src.config.settings import settings


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
