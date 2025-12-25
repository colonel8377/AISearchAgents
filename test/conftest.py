"""
Shared pytest fixtures and configuration for all tests.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """FastAPI test client fixture."""
    return TestClient(app)
