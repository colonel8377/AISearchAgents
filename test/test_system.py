"""Tests for System Management API endpoints."""

import pytest
from fastapi import status
from conftest import client


class TestSystemEndpoints:
    """Test system management endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "features" in data
        assert "agent_types" in data

    def test_reset_system(self, client):
        """Test resetting the system."""
        response = client.post("/api/v1/system/reset")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "success" in data or "message" in data

    def test_api_docs_available(self, client):
        """Test that API docs are available."""
        response = client.get("/docs")
        assert response.status_code == status.HTTP_200_OK

    def test_redoc_available(self, client):
        """Test that ReDoc is available."""
        response = client.get("/redoc")
        assert response.status_code == status.HTTP_200_OK

    def test_openapi_schema_available(self, client):
        """Test that OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data

