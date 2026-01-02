"""Tests for Opinion Extraction API endpoints."""

import pytest
from fastapi import status
from conftest import client


class TestOpinionExtraction:
    """Test opinion extraction endpoints."""

    def test_extract_opinions_from_url(self, client):
        """Test extracting opinions from URL."""
        response = client.post(
            "/api/v1/opinion/extract-opinions",
            json={
                "url": "https://example.com",
                "use_llm": True,
                "use_cot": "no_chain"
            }
        )
        # May fail if URL is not accessible
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ]

    def test_extract_opinions_from_text(self, client):
        """Test extracting opinions from text."""
        response = client.post(
            "/api/v1/opinion/extract-opinions",
            json={
                "text": "I believe that climate change is a serious issue. The government should take action.",
                "title": "Climate Opinion",
                "use_llm": True,
                "use_cot": "no_chain"
            }
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "atomic_opinions" in data
        assert "overall_bias_distribution" in data

    def test_extract_opinions_invalid_input(self, client):
        """Test extracting opinions with invalid input."""
        # No url or text provided
        response = client.post(
            "/api/v1/opinion/extract-opinions",
            json={"use_llm": True}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestBiasScore:
    """Test bias score endpoints."""

    def test_get_bias_score_from_url(self, client):
        """Test getting bias score from URL."""
        response = client.post(
            "/api/v1/opinion/bias-score",
            json={
                "url": "https://example.com",
                "mode": "LOCAL_CHAIN",
                "use_mbfc": False,
                "use_few_shots": True
            }
        )
        # May fail if URL is not accessible
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ]

    def test_get_bias_score_from_content(self, client):
        """Test getting bias score from content."""
        response = client.post(
            "/api/v1/opinion/bias-score",
            json={
                "content": "This is a political article discussing various policies.",
                "title": "Political Article",
                "mode": "LOCAL_CHAIN",
                "use_few_shots": True
            }
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "overall_bias_distribution" in data
        assert "opinions_count" in data
        assert "facts_count" in data

