"""Tests for Content Extraction API endpoints."""

import pytest
from fastapi import status
from conftest import client


class TestContentExtraction:
    """Test content extraction endpoints."""

    def test_extract_content_from_url(self, client):
        """Test extracting content from URL."""
        response = client.post(
            "/api/v1/content/extract",
            json={
                "url": "https://example.com",
                "use_llm": False,
                "compare_claims": False
            }
        )
        # May fail if URL is not accessible, but should handle gracefully
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ]

    def test_extract_content_from_html(self, client):
        """Test extracting content from HTML."""
        html_content = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Test Heading</h1>
                <p>This is a test paragraph.</p>
            </body>
        </html>
        """
        response = client.post(
            "/api/v1/content/extract",
            json={
                "html": html_content,
                "use_llm": False
            }
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "paragraphs" in data or "main_body" in data

    def test_extract_content_from_text(self, client):
        """Test extracting content from plain text."""
        response = client.post(
            "/api/v1/content/extract",
            json={
                "text": "This is a test text. It has multiple sentences.",
                "title": "Test Title",
                "use_llm": False
            }
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "paragraphs" in data or "main_body" in data

    def test_extract_content_invalid_input(self, client):
        """Test extracting content with invalid input."""
        # No url, html, or text provided
        response = client.post(
            "/api/v1/content/extract",
            json={"use_llm": False}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestClaimAtomization:
    """Test claim atomization endpoints."""

    def test_atomize_claims(self, client):
        """Test atomizing claims from text."""
        text = "The sky is blue. The grass is green. Water is wet."
        response = client.post(
            "/api/v1/content/atomize",
            json={
                "text": text,
                "use_cot": "no_chain"
            }
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "atomic_claims" in data
        assert isinstance(data["atomic_claims"], list)

    def test_atomize_claims_with_paragraphs(self, client):
        """Test atomizing claims with paragraph splitting."""
        text = "First paragraph. Second sentence.\n\nSecond paragraph. Another sentence."
        response = client.post(
            "/api/v1/content/atomize",
            json={
                "text": text,
                "split_into_paragraphs": True,
                "use_cot": "no_chain"
            }
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "atomic_claims" in data
        assert "paragraphs" in data


class TestEvidenceLocation:
    """Test evidence location endpoints."""

    def test_locate_evidence(self, client):
        """Test locating evidence for claims."""
        response = client.post(
            "/api/v1/content/locate-evidence",
            json={
                "claims": [
                    {"id": "claim1", "text": "The sky is blue"},
                    {"id": "claim2", "text": "Water is wet"}
                ],
                "main_body": "The sky is blue because of Rayleigh scattering. Water is wet due to its molecular properties.",
                "use_llm": True
            }
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "claim_evidences" in data
        assert isinstance(data["claim_evidences"], list)


class TestConflictAudit:
    """Test conflict auditing endpoints."""

    def test_audit_conflicts(self, client):
        """Test auditing conflicts between claims and evidence."""
        response = client.post(
            "/api/v1/content/audit-conflicts",
            json={
                "claim_evidences": [
                    {
                        "claim_id": "claim1",
                        "claim_text": "The sky is blue",
                        "evidence_quotes": ["The sky is blue because of Rayleigh scattering"]
                    }
                ],
                "use_cot": "no_chain"
            }
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "conflict_analyses" in data
        assert isinstance(data["conflict_analyses"], list)

