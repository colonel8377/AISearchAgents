"""Tests for Privacy Detection API endpoints."""

import pytest
from fastapi import status
from conftest import client


class TestPrivacyDetection:
    """Test privacy detection endpoints."""

    def test_detect_privacy_basic(self, client):
        """Test basic privacy detection."""
        response = client.post(
            "/api/v1/privacy/detect",
            json={
                "conversation_records": [
                    {"user": "My email is test@example.com"},
                    {"user": "My phone is 555-123-4567"}
                ],
                "use_few_shots": True
            }
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "privacy_detected" in data
        assert "privacy_leaks" in data
        assert "detection_id" in data
        assert isinstance(data["privacy_leaks"], list)

    def test_detect_privacy_with_account_id(self, client):
        """Test privacy detection with account ID."""
        response = client.post(
            "/api/v1/privacy/detect",
            json={
                "conversation_records": [
                    {"user": "My SSN is 123-45-6789"}
                ],
                "account_id": "user123",
                "use_few_shots": True
            }
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "detection_id" in data

    def test_detect_privacy_no_leaks(self, client):
        """Test privacy detection with no privacy leaks."""
        response = client.post(
            "/api/v1/privacy/detect",
            json={
                "conversation_records": [
                    {"user": "Hello, how are you?"},
                    {"user": "The weather is nice today."}
                ],
                "use_few_shots": True
            }
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "privacy_detected" in data
        # May or may not detect privacy, depending on implementation

    def test_detect_privacy_invalid_input(self, client):
        """Test privacy detection with invalid input."""
        # Missing conversation_records
        response = client.post(
            "/api/v1/privacy/detect",
            json={"use_few_shots": True}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestPrivacyResults:
    """Test privacy detection results endpoints."""

    def test_get_detection_result(self, client):
        """Test getting detection result by ID."""
        # First create a detection
        detect_response = client.post(
            "/api/v1/privacy/detect",
            json={
                "conversation_records": [
                    {"user": "My email is test@example.com"}
                ],
                "use_few_shots": True
            }
        )
        assert detect_response.status_code == status.HTTP_200_OK
        detection_id = detect_response.json()["detection_id"]
        
        # Get the result
        response = client.get(f"/api/v1/privacy/results/{detection_id}")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["detection_id"] == detection_id

    def test_get_detection_result_not_found(self, client):
        """Test getting non-existent detection result."""
        response = client.get("/api/v1/privacy/results/non-existent-id")
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_list_detection_ids(self, client):
        """Test listing detection IDs."""
        response = client.get("/api/v1/privacy/results?limit=10&offset=0")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "detection_ids" in data
        assert "count" in data
        assert isinstance(data["detection_ids"], list)


class TestPrivacyStatistics:
    """Test privacy detection statistics endpoints."""

    def test_get_statistics(self, client):
        """Test getting privacy detection statistics."""
        response = client.get("/api/v1/privacy/stats")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "total_detections" in data
        assert "detection_rate_percent" in data

    def test_get_statistics_with_account_filter(self, client):
        """Test getting statistics filtered by account."""
        response = client.get("/api/v1/privacy/stats?account_id=user123")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "total_detections" in data


class TestPrivacyDeletion:
    """Test privacy detection deletion endpoints."""

    def test_delete_detection_result(self, client):
        """Test deleting a detection result."""
        # First create a detection
        detect_response = client.post(
            "/api/v1/privacy/detect",
            json={
                "conversation_records": [
                    {"user": "My email is test@example.com"}
                ],
                "use_few_shots": True
            }
        )
        assert detect_response.status_code == status.HTTP_200_OK
        detection_id = detect_response.json()["detection_id"]
        
        # Delete the result
        response = client.delete(f"/api/v1/privacy/results/{detection_id}")
        assert response.status_code == status.HTTP_200_OK
        
        # Verify it's deleted
        get_response = client.get(f"/api/v1/privacy/results/{detection_id}")
        assert get_response.status_code == status.HTTP_404_NOT_FOUND

