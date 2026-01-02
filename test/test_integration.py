"""Integration tests for AI Search Agents Platform."""

import pytest
from fastapi import status
from conftest import client


class TestAgentWorkflow:
    """Test complete agent workflow."""

    def test_create_and_use_nudge_collapse_agent(self, client):
        """Test creating and using a nudge_collapse agent."""
        # Create agent
        create_response = client.post(
            "/api/v1/agents/create",
            json={
                "agent_type": "nudge_collapse",
                "use_memory": False
            }
        )
        assert create_response.status_code == status.HTTP_200_OK
        agent_id = create_response.json()["agent_id"]
        
        # Get agent status
        status_response = client.get(f"/api/v1/agents/{agent_id}/status")
        assert status_response.status_code == status.HTTP_200_OK
        
        # Reset agent
        reset_response = client.post(
            f"/api/v1/agents/{agent_id}/reset",
            json={"reset_conversation": True}
        )
        assert reset_response.status_code == status.HTTP_200_OK
        
        # Delete agent
        delete_response = client.delete(f"/api/v1/agents/{agent_id}")
        assert delete_response.status_code == status.HTTP_200_OK


class TestBotWorkflow:
    """Test complete bot workflow."""

    def test_create_bot_and_conversation(self, client):
        """Test creating a bot and having a conversation."""
        # Create bot
        create_response = client.post(
            "/api/v1/bot/create",
            json={"bot_name": "IntegrationBot"}
        )
        assert create_response.status_code == status.HTTP_200_OK
        bot_id = create_response.json()["bot_id"]
        
        # Create conversation
        conv_response = client.post(
            f"/api/v1/bot/{bot_id}/conversation",
            json={"title": "Integration Test Conversation"}
        )
        assert conv_response.status_code == status.HTTP_200_OK
        conversation_id = conv_response.json()["conversation_id"]
        
        # Chat in conversation
        chat_response = client.post(
            "/api/v1/bot/chat",
            json={
                "bot_id": bot_id,
                "message": "Hello, this is a test.",
                "conversation_id": conversation_id
            }
        )
        assert chat_response.status_code == status.HTTP_200_OK
        
        # Get conversation details
        detail_response = client.get(
            f"/api/v1/bot/{bot_id}/conversation/{conversation_id}"
        )
        assert detail_response.status_code == status.HTTP_200_OK


class TestContentAnalysisWorkflow:
    """Test complete content analysis workflow."""

    def test_content_extraction_and_analysis(self, client):
        """Test extracting content and analyzing it."""
        # Extract content
        extract_response = client.post(
            "/api/v1/content/extract",
            json={
                "text": "The sky is blue. The grass is green. Water is essential for life.",
                "title": "Test Content",
                "use_llm": False
            }
        )
        assert extract_response.status_code == status.HTTP_200_OK
        
        # Atomize claims
        atomize_response = client.post(
            "/api/v1/content/atomize",
            json={
                "text": "The sky is blue. The grass is green.",
                "use_cot": "no_chain"
            }
        )
        assert atomize_response.status_code == status.HTTP_200_OK
        data = atomize_response.json()
        assert "atomic_claims" in data


class TestPrivacyDetectionWorkflow:
    """Test complete privacy detection workflow."""

    def test_privacy_detection_and_retrieval(self, client):
        """Test detecting privacy and retrieving results."""
        # Detect privacy
        detect_response = client.post(
            "/api/v1/privacy/detect",
            json={
                "conversation_records": [
                    {"user": "My email is test@example.com"},
                    {"user": "My phone is 555-123-4567"}
                ],
                "account_id": "test-user",
                "use_few_shots": True
            }
        )
        assert detect_response.status_code == status.HTTP_200_OK
        detection_id = detect_response.json()["detection_id"]
        
        # Get detection result
        result_response = client.get(f"/api/v1/privacy/results/{detection_id}")
        assert result_response.status_code == status.HTTP_200_OK
        
        # Get statistics
        stats_response = client.get("/api/v1/privacy/stats")
        assert stats_response.status_code == status.HTTP_200_OK

