"""Tests for Summarizer API endpoints."""

import pytest
from fastapi import status
from conftest import client


class TestSummarizer:
    """Test summarizer endpoints."""

    def test_summarize_conversation(self, client):
        """Test summarizing a conversation."""
        # Create a summarizer agent first
        create_response = client.post(
            "/api/v1/agents/create",
            json={"agent_type": "summarizer"}
        )
        agent_id = create_response.json()["agent_id"]
        
        # Summarize conversation
        response = client.post(
            f"/api/v1/agent/{agent_id}/summarizer/summarize",
            json={
                "conversation_records": [
                    {"user": "What is AI?", "assistant": "AI is artificial intelligence."},
                    {"user": "How does it work?", "assistant": "It uses machine learning algorithms."}
                ],
                "use_cot": "chain_local",
                "use_few_shots": True
            }
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "summary" in data
        assert "conversation_length" in data

    def test_summarize_conversation_invalid_agent(self, client):
        """Test summarizing with invalid agent ID."""
        response = client.post(
            "/api/v1/agent/non-existent-id/summarizer/summarize",
            json={
                "conversation_records": [
                    {"user": "Test", "assistant": "Response"}
                ]
            }
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST

