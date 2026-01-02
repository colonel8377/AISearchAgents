"""Tests for Agent Management API endpoints."""

import pytest
from fastapi import status
from conftest import client, sample_agent_data


class TestAgentCreation:
    """Test agent creation endpoints."""

    def test_create_nudge_collapse_agent(self, client):
        """Test creating a nudge_collapse agent."""
        response = client.post(
            "/api/v1/agents/create",
            json={
                "agent_type": "nudge_collapse",
                "use_memory": False
            }
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "agent_id" in data
        assert data["agent_type"]["value"] == "nudge_collapse"
        assert data["status"] == "created"

    def test_create_summarizer_agent(self, client):
        """Test creating a summarizer agent."""
        response = client.post(
            "/api/v1/agents/create",
            json={
                "agent_type": "summarizer",
                "use_memory": True
            }
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "agent_id" in data
        assert data["agent_type"]["value"] == "summarizer"

    def test_create_bot_creator_agent(self, client):
        """Test creating a bot_creator agent."""
        response = client.post(
            "/api/v1/agents/create",
            json={
                "agent_type": "bot_creator",
                "persona_mode": "system_prompt"
            }
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "agent_id" in data
        assert data["agent_type"]["value"] == "bot_creator"

    def test_create_agent_with_custom_id(self, client):
        """Test creating an agent with custom ID."""
        custom_id = "custom-agent-123"
        response = client.post(
            "/api/v1/agents/create",
            json={
                "agent_type": "nudge_collapse",
                "agent_id": custom_id
            }
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["agent_id"] == custom_id

    def test_create_agent_invalid_type(self, client):
        """Test creating an agent with invalid type."""
        response = client.post(
            "/api/v1/agents/create",
            json={
                "agent_type": "invalid_type"
            }
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST


class TestAgentListing:
    """Test agent listing endpoints."""

    def test_list_agents_empty(self, client):
        """Test listing agents when none exist."""
        response = client.get("/api/v1/agents/list")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "agents" in data
        assert "total_count" in data
        assert isinstance(data["agents"], list)

    def test_list_agents_after_creation(self, client):
        """Test listing agents after creating some."""
        # Create a few agents
        for i in range(3):
            client.post(
                "/api/v1/agents/create",
                json={"agent_type": "nudge_collapse"}
            )
        
        response = client.get("/api/v1/agents/list")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total_count"] >= 3


class TestAgentStatus:
    """Test agent status endpoints."""

    def test_get_agent_status(self, client):
        """Test getting agent status."""
        # Create an agent first
        create_response = client.post(
            "/api/v1/agents/create",
            json={"agent_type": "nudge_collapse"}
        )
        agent_id = create_response.json()["agent_id"]
        
        # Get status
        response = client.get(f"/api/v1/agents/{agent_id}/status")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["agent_id"] == agent_id
        assert "status" in data
        assert "agent_type" in data

    def test_get_agent_status_not_found(self, client):
        """Test getting status for non-existent agent."""
        response = client.get("/api/v1/agents/non-existent-id/status")
        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestAgentReset:
    """Test agent reset endpoints."""

    def test_reset_agent(self, client):
        """Test resetting an agent."""
        # Create an agent
        create_response = client.post(
            "/api/v1/agents/create",
            json={"agent_type": "nudge_collapse"}
        )
        agent_id = create_response.json()["agent_id"]
        
        # Reset agent
        response = client.post(
            f"/api/v1/agents/{agent_id}/reset",
            json={
                "reset_conversation": True,
                "clear_memory": False
            }
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "success"

    def test_reset_agent_not_found(self, client):
        """Test resetting non-existent agent."""
        response = client.post(
            "/api/v1/agents/non-existent-id/reset",
            json={"reset_conversation": True}
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestAgentDeletion:
    """Test agent deletion endpoints."""

    def test_delete_agent(self, client):
        """Test deleting an agent."""
        # Create an agent
        create_response = client.post(
            "/api/v1/agents/create",
            json={"agent_type": "nudge_collapse"}
        )
        agent_id = create_response.json()["agent_id"]
        
        # Delete agent
        response = client.delete(f"/api/v1/agents/{agent_id}")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "deleted"
        
        # Verify agent is deleted
        status_response = client.get(f"/api/v1/agents/{agent_id}/status")
        assert status_response.status_code == status.HTTP_404_NOT_FOUND

    def test_delete_agent_not_found(self, client):
        """Test deleting non-existent agent."""
        response = client.delete("/api/v1/agents/non-existent-id")
        assert response.status_code == status.HTTP_404_NOT_FOUND

