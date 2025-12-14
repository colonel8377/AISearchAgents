"""
Integration tests for Multi-Agent Debate API endpoints.

Tests the FastAPI endpoints for the debate system.
"""

import pytest
from fastapi.testclient import TestClient


# We need to create a test client without dependencies that require external services
@pytest.fixture
def client():
    """Create a test client with mocked dependencies."""
    # Import inside fixture to avoid issues
    import os
    os.environ["API_KEY_REQUIRED"] = "false"
    os.environ["OPENAI_API_KEY"] = "test-key"
    
    from src.api.main import app
    return TestClient(app)


def test_debate_init_with_auto_agents(client):
    """Test initializing a debate with auto-generated agents."""
    response = client.post(
        "/debate/init",
        json={
            "topic": "Should we adopt renewable energy?",
            "custom_personas": [],
            "auto_agent_count": 3
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "session_id" in data
    assert "agents" in data
    assert len(data["agents"]) == 3
    assert data["topic"] == "Should we adopt renewable energy?"
    
    # Check agent structure
    for agent in data["agents"]:
        assert "agent_id" in agent
        assert "role_name" in agent
        assert "system_prompt" in agent
        assert "few_shot_example" in agent


def test_debate_init_with_custom_personas(client):
    """Test initializing a debate with custom personas."""
    response = client.post(
        "/debate/init",
        json={
            "topic": "Climate change policy",
            "custom_personas": [
                {
                    "name": "Dr. Scientist",
                    "description": "A climate scientist",
                    "style": "Critical"
                },
                {
                    "name": "Ms. Economist",
                    "description": "An economist",
                    "style": "Neutral"
                }
            ],
            "auto_agent_count": 0
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert len(data["agents"]) == 2
    assert data["agents"][0]["role_name"] == "Dr. Scientist"
    assert data["agents"][1]["role_name"] == "Ms. Economist"


def test_debate_init_invalid_no_agents(client):
    """Test that initialization fails without agents."""
    response = client.post(
        "/debate/init",
        json={
            "topic": "Test topic",
            "custom_personas": [],
            "auto_agent_count": 0
        }
    )
    
    assert response.status_code == 400
    assert "custom_personas or auto_agent_count" in response.json()["detail"]


def test_agent_chat(client):
    """Test interacting with a debate agent."""
    # First, create a debate session
    init_response = client.post(
        "/debate/init",
        json={
            "topic": "AI Ethics",
            "custom_personas": [],
            "auto_agent_count": 2
        }
    )
    
    assert init_response.status_code == 200
    data = init_response.json()
    session_id = data["session_id"]
    agent_id = data["agents"][0]["agent_id"]
    
    # Now interact with the agent
    chat_response = client.post(
        f"/agent/{agent_id}/chat",
        json={
            "session_id": session_id,
            "agent_id": agent_id,
            "history_context": "Other agents have expressed concerns about AI safety."
        }
    )
    
    assert chat_response.status_code == 200
    chat_data = chat_response.json()
    
    assert "agent_id" in chat_data
    assert "verdict" in chat_data
    assert "reasoning" in chat_data


def test_agent_chat_invalid_agent(client):
    """Test that chat fails with invalid agent ID."""
    response = client.post(
        "/agent/00000000-0000-0000-0000-000000000000/chat",
        json={
            "session_id": "fake-session",
            "agent_id": "00000000-0000-0000-0000-000000000000",
            "history_context": "Test context"
        }
    )
    
    assert response.status_code == 404


def test_stability_check(client):
    """Test checking debate stability."""
    # First, create a debate session
    init_response = client.post(
        "/debate/init",
        json={
            "topic": "Technology Impact",
            "custom_personas": [],
            "auto_agent_count": 3
        }
    )
    
    assert init_response.status_code == 200
    session_id = init_response.json()["session_id"]
    
    # Check stability with first round
    response1 = client.post(
        f"/debate/{session_id}/stability_check",
        json={
            "votes": [1, 1, 2]
        }
    )
    
    assert response1.status_code == 200
    assert response1.json()["stable"] == False  # Not enough rounds yet
    
    # Add more rounds
    response2 = client.post(
        f"/debate/{session_id}/stability_check",
        json={
            "votes": [1, 1, 2]
        }
    )
    
    assert response2.status_code == 200
    assert response2.json()["stable"] == False  # Still need one more
    
    # Add third round
    response3 = client.post(
        f"/debate/{session_id}/stability_check",
        json={
            "votes": [1, 1, 2]
        }
    )
    
    assert response3.status_code == 200
    # Now we have 3 rounds, should check stability (identical votes = stable)
    assert response3.json()["stable"] == True


def test_stability_check_invalid_session(client):
    """Test that stability check fails with invalid session."""
    response = client.post(
        "/debate/fake-session-id/stability_check",
        json={
            "votes": [1, 2, 3]
        }
    )
    
    assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
