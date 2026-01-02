"""Tests for Bot Management API endpoints."""

import pytest
from fastapi import status
from conftest import client


class TestBotCreation:
    """Test bot creation endpoints."""

    def test_create_bot(self, client):
        """Test creating a bot."""
        response = client.post(
            "/api/v1/bot/create",
            json={
                "bot_name": "TestBot",
                "use_few_shots": True
            }
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "bot_id" in data
        assert data["bot_name"] == "TestBot"
        assert data["status"] == "initialized"

    def test_create_bot_without_name(self, client):
        """Test creating a bot without name."""
        response = client.post(
            "/api/v1/bot/create",
            json={"use_few_shots": True}
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "bot_id" in data
        assert "bot_name" in data


class TestBotChat:
    """Test bot chat endpoints."""

    def test_chat_with_bot_incognito(self, client):
        """Test chatting with bot in incognito mode."""
        # Create a bot first
        create_response = client.post(
            "/api/v1/bot/create",
            json={"bot_name": "ChatBot"}
        )
        bot_id = create_response.json()["bot_id"]
        
        # Chat without conversation_id (incognito mode)
        response = client.post(
            "/api/v1/bot/chat",
            json={
                "bot_id": bot_id,
                "message": "Hello, how are you?"
            }
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["bot_id"] == bot_id
        assert "response" in data
        assert data["mode"] == "incognito"
        assert data["conversation_id"] is None

    def test_chat_with_bot_conversation(self, client):
        """Test chatting with bot in conversation mode."""
        # Create a bot
        create_response = client.post(
            "/api/v1/bot/create",
            json={"bot_name": "ChatBot"}
        )
        bot_id = create_response.json()["bot_id"]
        
        # Create a conversation
        conv_response = client.post(
            f"/api/v1/bot/{bot_id}/conversation",
            json={"title": "Test Conversation"}
        )
        conversation_id = conv_response.json()["conversation_id"]
        
        # Chat with conversation_id
        response = client.post(
            "/api/v1/bot/chat",
            json={
                "bot_id": bot_id,
                "message": "Hello!",
                "conversation_id": conversation_id
            }
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["mode"] == "conversation"
        assert data["conversation_id"] == conversation_id
        assert "conversation_history" in data

    def test_chat_with_invalid_bot(self, client):
        """Test chatting with non-existent bot."""
        response = client.post(
            "/api/v1/bot/chat",
            json={
                "bot_id": "non-existent-bot-id",
                "message": "Hello"
            }
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestConversationManagement:
    """Test conversation management endpoints."""

    def test_create_conversation(self, client):
        """Test creating a conversation."""
        # Create a bot
        create_response = client.post(
            "/api/v1/bot/create",
            json={"bot_name": "TestBot"}
        )
        bot_id = create_response.json()["bot_id"]
        
        # Create conversation
        response = client.post(
            f"/api/v1/bot/{bot_id}/conversation",
            json={"title": "My Conversation"}
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "conversation_id" in data
        assert data["title"] == "My Conversation"

    def test_list_conversations(self, client):
        """Test listing conversations."""
        # Create a bot
        create_response = client.post(
            "/api/v1/bot/create",
            json={"bot_name": "TestBot"}
        )
        bot_id = create_response.json()["bot_id"]
        
        # Create a conversation
        client.post(
            f"/api/v1/bot/{bot_id}/conversation",
            json={"title": "Test Conversation"}
        )
        
        # List conversations
        response = client.get(f"/api/v1/bot/{bot_id}/conversations")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "conversations" in data
        assert "total_count" in data
        assert len(data["conversations"]) >= 1

    def test_get_conversation_detail(self, client):
        """Test getting conversation details."""
        # Create a bot
        create_response = client.post(
            "/api/v1/bot/create",
            json={"bot_name": "TestBot"}
        )
        bot_id = create_response.json()["bot_id"]
        
        # Create a conversation
        conv_response = client.post(
            f"/api/v1/bot/{bot_id}/conversation",
            json={"title": "Test Conversation"}
        )
        conversation_id = conv_response.json()["conversation_id"]
        
        # Get conversation details
        response = client.get(
            f"/api/v1/bot/{bot_id}/conversation/{conversation_id}"
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["conversation_id"] == conversation_id
        assert "turns" in data
        assert "total_turns" in data

    def test_rename_conversation(self, client):
        """Test renaming a conversation."""
        # Create a bot
        create_response = client.post(
            "/api/v1/bot/create",
            json={"bot_name": "TestBot"}
        )
        bot_id = create_response.json()["bot_id"]
        
        # Create a conversation
        conv_response = client.post(
            f"/api/v1/bot/{bot_id}/conversation",
            json={"title": "Old Title"}
        )
        conversation_id = conv_response.json()["conversation_id"]
        
        # Rename conversation
        response = client.put(
            f"/api/v1/bot/{bot_id}/conversation/{conversation_id}",
            json={"title": "New Title"}
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["new_title"] == "New Title"

    def test_delete_conversation(self, client):
        """Test deleting a conversation."""
        # Create a bot
        create_response = client.post(
            "/api/v1/bot/create",
            json={"bot_name": "TestBot"}
        )
        bot_id = create_response.json()["bot_id"]
        
        # Create a conversation
        conv_response = client.post(
            f"/api/v1/bot/{bot_id}/conversation",
            json={"title": "Test Conversation"}
        )
        conversation_id = conv_response.json()["conversation_id"]
        
        # Delete conversation
        response = client.delete(
            f"/api/v1/bot/{bot_id}/conversation/{conversation_id}"
        )
        assert response.status_code == status.HTTP_200_OK
        
        # Verify conversation is deleted
        get_response = client.get(
            f"/api/v1/bot/{bot_id}/conversation/{conversation_id}"
        )
        assert get_response.status_code == status.HTTP_404_NOT_FOUND


class TestBotListing:
    """Test bot listing endpoints."""

    def test_list_bots(self, client):
        """Test listing all bots."""
        # Create a few bots
        for i in range(2):
            client.post(
                "/api/v1/bot/create",
                json={"bot_name": f"Bot{i}"}
            )
        
        # List bots
        response = client.get("/api/v1/bot/list")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "bots" in data
        assert "total_count" in data
        assert len(data["bots"]) >= 2

