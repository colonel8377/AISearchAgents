"""Pytest configuration and fixtures for AI Search Agents Platform tests."""

import pytest
import os
import sys
from pathlib import Path
from typing import Generator
from unittest.mock import Mock, MagicMock

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

# Set environment variables before imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["API_KEY_REQUIRED"] = "false"
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["VECTOR_STORE_TYPE"] = "chroma"
os.environ["ENABLE_PERSISTENCE"] = "false"

try:
    from src.shared.config.settings import Settings
    from src.presentation.api.main import app
    from fastapi.testclient import TestClient
except ImportError:
    # Fallback for test environment
    import sys
    sys.path.insert(0, str(BASE_DIR / "src"))
    from shared.config.settings import Settings
    from presentation.api.main import app
    from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Create test settings with minimal configuration."""
    return Settings(
        api_key_required=False,
        openai_api_key="test-key",
        vector_store_type="chroma",
        enable_persistence=False,
        cache_backend="local",
        use_llm_cache=False,
    )


@pytest.fixture(scope="function")
def client() -> Generator[TestClient, None, None]:
    """Create a test client for FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="function")
def mock_llm_response():
    """Mock LLM response for testing."""
    return {
        "choices": [{
            "message": {
                "content": "Test response from LLM",
                "role": "assistant"
            }
        }]
    }


@pytest.fixture(scope="function")
def mock_agent_manager():
    """Mock agent manager for testing."""
    manager = MagicMock()
    manager.create_agent = MagicMock(return_value="test-agent-id")
    manager.get_agent = MagicMock(return_value=MagicMock())
    manager.list_agents = MagicMock(return_value={})
    return manager


@pytest.fixture(scope="function")
def mock_bot_manager():
    """Mock bot manager for testing."""
    manager = MagicMock()
    manager.create_bot = MagicMock(return_value={
        "bot_id": "test-bot-id",
        "bot_name": "TestBot",
        "status": "initialized"
    })
    manager.chat_with_bot = MagicMock(return_value={
        "response": "Test response",
        "mode": "incognito"
    })
    return manager


@pytest.fixture(scope="function")
def sample_agent_data():
    """Sample agent data for testing."""
    return {
        "agent_id": "test-agent-123",
        "agent_type": "nudge_collapse",
        "status": "active",
        "current_turn": 0
    }


@pytest.fixture(scope="function")
def sample_bot_data():
    """Sample bot data for testing."""
    return {
        "bot_id": "test-bot-123",
        "bot_name": "TestBot",
        "status": "initialized",
        "conversations_count": 0
    }


@pytest.fixture(scope="function")
def sample_conversation_data():
    """Sample conversation data for testing."""
    return {
        "conversation_id": "test-conv-123",
        "bot_id": "test-bot-123",
        "title": "Test Conversation",
        "turns": []
    }


@pytest.fixture(scope="function")
def sample_content_extraction_request():
    """Sample content extraction request."""
    return {
        "url": "https://example.com/article",
        "use_llm": False,
        "compare_claims": False
    }


@pytest.fixture(scope="function")
def sample_privacy_detection_request():
    """Sample privacy detection request."""
    return {
        "conversation_records": [
            {"user": "My email is test@example.com"},
            {"user": "My phone is 555-123-4567"}
        ],
        "use_few_shots": True
    }


@pytest.fixture(scope="function")
def sample_opinion_extraction_request():
    """Sample opinion extraction request."""
    return {
        "url": "https://example.com/article",
        "use_llm": True,
        "use_cot": "no_chain"
    }


@pytest.fixture(scope="function")
def sample_debate_init_request():
    """Sample debate initialization request."""
    return {
        "topic": "Should AI be regulated?",
        "auto_agent_count": 3,
        "max_rounds": 5
    }


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment before each test."""
    # Clean up any test data
    yield
    # Cleanup after test if needed
    pass
