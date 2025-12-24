"""Tests for history mode and smart memory functionality."""

import pytest
from src.agents.bot_creator.agent import BotCreatorAgent
from src.agents.nudge_collapse.agent import NudgeCollapseAgent
from src.config.settings import settings


def test_bot_creator_history_mode_full():
    """Test BotCreatorAgent with full history mode."""
    agent = BotCreatorAgent(
        model_name=settings.openai_model,
        api_key=settings.openai_api_key,
        api_base=settings.openai_api_base,
        temperature=0.5
    )
    
    # Create a bot
    bot_result = agent.create_bot(
        persona_prompt="You are a helpful assistant that answers questions about technology."
    )
    assert "bot_id" in bot_result
    bot_id = bot_result["bot_id"]
    
    # First chat
    result1 = agent.chat_with_bot(
        bot_id=bot_id,
        user_message="What is Python?",
        conversation_history=None,
        history_mode=True
    )
    
    assert "response" in result1
    assert result1["history_mode"] == True
    assert len(result1["conversation_history"]) == 2  # User + assistant
    
    # Second chat with history
    result2 = agent.chat_with_bot(
        bot_id=bot_id,
        user_message="What are its main uses?",
        conversation_history=result1["conversation_history"],
        history_mode=True
    )
    
    assert "response" in result2
    assert len(result2["conversation_history"]) == 4  # 2 previous + 2 new
    print(f"✓ Full history mode test passed: {len(result2['conversation_history'])} messages in history")


def test_bot_creator_history_mode_none():
    """Test BotCreatorAgent with none history mode (stateless)."""
    agent = BotCreatorAgent(
        model_name=settings.openai_model,
        api_key=settings.openai_api_key,
        api_base=settings.openai_api_base,
        temperature=0.5
    )
    
    # Create a bot
    bot_result = agent.create_bot(
        persona_prompt="You are a helpful assistant that answers questions about technology."
    )
    assert "bot_id" in bot_result
    bot_id = bot_result["bot_id"]
    
    # First chat
    result1 = agent.chat_with_bot(
        bot_id=bot_id,
        user_message="What is Python?",
        conversation_history=None,
        history_mode=False
    )
    
    assert "response" in result1
    assert result1["history_mode"] == False
    assert len(result1["conversation_history"]) == 0  # No history when False
    
    # Second chat should also have no history
    result2 = agent.chat_with_bot(
        bot_id=bot_id,
        user_message="What are its main uses?",
        conversation_history=result1["conversation_history"],
        history_mode=False
    )
    
    assert "response" in result2
    assert len(result2["conversation_history"]) == 0  # Still no history
    print(f"✓ Stateless (none) mode test passed: history remains empty")


def test_nudge_collapse_history_mode_full():
    """Test NudgeCollapseAgent with full history mode."""
    agent = NudgeCollapseAgent(
        model_name=settings.openai_model,
        api_key=settings.openai_api_key,
        api_base=settings.openai_api_base,
        temperature=0.5
    )
    
    # First turn
    result1 = agent.generate_turn(
        user_query="What is climate change?",
        search_summary="Climate change refers to long-term shifts in temperatures.",
        history_mode=True
    )
    
    assert "response" in result1
    assert result1["history_mode"] == True
    assert result1["turn"] == 0
    
    # Second turn - should include history
    result2 = agent.generate_turn(
        user_query="What causes it?",
        search_summary="Main causes include greenhouse gas emissions.",
        history_mode=True
    )
    
    assert "response" in result2
    assert result2["turn"] == 1
    print(f"✓ NudgeCollapse full history mode test passed: turn {result2['turn']}")


def test_nudge_collapse_history_mode_none():
    """Test NudgeCollapseAgent with none history mode (stateless)."""
    agent = NudgeCollapseAgent(
        model_name=settings.openai_model,
        api_key=settings.openai_api_key,
        api_base=settings.openai_api_base,
        temperature=0.5
    )
    
    # First turn
    result1 = agent.generate_turn(
        user_query="What is climate change?",
        search_summary="Climate change refers to long-term shifts in temperatures.",
        history_mode=False
    )
    
    assert "response" in result1
    assert result1["history_mode"] == False
    
    # Second turn - should not use history from previous turn
    result2 = agent.generate_turn(
        user_query="What causes it?",
        search_summary="Main causes include greenhouse gas emissions.",
        history_mode=False
    )
    
    assert "response" in result2
    # Agent still tracks internally, but doesn't send to LLM
    print(f"✓ NudgeCollapse stateless mode test passed")


def test_smart_memory_detection():
    """Test smart memory heuristic detection."""
    from src.utils.smart_memory import SmartMemory
    
    memory = SmartMemory()
    
    # Test messages that should be stored
    assert memory.should_store_message("I'm really interested in climate change news")
    assert memory.should_store_message("I believe healthcare should be universal")
    assert memory.should_store_message("My favorite programming language is Python")
    assert memory.should_store_message("I think the political situation is concerning")
    
    # Test messages that should not be stored
    assert not memory.should_store_message("Hello")
    assert not memory.should_store_message("What's the weather?")
    assert not memory.should_store_message("Thanks")
    
    print(f"✓ Smart memory detection test passed")


if __name__ == "__main__":
    print("Running history mode and smart memory tests...")
    print()
    
    # Note: These tests require valid API credentials
    # Skip if not available
    if not settings.openai_api_key:
        print("⚠ Skipping tests - OPENAI_API_KEY not set")
    else:
        try:
            test_bot_creator_history_mode_full()
            test_bot_creator_history_mode_none()
            test_nudge_collapse_history_mode_full()
            test_nudge_collapse_history_mode_none()
            test_smart_memory_detection()
            print()
            print("✅ All tests passed!")
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
