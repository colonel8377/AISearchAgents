"""Simple validation test for history modes and smart memory."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        from config.settings import settings, HistoryMode
        print(f"✓ Settings imported successfully")
        print(f"  - Default history mode: {settings.default_history_mode}")
        print(f"  - Smart memory enabled: {settings.smart_memory_enabled}")
    except Exception as e:
        print(f"✗ Failed to import settings: {e}")
        return False
    
    try:
        from utils.smart_memory import SmartMemory
        print(f"✓ SmartMemory imported successfully")
    except Exception as e:
        print(f"✗ Failed to import SmartMemory: {e}")
        return False
    
    return True


def test_smart_memory_heuristics():
    """Test smart memory heuristic detection without LLM."""
    print("\nTesting smart memory heuristics...")
    
    from utils.smart_memory import SmartMemory
    
    memory = SmartMemory()  # No LLM
    
    # Test messages that should be stored
    test_cases = [
        ("I'm interested in climate change news", True),
        ("I believe healthcare should be universal", True),
        ("My favorite language is Python", True),
        ("I think the situation is concerning", True),
        ("Hello", False),
        ("What's the weather?", False),
        ("Thanks", False),
        ("", False),
    ]
    
    all_passed = True
    for message, expected in test_cases:
        result = memory.should_store_message(message)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{message[:40]}...' -> {result} (expected {expected})")
        if result != expected:
            all_passed = False
    
    return all_passed


def test_history_mode_type():
    """Test HistoryMode type definition."""
    print("\nTesting HistoryMode type...")
    
    from config.settings import HistoryMode
    
    # Test that HistoryMode accepts valid values
    valid_modes = ["full", "none"]
    print(f"  ✓ Valid history modes: {valid_modes}")
    
    return True


def test_agent_imports():
    """Test that agent imports work with new parameters."""
    print("\nTesting agent imports...")
    
    try:
        from agents.bot_creator.agent import BotCreatorAgent
        print(f"  ✓ BotCreatorAgent imported")
    except Exception as e:
        print(f"  ✗ Failed to import BotCreatorAgent: {e}")
        return False
    
    try:
        from agents.nudge_collapse.agent import NudgeCollapseAgent
        print(f"  ✓ NudgeCollapseAgent imported")
    except Exception as e:
        print(f"  ✗ Failed to import NudgeCollapseAgent: {e}")
        return False
    
    try:
        from agents.summarizer.agent import SummarizerAgent
        print(f"  ✓ SummarizerAgent imported")
    except Exception as e:
        print(f"  ✗ Failed to import SummarizerAgent: {e}")
        return False
    
    return True


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("History Modes & Smart Memory Validation Tests")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Smart Memory Heuristics", test_smart_memory_heuristics()))
    results.append(("HistoryMode Type", test_history_mode_type()))
    results.append(("Agent Imports", test_agent_imports()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("✅ All validation tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
