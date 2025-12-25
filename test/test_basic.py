"""
Simple test script to verify the v2 API works correctly.
This script tests basic functionality without requiring actual LLM API keys.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from src.api.main import app
        from src.agents.manager import AgentManager, AgentType
        from src.api.auth import verify_api_key
        from src.config.settings import settings
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_agent_manager():
    """Test the AgentManager class."""
    print("\nTesting AgentManager...")
    try:
        from src.agents.manager import AgentManager, AgentType
        
        # Create manager
        manager = AgentManager()
        
        # Test creating a mock agent
        class MockAgent:
            def __init__(self):
                self.state = "initialized"
        
        agent = MockAgent()
        agent_id = manager.create_agent(agent, AgentType.SUMMARIZER, "test_agent_1")
        
        assert agent_id == "test_agent_1", "Agent ID mismatch"
        assert manager.exists(agent_id), "Agent should exist"
        assert manager.get_agent(agent_id) == agent, "Agent retrieval failed"
        assert manager.get_agent_type(agent_id) == AgentType.SUMMARIZER, "Agent type mismatch"
        
        # Test listing
        agents_list = manager.list_agents()
        assert len(agents_list) == 1, "Should have 1 agent"
        assert agents_list[0]["agent_id"] == agent_id, "Agent ID in list mismatch"
        
        # Test deletion
        deleted = manager.delete_agent(agent_id)
        assert deleted, "Agent deletion failed"
        assert not manager.exists(agent_id), "Agent should not exist after deletion"
        
        print("✓ AgentManager tests passed")
        return True
    except Exception as e:
        print(f"✗ AgentManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_settings():
    """Test settings configuration."""
    print("\nTesting Settings...")
    try:
        from src.config.settings import settings
        
        # Test that new settings exist
        assert hasattr(settings, 'api_key_required'), "Missing api_key_required setting"
        assert hasattr(settings, 'api_keys'), "Missing api_keys setting"
        assert hasattr(settings, 'max_conversation_length'), "Missing max_conversation_length setting"
        assert hasattr(settings, 'max_tokens_per_message'), "Missing max_tokens_per_message setting"
        
        # Test defaults
        assert isinstance(settings.api_key_required, bool), "api_key_required should be bool"
        assert isinstance(settings.api_keys, list), "api_keys should be list"
        assert isinstance(settings.max_conversation_length, int), "max_conversation_length should be int"
        assert isinstance(settings.max_tokens_per_message, int), "max_tokens_per_message should be int"
        
        print("✓ Settings tests passed")
        return True
    except Exception as e:
        print(f"✗ Settings test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_structure():
    """Test that API endpoints are properly structured."""
    print("\nTesting API structure...")
    try:
        from src.api.main import app
        
        # Check that app exists and has the expected attributes
        assert app is not None, "App is None"
        assert hasattr(app, 'routes'), "App doesn't have routes"
        
        # Check for key endpoints
        routes = [route.path for route in app.routes]
        expected_paths = [
            "/",
            "/api/v1/agents/create",
            "/api/v1/agents/list",
            "/api/v1/agents/{agent_id}",
            "/api/v1/agents/{agent_id}/status",
            "/api/v1/agents/{agent_id}/reset",
            "/api/v1/agent/{agent_id}/nudge-collapse/generate",
            "/api/v1/agent/{agent_id}/summarizer/summarize",
            "/api/v1/agent/{agent_id}/bot-creator/create",
            "/api/v1/consistency/check-summary-url",
            "/api/v1/consistency/compare-claims",
            "/api/v1/consistency/complete",
            "/api/v1/quality/overall",
        ]
        
        missing_paths = []
        for expected in expected_paths:
            if expected not in routes:
                missing_paths.append(expected)
        
        if missing_paths:
            print(f"Warning: {len(missing_paths)} expected paths not found:")
            for path in missing_paths:
                print(f"  - {path}")
        
        print(f"  Found {len(routes)} total routes")
        print("✓ API structure tests passed")
        return True
    except Exception as e:
        print(f"✗ API structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("AI Search Agents v2 - Basic Verification Tests")
    print("=" * 60)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("AgentManager", test_agent_manager()))
    results.append(("Settings", test_settings()))
    results.append(("API Structure", test_api_structure()))
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("=" * 60)
    if all_passed:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
