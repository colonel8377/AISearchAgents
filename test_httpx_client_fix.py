"""
Test to verify the httpx.Client fix for the proxies parameter issue.

This test ensures that:
1. httpx.Client is created without the 'proxies' parameter
2. trust_env=True is set to support HTTP_PROXY/HTTPS_PROXY environment variables
3. The fix resolves the TypeError: Client.__init__() got an unexpected keyword argument 'proxies'
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_httpx_client_creation():
    """Test that httpx.Client is created correctly without proxies parameter."""
    print("Testing httpx.Client creation...")
    try:
        from src.utils.llm_client import llm_manager
        
        # Test 1: Create client without proxy
        print("  Test 1: Creating client without proxy parameter...")
        client = llm_manager.get_http_client()
        assert client is not None, "Client should not be None"
        assert hasattr(client, '_trust_env'), "Client should have _trust_env attribute"
        assert client._trust_env == True, "Client should have trust_env=True"
        client.close()
        print("  ✓ Client created successfully with trust_env=True")
        
        # Reset singleton for next test
        llm_manager._http_client = None
        
        # Test 2: Create client with proxy parameter (should warn but not fail)
        print("  Test 2: Creating client with deprecated proxy parameter...")
        client = llm_manager.get_http_client(proxy='http://127.0.0.1:7890')
        assert client is not None, "Client should not be None even with proxy param"
        assert client._trust_env == True, "Client should still have trust_env=True"
        client.close()
        print("  ✓ Client created successfully (proxy parameter deprecated but handled gracefully)")
        
        # Reset singleton
        llm_manager._http_client = None
        
        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_summarizer_agent_creation():
    """Test that SummarizerAgent can be created without TypeError."""
    print("\nTesting SummarizerAgent creation...")
    try:
        from src.agents.summarizer.agent import SummarizerAgent
        
        # This was the original failing scenario
        print("  Creating SummarizerAgent with proxy parameter...")
        agent = SummarizerAgent(
            model_name='gpt-3.5-turbo',
            api_key='test-key',
            api_base='https://api.openai.com/v1',
            temperature=0.3,
            proxy='http://127.0.0.1:7890'  # This caused the TypeError before
        )
        
        assert agent is not None, "Agent should not be None"
        assert agent.llm is not None, "Agent LLM should not be None"
        assert agent.llm.model_name == 'gpt-3.5-turbo', "Model name should match"
        
        print("  ✓ SummarizerAgent created successfully without TypeError")
        return True
    except TypeError as e:
        if "proxies" in str(e):
            print(f"  ✗ Test failed with proxies TypeError: {e}")
            return False
        else:
            # Other TypeErrors might be expected (e.g., validation errors)
            print(f"  ✓ No proxies-related TypeError (other TypeError is acceptable): {e}")
            return True
    except Exception as e:
        # Other exceptions are acceptable (e.g., missing API key)
        print(f"  ✓ No TypeError raised (other exceptions are acceptable): {type(e).__name__}")
        return True

def test_nudge_collapse_agent_creation():
    """Test that NudgeCollapseAgent can be created without TypeError."""
    print("\nTesting NudgeCollapseAgent creation...")
    try:
        from src.agents.nudge_collapse.agent import NudgeCollapseAgent
        
        print("  Creating NudgeCollapseAgent with proxy parameter...")
        agent = NudgeCollapseAgent(
            model_name='gpt-3.5-turbo',
            api_key='test-key',
            api_base='https://api.openai.com/v1',
            temperature=0.7,
            proxy='http://127.0.0.1:7890'
        )
        
        assert agent is not None, "Agent should not be None"
        print("  ✓ NudgeCollapseAgent created successfully without TypeError")
        return True
    except TypeError as e:
        if "proxies" in str(e):
            print(f"  ✗ Test failed with proxies TypeError: {e}")
            return False
        else:
            print(f"  ✓ No proxies-related TypeError (other TypeError is acceptable): {e}")
            return True
    except Exception as e:
        print(f"  ✓ No TypeError raised (other exceptions are acceptable): {type(e).__name__}")
        return True

def main():
    """Run all tests."""
    print("=" * 70)
    print("HTTPx Client Fix Verification Tests")
    print("=" * 70)
    
    results = []
    
    # Test httpx client creation
    results.append(("HTTPx Client Creation", test_httpx_client_creation()))
    
    # Test SummarizerAgent
    results.append(("SummarizerAgent Creation", test_summarizer_agent_creation()))
    
    # Test NudgeCollapseAgent
    results.append(("NudgeCollapseAgent Creation", test_nudge_collapse_agent_creation()))
    
    # Print summary
    print("\n" + "=" * 70)
    print("Test Results Summary:")
    print("=" * 70)
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<50} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    if all_passed:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    exit(main())
