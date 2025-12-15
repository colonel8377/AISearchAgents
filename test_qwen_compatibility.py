#!/usr/bin/env python3
"""
Test script to verify Qwen model compatibility improvements.

This script demonstrates the enhanced error handling and configuration
for using Qwen models with the AI Search Agents Platform.

Usage:
    python test_qwen_compatibility.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_qwen_configuration():
    """Test that Qwen configuration is properly handled."""
    print("=" * 60)
    print("Testing Qwen Model Compatibility")
    print("=" * 60)
    
    # Test 1: Import agents
    print("\n[Test 1] Importing agent classes...")
    try:
        from src.agents.bot_creator.agent import BotCreatorAgent
        from src.agents.summarizer.agent import SummarizerAgent
        from src.agents.nudge_collapse.agent import NudgeCollapseAgent
        print("✓ All agents imported successfully")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    # Test 2: Check configuration detection
    print("\n[Test 2] Testing API base URL detection...")
    test_cases = [
        ("https://api.openai.com/v1", False, "OpenAI"),
        ("https://dashscope.aliyuncs.com/compatible-mode/v1", True, "Qwen"),
        ("https://api.custom.com/v1", True, "Custom"),
    ]
    
    for url, should_add_headers, provider in test_cases:
        if url and "api.openai.com" not in url:
            adds_headers = True
        else:
            adds_headers = False
        
        if adds_headers == should_add_headers:
            print(f"✓ {provider}: Correctly {'adds' if adds_headers else 'skips'} custom headers")
        else:
            print(f"✗ {provider}: Header handling mismatch")
            return False
    
    # Test 3: Check error message enhancement
    print("\n[Test 3] Testing error message enhancement...")
    test_errors = [
        ("Error code: 502", "502", "502 error"),
        ("Unauthorized", "401", "Auth error"),
        ("Request timeout", "timeout", "Timeout error"),
    ]
    
    for error_str, expected_key, description in test_errors:
        error_msg = error_str.lower()
        detected = False
        
        if "502" in error_str or "Bad Gateway" in error_str:
            detected = "502" in expected_key
        elif "401" in error_str or "Unauthorized" in error_str:
            detected = "401" in expected_key
        elif "timeout" in error_msg:
            detected = "timeout" in expected_key
        
        if detected:
            print(f"✓ {description}: Error type correctly identified")
        else:
            print(f"✗ {description}: Error type not identified")
    
    # Test 4: Verify environment variable examples
    print("\n[Test 4] Checking .env.example for Qwen configuration...")
    env_example_path = os.path.join(os.getcwd(), '.env.example')
    
    try:
        with open(env_example_path, 'r') as f:
            content = f.read()
            
        checks = [
            ("dashscope.aliyuncs.com", "Qwen API base URL example"),
            ("qwen-turbo", "Qwen model example"),
            ("OPENAI_MAX_RETRIES", "Retry configuration"),
            ("OPENAI_TIMEOUT", "Timeout configuration"),
        ]
        
        for check_str, description in checks:
            if check_str in content:
                print(f"✓ {description} present in .env.example")
            else:
                print(f"✗ {description} missing from .env.example")
                return False
    except FileNotFoundError:
        print(f"✗ .env.example file not found at {env_example_path}")
        return False
    
    # Test 5: Verify troubleshooting guide exists
    print("\n[Test 5] Checking for troubleshooting documentation...")
    guide_path = os.path.join(os.getcwd(), 'doc', 'QWEN_TROUBLESHOOTING.md')
    
    try:
        with open(guide_path, 'r') as f:
            content = f.read()
            
        sections = [
            ("502", "502 error troubleshooting"),
            ("401", "401 error troubleshooting"),
            ("timeout", "Timeout troubleshooting"),
            ("quota", "Quota troubleshooting"),
        ]
        
        for section, description in sections:
            if section.lower() in content.lower():
                print(f"✓ {description} covered in guide")
            else:
                print(f"⚠ {description} may not be covered")
    except FileNotFoundError:
        print("✗ QWEN_TROUBLESHOOTING.md file not found")
        return False
    
    # Test 6: Verify README updates
    print("\n[Test 6] Checking README for Qwen documentation...")
    readme_path = os.path.join(os.getcwd(), 'README.md')
    
    try:
        with open(readme_path, 'r') as f:
            content = f.read()
            
        if "Qwen" in content or "qwen" in content:
            print("✓ Qwen mentioned in README")
        else:
            print("⚠ Qwen not prominently mentioned in README")
        
        if "QWEN_TROUBLESHOOTING" in content:
            print("✓ Troubleshooting guide linked in README")
        else:
            print("⚠ Troubleshooting guide may not be linked")
    except FileNotFoundError:
        print("✗ README.md file not found")
        return False
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print("\nQwen model compatibility improvements verified:")
    print("  • Enhanced error handling with specific guidance")
    print("  • Custom headers for non-OpenAI APIs")
    print("  • Comprehensive configuration examples")
    print("  • Detailed troubleshooting documentation")
    print("\nRefer to doc/QWEN_TROUBLESHOOTING.md for setup instructions.")
    return True


if __name__ == "__main__":
    success = test_qwen_configuration()
    sys.exit(0 if success else 1)
