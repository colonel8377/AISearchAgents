"""
Example usage of the extractandclean API endpoint.

This demonstrates how the endpoint saves tokens and supports optional proxy configuration.
"""

import requests
import json


def example_extractandclean():
    """
    Example: Use the extractandclean endpoint.
    
    This single endpoint handles both HTML extraction and cleaning,
    with optional proxy support for fetching and LLM requests.
    """
    url = "http://localhost:8000/api/v1/web-opinion/extractandclean"
    
    # Basic request
    payload = {
        "url": "https://example.com/article"
    }
    
    # With custom proxy
    payload_with_proxy = {
        "url": "https://example.com/article",
        "proxy": "http://proxy.example.com:8080"
    }
    
    headers = {
        "Content-Type": "application/json",
        # "X-API-Key": "your-api-key-here"  # Uncomment if authentication is enabled
    }
    
    print("=== Example: Extract and Clean ===")
    print(f"Requesting: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print()
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get("error"):
                print(f"Error: {data.get('error')}")
                print(f"Message: {data.get('error_message')}")
            else:
                print(f"✓ Success!")
                print(f"URL: {data['url']}")
                print(f"Title: {data['title']}")
                print(f"Text length: {data['text_length']} characters")
                print(f"\nFirst 200 characters of text:")
                print(f"{data['text'][:200]}...")
        else:
            print(f"HTTP Error: {response.status_code}")
            print(response.text)
    
    except requests.RequestException as e:
        print(f"Request failed: {e}")


def compare_approaches():
    """
    Show the efficiency of the single endpoint approach.
    
    SINGLE ENDPOINT APPROACH:
    1. POST /extractandclean with URL → returns cleaned text directly
    
    Benefits:
    - Single API call
    - Optional proxy support for fetching and LLM
    - Server-side processing with BeautifulSoup
    - Token efficient
    """
    print("\n=== API Approach ===")
    print("\nSINGLE ENDPOINT:")
    print("  POST /extractandclean")
    print("    Request: ~50 tokens (URL + optional proxy)")
    print("    Response: ~500 tokens (cleaned text)")
    print("  TOTAL: ~550 tokens")
    print()
    print("FEATURES:")
    print("  - Single API call")
    print("  - Optional proxy parameter")
    print("  - Server-side BeautifulSoup filtering")
    print("  - No HTML transmission required")
    print("\nNote: HTML is processed server-side with BeautifulSoup,")
    print("      avoiding the need to transmit it to the client.")


if __name__ == "__main__":
    # Show the API approach
    compare_approaches()
    
    print("\n" + "="*60 + "\n")
    
    # Example API call (requires server to be running)
    print("To test the API, start the server with:")
    print("  uvicorn src.api.main:app --reload")
    print("\nThen run this script again to make the API call.")
    print()
    
    # Uncomment to actually make the API call:
    # example_extractandclean()
