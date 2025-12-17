"""
Example usage of the extractandclean API endpoint.

This demonstrates how the endpoint saves tokens by processing HTML server-side.
Proxy configuration is handled via OPENAI_PROXY setting or environment variables.
"""

import requests
import json


def example_extractandclean():
    """
    Example: Use the extractandclean endpoint.
    
    This single endpoint handles both HTML extraction and cleaning.
    Proxy configuration is handled via OPENAI_PROXY setting or 
    HTTP_PROXY/HTTPS_PROXY environment variables.
    """
    url = "http://localhost:8000/api/v1/web-opinion/extractandclean"
    
    # Basic request
    payload = {
        "url": "https://example.com/article"
    }
    
    headers = {
        "Content-Type": "application/json",
        # "X-API-Key": "your-api-key-here"  # Uncomment if authentication is enabled
    }
    
    print("=== Example: Extract and Clean ===")
    print(f"Requesting: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print()
    print("Note: To use a proxy, configure OPENAI_PROXY in .env or set")
    print("      HTTP_PROXY/HTTPS_PROXY environment variables")
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
    - Server-side processing with BeautifulSoup
    - Token efficient
    - Proxy support via settings/environment variables
    """
    print("\n=== API Approach ===")
    print("\nSINGLE ENDPOINT:")
    print("  POST /extractandclean")
    print("    Request: ~50 tokens (URL)")
    print("    Response: ~500 tokens (cleaned text)")
    print("  TOTAL: ~550 tokens")
    print()
    print("FEATURES:")
    print("  - Single API call")
    print("  - Proxy via OPENAI_PROXY or HTTP_PROXY env vars")
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
