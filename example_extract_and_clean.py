"""
Example usage of the new combined extract-and-clean API endpoint.

This demonstrates how the new endpoint saves tokens by combining
HTML extraction and cleaning into a single API call.
"""

import requests
import json


def example_extract_and_clean():
    """
    Example: Use the combined extract-and-clean endpoint.
    
    This is more efficient than calling extract-html followed by clean-html
    because it avoids passing large HTML content between API calls.
    """
    url = "http://localhost:8000/api/v1/web-opinion/extract-and-clean"
    
    payload = {
        "url": "https://example.com/article"
    }
    
    headers = {
        "Content-Type": "application/json",
        # "X-API-Key": "your-api-key-here"  # Uncomment if authentication is enabled
    }
    
    print("=== Example: Combined Extract and Clean ===")
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


def compare_old_vs_new_approach():
    """
    Compare the old two-step approach vs the new combined approach.
    
    OLD APPROACH (two API calls):
    1. POST /extract-html with URL → returns HTML (can be 100KB+)
    2. POST /clean-html with HTML → returns cleaned text
    
    NEW APPROACH (one API call):
    1. POST /extract-and-clean with URL → returns cleaned text directly
    
    Token savings: The HTML content doesn't need to be transmitted in the request,
    saving potentially thousands of tokens per request.
    """
    print("\n=== Token Savings Comparison ===")
    print("\nOLD APPROACH (two-step):")
    print("  Step 1: POST /extract-html")
    print("    Request: ~50 tokens (URL)")
    print("    Response: ~25,000 tokens (100KB HTML)")
    print("  Step 2: POST /clean-html")
    print("    Request: ~25,000 tokens (100KB HTML)")
    print("    Response: ~500 tokens (cleaned text)")
    print("  TOTAL: ~50,550 tokens")
    print()
    print("NEW APPROACH (combined):")
    print("  POST /extract-and-clean")
    print("    Request: ~50 tokens (URL)")
    print("    Response: ~500 tokens (cleaned text)")
    print("  TOTAL: ~550 tokens")
    print()
    print("SAVINGS: ~50,000 tokens per request! (99% reduction)")
    print("\nNote: HTML is processed server-side with BeautifulSoup,")
    print("      avoiding the need to transmit it to the client.")


if __name__ == "__main__":
    # Show the token savings comparison
    compare_old_vs_new_approach()
    
    print("\n" + "="*60 + "\n")
    
    # Example API call (requires server to be running)
    print("To test the API, start the server with:")
    print("  uvicorn src.api.main:app --reload")
    print("\nThen run this script again to make the API call.")
    print()
    
    # Uncomment to actually make the API call:
    # example_extract_and_clean()
