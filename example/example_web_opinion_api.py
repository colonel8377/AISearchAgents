#!/usr/bin/env python
"""
Example usage of Web Opinion Extract API endpoints.

This script demonstrates how to use the new FastAPI endpoints for
web opinion extraction and analysis.
"""

import requests
import json
from typing import Optional

# Configuration
BASE_URL = "http://localhost:8000"  # Change to your server URL
API_KEY = "your-api-key-here"  # Optional, set if API key is required

# Headers
headers = {
    "Content-Type": "application/json"
}

# Add API key if required
if API_KEY != "your-api-key-here":
    headers["X-API-Key"] = API_KEY


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def example_1_extract_html():
    """
    Example 1: Extract HTML from URL
    
    API Endpoint: POST /api/v1/web-opinion/extract-html
    Input: URL only
    Output: Raw HTML content
    """
    print_section("Example 1: Extract HTML from URL")
    
    url = "https://example.com/article"
    
    request_data = {
        "url": url
    }
    
    print(f"\nRequest:")
    print(f"  POST {BASE_URL}/api/v1/web-opinion/extract-html")
    print(f"  Body: {json.dumps(request_data, indent=2)}")
    
    # Uncomment to make actual request:
    # response = requests.post(
    #     f"{BASE_URL}/api/v1/web-opinion/extract-html",
    #     headers=headers,
    #     json=request_data
    # )
    # result = response.json()
    # print(f"\nResponse:")
    # print(f"  Status: {response.status_code}")
    # if result.get("error"):
    #     print(f"  Error: {result['error']}")
    #     print(f"  Message: {result['error_message']}")
    # else:
    #     print(f"  HTML Length: {result['html_length']} characters")
    #     print(f"  HTML Preview: {result['html'][:200]}...")
    
    print("\nExpected Response:")
    print("""{
  "url": "https://example.com/article",
  "html": "<html>...</html>",
  "html_length": 12345,
  "error": null,
  "error_message": null
}""")


def example_2_clean_html():
    """
    Example 2: Clean HTML to extract main content
    
    API Endpoint: POST /api/v1/web-opinion/clean-html
    Input: Raw HTML
    Output: Cleaned text and title
    """
    print_section("Example 2: Clean HTML Content")
    
    html = """
    <html>
    <head><title>Article Title</title></head>
    <body>
        <nav>Navigation</nav>
        <article>
            <h1>Breaking News</h1>
            <p>This is the main article content.</p>
        </article>
        <footer>Footer</footer>
    </body>
    </html>
    """
    
    request_data = {
        "html": html
    }
    
    print(f"\nRequest:")
    print(f"  POST {BASE_URL}/api/v1/web-opinion/clean-html")
    print(f"  Body: (HTML content)")
    
    print("\nExpected Response:")
    print("""{
  "text": "Breaking News\\n\\nThis is the main article content.",
  "title": "Breaking News",
  "text_length": 45,
  "error": null,
  "error_message": null
}""")


def example_3_extract_opinions():
    """
    Example 3: Extract atomic opinions from text
    
    API Endpoint: POST /api/v1/web-opinion/extract-opinions
    Input: Text content
    Output: Atomic opinions with bias scores per opinion
    """
    print_section("Example 3: Extract Atomic Opinions from Text")
    
    text = """
    I strongly support this new policy initiative. It will help workers.
    The bill was passed on January 5th. However, the implementation timeline
    seems unrealistic and could lead to problems.
    """
    
    request_data = {
        "text": text,
        "url": "https://example.com",
        "title": "Policy Analysis",
        "execution_mode": "chain_local"
    }
    
    print(f"\nRequest:")
    print(f"  POST {BASE_URL}/api/v1/web-opinion/extract-opinions")
    print(f"  Body:")
    print(f"    text: (article text)")
    print(f"    execution_mode: chain_local")
    
    print("\nExpected Response:")
    print("""{
  "url": "https://example.com",
  "title": "Policy Analysis",
  "atomic_opinions": [
    {
      "text": "Support for new policy initiative",
      "opinion_type": "opinion",
      "bias_probabilities": {
        "left": 0.6,
        "right": 0.2,
        "neutral": 0.2,
        "dominant_bias": "left",
        "bias_score": -0.4
      },
      "confidence": 0.9,
      "reasoning": "Strong support indicates progressive stance..."
    },
    {
      "text": "The bill was passed on January 5th",
      "opinion_type": "fact",
      "bias_probabilities": {
        "left": 0.1,
        "right": 0.1,
        "neutral": 0.8,
        "dominant_bias": "neutral",
        "bias_score": 0.0
      },
      "confidence": 0.95
    }
  ],
  "facts": [...],
  "opinions": [...],
  "overall_bias_distribution": {
    "left": 0.6,
    "right": 0.2,
    "neutral": 0.2,
    "dominant_bias": "left",
    "bias_score": -0.4
  },
  "text_length": 150,
  "truncated": false
}""")


def example_4_analyze_url():
    """
    Example 4: Complete analysis pipeline from URL
    
    API Endpoint: POST /api/v1/web-opinion/analyze
    Input: URL
    Output: Complete analysis with atomic opinions and bias scores
    """
    print_section("Example 4: Complete URL Analysis")
    
    url = "https://example.com/article"
    
    request_data = {
        "url": url,
        "execution_mode": "chain_local"
    }
    
    print(f"\nRequest:")
    print(f"  POST {BASE_URL}/api/v1/web-opinion/analyze")
    print(f"  Body: {json.dumps(request_data, indent=2)}")
    
    print("\nExpected Response:")
    print("""{
  "url": "https://example.com/article",
  "title": "Article Title",
  "atomic_opinions": [
    {
      "text": "Opinion text",
      "opinion_type": "opinion",
      "bias_probabilities": {
        "left": 0.7,
        "right": 0.1,
        "neutral": 0.2,
        "dominant_bias": "left",
        "bias_score": -0.6
      }
    }
  ],
  "facts": [...],
  "opinions": [...],
  "overall_bias_distribution": {
    "left": 0.7,
    "right": 0.1,
    "neutral": 0.2,
    "dominant_bias": "left",
    "bias_score": -0.6
  }
}""")


def example_5_bias_score():
    """
    Example 5: Get overall bias score from URL (simplified API)
    
    API Endpoint: POST /api/v1/web-opinion/bias-score
    Input: URL only
    Output: Overall bias score only (no detailed opinions)
    """
    print_section("Example 5: Get Overall Bias Score from URL")
    
    url = "https://example.com/article"
    
    request_data = {
        "url": url
    }
    
    print(f"\nRequest:")
    print(f"  POST {BASE_URL}/api/v1/web-opinion/bias-score")
    print(f"  Body: {json.dumps(request_data, indent=2)}")
    
    print("\nExpected Response:")
    print("""{
  "url": "https://example.com/article",
  "overall_bias_distribution": {
    "left": 0.45,
    "right": 0.35,
    "neutral": 0.20,
    "dominant_bias": "left",
    "bias_score": -0.1
  },
  "opinions_count": 5,
  "facts_count": 3,
  "error": null,
  "error_message": null
}""")


def example_6_execution_modes():
    """
    Example 6: Different execution modes for Chain of Thought
    
    Demonstrates the three execution modes:
    - no_chain: Fastest, no reasoning
    - chain_local: Balanced, includes reasoning
    - chain_online: Most detailed reasoning
    """
    print_section("Example 6: Execution Modes")
    
    print("\nMode 1: no_chain (Fastest)")
    print("  - No reasoning field in response")
    print("  - Fastest processing")
    print("  - Best for high-volume production")
    print('  - Request: {"url": "...", "execution_mode": "no_chain"}')
    
    print("\nMode 2: chain_local (Recommended)")
    print("  - Includes reasoning field")
    print("  - Good balance of speed and interpretability")
    print("  - Recommended for most use cases")
    print('  - Request: {"url": "...", "execution_mode": "chain_local"}')
    
    print("\nMode 3: chain_online (Most Detailed)")
    print("  - Full LLM-driven CoT reasoning")
    print("  - Most detailed analysis")
    print("  - Best for research and detailed analysis")
    print('  - Request: {"url": "...", "execution_mode": "chain_online"}')


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("Web Opinion Extract API - Usage Examples")
    print("=" * 80)
    print("\nThese examples demonstrate the 5 main API endpoints:")
    print("  1. /api/v1/web-opinion/extract-html - Extract HTML from URL")
    print("  2. /api/v1/web-opinion/clean-html - Clean HTML to text")
    print("  3. /api/v1/web-opinion/extract-opinions - Extract opinions from text")
    print("  4. /api/v1/web-opinion/analyze - Complete analysis from URL")
    print("  5. /api/v1/web-opinion/bias-score - Get overall bias score from URL")
    
    print("\nNote: To make actual API calls, uncomment the request code in each example")
    print("      and update BASE_URL and API_KEY at the top of this file.")
    
    # Run examples
    example_1_extract_html()
    example_2_clean_html()
    example_3_extract_opinions()
    example_4_analyze_url()
    example_5_bias_score()
    example_6_execution_modes()
    
    print("\n" + "=" * 80)
    print("Examples Complete!")
    print("=" * 80)
    print("\nTo start the API server:")
    print("  cd src")
    print("  uvicorn api.main:app --reload")
    print("\nThen test with curl:")
    print('  curl -X POST http://localhost:8000/api/v1/web-opinion/bias-score \\')
    print('       -H "Content-Type: application/json" \\')
    print('       -d \'{"url": "https://example.com/article"}\'')
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
