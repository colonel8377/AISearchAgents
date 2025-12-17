# Web Opinion Extract API - Quick Start Guide

## Overview

This implementation adds 5 comprehensive REST API endpoints for web opinion extraction and bias analysis. The APIs can extract HTML, clean content, analyze opinions, and calculate bias scores.

## Available APIs

| # | Endpoint | Input | Output | Use Case |
|---|----------|-------|--------|----------|
| 1 | `/api/v1/web-opinion/extract-html` | URL | Raw HTML | Fetch HTML from URL |
| 2 | `/api/v1/web-opinion/clean-html` | HTML | Clean text | Extract main content |
| 3 | `/api/v1/web-opinion/extract-opinions` | Text | Atomic opinions | Analyze text for opinions |
| 4 | `/api/v1/web-opinion/analyze` | URL | Complete analysis | One-step full pipeline |
| 5 | `/api/v1/web-opinion/bias-score` | URL | Bias score only | Quick bias check |

## Quick Start

### 1. Start the Server

```bash
cd src
uvicorn api.main:app --reload
```

The server will start at `http://localhost:8000`

### 2. Test with curl

#### Quick Bias Check (Simplest)
```bash
curl -X POST http://localhost:8000/api/v1/web-opinion/bias-score \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/article"}'
```

#### Complete Analysis
```bash
curl -X POST http://localhost:8000/api/v1/web-opinion/analyze \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/article", "execution_mode": "chain_local"}'
```

### 3. Python Example

```python
import requests

# Quick bias check
response = requests.post(
    "http://localhost:8000/api/v1/web-opinion/bias-score",
    json={"url": "https://example.com/article"}
)

result = response.json()
if not result.get("error"):
    bias = result["overall_bias_distribution"]
    print(f"Dominant Bias: {bias['dominant_bias']}")
    print(f"Left: {bias['left']:.2%}")
    print(f"Right: {bias['right']:.2%}")
    print(f"Neutral: {bias['neutral']:.2%}")
```

## Response Format

### Bias Score Response
```json
{
  "url": "https://example.com/article",
  "overall_bias_distribution": {
    "left": 0.45,
    "right": 0.35,
    "neutral": 0.20,
    "dominant_bias": "left",
    "bias_score": -0.1
  },
  "opinions_count": 5,
  "facts_count": 3
}
```

### Complete Analysis Response
```json
{
  "url": "https://example.com/article",
  "title": "Article Title",
  "atomic_opinions": [
    {
      "text": "Support for policy initiative",
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
    }
  ],
  "facts": [...],
  "opinions": [...],
  "overall_bias_distribution": {...}
}
```

## Key Features

### 1. Atomic Opinion Extraction
- Breaks compound sentences into atomic opinions
- Example: "I support A but oppose B" → 2 atomic opinions

### 2. Bias Probability Distribution
- Each opinion has left/right/neutral probabilities
- Probabilities sum to 1.0
- Includes dominant bias and single score

### 3. Fact vs Opinion Classification
- Separates verifiable facts from subjective opinions
- Helps identify objective vs subjective content

### 4. Chain of Thought Support
Three execution modes:
- `no_chain` - Fastest (no reasoning)
- `chain_local` - Balanced (includes reasoning)
- `chain_online` - Most detailed

### 5. Robust Error Handling
All endpoints return structured errors:
```json
{
  "error": "network_error",
  "error_message": "Connection timeout",
  ...
}
```

## Documentation

- **API Reference:** [WEB_OPINION_API_DOCS.md](WEB_OPINION_API_DOCS.md)
- **Usage Examples:** [example_web_opinion_api.py](example_web_opinion_api.py)
- **Test Suite:** [test_web_opinion_api.py](test_web_opinion_api.py)
- **Implementation Summary:** [WEB_OPINION_EXTRACT_API_SUMMARY.md](WEB_OPINION_EXTRACT_API_SUMMARY.md)

## API Endpoints Details

### 1. Extract HTML
```bash
POST /api/v1/web-opinion/extract-html
Body: {"url": "https://example.com"}
```
Returns raw HTML content from URL.

### 2. Clean HTML
```bash
POST /api/v1/web-opinion/clean-html
Body: {"html": "<html>...</html>"}
```
Returns cleaned text and title.

### 3. Extract Opinions
```bash
POST /api/v1/web-opinion/extract-opinions
Body: {
  "text": "Article text...",
  "execution_mode": "chain_local"
}
```
Returns atomic opinions with bias scores.

### 4. Complete Analysis
```bash
POST /api/v1/web-opinion/analyze
Body: {
  "url": "https://example.com",
  "execution_mode": "chain_local"
}
```
Complete pipeline from URL to opinions.

### 5. Bias Score Only
```bash
POST /api/v1/web-opinion/bias-score
Body: {"url": "https://example.com"}
```
Quick bias check without opinion details.

## Common Use Cases

### Use Case 1: Quick Content Screening
```python
# Screen multiple URLs for bias
urls = ["url1", "url2", "url3"]
for url in urls:
    response = requests.post(
        f"{base_url}/api/v1/web-opinion/bias-score",
        json={"url": url}
    )
    print(f"{url}: {response.json()['overall_bias_distribution']['dominant_bias']}")
```

### Use Case 2: Detailed Analysis
```python
# Get detailed opinion breakdown
response = requests.post(
    f"{base_url}/api/v1/web-opinion/analyze",
    json={"url": url, "execution_mode": "chain_local"}
)
result = response.json()
for opinion in result["opinions"]:
    print(f"Opinion: {opinion['text']}")
    print(f"Bias: {opinion['bias_probabilities']['dominant_bias']}")
    if opinion.get('reasoning'):
        print(f"Why: {opinion['reasoning'][:100]}...")
```

### Use Case 3: Manual Pipeline
```python
# Step-by-step processing
# 1. Extract HTML
html_response = requests.post(f"{base_url}/api/v1/web-opinion/extract-html",
                              json={"url": url})
html = html_response.json()["html"]

# 2. Clean HTML
clean_response = requests.post(f"{base_url}/api/v1/web-opinion/clean-html",
                               json={"html": html})
text = clean_response.json()["text"]

# 3. Extract opinions
opinions_response = requests.post(f"{base_url}/api/v1/web-opinion/extract-opinions",
                                  json={"text": text})
opinions = opinions_response.json()["opinions"]
```

## Authentication

If API key is required:
```bash
curl -X POST http://localhost:8000/api/v1/web-opinion/bias-score \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"url": "https://example.com"}'
```

## Environment Setup

Make sure these environment variables are set:
```bash
export OPENAI_API_KEY=your_openai_key
export OPENAI_API_BASE=https://api.openai.com/v1  # Optional
export OPENAI_MODEL=gpt-3.5-turbo  # Optional
```

## Troubleshooting

### Error: "network_error"
- Check URL is accessible
- Verify network connectivity
- Check timeout settings

### Error: "content_extraction_error"
- URL may have no meaningful content
- Try a different news article URL
- Check if page requires JavaScript

### Error: "llm_analysis_error"
- Verify OPENAI_API_KEY is set
- Check API quota/limits
- Verify LLM service is accessible

## Next Steps

1. Review [WEB_OPINION_API_DOCS.md](WEB_OPINION_API_DOCS.md) for complete API reference
2. Run [example_web_opinion_api.py](example_web_opinion_api.py) to see examples
3. Explore [test_web_opinion_api.py](test_web_opinion_api.py) for testing patterns
4. Start building your application with the APIs!

## Support

For issues or questions:
1. Check [WEB_OPINION_API_DOCS.md](WEB_OPINION_API_DOCS.md)
2. Review [example_web_opinion_api.py](example_web_opinion_api.py)
3. Run tests in [test_web_opinion_api.py](test_web_opinion_api.py)
4. Check implementation in `src/api/main.py`

---

**Implementation Status:** ✅ Complete and Ready for Use

All 5 required APIs are implemented, tested, and documented.
