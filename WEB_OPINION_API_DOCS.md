# Web Opinion Extract Module - API Documentation

## Overview

The Web Opinion Extract Module provides a comprehensive REST API for extracting and analyzing opinions from web content. The API includes endpoints for HTML extraction, text cleaning, atomic opinion extraction, bias analysis, and overall bias scoring.

## Base URL

```
http://localhost:8000/api/v1/web-opinion
```

## Authentication

If API key authentication is enabled:
- Add header: `X-API-Key: your-api-key`

## API Endpoints

### 1. Extract HTML from URL

Extract raw HTML content from a URL.

**Endpoint:** `POST /api/v1/web-opinion/extract-html`

**Request Body:**
```json
{
  "url": "https://example.com/article"
}
```

**Response:**
```json
{
  "url": "https://example.com/article",
  "html": "<html>...</html>",
  "html_length": 12345,
  "error": null,
  "error_message": null
}
```

**Use Case:** 
- Step 1 of manual pipeline
- Verify HTML fetching independently
- Debug content extraction issues

---

### 2. Clean HTML Content

Extract clean text from raw HTML, removing scripts, styles, navigation, and other non-content elements.

**Endpoint:** `POST /api/v1/web-opinion/clean-html`

**Request Body:**
```json
{
  "html": "<html><body><article>...</article></body></html>"
}
```

**Response:**
```json
{
  "text": "Cleaned article text...",
  "title": "Article Title",
  "text_length": 1234,
  "error": null,
  "error_message": null
}
```

**Use Case:**
- Step 2 of manual pipeline
- Verify HTML cleaning process
- Extract main content for further processing

---

### 3. Extract Atomic Opinions from Text

Analyze text to extract atomic opinions with bias scores.

**Endpoint:** `POST /api/v1/web-opinion/extract-opinions`

**Request Body:**
```json
{
  "text": "Article text to analyze...",
  "url": "https://example.com",
  "title": "Article Title",
  "execution_mode": "chain_local"
}
```

**Parameters:**
- `text` (required): Text content to analyze
- `url` (optional): Source URL for metadata
- `title` (optional): Article title for metadata
- `execution_mode` (optional): CoT mode - "no_chain", "chain_local", or "chain_online"

**Response:**
```json
{
  "url": "https://example.com",
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
      "original_sentence": "I support this policy initiative",
      "confidence": 0.9,
      "reasoning": "Strong support indicates progressive stance..."
    }
  ],
  "facts": [...],
  "opinions": [...],
  "overall_bias_distribution": {
    "left": 0.5,
    "right": 0.3,
    "neutral": 0.2,
    "dominant_bias": "left",
    "bias_score": -0.2
  },
  "text_length": 1234,
  "truncated": false,
  "error": null
}
```

**Key Features:**
- **Atomic Opinions:** Each opinion is broken into atomic units (one stance per opinion)
- **Bias Per Opinion:** Each opinion has its own bias probability distribution
- **Fact vs Opinion:** Separates verifiable facts from subjective opinions
- **Chain of Thought:** Optional reasoning explaining the analysis (when CoT enabled)
- **Overall Bias:** Aggregated bias distribution across all opinions

---

### 4. Complete URL Analysis

One-step complete pipeline: fetch HTML, clean, extract opinions, and calculate bias scores.

**Endpoint:** `POST /api/v1/web-opinion/analyze`

**Request Body:**
```json
{
  "url": "https://example.com/article",
  "execution_mode": "chain_local"
}
```

**Parameters:**
- `url` (required): URL to analyze
- `execution_mode` (optional): CoT mode - "no_chain", "chain_local", or "chain_online"

**Response:** Same as `/extract-opinions` endpoint

**Use Case:**
- Main entry point for complete analysis
- Simplest way to get full opinion analysis from URL
- Production use for automated bias detection

**Pipeline Steps:**
1. Fetch HTML from URL
2. Clean and extract main content
3. Analyze with LLM to extract atomic opinions
4. Calculate bias scores per opinion
5. Calculate overall bias distribution

---

### 5. Get Overall Bias Score from URL

Simplified API that returns only the overall bias score without detailed opinion breakdowns.

**Endpoint:** `POST /api/v1/web-opinion/bias-score`

**Request Body:**
```json
{
  "url": "https://example.com/article",
  "execution_mode": "chain_local"
}
```

**Parameters:**
- `url` (required): URL to analyze
- `execution_mode` (optional): CoT mode

**Response:**
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
  "facts_count": 3,
  "error": null,
  "error_message": null
}
```

**Use Case:**
- Quick bias assessment without detailed analysis
- High-volume bias screening
- Dashboard metrics and aggregation

---

## Data Models

### BiasDistribution

Probability distribution of political bias across three categories.

```json
{
  "left": 0.6,        // Probability of Left/Progressive bias (0.0 to 1.0)
  "right": 0.2,       // Probability of Right/Conservative bias (0.0 to 1.0)
  "neutral": 0.2,     // Probability of Neutral/Centrist bias (0.0 to 1.0)
  "dominant_bias": "left",  // The category with highest probability
  "bias_score": -0.4  // Single score: -1.0 (left) to +1.0 (right)
}
```

**Note:** Probabilities always sum to 1.0 (normalized automatically)

### AtomicOpinion

A single atomic opinion extracted from content.

```json
{
  "text": "Support for universal healthcare",
  "opinion_type": "opinion",  // "fact" or "opinion"
  "bias_probabilities": { ... },
  "original_sentence": "I strongly support universal healthcare",
  "confidence": 0.9,
  "reasoning": "This expresses support for universal healthcare..." // Optional, only with CoT
}
```

### Error Responses

When errors occur, the API returns a valid response with error information:

```json
{
  "error": "network_error",  // Error type
  "error_message": "Connection timeout",  // Detailed message
  ...other fields set to null or empty...
}
```

**Error Types:**
- `network_error` - Failed to fetch URL (timeout, 404, connection refused)
- `content_extraction_error` - No meaningful content found
- `llm_analysis_error` - LLM analysis failed
- `analysis_failed` - Unexpected error during processing
- `fetch_failed` - Failed to fetch HTML
- `cleaning_failed` - Failed to clean HTML

---

## Execution Modes

The API supports three Chain of Thought (CoT) execution modes:

### 1. no_chain (Fastest)

```json
{
  "execution_mode": "no_chain"
}
```

- **No reasoning field** in response
- **Fastest processing**
- **Best for:** High-volume production, cost optimization

### 2. chain_local (Recommended)

```json
{
  "execution_mode": "chain_local"
}
```

- **Includes reasoning field** with analysis explanation
- **Good balance** of speed and interpretability
- **Best for:** Most use cases requiring explainability

### 3. chain_online (Most Detailed)

```json
{
  "execution_mode": "chain_online"
}
```

- **Full LLM-driven CoT** reasoning
- **Most detailed analysis**
- **Best for:** Research, detailed analysis, maximum explainability

---

## Usage Examples

### Example 1: Quick Bias Check

```bash
curl -X POST http://localhost:8000/api/v1/web-opinion/bias-score \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/article"
  }'
```

### Example 2: Complete Analysis with CoT

```bash
curl -X POST http://localhost:8000/api/v1/web-opinion/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/article",
    "execution_mode": "chain_local"
  }'
```

### Example 3: Manual Pipeline

```bash
# Step 1: Extract HTML
curl -X POST http://localhost:8000/api/v1/web-opinion/extract-html \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}' > response1.json

# Step 2: Clean HTML (use html from response1)
curl -X POST http://localhost:8000/api/v1/web-opinion/clean-html \
  -H "Content-Type: application/json" \
  -d '{"html": "..."}' > response2.json

# Step 3: Extract opinions (use text from response2)
curl -X POST http://localhost:8000/api/v1/web-opinion/extract-opinions \
  -H "Content-Type: application/json" \
  -d '{"text": "..."}' > response3.json
```

### Example 4: Python Client

```python
import requests

# Complete analysis
response = requests.post(
    "http://localhost:8000/api/v1/web-opinion/analyze",
    json={
        "url": "https://example.com/article",
        "execution_mode": "chain_local"
    }
)

result = response.json()

# Check for errors
if result.get("error"):
    print(f"Error: {result['error']} - {result['error_message']}")
else:
    # Print overall bias
    bias = result["overall_bias_distribution"]
    print(f"Overall Bias: {bias['dominant_bias']}")
    print(f"  Left: {bias['left']:.2%}")
    print(f"  Right: {bias['right']:.2%}")
    print(f"  Neutral: {bias['neutral']:.2%}")
    
    # Print opinions
    print(f"\nFound {len(result['opinions'])} opinions:")
    for opinion in result["opinions"]:
        print(f"  - {opinion['text']}")
        print(f"    Bias: {opinion['bias_probabilities']['dominant_bias']}")
        if opinion.get("reasoning"):
            print(f"    Reasoning: {opinion['reasoning'][:100]}...")
```

---

## Best Practices

1. **Use `/analyze` for complete pipeline** - Simplest and most reliable
2. **Use `/bias-score` for quick assessments** - When you only need overall bias
3. **Use manual pipeline for debugging** - When you need to inspect intermediate steps
4. **Choose appropriate execution mode:**
   - Production/high-volume: `no_chain`
   - Default use: `chain_local`
   - Research/detailed analysis: `chain_online`
5. **Handle errors gracefully** - Always check for `error` field in response
6. **Cache results** - Consider caching for frequently accessed URLs

---

## Rate Limiting

The API respects LLM provider rate limits. Consider:
- Implementing client-side rate limiting
- Using batch processing for multiple URLs
- Caching results to reduce API calls

---

## Performance

- **HTML Extraction:** ~1-3 seconds
- **Text Cleaning:** <1 second
- **Opinion Extraction:**
  - `no_chain`: ~5-10 seconds
  - `chain_local`: ~8-15 seconds
  - `chain_online`: ~10-20 seconds

Times vary based on:
- Article length
- LLM provider response time
- Network conditions

---

## Error Handling

All endpoints return HTTP 200 with error information in the response body:

```json
{
  "error": "network_error",
  "error_message": "Connection timeout after 30 seconds",
  "url": "https://example.com",
  ...
}
```

This design ensures consistent error handling across all endpoints.

---

## API Versioning

Current version: `v1`

Base path: `/api/v1/web-opinion`

Future versions will use different paths (e.g., `/api/v2/web-opinion`)

---

## See Also

- [Web Opinion Analyzer Documentation](WEB_OPINION_ANALYZER_DOCS.md)
- [Example Scripts](example_web_opinion_api.py)
- [Test Suite](test_web_opinion_api.py)
