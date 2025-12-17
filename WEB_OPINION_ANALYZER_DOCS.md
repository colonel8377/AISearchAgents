# WebOpinionAnalyzer Module Documentation

## Overview

The **WebOpinionAnalyzer** module provides production-ready web content analysis with opinion extraction, fact/opinion classification, and political bias detection using probability distributions.

## Key Features

### ✅ Already Implemented

1. **Surgical HTML Extraction**
   - Extracts only article title and main content text
   - Removes all noise: `<script>`, `<style>`, `<nav>`, `<footer>`, `<header>`, `<aside>`
   - Removes media: `<button>`, `<img>`, `<video>`, `<source>`, `<figure>`, `<input>`
   - Removes navigation links while preserving content flow
   - Uses heuristics to locate main article container (e.g., `<article>`, high text density divs)

2. **Chain of Thought (CoT) Support**
   - Three execution modes:
     - `chain_online`: LLM handles full CoT reasoning
     - `chain_local`: Local task decomposition with CoT (recommended)
     - `no_chain`: No CoT reasoning (fastest for production)
   - Each atomic opinion includes optional `reasoning` field with step-by-step analysis
   - Explains WHY classification was made and HOW bias probabilities were determined

3. **Bias Probability Distribution**
   - Replaces single bias score with probability distribution:
     - `left`: Probability of Left/Progressive bias (0.0 to 1.0)
     - `right`: Probability of Right/Conservative bias (0.0 to 1.0)
     - `neutral`: Probability of Neutral/Centrist stance (0.0 to 1.0)
   - Probabilities automatically normalized to sum to 1.0
   - Backward compatible: can still get single bias score via `bias_score` property

4. **High-Level Public API**
   - `extract_and_analyze(url)`: Main entry point, complete pipeline
   - `extract_html(url)`: Step 1 verification - fetch HTML
   - `clean_html(html)`: Step 2 verification - extract clean text
   - `analyze_text(text)`: Step 3 verification - LLM analysis
   - Robust error handling: returns valid error states instead of crashing

5. **Atomic Opinion Extraction**
   - Breaks compound sentences into separate atomic opinions
   - Example: "I support the tax cut but oppose the trade war" → 2 atomic opinions
   - Each atomic opinion expresses ONE and only ONE stance

6. **Fact vs. Opinion Classification**
   - FACTS: Verifiable, objective statements
   - OPINIONS: Subjective viewpoints, beliefs, judgments

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Environment Setup

```bash
# Required
export OPENAI_API_KEY=your_api_key_here

# Optional
export OPENAI_API_BASE=https://api.openai.com/v1  # Default
export OPENAI_MODEL=gpt-3.5-turbo  # Default
```

### Basic Usage

```python
from src.agents.web_opinion_extractor import WebOpinionAnalyzer

# Create analyzer with CoT support
analyzer = WebOpinionAnalyzer(execution_mode="chain_local")

# Analyze a URL (complete pipeline)
result = analyzer.extract_and_analyze("https://example.com/article")

# Check for errors
if result.extraction_metadata and "error" in result.extraction_metadata:
    print(f"Error: {result.extraction_metadata['error']}")
    print(f"Message: {result.extraction_metadata['error_message']}")
else:
    # Print results
    print(f"Title: {result.title}")
    print(f"Found {len(result.opinions)} opinions")
    
    for opinion in result.opinions:
        print(f"\nOpinion: {opinion.text}")
        bias = opinion.bias_probabilities
        print(f"  Bias: Left={bias.left:.2%}, Right={bias.right:.2%}, Neutral={bias.neutral:.2%}")
        print(f"  Dominant: {bias.dominant_bias.upper()}")
        if opinion.reasoning:
            print(f"  Reasoning: {opinion.reasoning[:150]}...")
```

## API Reference

### WebOpinionAnalyzer

#### Constructor

```python
analyzer = WebOpinionAnalyzer(
    model_name: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float = 0.3,
    proxy: Optional[str] = None,
    request_timeout: float = 30.0,
    execution_mode: Optional[ExecutionMode] = None  # "chain_online", "chain_local", "no_chain"
)
```

**Parameters:**
- `model_name`: LLM model to use (default: "gpt-3.5-turbo")
- `api_key`: OpenAI API key (default: from OPENAI_API_KEY env var)
- `api_base`: API base URL (default: from OPENAI_API_BASE env var)
- `temperature`: Temperature for LLM responses (default: 0.3, lower = more consistent)
- `proxy`: HTTP proxy for API requests (optional)
- `request_timeout`: Timeout for HTTP requests in seconds (default: 30.0)
- `execution_mode`: CoT mode - "chain_local" (recommended), "chain_online", or "no_chain"

#### Methods

##### extract_and_analyze(url: str) -> OpinionExtractionResult

Main entry point: Complete pipeline from URL to analyzed opinions.

**Returns:** `OpinionExtractionResult` with extracted opinions and metadata

**Error Handling:** Returns valid result with error metadata on failure (never crashes)

```python
result = analyzer.extract_and_analyze("https://example.com/article")

# Check for errors
if result.extraction_metadata and "error" in result.extraction_metadata:
    error_type = result.extraction_metadata["error"]
    # Types: "network_error", "content_extraction_error", "unexpected_error"
    error_msg = result.extraction_metadata["error_message"]
    print(f"Error: {error_type} - {error_msg}")
```

##### extract_html(url: str) -> Optional[str]

Step 1 verification: Fetch HTML from URL.

**Returns:** HTML string or None on error

```python
html = analyzer.extract_html("https://example.com/article")
if html is None:
    print("Failed to fetch HTML")
```

##### clean_html(html: str) -> Tuple[Optional[str], Optional[str]]

Step 2 verification: Clean HTML and extract main content.

**Returns:** Tuple of (cleaned_text, title) or (None, None) on error

```python
text, title = analyzer.clean_html(html)
if text is None:
    print("Failed to clean HTML")
else:
    print(f"Title: {title}")
    print(f"Text length: {len(text)} characters")
```

##### analyze_text(text: str, url: Optional[str] = None, title: Optional[str] = None) -> OpinionExtractionResult

Step 3 verification: Analyze cleaned text with LLM.

**Returns:** `OpinionExtractionResult` with extracted opinions

```python
result = analyzer.analyze_text(text, url=url, title=title)
if result.extraction_metadata and "error" in result.extraction_metadata:
    print(f"Analysis failed: {result.extraction_metadata['error']}")
```

### OpinionExtractionResult

Result object containing all extracted opinions and metadata.

**Fields:**
- `url: Optional[str]` - Source URL
- `title: Optional[str]` - Page title
- `atomic_opinions: List[AtomicOpinion]` - All extracted atomic opinions
- `facts: List[AtomicOpinion]` - Objective facts (filtered from atomic_opinions)
- `opinions: List[AtomicOpinion]` - Subjective opinions (filtered from atomic_opinions)
- `overall_bias_distribution: Optional[BiasDistribution]` - Aggregated bias across all opinions
- `text_length: int` - Length of cleaned text
- `truncated: bool` - Whether text was truncated
- `extraction_metadata: Optional[dict]` - Additional metadata (including errors)

### AtomicOpinion

Represents a single atomic opinion.

**Fields:**
- `text: str` - The atomic opinion text
- `opinion_type: Literal["fact", "opinion"]` - Classification
- `bias_probabilities: BiasDistribution` - Bias probability distribution
- `original_sentence: Optional[str]` - Original sentence from which extracted
- `confidence: Optional[float]` - Confidence score (0.0 to 1.0)
- `reasoning: Optional[str]` - Chain of Thought reasoning (when CoT enabled)

**Properties:**
- `bias_score: float` - Single bias score for backward compatibility (-1.0 to +1.0)

### BiasDistribution

Probability distribution of political bias.

**Fields:**
- `left: float` - Probability of Left/Progressive bias (0.0 to 1.0)
- `right: float` - Probability of Right/Conservative bias (0.0 to 1.0)
- `neutral: float` - Probability of Neutral/Centrist bias (0.0 to 1.0)

**Properties:**
- `dominant_bias: str` - The dominant bias category ("left", "right", or "neutral")
- `bias_score: float` - Single score for backward compatibility (-1.0 to +1.0)

**Note:** Probabilities are automatically normalized to sum to 1.0

## Execution Modes

### no_chain (Fastest)

```python
analyzer = WebOpinionAnalyzer(execution_mode="no_chain")
```

- **No CoT reasoning** - `reasoning` field will be None
- **Fastest processing** - Minimal LLM prompt
- **Best for:** High-volume production use, cost optimization

### chain_local (Recommended)

```python
analyzer = WebOpinionAnalyzer(execution_mode="chain_local")
```

- **Includes CoT reasoning** - `reasoning` field populated with analysis
- **Good balance** - Speed vs. interpretability
- **Best for:** Most use cases requiring explainability

### chain_online (Most Detailed)

```python
analyzer = WebOpinionAnalyzer(execution_mode="chain_online")
```

- **Full LLM-driven CoT** - Most detailed reasoning
- **Slower processing** - More comprehensive prompts
- **Best for:** Research, detailed analysis, maximum explainability

## Understanding Bias Probability Distribution

Unlike traditional single-score classifiers, this module provides a probability distribution:

### Example 1: Left-leaning Opinion

```python
{
    "left": 0.7,
    "right": 0.1,
    "neutral": 0.2
}
# Interpretation: 70% left-aligned, 10% right-aligned, 20% neutral
# Dominant: LEFT
```

### Example 2: Right-leaning Opinion

```python
{
    "left": 0.1,
    "right": 0.8,
    "neutral": 0.1
}
# Interpretation: 10% left-aligned, 80% right-aligned, 10% neutral
# Dominant: RIGHT
```

### Example 3: Neutral Opinion

```python
{
    "left": 0.2,
    "right": 0.2,
    "neutral": 0.6
}
# Interpretation: 20% left-aligned, 20% right-aligned, 60% neutral
# Dominant: NEUTRAL
```

### Example 4: Mixed/Ambiguous Opinion

```python
{
    "left": 0.4,
    "right": 0.4,
    "neutral": 0.2
}
# Interpretation: 40% left-aligned, 40% right-aligned, 20% neutral
# Dominant: LEFT (by alphabetical order when tied)
```

## Error Handling

The module provides robust error handling with three error types:

### 1. Network Error

Occurs when URL cannot be fetched (timeout, 404, connection refused, etc.)

```python
result = analyzer.extract_and_analyze("http://invalid.url")
if result.extraction_metadata and "error" in result.extraction_metadata:
    assert result.extraction_metadata["error"] == "network_error"
    print(result.extraction_metadata["error_message"])
    print(result.extraction_metadata.get("status_code"))  # HTTP status if available
```

### 2. Content Extraction Error

Occurs when HTML has no meaningful content to extract.

```python
result = analyzer.extract_and_analyze("https://empty-page.com")
if result.extraction_metadata and "error" in result.extraction_metadata:
    assert result.extraction_metadata["error"] == "content_extraction_error"
    print(result.extraction_metadata["error_message"])
```

### 3. Unexpected Error

Catch-all for any unexpected errors during processing.

```python
result = analyzer.extract_and_analyze(url)
if result.extraction_metadata and "error" in result.extraction_metadata:
    if result.extraction_metadata["error"] == "unexpected_error":
        print(f"Unexpected: {result.extraction_metadata['error_message']}")
```

## HTML Extraction Details

The module uses sophisticated heuristics to extract only the main article content:

### Removed Elements

- **Scripts & Styles:** `<script>`, `<style>`
- **Navigation:** `<nav>`, `<header>`, `<aside>`, `<footer>`
- **Forms:** `<button>`, `<input>`, `<select>`, `<textarea>`, `<label>`, `<form>`
- **Media:** `<img>`, `<figure>`, `<figcaption>`, `<picture>`, `<video>`, `<audio>`, `<source>`, `<track>`
- **Graphics:** `<svg>`, `<canvas>`, `<map>`, `<area>`
- **Other:** `<iframe>`, `<noscript>`, `<menu>`, `<dialog>`, `<template>`
- **Navigation lists:** `<ul>`, `<ol>` when most items are links

### Content Selectors (in priority order)

1. `<article>`
2. `[role='main']`
3. `<main>`
4. `.article-content`, `.article-body`
5. `.post-content`, `.post-body`
6. `.entry-content`
7. `.content-body`
8. `.story-body`, `.news-body`
9. `#article-body`, `#main-content`, `#content`

### Title Extraction (in priority order)

1. `<h1>` within article
2. `.article-title`
3. `.post-title`
4. `.entry-title`
5. `.headline`
6. `[itemprop='headline']`
7. `<title>` tag (fallback)

## Testing

Run the test suite:

```bash
# All tests
python -m pytest test_web_opinion_extractor.py -v

# Specific test classes
python -m pytest test_web_opinion_extractor.py::TestChainOfThoughtSupport -v
python -m pytest test_web_opinion_extractor.py::TestWebOpinionAnalyzer -v
python -m pytest test_web_opinion_extractor.py::TestBiasDistribution -v
```

## Examples

See `example_web_opinion_analyzer.py` for comprehensive examples:

```bash
python example_web_opinion_analyzer.py
```

The example script demonstrates:
- High-level API usage
- Step-by-step verification
- All three CoT modes
- Error handling patterns
- Bias distribution interpretation

## Migration from Legacy Code

If you have code using the old single bias score:

### Old Code
```python
from src.agents.web_opinion_extractor import WebOpinionExtractor

extractor = WebOpinionExtractor()
result = extractor.extract_from_url(url)

for opinion in result.opinions:
    bias_score = opinion.bias_score  # Single score: -1.0 to +1.0
```

### New Code (with backward compatibility)
```python
from src.agents.web_opinion_extractor import WebOpinionAnalyzer

analyzer = WebOpinionAnalyzer(execution_mode="chain_local")
result = analyzer.extract_and_analyze(url)

for opinion in result.opinions:
    # New: Probability distribution
    left = opinion.bias_probabilities.left
    right = opinion.bias_probabilities.right
    neutral = opinion.bias_probabilities.neutral
    
    # Old: Still works for backward compatibility
    bias_score = opinion.bias_score  # Computed from distribution
    
    # New: CoT reasoning (when enabled)
    if opinion.reasoning:
        print(f"Reasoning: {opinion.reasoning}")
```

## Performance Considerations

1. **Use appropriate execution mode:**
   - `no_chain` for high-volume production (fastest)
   - `chain_local` for most use cases (balanced)
   - `chain_online` for research/detailed analysis (slowest)

2. **Connection pooling:** The module uses shared HTTP client for better performance

3. **Text truncation:** Long articles are truncated to ~32,000 characters (~8,000 tokens)

4. **Caching:** Consider caching results for frequently accessed URLs

## Limitations

1. **Requires LLM API:** Needs OpenAI-compatible API for analysis
2. **Language support:** Optimized for English content
3. **Context window:** Very long articles may be truncated
4. **JavaScript-heavy sites:** May not extract dynamic content
5. **Paywalled content:** Cannot access content behind authentication

## License

Experimental research platform. Use responsibly.
