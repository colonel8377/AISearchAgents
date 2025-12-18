# WebOpinionEngine Implementation Summary

## Overview
Successfully refactored the web analysis tool into a fully modular, "glass box" **WebOpinionEngine** with optional MBFC (Media Bias/Fact Check) database integration.

## What Was Implemented

### 1. WebOpinionEngine Class (`src/agents/web_opinion_extractor/engine.py`)

A modular 4-agent pipeline with public APIs for debugging:

#### Agent 1: `extract_content(url) -> ArticleContent`
- **Purpose**: Deterministic content extraction (no LLM to avoid hallucination)
- **Method**: Uses trafilatura (preferred) with BeautifulSoup fallback
- **Features**: 
  - Full text extraction (no truncation)
  - User-Agent headers to prevent blocking
  - Retry with exponential backoff
  - Returns: title, full_text, domain, url

#### Agent 2: `resolve_metadata(url) -> SourceMetadata`
- **Purpose**: Optional MBFC database lookup for historical bias data
- **Method**: SQLite query with smart domain matching
- **Features**:
  - Domain normalization (www. stripping, aliases)
  - Fuzzy matching with heuristics
  - Graceful degradation (works without database)
  - SQL injection protection
  - Returns: source_name, match_type, bias_rating, factual_reporting

#### Agent 3: `atomize_text(text) -> List[AtomicUnit]`
- **Purpose**: Split complex text into atomic fact/opinion units
- **Method**: LLM-based with automatic chunking for large texts
- **Features**:
  - Handles texts >15k tokens via chunking
  - Few-shot prompt examples
  - Retry with exponential backoff
  - Returns: List of (statement, type, original_sentence)

#### Agent 4: `calculate_bias(units, metadata) -> BiasResult`
- **Purpose**: Bayesian bias scoring with optional MBFC prior
- **Method**: LLM analysis with probability distribution
- **Features**:
  - Uses MBFC metadata as Bayesian prior when available
  - Falls back to zero-shot analysis without metadata
  - Returns probability distribution (left, right, neutral)
  - Includes Chain of Thought reasoning

#### Production Orchestrator: `run_pipeline(url, use_mbfc=True) -> dict`
- Orchestrates all 4 agents
- Returns consolidated JSON with complete analysis

### 2. New Pydantic Models (`src/agents/web_opinion_extractor/models.py`)

- **ArticleContent**: title, full_text, domain, url
- **SourceMetadata**: source_name, raw_db_row, match_type, bias_rating, factual_reporting
- **AtomicUnit**: statement, type (fact/opinion), original_sentence, confidence
- **BiasResult**: bias_distribution, reasoning, metadata_used, individual_biases

### 3. Tenacity Retry Integration

Added `@retry` decorators to all agents:
- **BotCreatorAgent.create_bot**: Max 3 attempts, exponential backoff
- **SummarizerAgent.summarize_conversation**: Max 3 attempts, exponential backoff
- **NudgeCollapseAgent.generate_turn**: Max 3 attempts, exponential backoff
- **WebOpinionEngine**: All 4 agent methods with retry

Configuration:
```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception)
)
```

### 4. Few-Shot Prompts in Dedicated Folders

Created organized prompt structure:
```
src/prompts/
├── __init__.py
├── bot_creator/
│   └── few_shots.py          # 3 examples: tech support, financial advisor, writing coach
├── summarizer/
│   └── few_shots.py          # 3 examples: renewable energy, programming, history
├── nudge_collapse/
│   └── few_shots.py          # 4 turn examples: neutral, focus shift, source attack, echo chamber
└── web_opinion_extractor/
    ├── atomizer_shots.py     # Atomization examples
    └── scorer_shots.py       # Bias scoring examples
```

All agents now import and use these few-shot examples in their system prompts.

### 5. MBFC Database Features

- **Context Manager**: Safe SQLite connection handling
- **Domain Aliases Map**: bbc.co.uk→bbc.com, etc.
- **Match Types**: exact, fuzzy, none, disabled
- **SQL Injection Protection**: Domain sanitization before queries
- **Graceful Degradation**: Works without database or on connection errors

### 6. Dependencies Added

```
trafilatura==1.6.2    # Deterministic web content extraction
tenacity==8.2.3       # Retry decorators with exponential backoff
```

## Testing

### Test Coverage (`test_web_opinion_engine.py`)
- **18 tests total, all passing**
- Model tests: ArticleContent, SourceMetadata, AtomicUnit, BiasResult
- Engine tests: initialization, domain normalization, extraction, metadata resolution
- Agent tests: atomization, bias calculation (with/without MBFC)
- Integration tests: full pipeline orchestration

### Example Scripts
1. **example_web_opinion_engine.py**: Demonstrates all features
   - Individual agent testing
   - Full pipeline orchestration
   - MBFC integration
   - Error handling
   - No truncation feature

## Security

### Code Review Fixes
- ✅ Removed duplicate line in bot_creator
- ✅ Added User-Agent header to prevent blocking
- ✅ Improved domain matching (prevent false positives like evil-cnn.com matching cnn.com)
- ✅ Removed redundant BeautifulSoup import
- ✅ Sanitized SQL inputs (strip %, _ wildcards)

### CodeQL Analysis
- **Result: 0 vulnerabilities found**
- No SQL injection risks
- No security alerts

## Usage Examples

### Basic Usage
```python
from src.agents.web_opinion_extractor import WebOpinionEngine

# Initialize engine
engine = WebOpinionEngine(
    db_path='/path/to/mbfc.db',  # Optional
    max_chunk_tokens=15000
)

# Run complete pipeline
result = engine.run_pipeline(url, use_mbfc=True)
```

### Glass Box Debugging
```python
# Call agents individually
article = engine.extract_content(url)
metadata = engine.resolve_metadata(url)
units = engine.atomize_text(article.full_text)
bias = engine.calculate_bias(units, metadata)
```

### Without MBFC Database
```python
# Works perfectly without database
engine = WebOpinionEngine(db_path=None)
result = engine.run_pipeline(url, use_mbfc=False)
# metadata.match_type will be "disabled"
```

## Key Design Decisions

1. **Glass Box Design**: All agents are public methods for transparency and debugging
2. **MBFC Optional**: Engine works with or without database - graceful degradation
3. **No LLM for Agent 1**: Deterministic extraction prevents hallucination and token limits
4. **Bayesian Integration**: MBFC data used as prior probability, not hard constraint
5. **Full Text**: No truncation - Agent 3 handles large texts via chunking
6. **Retry Everything**: Network and LLM calls all have retry logic
7. **Dedicated Prompts**: Few-shots organized by agent, easy to maintain

## Files Changed

**New Files:**
- `src/agents/web_opinion_extractor/engine.py` (915 lines)
- `src/agents/web_opinion_extractor/models.py` (additions)
- `src/prompts/bot_creator/few_shots.py`
- `src/prompts/summarizer/few_shots.py`
- `src/prompts/nudge_collapse/few_shots.py`
- `src/prompts/web_opinion_extractor/atomizer_shots.py`
- `src/prompts/web_opinion_extractor/scorer_shots.py`
- `example_web_opinion_engine.py`
- `test_web_opinion_engine.py`

**Modified Files:**
- `requirements.txt` (added trafilatura, tenacity)
- `src/agents/bot_creator/agent.py` (added retry, few-shots)
- `src/agents/summarizer/agent.py` (added retry, few-shots)
- `src/agents/nudge_collapse/agent.py` (added retry, few-shots)
- `src/agents/web_opinion_extractor/__init__.py` (export WebOpinionEngine)

## Performance Considerations

- **Connection Pooling**: Uses shared HTTP client from llm_manager
- **Retry Logic**: Exponential backoff prevents overwhelming servers
- **Chunking**: Large texts split into manageable pieces
- **Context Manager**: Proper SQLite connection handling
- **Caching**: Domain normalization map for fast lookups

## Future Enhancements

Possible improvements:
1. Add caching layer for MBFC lookups
2. Implement parallel processing for multiple articles
3. Add more sophisticated fuzzy matching algorithms
4. Extend MBFC schema to support more bias dimensions
5. Add confidence scores to metadata matching
6. Implement adaptive chunking based on content structure

## Conclusion

The WebOpinionEngine successfully delivers on all requirements:
- ✅ Modular "glass box" design
- ✅ Optional MBFC integration
- ✅ Deterministic extraction
- ✅ Bayesian bias scoring
- ✅ Tenacity retry decorators
- ✅ Few-shot prompts organized
- ✅ Comprehensive testing
- ✅ Zero security vulnerabilities

The implementation is production-ready and provides a solid foundation for web opinion analysis with optional historical bias context.
