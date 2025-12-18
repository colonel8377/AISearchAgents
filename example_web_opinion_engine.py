#!/usr/bin/env python
"""
Example usage of WebOpinionEngine - Modular "Glass Box" Design.

This script demonstrates the new WebOpinionEngine with:
1. Public API for each agent (extract_content, resolve_metadata, atomize_text, calculate_bias)
2. Optional MBFC database support
3. Full pipeline orchestration with run_pipeline()
4. Deterministic extraction using trafilatura (no LLM for Agent 1)
"""

import os
import sys
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.web_opinion_extractor import WebOpinionEngine


def example_1_individual_agents():
    """
    Example 1: Call each agent individually for debugging.
    
    This demonstrates the "Glass Box" design - you can inspect
    and debug each step of the pipeline independently.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Individual Agent Testing (Glass Box Design)")
    print("="*80)
    
    # Initialize engine (no MBFC database for this example)
    engine = WebOpinionEngine(
        temperature=0.3,
        db_path=None  # No MBFC database
    )
    
    print("\nDemo: Calling each agent individually")
    print("\n1. Agent 1: extract_content(url)")
    print("   Purpose: Deterministic extraction using trafilatura")
    print("   Input:   URL string")
    print("   Output:  ArticleContent(title, full_text, domain, url)")
    print("   Example: article = engine.extract_content('https://example.com/news')")
    
    print("\n2. Agent 2: resolve_metadata(url)")
    print("   Purpose: Optional MBFC database lookup")
    print("   Input:   URL string")
    print("   Output:  SourceMetadata(source_name, raw_db_row, match_type, bias_rating)")
    print("   Example: metadata = engine.resolve_metadata('https://example.com/news')")
    
    print("\n3. Agent 3: atomize_text(text)")
    print("   Purpose: Split text into atomic fact/opinion units")
    print("   Input:   Full text string")
    print("   Output:  List[AtomicUnit(statement, type, original_sentence)]")
    print("   Example: units = engine.atomize_text(article.full_text)")
    
    print("\n4. Agent 4: calculate_bias(units, metadata)")
    print("   Purpose: Bayesian scoring with optional MBFC prior")
    print("   Input:   List[AtomicUnit], SourceMetadata")
    print("   Output:  BiasResult(bias_distribution, reasoning, metadata_used)")
    print("   Example: result = engine.calculate_bias(units, metadata)")


def example_2_full_pipeline():
    """
    Example 2: Run the complete pipeline with run_pipeline().
    
    This is the production-ready method that orchestrates all 4 agents.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Full Pipeline Orchestration")
    print("="*80)
    
    # Initialize engine
    engine = WebOpinionEngine(
        temperature=0.3,
        db_path=None  # No MBFC database
    )
    
    print("\nUsage:")
    print("  engine = WebOpinionEngine(db_path='/path/to/mbfc.db')")
    print("  result = engine.run_pipeline(url, use_mbfc=True)")
    print("\nResult structure:")
    print("""  {
    "url": "https://example.com/article",
    "article": {
      "title": "Article Title",
      "domain": "example.com",
      "text_length": 5000
    },
    "metadata": {
      "source_name": "Example News",
      "match_type": "exact",
      "bias_rating": "left-center",
      "factual_reporting": "high"
    },
    "atomic_units": [
      {
        "statement": "Support for policy X",
        "type": "opinion",
        "original_sentence": "..."
      }
    ],
    "bias_analysis": {
      "distribution": {
        "left": 0.6,
        "right": 0.2,
        "neutral": 0.2
      },
      "dominant_bias": "left",
      "reasoning": "...",
      "metadata_used": true
    },
    "pipeline_metadata": {
      "use_mbfc": true,
      "num_atomic_units": 15,
      "model": "gpt-3.5-turbo"
    }
  }""")


def example_3_with_mbfc():
    """
    Example 3: Using MBFC database as prior probability.
    
    This shows how the engine uses historical bias data to inform analysis.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: MBFC Database Integration (Bayesian Prior)")
    print("="*80)
    
    print("\nSetup:")
    print("  1. Create SQLite database with table 'media_sources'")
    print("     Required columns: source_name, source_url, bias_rating, factual_reporting")
    print("  2. Initialize engine with db_path parameter")
    print("     engine = WebOpinionEngine(db_path='/path/to/mbfc.db')")
    
    print("\nHow it works:")
    print("  1. Agent 2 queries: SELECT * FROM media_sources WHERE source_url LIKE '%domain%'")
    print("  2. Smart matching:")
    print("     - 0 rows: match_type='none' → No prior, analyze text directly")
    print("     - 1 row:  match_type='exact' → Use as strong prior")
    print("     - >1 rows: match_type='fuzzy' → Use heuristics to pick best match")
    print("  3. Agent 4 uses metadata as Bayesian prior:")
    print("     - Prior: Historical bias from MBFC (e.g., 'left-center')")
    print("     - Evidence: Current article's atomic units")
    print("     - Posterior: Updated bias distribution")
    
    print("\nExample Database Schema:")
    print("""  CREATE TABLE media_sources (
    id INTEGER PRIMARY KEY,
    source_name TEXT NOT NULL,
    source_url TEXT NOT NULL,
    bias_rating TEXT,
    factual_reporting TEXT,
    notes TEXT
  );
  
  INSERT INTO media_sources VALUES
    (1, 'CNN', 'cnn.com', 'left-center', 'high', '...'),
    (2, 'Fox News', 'foxnews.com', 'right', 'mixed', '...'),
    (3, 'BBC', 'bbc.com', 'center', 'high', '...');""")


def example_4_no_mbfc_mode():
    """
    Example 4: Engine works without MBFC database.
    
    This demonstrates that MBFC is optional - the engine gracefully
    handles missing database and disabled mode.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: No MBFC Mode (Optional Feature)")
    print("="*80)
    
    print("\nScenario 1: No database path configured")
    print("  engine = WebOpinionEngine(db_path=None)")
    print("  result = engine.run_pipeline(url, use_mbfc=True)")
    print("  → metadata.match_type = 'disabled'")
    print("  → Agent 4 uses neutral prior: {left: 0.33, neutral: 0.34, right: 0.33}")
    
    print("\nScenario 2: Database file missing")
    print("  engine = WebOpinionEngine(db_path='/nonexistent.db')")
    print("  result = engine.run_pipeline(url, use_mbfc=True)")
    print("  → Logs warning: 'Database connection failed'")
    print("  → metadata.match_type = 'none'")
    print("  → Agent 4 continues with text-only analysis")
    
    print("\nScenario 3: Explicitly disable MBFC")
    print("  engine = WebOpinionEngine(db_path='/valid.db')")
    print("  result = engine.run_pipeline(url, use_mbfc=False)")
    print("  → metadata.match_type = 'disabled'")
    print("  → Agent 2 skipped entirely")


def example_5_no_truncation():
    """
    Example 5: Full text extraction without truncation.
    
    Agent 1 returns the FULL text - no context window limits.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: No Truncation (Full Text Extraction)")
    print("="*80)
    
    print("\nKey Feature: Agent 1 returns FULL text")
    print("  - Old WebOpinionExtractor: Truncated at ~32k characters")
    print("  - New WebOpinionEngine: Returns complete text")
    print("  - Agent 3 handles large texts: Automatic chunking")
    
    print("\nChunking Logic (Agent 3):")
    print("  1. If text > 15k tokens → split into chunks")
    print("  2. Process each chunk separately with LLM")
    print("  3. Aggregate atomic units from all chunks")
    print("  4. Return combined list")
    
    print("\nConfiguration:")
    print("  engine = WebOpinionEngine(max_chunk_tokens=15000)")
    print("  # Adjust based on your LLM's context window")


def example_6_error_handling():
    """
    Example 6: Robust error handling.
    
    Shows how the engine handles various error scenarios gracefully.
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: Error Handling")
    print("="*80)
    
    print("\nNetwork Errors (Agent 1):")
    print("  - Timeout: Retry with exponential backoff (max 3 attempts)")
    print("  - 404/403: Raise NetworkError immediately")
    print("  - Connection error: Retry with exponential backoff")
    
    print("\nExtraction Errors (Agent 1):")
    print("  - Trafilatura fails → Fallback to BeautifulSoup")
    print("  - Both fail → Raise ContentExtractionError")
    print("  - Insufficient content (<50 chars) → Raise ContentExtractionError")
    
    print("\nDatabase Errors (Agent 2):")
    print("  - Connection failed → Log warning, return match_type='none'")
    print("  - Query error → Log error, return match_type='none'")
    print("  - Does NOT crash the pipeline")
    
    print("\nLLM Errors (Agents 3 & 4):")
    print("  - Timeout/rate limit: Retry with exponential backoff (max 3 attempts)")
    print("  - JSON parse error: Log warning, return default/empty result")
    print("  - Unexpected error: Raise exception with traceback")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("WebOpinionEngine - Modular 'Glass Box' Design Examples")
    print("="*80)
    print("\nFeatures:")
    print("  ✓ 4 Public Agent APIs (extract_content, resolve_metadata, atomize_text, calculate_bias)")
    print("  ✓ Optional MBFC database support (works without it)")
    print("  ✓ Deterministic extraction with trafilatura (no LLM hallucination)")
    print("  ✓ Bayesian bias scoring with optional prior")
    print("  ✓ Full text extraction (no truncation)")
    print("  ✓ Automatic text chunking for large articles")
    print("  ✓ Retry decorators with exponential backoff")
    print("  ✓ SQLite context manager for safe DB access")
    print("  ✓ Domain normalization and fuzzy matching")
    
    # Run examples
    example_1_individual_agents()
    example_2_full_pipeline()
    example_3_with_mbfc()
    example_4_no_mbfc_mode()
    example_5_no_truncation()
    example_6_error_handling()
    
    print("\n" + "="*80)
    print("Examples Complete!")
    print("="*80)
    print("\nTo use with real URLs:")
    print("  1. Set OPENAI_API_KEY environment variable")
    print("  2. Install dependencies: pip install -r requirements.txt")
    print("  3. (Optional) Create MBFC SQLite database")
    print("  4. Run: python example_web_opinion_engine.py")
    print("\nFor more details, see:")
    print("  - src/agents/web_opinion_extractor/engine.py")
    print("  - src/agents/web_opinion_extractor/models.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
