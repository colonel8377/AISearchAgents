#!/usr/bin/env python
"""
Example usage of WebOpinionAnalyzer module.

This script demonstrates:
1. High-level API: extract_and_analyze(url)
2. Step-by-step verification: extract_html, clean_html, analyze_text
3. Chain of Thought (CoT) modes: chain_online, chain_local, no_chain
4. Robust error handling with error states
"""

import os
import sys
from typing import Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.web_opinion_extractor import WebOpinionAnalyzer, OpinionExtractionResult


def print_result(result: OpinionExtractionResult, title: str = "Analysis Results"):
    """Pretty print analysis results."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    
    # Check for errors
    if result.extraction_metadata and "error" in result.extraction_metadata:
        print(f"\n❌ ERROR: {result.extraction_metadata['error']}")
        print(f"   Message: {result.extraction_metadata['error_message']}")
        return
    
    # Print metadata
    print(f"\nURL: {result.url or 'N/A'}")
    print(f"Title: {result.title or 'N/A'}")
    print(f"Text length: {result.text_length} characters")
    print(f"Truncated: {result.truncated}")
    
    # Print overall bias
    if result.overall_bias_distribution:
        bias = result.overall_bias_distribution
        print(f"\nOverall Bias Distribution:")
        print(f"  Left:    {bias.left:.2%}")
        print(f"  Right:   {bias.right:.2%}")
        print(f"  Neutral: {bias.neutral:.2%}")
        print(f"  Dominant: {bias.dominant_bias.upper()}")
    
    # Print facts
    print(f"\n📋 FACTS ({len(result.facts)}):")
    for i, fact in enumerate(result.facts, 1):
        print(f"\n  {i}. {fact.text}")
        print(f"     Type: {fact.opinion_type}")
        if fact.confidence:
            print(f"     Confidence: {fact.confidence:.2%}")
    
    # Print opinions
    print(f"\n💭 OPINIONS ({len(result.opinions)}):")
    for i, opinion in enumerate(result.opinions, 1):
        print(f"\n  {i}. {opinion.text}")
        print(f"     Type: {opinion.opinion_type}")
        bias = opinion.bias_probabilities
        print(f"     Bias: Left={bias.left:.2%}, Right={bias.right:.2%}, Neutral={bias.neutral:.2%}")
        print(f"     Dominant: {bias.dominant_bias.upper()}")
        if opinion.confidence:
            print(f"     Confidence: {opinion.confidence:.2%}")
        if opinion.reasoning:
            print(f"     Reasoning: {opinion.reasoning[:150]}...")
        if opinion.original_sentence:
            print(f"     From: \"{opinion.original_sentence}\"")


def example_1_high_level_api():
    """
    Example 1: Using the high-level API - extract_and_analyze()
    
    This is the simplest way to use the analyzer. Just provide a URL
    and get back a complete analysis with robust error handling.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: High-Level API - extract_and_analyze()")
    print("="*80)
    
    # Create analyzer with CoT enabled (chain_local mode)
    analyzer = WebOpinionAnalyzer(
        execution_mode="chain_local",
        temperature=0.3
    )
    
    # Note: This would require a real URL and API key to work
    # For demonstration, we'll show the API usage
    print("\nUsage:")
    print("  analyzer = WebOpinionAnalyzer(execution_mode='chain_local')")
    print("  result = analyzer.extract_and_analyze('https://example.com/article')")
    print("  if result.extraction_metadata and 'error' in result.extraction_metadata:")
    print("      print(f\"Error: {result.extraction_metadata['error']}\")")
    print("  else:")
    print("      print(f\"Found {len(result.opinions)} opinions\")")


def example_2_step_by_step():
    """
    Example 2: Step-by-step verification
    
    Use individual methods to verify each processing step independently.
    This is useful for debugging or when you need fine-grained control.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Step-by-Step Verification")
    print("="*80)
    
    analyzer = WebOpinionAnalyzer(execution_mode="chain_local")
    
    print("\nStep-by-step usage:")
    print("\n1. Extract HTML:")
    print("   html = analyzer.extract_html('https://example.com/article')")
    print("   if html is None:")
    print("       print('Failed to fetch HTML')")
    
    print("\n2. Clean HTML:")
    print("   text, title = analyzer.clean_html(html)")
    print("   if text is None:")
    print("       print('Failed to clean HTML')")
    print("   else:")
    print("       print(f'Title: {title}')")
    print("       print(f'Text length: {len(text)} chars')")
    
    print("\n3. Analyze Text:")
    print("   result = analyzer.analyze_text(text, url=url, title=title)")
    print("   if result.extraction_metadata and 'error' in result.extraction_metadata:")
    print("       print(f\"Analysis failed: {result.extraction_metadata['error']}\")")
    print("   else:")
    print("       print(f'Found {len(result.opinions)} opinions')")


def example_3_cot_modes():
    """
    Example 3: Chain of Thought (CoT) Modes
    
    Demonstrates the three different execution modes:
    - no_chain: Fastest, no reasoning (good for production/high-volume)
    - chain_local: Local CoT, good balance of speed and interpretability
    - chain_online: Full LLM-driven CoT, most detailed reasoning
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Chain of Thought (CoT) Modes")
    print("="*80)
    
    print("\nMode 1: no_chain (Fastest)")
    print("  analyzer = WebOpinionAnalyzer(execution_mode='no_chain')")
    print("  - No reasoning field")
    print("  - Fastest processing")
    print("  - Best for high-volume production use")
    
    print("\nMode 2: chain_local (Balanced)")
    print("  analyzer = WebOpinionAnalyzer(execution_mode='chain_local')")
    print("  - Includes reasoning field")
    print("  - Good balance of speed and interpretability")
    print("  - Recommended for most use cases")
    
    print("\nMode 3: chain_online (Most Detailed)")
    print("  analyzer = WebOpinionAnalyzer(execution_mode='chain_online')")
    print("  - Full LLM-driven CoT reasoning")
    print("  - Most detailed analysis")
    print("  - Best for research and detailed analysis")
    
    # Example with mock text
    print("\nExample with sample text:")
    sample_text = """
    I strongly support this new policy initiative. It addresses critical issues
    that have been overlooked for years. However, the implementation timeline
    seems unrealistic and could lead to problems.
    """
    
    print("\nInput text:", sample_text.strip())
    print("\nExpected output (chain_local mode):")
    print("  Opinion 1: 'Support for new policy initiative'")
    print("    - Type: opinion")
    print("    - Bias: Left=0.6, Right=0.2, Neutral=0.2")
    print("    - Reasoning: 'Strong support indicates progressive stance...'")
    print("    - From: 'I strongly support this new policy initiative...'")
    print("\n  Opinion 2: 'Concern about implementation timeline'")
    print("    - Type: opinion")
    print("    - Bias: Left=0.2, Right=0.2, Neutral=0.6")
    print("    - Reasoning: 'Critical assessment without partisan alignment...'")
    print("    - From: 'However, the implementation timeline seems unrealistic...'")


def example_4_error_handling():
    """
    Example 4: Robust Error Handling
    
    Shows how the analyzer gracefully handles errors without crashing.
    All errors return valid OpinionExtractionResult objects with error metadata.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Robust Error Handling")
    print("="*80)
    
    analyzer = WebOpinionAnalyzer()
    
    print("\nHandling Network Errors:")
    print("  result = analyzer.extract_and_analyze('http://invalid.url')")
    print("  if result.extraction_metadata and 'error' in result.extraction_metadata:")
    print("      error_type = result.extraction_metadata['error']")
    print("      error_msg = result.extraction_metadata['error_message']")
    print("      print(f'Error: {error_type} - {error_msg}')")
    print("      # Error types: network_error, content_extraction_error, unexpected_error")
    
    print("\nHandling Content Extraction Errors:")
    print("  # When URL returns no meaningful content")
    print("  result = analyzer.extract_and_analyze('https://empty-page.com')")
    print("  if result.extraction_metadata and 'error' in result.extraction_metadata:")
    print("      print('No content extracted from page')")
    
    print("\nHandling LLM Analysis Errors:")
    print("  # When LLM analysis fails")
    print("  result = analyzer.analyze_text('Very short.')")
    print("  if result.extraction_metadata and 'error' in result.extraction_metadata:")
    print("      print('LLM analysis failed')")


def example_5_bias_distribution():
    """
    Example 5: Understanding Bias Probability Distribution
    
    Shows how to interpret the bias probability distribution which is
    the core feature that distinguishes this from simpler classifiers.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Understanding Bias Probability Distribution")
    print("="*80)
    
    print("\nWhat is Bias Probability Distribution?")
    print("  Instead of a single bias score, we provide a probability distribution:")
    print("    - left:    Probability of Left/Progressive bias (0.0 to 1.0)")
    print("    - right:   Probability of Right/Conservative bias (0.0 to 1.0)")
    print("    - neutral: Probability of Neutral/Centrist stance (0.0 to 1.0)")
    print("    - The three probabilities MUST sum to 1.0")
    
    print("\nExample Interpretations:")
    print("\n  Left-leaning opinion:")
    print("    {'left': 0.7, 'right': 0.1, 'neutral': 0.2}")
    print("    → 70% left-aligned, 10% right-aligned, 20% neutral")
    print("    → Dominant bias: LEFT")
    
    print("\n  Right-leaning opinion:")
    print("    {'left': 0.1, 'right': 0.8, 'neutral': 0.1}")
    print("    → 10% left-aligned, 80% right-aligned, 10% neutral")
    print("    → Dominant bias: RIGHT")
    
    print("\n  Neutral/Centrist opinion:")
    print("    {'left': 0.2, 'right': 0.2, 'neutral': 0.6}")
    print("    → 20% left-aligned, 20% right-aligned, 60% neutral")
    print("    → Dominant bias: NEUTRAL")
    
    print("\n  Mixed/Ambiguous opinion:")
    print("    {'left': 0.4, 'right': 0.4, 'neutral': 0.2}")
    print("    → 40% left-aligned, 40% right-aligned, 20% neutral")
    print("    → Dominant bias: LEFT (by alphabetical order, effectively a tie)")
    
    print("\nBackward Compatibility:")
    print("  For legacy code, you can still get a single bias score:")
    print("    bias_score = opinion.bias_probabilities.bias_score")
    print("    # Returns value from -1.0 (left) to +1.0 (right)")
    print("    # Calculated as: -1.0*left + 0.0*neutral + 1.0*right")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("WebOpinionAnalyzer - Usage Examples")
    print("="*80)
    print("\nThis script demonstrates the new WebOpinionAnalyzer module with:")
    print("  ✓ High-level API: extract_and_analyze(url)")
    print("  ✓ Step-by-step verification: extract_html, clean_html, analyze_text")
    print("  ✓ Chain of Thought (CoT) support with 3 modes")
    print("  ✓ Bias probability distribution (left, right, neutral)")
    print("  ✓ Robust error handling")
    print("  ✓ Atomic opinion extraction")
    print("  ✓ Fact vs. opinion classification")
    
    # Run examples
    example_1_high_level_api()
    example_2_step_by_step()
    example_3_cot_modes()
    example_4_error_handling()
    example_5_bias_distribution()
    
    print("\n" + "="*80)
    print("Examples Complete!")
    print("="*80)
    print("\nTo use with real URLs, you need to:")
    print("  1. Set OPENAI_API_KEY environment variable")
    print("  2. Provide a valid news article URL")
    print("  3. Run: python example_web_opinion_analyzer.py")
    print("\nFor more details, see:")
    print("  - src/agents/web_opinion_extractor/agent.py")
    print("  - test_web_opinion_extractor.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
