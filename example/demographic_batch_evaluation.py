"""
Example: Batch Demographic Evaluation with Feather Files

This script demonstrates how to use the Demographic Evaluator API to run
batch evaluations across multiple demographic combinations on articles
stored in a feather file.

Requirements:
    pip install pandas pyarrow requests

Usage:
    1. Start the API server: uvicorn src.presentation.api.main:app --reload
    2. Update FEATHER_PATH to point to your feather file
    3. Run: python example/demographic_batch_evaluation.py
"""

import json
import time
from itertools import combinations, product
from typing import Generator, Dict, List, Any, Optional

import requests

# Optional: for feather file support
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not installed. Install with: pip install pandas pyarrow")

# =============================================================================
# Configuration
# =============================================================================

# API Configuration
API_BASE_URL = "http://localhost:8000"
API_ENDPOINT = "/api/v1/agent/demographic-evaluator/evaluate"
API_KEY = None  # Set if API_KEY_REQUIRED=true in your .env

# Feather file path (update this to your file)
FEATHER_PATH = "your_articles.feather"

# Rate limiting (requests per second)
RATE_LIMIT_DELAY = 0.5  # seconds between API calls

# =============================================================================
# Demographic Combinations Generator
# =============================================================================

DEMOGRAPHIC_DETAILS = {
    'gender': ['male', 'female', 'non-binary'],
    'age': ['18to24', '25to34', '35to44', '45to54', '55to64', '65plus'],
    'ethnicity': ['white', 'black', 'hispanic', 'asian', 'other'],
    'degree': ['yes', 'no'],
    'politicsParty': ['democrats', 'republican', 'independent'],
    'politicalStance': ['liberal', 'conservative', 'neutral'],
    'parent': ['yes', 'no']
}


def generate_demographic_combinations(
    demographic_dict: Dict[str, List[str]],
    min_features: int = 1,
    max_features: Optional[int] = None
) -> Generator[Dict[str, str], None, None]:
    """
    Generate all possible combinations of demographic attributes.
    
    Args:
        demographic_dict: Dictionary with demographic features and their values
        min_features: Minimum number of features in each combination (default: 1)
        max_features: Maximum number of features (default: all features)
    
    Yields:
        dict: Each unique combination as a dictionary
    """
    features = list(demographic_dict.keys())
    max_features = max_features or len(features)
    
    for r in range(min_features, max_features + 1):
        for feature_combo in combinations(features, r):
            values_for_combo = [demographic_dict[feature] for feature in feature_combo]
            for value_combo in product(*values_for_combo):
                yield dict(zip(feature_combo, value_combo))


def get_full_demographic_combinations() -> List[Dict[str, str]]:
    """Get all combinations with all 7 demographic features (3240 combinations)."""
    return [
        combo for combo in generate_demographic_combinations(DEMOGRAPHIC_DETAILS)
        if len(combo) == 7
    ]


# =============================================================================
# Article Loading
# =============================================================================

def load_article_from_feather(feather_path: str, text_column: str = "text") -> str:
    """
    Load article text from a feather file.
    
    Args:
        feather_path: Path to the feather file
        text_column: Name of the column containing article text
    
    Returns:
        str: Article text
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required. Install with: pip install pandas pyarrow")
    
    df = pd.read_feather(feather_path)
    
    if text_column not in df.columns:
        available = ", ".join(df.columns.tolist())
        raise ValueError(f"Column '{text_column}' not found. Available columns: {available}")
    
    # Return the first article or concatenate all
    return df[text_column].iloc[0]


def load_articles_from_feather(feather_path: str, text_column: str = "text") -> List[str]:
    """
    Load all articles from a feather file.
    
    Args:
        feather_path: Path to the feather file
        text_column: Name of the column containing article text
    
    Returns:
        List[str]: List of article texts
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required. Install with: pip install pandas pyarrow")
    
    df = pd.read_feather(feather_path)
    return df[text_column].tolist()


# =============================================================================
# API Client
# =============================================================================

def evaluate_sentences_api(
    demography_json: Dict[str, Any],
    sentences: List[str],
    use_cot: str = "no_chain",
    use_few_shots: bool = True,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Call the Demographic Evaluator API.
    
    Args:
        demography_json: Demographic profile as dictionary
        sentences: List of sentences to evaluate
        use_cot: Chain of Thought mode ('chain_online', 'chain_local', 'no_chain')
        use_few_shots: Whether to use few-shot examples
        api_key: Optional API key for authentication
    
    Returns:
        dict: API response with judgments
    """
    url = f"{API_BASE_URL}{API_ENDPOINT}"
    
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    
    payload = {
        "demography_json": demography_json,
        "sentences": sentences,
        "use_cot": use_cot,
        "use_few_shots": use_few_shots
    }
    
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    
    return response.json()


# =============================================================================
# Batch Evaluation
# =============================================================================

def run_batch_evaluation(
    article_text: str,
    demographic_combinations: List[Dict[str, str]],
    use_cot: str = "no_chain",
    use_few_shots: bool = True,
    save_results: bool = True,
    output_file: str = "evaluation_results.json"
) -> List[Dict[str, Any]]:
    """
    Run batch evaluation across multiple demographic combinations.
    
    Args:
        article_text: Article text (will be split into sentences by the API)
        demographic_combinations: List of demographic profiles to test
        use_cot: Chain of Thought mode
        use_few_shots: Whether to use few-shot examples
        save_results: Whether to save results to a file
        output_file: Output file path for results
    
    Returns:
        List of results for each demographic combination
    """
    results = []
    total = len(demographic_combinations)
    
    print(f"Starting batch evaluation with {total} demographic combinations...")
    print(f"Article length: {len(article_text)} characters")
    print("-" * 50)
    
    for i, persona in enumerate(demographic_combinations, 1):
        print(f"[{i}/{total}] Evaluating persona: {persona}")
        
        try:
            response = evaluate_sentences_api(
                demography_json=persona,
                sentences=article_text,  # API will split into sentences
                use_cot=use_cot,
                use_few_shots=use_few_shots,
                api_key=API_KEY
            )
            
            result = {
                "persona": persona,
                "judgments": response.get("judgments", []),
                "success": True,
                "error": None
            }
            
            # Summary stats
            judgments = response.get("judgments", [])
            agree_count = sum(1 for j in judgments if j.get("agree") == 1)
            disagree_count = len(judgments) - agree_count
            print(f"    → {len(judgments)} sentences evaluated: {agree_count} agree, {disagree_count} disagree")
            
        except requests.exceptions.RequestException as e:
            print(f"    → ERROR: {e}")
            result = {
                "persona": persona,
                "judgments": [],
                "success": False,
                "error": str(e)
            }
        
        results.append(result)
        
        # Rate limiting
        if i < total:
            time.sleep(RATE_LIMIT_DELAY)
    
    print("-" * 50)
    print(f"Batch evaluation complete. {sum(1 for r in results if r['success'])}/{total} successful.")
    
    # Save results
    if save_results:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_file}")
    
    return results


def analyze_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze batch evaluation results.
    
    Args:
        results: List of evaluation results
    
    Returns:
        dict: Analysis summary
    """
    successful = [r for r in results if r['success']]
    
    if not successful:
        return {"error": "No successful evaluations"}
    
    # Aggregate agreement rates by demographic feature
    feature_agreement = {}
    
    for result in successful:
        persona = result['persona']
        judgments = result['judgments']
        
        if not judgments:
            continue
        
        agree_rate = sum(1 for j in judgments if j.get('agree') == 1) / len(judgments)
        
        for feature, value in persona.items():
            if feature not in feature_agreement:
                feature_agreement[feature] = {}
            if value not in feature_agreement[feature]:
                feature_agreement[feature][value] = []
            feature_agreement[feature][value].append(agree_rate)
    
    # Calculate averages
    summary = {}
    for feature, values in feature_agreement.items():
        summary[feature] = {
            value: {
                "mean_agreement_rate": sum(rates) / len(rates),
                "sample_count": len(rates)
            }
            for value, rates in values.items()
        }
    
    return {
        "total_evaluations": len(results),
        "successful_evaluations": len(successful),
        "agreement_by_demographic": summary
    }


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main function demonstrating batch demographic evaluation."""
    
    # Example 1: Using a sample article text directly
    sample_article = """
    The government should increase funding for public education.
    Climate change requires immediate global action.
    Tax cuts for corporations stimulate economic growth.
    Universal healthcare should be a fundamental right.
    Gun ownership is a constitutional right that must be protected.
    Immigration strengthens our economy and cultural diversity.
    Traditional family values are essential for society.
    Social media platforms need stronger content moderation.
    """
    
    # Example 2: Load from feather file (uncomment when you have a feather file)
    # article_text = load_article_from_feather(FEATHER_PATH, text_column="article_text")
    
    # Use sample article for this demo
    article_text = sample_article
    
    # Get a subset of demographic combinations for testing
    # Full run: all_personas = get_full_demographic_combinations()  # 3240 combinations
    # Test run: just a few combinations
    test_personas = [
        {"gender": "male", "age": "35to44", "politicsParty": "republican", "politicalStance": "conservative"},
        {"gender": "female", "age": "25to34", "politicsParty": "democrats", "politicalStance": "liberal"},
        {"gender": "non-binary", "age": "18to24", "politicsParty": "independent", "politicalStance": "neutral"},
    ]
    
    # Run batch evaluation
    results = run_batch_evaluation(
        article_text=article_text,
        demographic_combinations=test_personas,
        use_cot="no_chain",  # Use "chain_local" for more detailed reasoning
        use_few_shots=True,
        save_results=True,
        output_file="demographic_evaluation_results.json"
    )
    
    # Analyze results
    analysis = analyze_results(results)
    print("\n=== Analysis Summary ===")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
