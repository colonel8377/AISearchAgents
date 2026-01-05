"""
Example: Batch Demographic Evaluation with Feather Files

This script demonstrates how to use the Demographic Evaluator API to run
batch evaluations across multiple demographic combinations on pre-split
sentences stored in a feather file.

IMPORTANT: Pass sentences as an array (List[str]) to the API.
The API will NOT split the sentences when a list is provided - use pre-split sentences.

DataFrame format expected:
    Columns: ['article_id', 'sentence', 'target', 'source_bias']
    Example row: 5.0, "[0]: The wife of Kentucky...", 2, right

Results are stored in SQLite database at: data/demographic_evaluation.db

Requirements:
    pip install pandas pyarrow requests

Usage:
    1. Start the API server: uvicorn src.presentation.api.main:app --reload
    2. Update FEATHER_PATH to point to your feather file
    3. Run: python example/demographic_batch_evaluation.py

Quick integration example:
    >>> import pandas as pd
    >>> import requests
    >>>
    >>> # Load pre-split sentences from feather
    >>> df = pd.read_feather("your_file.feather")
    >>> sentences = df["sentence"].tolist()  # Get as list
    >>>
    >>> response = requests.post(
    ...     "http://localhost:8000/api/v1/agent/demographic-evaluator/evaluate",
    ...     json={
    ...         "demography_json": {"gender": "male", "age": "25to34", ...},
    ...         "sentences": sentences,  # Pass as array!
    ...         "use_cot": "no_chain",
    ...         "use_few_shots": False  # Disable few-shot examples
    ...     }
    ... )
    >>> result = response.json()
    >>> # result["judgments"] contains: index, sentence, agree (0/1), reason
"""

import json
import os
import sqlite3
import time
from datetime import datetime
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

# SQLite database path (in data directory)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SQLITE_DB_PATH = os.path.join(DATA_DIR, "demographic_evaluation.db")

# Rate limiting (requests per second)
RATE_LIMIT_DELAY = 0.5  # seconds between API calls


# =============================================================================
# SQLite Storage
# =============================================================================

def init_sqlite_db(db_path: str = SQLITE_DB_PATH) -> sqlite3.Connection:
    """
    Initialize SQLite database with required tables.
    
    Args:
        db_path: Path to the SQLite database file
    
    Returns:
        sqlite3.Connection: Database connection
    """
    # Ensure data directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create evaluation_runs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS evaluation_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_personas INTEGER,
            successful_evaluations INTEGER,
            failed_evaluations INTEGER,
            num_sentences INTEGER,
            use_cot TEXT,
            use_few_shots INTEGER
        )
    ''')
    
    # Create personas table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS personas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            persona_json TEXT NOT NULL,
            gender TEXT,
            age TEXT,
            ethnicity TEXT,
            degree TEXT,
            politicsParty TEXT,
            politicalStance TEXT,
            parent TEXT,
            success INTEGER,
            error TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES evaluation_runs(run_id)
        )
    ''')
    
    # Create judgments table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS judgments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            persona_id INTEGER NOT NULL,
            sentence_index INTEGER,
            sentence TEXT,
            agree INTEGER,
            reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES evaluation_runs(run_id),
            FOREIGN KEY (persona_id) REFERENCES personas(id)
        )
    ''')
    
    # Create indexes for faster queries
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_judgments_run_id ON judgments(run_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_judgments_persona_id ON judgments(persona_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_personas_run_id ON personas(run_id)')
    
    conn.commit()
    return conn


def save_results_to_sqlite(
    results: List[Dict[str, Any]],
    num_sentences: int,
    use_cot: str,
    use_few_shots: bool,
    db_path: str = SQLITE_DB_PATH
) -> str:
    """
    Save evaluation results to SQLite database.
    
    Args:
        results: List of evaluation results
        num_sentences: Number of sentences evaluated
        use_cot: Chain of Thought mode used
        use_few_shots: Whether few-shot examples were used
        db_path: Path to the SQLite database file
    
    Returns:
        str: Run ID for this batch
    """
    conn = init_sqlite_db(db_path)
    cursor = conn.cursor()
    
    # Generate unique run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Count successful/failed evaluations
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    # Insert evaluation run
    cursor.execute('''
        INSERT INTO evaluation_runs (run_id, total_personas, successful_evaluations, 
                                      failed_evaluations, num_sentences, use_cot, use_few_shots)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (run_id, len(results), successful, failed, num_sentences, use_cot, int(use_few_shots)))
    
    # Insert personas and judgments
    for result in results:
        persona = result['persona']
        
        # Insert persona
        cursor.execute('''
            INSERT INTO personas (run_id, persona_json, gender, age, ethnicity, degree,
                                   politicsParty, politicalStance, parent, success, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            run_id,
            json.dumps(persona),
            persona.get('gender'),
            persona.get('age'),
            persona.get('ethnicity'),
            persona.get('degree'),
            persona.get('politicsParty'),
            persona.get('politicalStance'),
            persona.get('parent'),
            int(result['success']),
            result.get('error')
        ))
        
        persona_id = cursor.lastrowid
        
        # Insert judgments
        for judgment in result.get('judgments', []):
            cursor.execute('''
                INSERT INTO judgments (run_id, persona_id, sentence_index, sentence, agree, reason)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                run_id,
                persona_id,
                judgment.get('index'),
                judgment.get('sentence'),
                judgment.get('agree'),
                judgment.get('reason')
            ))
    
    conn.commit()
    conn.close()
    
    print(f"Results saved to SQLite: {db_path}")
    print(f"Run ID: {run_id}")
    
    return run_id


def query_results_from_sqlite(
    run_id: Optional[str] = None,
    db_path: str = SQLITE_DB_PATH
) -> Dict[str, Any]:
    """
    Query evaluation results from SQLite database.
    
    Args:
        run_id: Specific run ID to query (None for latest)
        db_path: Path to the SQLite database file
    
    Returns:
        dict: Query results
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get run info
    if run_id:
        cursor.execute('SELECT * FROM evaluation_runs WHERE run_id = ?', (run_id,))
    else:
        cursor.execute('SELECT * FROM evaluation_runs ORDER BY created_at DESC LIMIT 1')
    
    run = cursor.fetchone()
    if not run:
        conn.close()
        return {"error": "No runs found"}
    
    run_id = run['run_id']
    
    # Get personas
    cursor.execute('SELECT * FROM personas WHERE run_id = ?', (run_id,))
    personas = cursor.fetchall()
    
    # Get judgments
    cursor.execute('SELECT * FROM judgments WHERE run_id = ?', (run_id,))
    judgments = cursor.fetchall()
    
    conn.close()
    
    return {
        "run_id": run_id,
        "created_at": run['created_at'],
        "total_personas": run['total_personas'],
        "successful_evaluations": run['successful_evaluations'],
        "failed_evaluations": run['failed_evaluations'],
        "num_sentences": run['num_sentences'],
        "use_cot": run['use_cot'],
        "use_few_shots": bool(run['use_few_shots']),
        "personas": [dict(p) for p in personas],
        "judgments": [dict(j) for j in judgments]
    }


def list_runs_from_sqlite(db_path: str = SQLITE_DB_PATH) -> List[Dict[str, Any]]:
    """
    List all evaluation runs from SQLite database.
    
    Args:
        db_path: Path to the SQLite database file
    
    Returns:
        List of run summaries
    """
    if not os.path.exists(db_path):
        return []
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM evaluation_runs ORDER BY created_at DESC')
    runs = cursor.fetchall()
    
    conn.close()
    
    return [dict(r) for r in runs]

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
# Article/Sentences Loading
# =============================================================================

def load_sentences_from_feather(feather_path: str, sentence_column: str = "sentence") -> List[str]:
    """
    Load pre-split sentences from a feather file.
    
    Args:
        feather_path: Path to the feather file
        sentence_column: Name of the column containing sentences
    
    Returns:
        List[str]: List of sentences
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required. Install with: pip install pandas pyarrow")
    
    df = pd.read_feather(feather_path)
    
    if sentence_column not in df.columns:
        available = ", ".join(df.columns.tolist())
        raise ValueError(f"Column '{sentence_column}' not found. Available columns: {available}")
    
    return df[sentence_column].tolist()


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
    use_few_shots: bool = False,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Call the Demographic Evaluator API.
    
    Args:
        demography_json: Demographic profile as dictionary
        sentences: List of pre-split sentences to evaluate (passed as array to API)
        use_cot: Chain of Thought mode ('chain_online', 'chain_local', 'no_chain')
        use_few_shots: Whether to use few-shot examples (default: False)
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
    sentences: List[str],
    demographic_combinations: List[Dict[str, str]],
    use_cot: str = "no_chain",
    use_few_shots: bool = False,
    save_results: bool = True,
    save_to_sqlite: bool = True,
    output_file: str = "evaluation_results.json"
) -> List[Dict[str, Any]]:
    """
    Run batch evaluation across multiple demographic combinations.
    
    Args:
        sentences: List of pre-split sentences to evaluate (passed as array to API)
        demographic_combinations: List of demographic profiles to test
        use_cot: Chain of Thought mode
        use_few_shots: Whether to use few-shot examples (default: False)
        save_results: Whether to save results to a JSON file
        save_to_sqlite: Whether to save results to SQLite database (default: True)
        output_file: Output file path for JSON results
    
    Returns:
        List of results for each demographic combination
    """
    results = []
    total = len(demographic_combinations)
    
    print(f"Starting batch evaluation with {total} demographic combinations...")
    print(f"Number of sentences: {len(sentences)}")
    print("-" * 50)
    
    for i, persona in enumerate(demographic_combinations, 1):
        print(f"[{i}/{total}] Evaluating persona: {persona}")
        
        try:
            response = evaluate_sentences_api(
                demography_json=persona,
                sentences=sentences,  # Pass as array (pre-split)
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
    
    # Save results to JSON
    if save_results:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to JSON: {output_file}")
    
    # Save results to SQLite
    if save_to_sqlite:
        run_id = save_results_to_sqlite(
            results=results,
            num_sentences=len(sentences),
            use_cot=use_cot,
            use_few_shots=use_few_shots
        )
        print(f"Results saved to SQLite with run_id: {run_id}")
    
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
    
    # Example 1: Using pre-split sentences as a list (RECOMMENDED)
    # Sentences from user's feather format: "[0]: The wife of Kentucky..."
    # The API receives the sentences exactly as provided (no splitting)
    sample_sentences = [
        "[0]: The wife of Kentucky State Rep. Dan Johnson announced Thursday that she would pursue her husband's seat.",
        "[1]: Dan Johnson, a preacher and a Republican, committed suicide Wednesday on a bridge.",
        "[2]: Washington, according to Bullitt County Sheriff Donnie Tinnell.",
        "[3]: The government should increase funding for public education.",
        "[4]: Climate change requires immediate global action.",
        "[5]: Tax cuts for corporations stimulate economic growth.",
        "[6]: Universal healthcare should be a fundamental right.",
        "[7]: Gun ownership is a constitutional right that must be protected.",
    ]
    
    # Example 2: Load pre-split sentences from feather file
    # DataFrame columns: ['article_id', 'sentence', 'target', 'source_bias']
    # sentences = load_sentences_from_feather(FEATHER_PATH, sentence_column="sentence")
    
    # Use sample sentences for this demo
    sentences = sample_sentences
    
    # Get a subset of demographic combinations for testing
    # Full run: all_personas = get_full_demographic_combinations()  # 3240 combinations
    # Test run: just a few combinations
    test_personas = [
        {"gender": "male", "age": "35to44", "politicsParty": "republican", "politicalStance": "conservative"},
        {"gender": "female", "age": "25to34", "politicsParty": "democrats", "politicalStance": "liberal"},
        {"gender": "non-binary", "age": "18to24", "politicsParty": "independent", "politicalStance": "neutral"},
    ]
    
    # Run batch evaluation
    # NOTE: use_few_shots=False as requested - no few-shot examples in prompt
    # Results are saved to both JSON file and SQLite database (data/demographic_evaluation.db)
    results = run_batch_evaluation(
        sentences=sentences,  # Pass as array (pre-split sentences)
        demographic_combinations=test_personas,
        use_cot="no_chain",  # Use "chain_local" for more detailed reasoning
        use_few_shots=False,  # Disable few-shot examples
        save_results=True,  # Save to JSON
        save_to_sqlite=True,  # Save to SQLite (data/demographic_evaluation.db)
        output_file="demographic_evaluation_results.json"
    )
    
    # Analyze results
    analysis = analyze_results(results)
    print("\n=== Analysis Summary ===")
    print(json.dumps(analysis, indent=2))
    
    # Show how to query results from SQLite
    print("\n=== SQLite Query Example ===")
    print(f"Database location: {SQLITE_DB_PATH}")
    runs = list_runs_from_sqlite()
    print(f"Total runs in database: {len(runs)}")
    if runs:
        print(f"Latest run: {runs[0]['run_id']}")


if __name__ == "__main__":
    main()
