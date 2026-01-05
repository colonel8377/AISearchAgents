"""
Asynchronous batch demographic evaluation script.

This script performs efficient, asynchronous batch demographic evaluation using:
- httpx (async) for API calls
- asyncio and Semaphore for concurrency and rate limiting
- "Fetch one, save one" strategy: save each result immediately to SQLite
- Optional Pandas/Feather support for loading sentences

Usage:
    python example/demographic_batch_evaluation.py
"""

import asyncio
import json
import os
import sqlite3
import time
from datetime import datetime
from itertools import combinations, product
from typing import Generator, Dict, List, Any, Optional

import httpx

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

API_BASE_URL = "http://localhost:8000"
API_ENDPOINT = "/api/v1/agent/demographic-evaluator/evaluate"
# Using a semaphore to limit concurrent requests (effectively rate limiting)
MAX_CONCURRENT_REQUESTS = 5
# Database path
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SQLITE_DB_PATH = os.path.join(DATA_DIR, "demographic_evaluation.db")

# =============================================================================
# SQLite Storage
# =============================================================================

def init_db(db_path: str = SQLITE_DB_PATH) -> None:
    """Initialize database tables."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # evaluation_runs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS evaluation_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_personas INTEGER,
            successful_evaluations INTEGER DEFAULT 0,
            failed_evaluations INTEGER DEFAULT 0,
            num_sentences INTEGER,
            use_cot TEXT,
            use_few_shots INTEGER  -- 0 for False, 1 for True
        )
    ''')
    
    # personas table
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
    
    # judgments table
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
    
    # Indexes
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_judgments_run_id ON judgments(run_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_judgments_persona_id ON judgments(persona_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_personas_run_id ON personas(run_id)')
    
    conn.commit()
    conn.close()

def create_run_entry(num_sentences: int, use_cot: str, use_few_shots: bool, total_personas: int, db_path: str = SQLITE_DB_PATH) -> str:
    """Create a new run entry and return the run_id."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO evaluation_runs (run_id, total_personas, num_sentences, use_cot, use_few_shots)
        VALUES (?, ?, ?, ?, ?)
    ''', (run_id, total_personas, num_sentences, use_cot, int(use_few_shots)))
    conn.commit()
    conn.close()
    return run_id

def save_single_result(run_id: str, result: Dict[str, Any], db_path: str = SQLITE_DB_PATH) -> None:
    """Save a single evaluation result (persona + judgments) to SQLite."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
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
    if result.get('judgments'):
        judgment_data = []
        for judgment in result['judgments']:
            judgment_data.append((
                run_id,
                persona_id,
                judgment.get('index'),
                judgment.get('sentence'),
                judgment.get('agree'),
                judgment.get('reason')
            ))
        
        cursor.executemany('''
            INSERT INTO judgments (run_id, persona_id, sentence_index, sentence, agree, reason)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', judgment_data)
        
    conn.commit()
    conn.close()

def update_run_stats(run_id: str, successful: int, failed: int, db_path: str = SQLITE_DB_PATH) -> None:
    """Update the run statistics after completion."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE evaluation_runs 
        SET successful_evaluations = ?, failed_evaluations = ?
        WHERE run_id = ?
    ''', (successful, failed, run_id))
    conn.commit()
    conn.close()

# =============================================================================
# Sentence Loading Helpers
# =============================================================================

def extract_sentence_index(sentence: str) -> Optional[int]:
    """Extract sentence index from bracketed prefix like '[1] sentence text'.
    
    Returns:
        int: The extracted index, or None if no valid index is found
    """
    if sentence.startswith('['):
        end_bracket = sentence.find(']')
        if end_bracket > 0:
            try:
                return int(sentence[1:end_bracket])
            except ValueError:
                pass
    return None

def remove_bracket_prefix(sentence: str) -> str:
    """Remove bracketed index prefix from sentence."""
    if sentence.startswith('['):
        end_bracket = sentence.find(']')
        if end_bracket > 0:
            return sentence[end_bracket + 1:].strip()
    return sentence

def load_sentences_by_article(feather_path: str, article_id: str) -> List[str]:
    """Load sentences for a specific article from feather file."""
    if not HAS_PANDAS:
        raise ImportError("pandas is required to load feather files. Install with: pip install pandas pyarrow")
    
    df = pd.read_feather(feather_path)
    article_df = df[df['article_id'] == article_id]
    
    if article_df.empty:
        return []
    
    sentences = article_df['sentence'].tolist()
    return [remove_bracket_prefix(s) for s in sentences]

def load_all_articles_sentences(feather_path: str) -> Dict[str, List[str]]:
    """Load all sentences grouped by article from feather file."""
    if not HAS_PANDAS:
        raise ImportError("pandas is required to load feather files. Install with: pip install pandas pyarrow")
    
    df = pd.read_feather(feather_path)
    articles = {}
    
    for article_id in df['article_id'].unique():
        articles[article_id] = load_sentences_by_article(feather_path, article_id)
    
    return articles

def load_sentences_from_feather(feather_path: str, limit: Optional[int] = None) -> List[str]:
    """Load sentences from feather file (flatten all articles)."""
    if not HAS_PANDAS:
        raise ImportError("pandas is required to load feather files. Install with: pip install pandas pyarrow")
    
    df = pd.read_feather(feather_path)
    sentences = df['sentence'].tolist()
    sentences = [remove_bracket_prefix(s) for s in sentences]
    
    if limit:
        sentences = sentences[:limit]
    
    return sentences

# =============================================================================
# Demographic Combination Helpers
# =============================================================================

def generate_demographic_combinations(demographics: Dict[str, List[str]]) -> Generator[Dict[str, str], None, None]:
    """
    Generate all combinations of demographic attributes.
    
    Args:
        demographics: Dict mapping attribute names to lists of possible values
        
    Yields:
        Dict representing a single demographic persona
    """
    keys = list(demographics.keys())
    values = list(demographics.values())
    
    for combination in product(*values):
        yield dict(zip(keys, combination))

def get_full_demographic_combinations() -> List[Dict[str, str]]:
    """
    Generate the full set of demographic combinations.
    
    Returns:
        List of demographic persona dictionaries
    """
    demographics = {
        'gender': ['male', 'female'],
        'age': ['18-24', '25-34', '35-44', '45-54', '55-64', '65+'],
        'ethnicity': ['white', 'black', 'hispanic', 'asian', 'other'],
        'degree': ['high_school', 'bachelor', 'master', 'phd'],
        'politicsParty': ['democrat', 'republican', 'independent'],
        'politicalStance': ['liberal', 'moderate', 'conservative'],
        'parent': ['yes', 'no']
    }
    
    return list(generate_demographic_combinations(demographics))

# =============================================================================
# Async Evaluation Logic
# =============================================================================

async def evaluate_persona_async(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    persona: Dict[str, str],
    sentences: List[str],
    use_cot: str,
    use_few_shots: bool,
    run_id: str
) -> Dict[str, Any]:
    """Evaluate a single persona asynchronously and save result immediately."""
    url = f"{API_BASE_URL}{API_ENDPOINT}"
    payload = {
        "demography_json": persona,
        "sentences": sentences,
        "use_cot": use_cot,
        "use_few_shots": use_few_shots
    }
    
    result = {
        "persona": persona,
        "judgments": [],
        "success": False,
        "error": None
    }
    
    async with semaphore:
        try:
            response = await client.post(url, json=payload, timeout=60.0)
            response.raise_for_status()
            data = response.json()
            result["judgments"] = data.get("judgments", [])
            result["success"] = True
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            result["error"] = f"HTTP error: {str(e)}"
            print(f"\nHTTP error evaluating persona {persona}: {e}")
        except Exception as e:
            result["error"] = f"Unexpected error: {str(e)}"
            print(f"\nUnexpected error evaluating persona {persona}: {e}")

    # Fetch one, save one
    save_single_result(run_id, result)
    
    # Progress indicator
    print(".", end="", flush=True)
    return result

async def run_batch_evaluation_async(
    sentences: List[str],
    demographic_combinations: List[Dict[str, str]],
    use_cot: str = "no_chain",
    use_few_shots: bool = False
) -> None:
    """Run batch evaluation asynchronously."""
    
    # Init DB
    init_db()
    
    # Create Run
    run_id = create_run_entry(
        num_sentences=len(sentences),
        use_cot=use_cot,
        use_few_shots=use_few_shots,
        total_personas=len(demographic_combinations)
    )
    print(f"Started Run ID: {run_id}")
    print(f"Evaluating {len(demographic_combinations)} personas with {len(sentences)} sentences...")
    print(f"Max concurrent requests: {MAX_CONCURRENT_REQUESTS}")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    start_time = time.time()
    
    async with httpx.AsyncClient() as client:
        tasks = [
            evaluate_persona_async(
                client, semaphore, persona, sentences, use_cot, use_few_shots, run_id
            )
            for persona in demographic_combinations
        ]
        
        results = await asyncio.gather(*tasks)
    
    elapsed_time = time.time() - start_time
    
    # Calculate stats
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    update_run_stats(run_id, successful, failed)
    
    print(f"\n\nBatch evaluation complete!")
    print(f"Run ID: {run_id}")
    print(f"Total personas: {len(demographic_combinations)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Average time per persona: {elapsed_time/len(demographic_combinations):.2f} seconds")
    print(f"Database: {SQLITE_DB_PATH}")

# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for the script."""
    
    # Example sentences if no feather file
    sentences = [
        "The government should increase funding for public education.",
        "Climate change requires immediate global action.",
        "Healthcare should be a universal right.",
        "Tax cuts stimulate economic growth.",
        "Immigration policies should prioritize border security."
    ]
    
    # If you have a feather file with sentences, uncomment and modify:
    # feather_path = os.path.join(DATA_DIR, "sentences.feather")
    # if os.path.exists(feather_path) and HAS_PANDAS:
    #     try:
    #         sentences = load_sentences_from_feather(feather_path, limit=10)
    #         print(f"Loaded {len(sentences)} sentences from feather file")
    #     except Exception as e:
    #         print(f"Error loading feather file: {e}")
    #         print("Using default sentences instead")
    
    # Generate personas
    # For full demographic combinations (warning: this generates many combinations):
    # test_personas = get_full_demographic_combinations()
    
    # For testing, use a smaller subset:
    test_personas = list(generate_demographic_combinations({
        'gender': ['male', 'female'],
        'age': ['25-34', '45-54'],
        'politicalStance': ['liberal', 'conservative']
    }))
    
    print(f"\n{'='*60}")
    print(f"Demographic Batch Evaluation")
    print(f"{'='*60}")
    print(f"Number of personas to evaluate: {len(test_personas)}")
    print(f"Number of sentences per persona: {len(sentences)}")
    print(f"Total API calls: {len(test_personas)}")
    print(f"{'='*60}\n")
    
    # Run async batch evaluation
    asyncio.run(run_batch_evaluation_async(
        sentences=sentences,
        demographic_combinations=test_personas,
        use_cot="no_chain",
        use_few_shots=False
    ))

if __name__ == "__main__":
    main()
