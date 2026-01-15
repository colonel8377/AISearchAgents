import asyncio
import hashlib
import json
import os
import re
import sqlite3
import time
from datetime import datetime
from itertools import product
from typing import List, Dict, Any, Optional

import aiohttp
import pandas as pd

API_BASE_URL = "http://localhost:8000"
API_ENDPOINT = "/api/v1/agent/demographic-evaluator/evaluate"
API_KEY = None

FEATHER_PATH = "../data/biased_sentences.feather"

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SQLITE_DB_PATH = os.path.join(DATA_DIR, "demographic_evaluation.db")

MAX_CONCURRENT_REQUESTS = 10

FORCE_NEW_RUN = False
RESUME_RUN_ID: Optional[str] = '20260107_181322'


def extract_index_from_sentence(text: str) -> int:
    if not isinstance(text, str):
        return 999999
    match = re.match(r'\[\s*(\d+)\s*\]:', text.strip())
    return int(match.group(1)) if match else 999999


def clean_sentence(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    cleaned = re.sub(r'^\[\s*\d+\s*\]:\s*', '', text).strip()
    return cleaned if cleaned else None


DEMOGRAPHIC_DETAILS = {
    'gender': ['male', 'female', 'non-binary'],
    'age': ['18to24', '25to34', '35to44', '45to54', '55to64', '65plus'],
    'ethnicity': ['white', 'black', 'hispanic', 'asian', 'other'],
    'degree': ['yes', 'no'],
    'politicsParty': ['democrats', 'republican', 'independent'],
    'politicalStance': ['liberal', 'conservative', 'neutral'],
    'parent': ['yes', 'no']
}

REQUIRED_FIELDS = list(DEMOGRAPHIC_DETAILS.keys())


def get_full_demographic_combinations() -> List[Dict[str, str]]:
    values = [DEMOGRAPHIC_DETAILS[k] for k in REQUIRED_FIELDS]
    return [dict(zip(REQUIRED_FIELDS, combo)) for combo in product(*values)]


def init_sqlite_db(db_path: str = SQLITE_DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute('''
        CREATE TABLE IF NOT EXISTS evaluation_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_personas INTEGER,
            successful_evaluations INTEGER DEFAULT 0,
            failed_evaluations INTEGER DEFAULT 0,
            num_articles INTEGER,
            use_cot TEXT,
            use_few_shots INTEGER,
            per_message boolean DEFAULT 0,
            is_binary_agreement boolean DEFAULT 0
        )
    ''')

    cur.execute('''
        CREATE TABLE IF NOT EXISTS personas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            persona_json TEXT NOT NULL,
            persona_hash TEXT NOT NULL,
            gender TEXT NOT NULL,
            age TEXT NOT NULL,
            ethnicity TEXT NOT NULL,
            degree TEXT NOT NULL,
            politicsParty TEXT NOT NULL,
            politicalStance TEXT NOT NULL,
            parent TEXT NOT NULL,
            success INTEGER,
            error TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(run_id, persona_hash)
        )
    ''')

    cur.execute('''
        CREATE TABLE IF NOT EXISTS judgments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            persona_id INTEGER NOT NULL,
            article_id TEXT,
            sentence_index INTEGER,
            sentence TEXT,
            agree FLOAT,
            reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cur.execute('CREATE INDEX IF NOT EXISTS idx_personas_run_id ON personas(run_id)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_judgments_persona ON judgments(persona_id)')

    conn.commit()
    return conn


def get_run_config(conn: sqlite3.Connection, run_id: str) -> Optional[Dict[str, Any]]:
    """Get configuration for an existing run from database."""
    cur = conn.cursor()
    cur.execute('''
        SELECT use_cot, use_few_shots, per_message, is_binary_agreement
        FROM evaluation_runs WHERE run_id = ?
    ''', (run_id,))
    row = cur.fetchone()
    if row:
        return {
            'use_cot': row[0],
            'use_few_shots': bool(row[1]),
            'per_message': bool(row[2]),
            'is_binary_agreement': bool(row[3])
        }
    return None


def get_or_create_run(
        conn: sqlite3.Connection,
        total_personas: int,
        num_articles: int,
        use_cot: str,
        use_few_shots: bool,
        per_message: bool = True,
        is_binary_agreement: bool = False
) -> tuple[str, Optional[Dict[str, Any]]]:
    """
    Get or create a run. Returns (run_id, config_dict).
    If RESUME_RUN_ID is provided, returns its config from database.
    Otherwise, creates a new run.
    """
    cur = conn.cursor()

    if RESUME_RUN_ID:
        config = get_run_config(conn, RESUME_RUN_ID)
        if config:
            print(f"Resuming run: {RESUME_RUN_ID}")
            print(f"Using config from DB: use_cot={config['use_cot']}, use_few_shots={config['use_few_shots']}, "
                  f"per_message={config['per_message']}, is_binary_agreement={config['is_binary_agreement']}")
            return RESUME_RUN_ID, config
        else:
            print(f"Warning: RESUME_RUN_ID '{RESUME_RUN_ID}' not found in database. Creating new run instead.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    cur.execute('''
        INSERT INTO evaluation_runs
        (run_id, total_personas, num_articles, use_cot, use_few_shots, per_message, is_binary_agreement)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
    run_id, total_personas, num_articles, use_cot, int(use_few_shots), int(per_message), int(is_binary_agreement)))
    conn.commit()
    print(f"Created new run: {run_id}")
    return run_id, None


def get_completed_persona_article_pairs(conn: sqlite3.Connection, run_id: str) -> set:
    """
    Get set of (persona_hash, article_id) pairs that have been successfully completed.
    A pair is considered completed if judgments exist for that persona-article combination.
    We don't check persona.success because a persona may succeed on some articles and fail on others.
    """
    cur = conn.cursor()
    cur.execute('''
        SELECT DISTINCT p.persona_hash, j.article_id
        FROM personas p
        JOIN judgments j ON p.id = j.persona_id
        WHERE p.run_id = ? AND j.run_id = ?
    ''', (run_id, run_id))
    return {(row[0], row[1]) for row in cur.fetchall()}


def save_article_judgments(
        conn: sqlite3.Connection,
        run_id: str,
        persona: Dict[str, str],
        article_id: int,
        judgments: List[Dict],
        success: bool,
        error: Optional[str] = None
) -> None:
    """
    Save judgments for a persona-article combination.
    - If persona doesn't exist, create it with success status
    - If persona exists, update success status (but this is per-persona, not per-article)
    - Delete old judgments for this specific article before inserting new ones
    - Only save judgments if successful
    Note: persona.success reflects the last evaluation status, but we check completion
    at the persona-article level by checking if judgments exist.
    """
    cur = conn.cursor()
    persona_json = json.dumps(persona, sort_keys=True)
    persona_hash = hashlib.sha256(persona_json.encode()).hexdigest()

    cur.execute('SELECT id FROM personas WHERE run_id = ? AND persona_hash = ?', (run_id, persona_hash))
    row = cur.fetchone()

    if row:
        persona_id = row[0]
        # Update persona success status (for tracking, but not used for completion check)
        cur.execute('''
            UPDATE personas 
            SET success = ?, error = ?
            WHERE id = ?
        ''', (int(success), error if not success else None, persona_id))

        # Delete old judgments for this specific article before inserting new ones
        cur.execute('''
            DELETE FROM judgments 
            WHERE run_id = ? AND persona_id = ? AND article_id = ?
        ''', (run_id, persona_id, article_id))
    else:
        # Create new persona record
        cur.execute('''
            INSERT INTO personas
            (run_id, persona_json, persona_hash, gender, age, ethnicity, degree,
             politicsParty, politicalStance, parent, success, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            run_id, persona_json, persona_hash,
            persona['gender'], persona['age'], persona['ethnicity'], persona['degree'],
            persona['politicsParty'], persona['politicalStance'], persona['parent'],
            int(success), error if not success else None
        ))
        persona_id = cur.lastrowid

    # Only save judgments if successful
    if success and judgments:
        for j in judgments:
            cur.execute('''
                INSERT INTO judgments
                (run_id, persona_id, article_id, sentence_index, sentence, agree, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                run_id, persona_id, article_id,
                j.get('index'), j.get('sentence'),
                j.get('agree'), j.get('reason')
            ))

    col = 'successful_evaluations' if success else 'failed_evaluations'
    cur.execute(f'UPDATE evaluation_runs SET {col} = {col} + 1 WHERE run_id = ?', (run_id,))
    conn.commit()


# =============================================================================
# Async API Client
# =============================================================================

async def evaluate_article_async(
        session: aiohttp.ClientSession,
        persona: Dict[str, str],
        sentences: List[str],
        use_cot: str,
        use_few_shots: bool,
        per_message: bool,
        is_binary_agreement: bool = False
) -> Dict[str, Any]:
    url = f"{API_BASE_URL}{API_ENDPOINT}"
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY

    payload = {
        "demography_json": persona,
        "sentences": sentences,
        "use_cot": use_cot,
        "use_few_shots": use_few_shots,
        "per_message": per_message,
        "is_binary_agreement": is_binary_agreement
    }

    async with session.post(url, json=payload, headers=headers, timeout=3600) as resp:
        resp.raise_for_status()
        return await resp.json()


def get_existing_judgments(
        conn: sqlite3.Connection,
        run_id: str,
        persona: Dict[str, str],
        article_id: int
) -> Optional[List[Dict[str, Any]]]:
    """
    Get existing judgments from database for a persona-article combination.
    Only returns judgments if judgments exist for this specific persona-article combination.
    Returns None if no judgments exist (failed or not yet run), so it will be retried.
    We don't check persona.success because a persona may succeed on some articles and fail on others.
    """
    cur = conn.cursor()
    persona_json = json.dumps(persona, sort_keys=True)
    persona_hash = hashlib.sha256(persona_json.encode()).hexdigest()

    # Find persona_id
    cur.execute('''
        SELECT id FROM personas
        WHERE run_id = ? AND persona_hash = ?
    ''', (run_id, persona_hash))
    persona_row = cur.fetchone()

    if not persona_row:
        # Persona doesn't exist yet, need to run
        return None

    persona_id = persona_row[0]

    # Check if judgments exist for this persona-article combination
    # If judgments exist, the combination was successfully completed
    cur.execute('''
        SELECT sentence_index, sentence, agree, reason
        FROM judgments
        WHERE run_id = ? AND persona_id = ? AND article_id = ?
        ORDER BY sentence_index
    ''', (run_id, persona_id, article_id))

    judgments = []
    for row in cur.fetchall():
        judgments.append({
            'index': row[0],
            'sentence': row[1],
            'agree': row[2],
            'reason': row[3]
        })

    # Return judgments if they exist, otherwise None (will trigger retry)
    return judgments if judgments else None


async def process_persona_article(
        session: aiohttp.ClientSession,
        persona: Dict[str, str],
        article_id: int,
        sentences: List[str],
        use_cot: str,
        use_few_shots: bool,
        per_message: bool,
        is_binary_agreement: bool,
        semaphore: asyncio.Semaphore,
        conn: sqlite3.Connection,
        run_id: str
) -> None:
    async with semaphore:
        assert all(k in persona for k in REQUIRED_FIELDS)

        # Check if results already exist in database
        existing_judgments = get_existing_judgments(conn, run_id, persona, article_id)
        if existing_judgments:
            agree = sum(j.get("agree") for j in existing_judgments)
            print(f"⊘ {persona} | Article {article_id} | Agree: {agree}/{len(sentences)} (from DB)")
            return

        try:
            result = await evaluate_article_async(
                session=session,
                persona=persona,
                sentences=sentences,
                use_cot=use_cot,
                use_few_shots=use_few_shots,
                per_message=per_message,
                is_binary_agreement=is_binary_agreement
            )

            judgments = result.get("judgments", [])
            agree = sum(j.get("agree") for j in judgments)

            save_article_judgments(
                conn=conn, run_id=run_id, persona=persona,
                article_id=article_id, judgments=judgments, success=True
            )

            print(f"✓ {persona} | Article {article_id} | Agree: {agree}/{len(sentences)}")

        except Exception as e:
            print(f"✗ {persona} | Article {article_id} | Error: {e}")
            save_article_judgments(
                conn=conn, run_id=run_id, persona=persona,
                article_id=article_id, judgments=[], success=False, error=str(e)
            )


async def run_batch_evaluation_async(
        article_groups: Dict[int, List[str]],
        personas: List[Dict[str, str]],
        use_cot: str = "no_chain",
        use_few_shots: bool = False,
        per_message: bool = True,
        is_binary_agreement: bool = False
) -> None:
    total_personas = len(personas)
    num_articles = len(article_groups)
    print(f"\nTotal personas: {total_personas} | Valid articles: {num_articles}")
    conn = None

    try:
        conn = init_sqlite_db()
        run_id, existing_config = get_or_create_run(
            conn=conn,
            total_personas=total_personas,
            num_articles=num_articles,
            use_cot=use_cot,
            use_few_shots=use_few_shots,
            per_message=per_message,
            is_binary_agreement=is_binary_agreement
        )

        # Use existing config if run already exists
        if existing_config:
            use_cot = existing_config['use_cot']
            use_few_shots = existing_config['use_few_shots']
            per_message = existing_config['per_message']
            is_binary_agreement = existing_config['is_binary_agreement']

        completed_pairs = get_completed_persona_article_pairs(conn, run_id)

        # Build task coroutines properly
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        async with aiohttp.ClientSession() as session:
            tasks = []

            for persona in personas:
                persona_hash = hashlib.sha256(json.dumps(persona, sort_keys=True).encode()).hexdigest()

                for article_id, sentences in article_groups.items():
                    if not sentences:
                        continue
                    if (persona_hash, article_id) in completed_pairs:
                        continue

                    tasks.append(
                        process_persona_article(
                            session=session,
                            persona=persona,
                            article_id=article_id,
                            sentences=sentences,
                            use_cot=use_cot,
                            use_few_shots=use_few_shots,
                            per_message=per_message,
                            is_binary_agreement=is_binary_agreement,
                            semaphore=semaphore,
                            conn=conn,
                            run_id=run_id
                        )
                    )

            print(f"Launching {len(tasks)} evaluation tasks...\n")

            if tasks:
                await asyncio.gather(*tasks)
            else:
                print("No new tasks — all evaluations already completed.")
            print(f"\nRun {run_id} finished.")
    finally:
        conn.close()


def main():
    print("Loading and cleaning feather file...")
    df = pd.read_feather(FEATHER_PATH)

    if 'sentence' not in df.columns or 'article_id' not in df.columns:
        raise ValueError("Feather must contain 'sentence' and 'article_id' columns")

    original = len(df)
    df = df[['article_id', 'sentence']].dropna(how='any')
    df['sentence'] = df['sentence'].astype(str)
    print(f"Dropped null rows: {original - len(df)} → {len(df)} remaining")

    article_groups = {}
    dropped = 0

    for article_id, group in df.groupby('article_id'):
        sorted_group = group.sort_values(
            by='sentence',
            key=lambda x: x.map(extract_index_from_sentence)
        )
        cleaned = [clean_sentence(s) for s in sorted_group['sentence']]
        valid = [s for s in cleaned if s is not None]
        dropped += len(cleaned) - len(valid)

        if valid:
            article_groups[int(article_id)] = valid
        else:
            print(f"Skipped article {article_id}: no valid sentences after cleaning")

    total_sents = sum(len(v) for v in article_groups.values())
    print(f"Final: {total_sents} valid sentences in {len(article_groups)} articles")
    print(f"Dropped empty/invalid sentences: {dropped}\n")

    if article_groups:
        ex_id = next(iter(article_groups))
        print(f"Example (Article {ex_id}):")
        for s in article_groups[ex_id][:5]:
            print(f"  → {s}")

    personas = get_full_demographic_combinations()

    start = time.time()
    asyncio.run(run_batch_evaluation_async(
        article_groups=article_groups,
        personas=personas,
        use_cot="no_chain",
        use_few_shots=False,
        per_message=True,
        is_binary_agreement=False
    ))
    print(f"\nTotal time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()