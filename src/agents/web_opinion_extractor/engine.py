"""WebOpinionEngine - Modular "Glass Box" Web Analysis Tool.

This engine refactors web analysis into a fully exposed, modular class with MBFC
(Media Bias/Fact Check) database support as optional prior probability.

Core Philosophy:
- "Glass Box" Design: Every agent/step is a public method for debugging
- MBFC is Optional: Works with or without database (use_mbfc=False)
- No Hallucination: Use deterministic extraction (trafilatura) for Agent 1
- Bayesian Integration: Merge MBFC as prior probability in Agent 4
"""

import json
import logging
import re
import sqlite3
from contextlib import contextmanager
from typing import List, Optional, Dict, Any, Tuple
from urllib.parse import urlparse

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ...utils.logger import get_logger
from ...config.settings import settings
from ...utils.llm_client import llm_manager
from .models import (
    ArticleContent,
    SourceMetadata,
    AtomicUnit,
    BiasResult,
    BiasDistribution
)
from .exceptions import NetworkError, ContentExtractionError

# Try to import trafilatura (optional dependency)
try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False
    trafilatura = None

# Try to import beautifulsoup4 (fallback)
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    BeautifulSoup = None

import httpx

logger = get_logger(__name__)

# ========== PROMPT CONSTANTS ==========

ATOMIZER_FEW_SHOTS = """You are an expert text analyzer. Your task is to split complex text into atomic units.

INSTRUCTIONS:
1. Break down compound sentences into separate atomic units
2. Each atomic unit should express ONE and only ONE statement
3. Classify each unit as either "fact" or "opinion"
   - FACT: Verifiable, objective statement (e.g., "The bill was passed on January 5th")
   - OPINION: Subjective viewpoint, belief, or judgment (e.g., "The policy is harmful")

EXAMPLES:

Input: "I support the tax cut but oppose the trade war."
Output:
[
  {"statement": "Support for the tax cut", "type": "opinion", "original_sentence": "I support the tax cut but oppose the trade war."},
  {"statement": "Opposition to the trade war", "type": "opinion", "original_sentence": "I support the tax cut but oppose the trade war."}
]

Input: "The unemployment rate fell to 3.5% in October, which shows excellent economic management."
Output:
[
  {"statement": "Unemployment rate fell to 3.5% in October", "type": "fact", "original_sentence": "The unemployment rate fell to 3.5% in October, which shows excellent economic management."},
  {"statement": "This shows excellent economic management", "type": "opinion", "original_sentence": "The unemployment rate fell to 3.5% in October, which shows excellent economic management."}
]

Return ONLY a JSON array of atomic units."""

SCORER_FEW_SHOTS = """You are an expert political bias analyzer. Your task is to calculate bias probability distributions.

INSTRUCTIONS:
1. Analyze the provided atomic units (facts and opinions)
2. For each unit, determine bias probability distribution:
   - left: Probability of Left/Progressive bias (0.0 to 1.0)
   - right: Probability of Right/Conservative bias (0.0 to 1.0)
   - neutral: Probability of Neutral/Centrist stance (0.0 to 1.0)
   - The three probabilities MUST sum to 1.0
3. If Source Metadata (MBFC prior) is provided, use it to inform your analysis
4. Provide Chain of Thought reasoning explaining your probability assignments

EXAMPLES:

Example 1 - WITH MBFC Prior:
Source History (MBFC): "CNN - Left-Center bias, High factual reporting"
Atomic Units: ["Support for universal healthcare", "Opposition to corporate tax cuts"]

Analysis:
- Prior from MBFC: Left-Center suggests baseline {left: 0.6, neutral: 0.3, right: 0.1}
- Unit 1 "Support for universal healthcare": Progressive policy stance
  → Evidence strengthens left bias: {left: 0.8, neutral: 0.15, right: 0.05}
- Unit 2 "Opposition to corporate tax cuts": Progressive economic view
  → Evidence confirms left bias: {left: 0.8, neutral: 0.15, right: 0.05}
- Overall: {left: 0.8, neutral: 0.15, right: 0.05}

Example 2 - WITHOUT MBFC Prior:
Atomic Units: ["The policy is well-balanced", "Both sides have valid concerns"]

Analysis:
- No prior available, analyze text evidence directly
- Unit 1 "The policy is well-balanced": Neutral framing
  → {left: 0.2, neutral: 0.6, right: 0.2}
- Unit 2 "Both sides have valid concerns": Centrist perspective
  → {left: 0.2, neutral: 0.6, right: 0.2}
- Overall: {left: 0.2, neutral: 0.6, right: 0.2}

Return a JSON object with bias_distribution, reasoning, and metadata_used fields."""

# ========== DOMAIN NORMALIZATION MAP ==========

DOMAIN_ALIASES = {
    "bbc.co.uk": "bbc.com",
    "www.bbc.co.uk": "bbc.com",
    "cnn.co.uk": "cnn.com",
    "nytimes.com": "nyt.com",
    "washingtonpost.com": "wapo.com",
}

# ========== TOKEN ESTIMATION ==========

def estimate_tokens(text: str) -> int:
    """Rough estimation: ~4 characters per token."""
    return len(text) // 4


# ========== WebOpinionEngine CLASS ==========

class WebOpinionEngine:
    """
    Modular "Glass Box" Web Opinion Analysis Engine.
    
    This engine implements a 4-agent pipeline with optional MBFC database integration:
    
    Agent 1: extract_content(url) -> ArticleContent
        - Deterministic extraction using trafilatura (no LLM)
        - Falls back to BeautifulSoup if trafilatura fails
        - Returns FULL text (no truncation)
    
    Agent 2: resolve_metadata(url) -> SourceMetadata
        - Optional MBFC database lookup
        - Smart domain matching with normalization
        - Gracefully handles missing database
    
    Agent 3: atomize_text(text) -> List[AtomicUnit]
        - LLM-based atomization with chunking for large texts
        - Splits compound sentences into atomic units
        - Classifies as fact or opinion
    
    Agent 4: calculate_bias(units, metadata) -> BiasResult
        - Bayesian scoring with optional MBFC prior
        - Returns probability distribution (left, right, neutral)
        - Includes Chain of Thought reasoning
    
    Production: run_pipeline(url, use_mbfc=True) -> dict
        - Orchestrates all 4 agents
        - Returns consolidated JSON result
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.3,
        proxy: Optional[str] = None,
        request_timeout: float = 30.0,
        db_path: Optional[str] = None,
        max_chunk_tokens: int = 15000
    ):
        """
        Initialize the WebOpinionEngine.
        
        Args:
            model_name: LLM model name (defaults to settings.openai_model)
            api_key: OpenAI API key or compatible
            api_base: Base URL for the API
            temperature: Temperature for LLM responses
            proxy: Optional HTTP proxy
            request_timeout: Timeout for HTTP requests in seconds
            db_path: Path to MBFC SQLite database (optional)
            max_chunk_tokens: Maximum tokens per chunk for text atomization
        """
        model_name = model_name or settings.openai_model
        logger.info(f"Initializing WebOpinionEngine: model={model_name}, db_path={db_path}")
        
        # Check for trafilatura
        if not HAS_TRAFILATURA:
            logger.warning("trafilatura not installed. Will use BeautifulSoup fallback only.")
        
        # Initialize LLM client
        http_client = llm_manager.get_http_client(proxy=proxy)
        self.llm = ChatOpenAI(
            model_name=model_name,
            api_key=api_key or settings.openai_api_key,
            base_url=api_base or settings.openai_api_base,
            temperature=temperature,
            max_retries=settings.openai_max_retries,
            timeout=settings.openai_timeout,
            http_client=http_client
        )
        
        self.request_timeout = request_timeout
        self.db_path = db_path
        self.max_chunk_tokens = max_chunk_tokens
        
        logger.debug("WebOpinionEngine initialized successfully")
    
    # ========== AGENT 1: EXTRACTION API ==========
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.RequestError))
    )
    def extract_content(self, url: str) -> ArticleContent:
        """
        Agent 1: Extract article content from URL using deterministic methods.
        
        Logic:
        1. Fetch HTML using httpx
        2. Extract using trafilatura (preferred)
        3. Fallback to BeautifulSoup if trafilatura fails
        4. Return FULL text (no truncation)
        
        Args:
            url: The URL to extract from
            
        Returns:
            ArticleContent with title, full_text, domain, url
            
        Raises:
            NetworkError: If fetching fails
            ContentExtractionError: If extraction fails
        """
        logger.info(f"[Agent 1] extract_content: {url}")
        
        # Extract domain
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Fetch HTML
        try:
            with httpx.Client(timeout=self.request_timeout) as client:
                response = client.get(url, follow_redirects=True)
                response.raise_for_status()
                html = response.text
                logger.debug(f"Fetched {len(html)} characters from {url}")
        except httpx.TimeoutException as e:
            logger.error(f"Timeout fetching {url}: {e}")
            raise NetworkError(
                message=f"Request timed out after {self.request_timeout} seconds",
                url=url
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching {url}: {e}")
            raise NetworkError(
                message=f"HTTP error: {e.response.status_code}",
                url=url,
                status_code=e.response.status_code
            )
        except httpx.RequestError as e:
            logger.error(f"Request error fetching {url}: {e}")
            raise NetworkError(
                message=f"Request failed: {str(e)}",
                url=url
            )
        
        # Extract using trafilatura (preferred)
        title = None
        full_text = None
        
        if HAS_TRAFILATURA and trafilatura is not None:
            try:
                # Extract with trafilatura
                downloaded = trafilatura.fetch_url(url)
                if downloaded:
                    # Extract main text
                    extracted = trafilatura.extract(
                        downloaded,
                        include_comments=False,
                        include_tables=True,
                        no_fallback=False
                    )
                    
                    # Extract metadata for title
                    metadata = trafilatura.extract_metadata(downloaded)
                    if metadata and metadata.title:
                        title = metadata.title
                    
                    if extracted and len(extracted.strip()) > 50:
                        full_text = extracted
                        logger.info(f"Trafilatura extracted {len(full_text)} chars")
            except Exception as e:
                logger.warning(f"Trafilatura extraction failed: {e}, falling back to BS4")
        
        # Fallback to BeautifulSoup
        if not full_text and HAS_BS4 and BeautifulSoup is not None:
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, "html.parser")
                
                # Extract title
                if not title:
                    title_tag = soup.find("title")
                    if title_tag:
                        title = title_tag.get_text(strip=True)
                    # Try h1 as fallback
                    if not title:
                        h1_tag = soup.find("h1")
                        if h1_tag:
                            title = h1_tag.get_text(strip=True)
                
                # Remove non-content tags
                for tag in soup(["script", "style", "nav", "footer", "iframe", "noscript"]):
                    tag.decompose()
                
                # Try to find main content
                main_content = None
                for selector in ["article", "main", "[role='main']", ".article-content", ".post-content"]:
                    main_content = soup.select_one(selector)
                    if main_content:
                        break
                
                # Extract text
                if main_content:
                    full_text = main_content.get_text(separator="\n", strip=True)
                else:
                    body = soup.find("body")
                    if body:
                        full_text = body.get_text(separator="\n", strip=True)
                    else:
                        full_text = soup.get_text(separator="\n", strip=True)
                
                # Clean whitespace
                full_text = re.sub(r'\n\s*\n+', '\n\n', full_text)
                full_text = re.sub(r'[ \t]+', ' ', full_text)
                full_text = full_text.strip()
                
                logger.info(f"BeautifulSoup extracted {len(full_text)} chars")
            except Exception as e:
                logger.error(f"BeautifulSoup extraction failed: {e}")
                raise ContentExtractionError(
                    message=f"Failed to extract content: {str(e)}",
                    url=url
                )
        
        if not full_text or len(full_text.strip()) < 50:
            raise ContentExtractionError(
                message="Insufficient content extracted from page",
                url=url
            )
        
        logger.info(f"[Agent 1] Extracted: title={title}, text_len={len(full_text)}, domain={domain}")
        
        return ArticleContent(
            title=title,
            full_text=full_text,
            domain=domain,
            url=url
        )
    
    # ========== AGENT 2: METADATA RESOLVER API ==========
    
    @contextmanager
    def _get_db_connection(self):
        """Context manager for SQLite database connections."""
        if not self.db_path:
            yield None
            return
        
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            yield conn
        except sqlite3.Error as e:
            logger.warning(f"Database connection error: {e}")
            yield None
        finally:
            if conn:
                conn.close()
    
    def _normalize_domain(self, domain: str) -> str:
        """
        Normalize domain for matching.
        
        Logic:
        - Strip 'www.'
        - Apply alias mappings
        - Convert to lowercase
        """
        domain = domain.lower()
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Apply aliases
        if domain in DOMAIN_ALIASES:
            domain = DOMAIN_ALIASES[domain]
        
        return domain
    
    def resolve_metadata(self, url: str) -> SourceMetadata:
        """
        Agent 2: Resolve source metadata from MBFC database (optional).
        
        Logic:
        1. Extract and normalize domain from URL
        2. Query database: SELECT * FROM media_sources WHERE source_url LIKE %domain%
        3. Smart matching:
           - 0 rows: Return generic metadata (match_type="none")
           - 1 row: Exact match (match_type="exact")
           - >1 rows: Fuzzy match, use heuristics (match_type="fuzzy")
        4. If DB missing or error: Return empty metadata, log warning
        
        Args:
            url: The URL to resolve metadata for
            
        Returns:
            SourceMetadata with match information
        """
        logger.info(f"[Agent 2] resolve_metadata: {url}")
        
        # Extract and normalize domain
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith('www.'):
            domain = domain[4:]
        domain = self._normalize_domain(domain)
        
        # Check if database is available
        if not self.db_path:
            logger.info("No database path configured, returning disabled metadata")
            return SourceMetadata(
                source_name=None,
                raw_db_row=None,
                match_type="disabled",
                bias_rating=None,
                factual_reporting=None
            )
        
        # Query database
        with self._get_db_connection() as conn:
            if conn is None:
                logger.warning("Database connection failed, returning none metadata")
                return SourceMetadata(
                    source_name=None,
                    raw_db_row=None,
                    match_type="none",
                    bias_rating=None,
                    factual_reporting=None
                )
            
            try:
                cursor = conn.cursor()
                query = "SELECT * FROM media_sources WHERE source_url LIKE ?"
                cursor.execute(query, (f"%{domain}%",))
                rows = cursor.fetchall()
                
                if len(rows) == 0:
                    # No match
                    logger.info(f"No database match for domain: {domain}")
                    return SourceMetadata(
                        source_name=None,
                        raw_db_row=None,
                        match_type="none",
                        bias_rating=None,
                        factual_reporting=None
                    )
                elif len(rows) == 1:
                    # Exact match
                    row = dict(rows[0])
                    logger.info(f"Exact match found: {row.get('source_name', 'Unknown')}")
                    return SourceMetadata(
                        source_name=row.get('source_name'),
                        raw_db_row=row,
                        match_type="exact",
                        bias_rating=row.get('bias_rating'),
                        factual_reporting=row.get('factual_reporting')
                    )
                else:
                    # Multiple matches - use heuristics
                    logger.info(f"Fuzzy match: {len(rows)} candidates for {domain}")
                    
                    # Heuristic: Check if any row's URL path is in the user URL
                    best_match = rows[0]  # Default to first
                    for row in rows:
                        row_url = row['source_url'] if 'source_url' in row.keys() else ''
                        if row_url and row_url in url:
                            best_match = row
                            logger.info(f"Found specific path match: {row_url}")
                            break
                    
                    row = dict(best_match)
                    return SourceMetadata(
                        source_name=row.get('source_name'),
                        raw_db_row=row,
                        match_type="fuzzy",
                        bias_rating=row.get('bias_rating'),
                        factual_reporting=row.get('factual_reporting')
                    )
            except sqlite3.Error as e:
                logger.error(f"Database query error: {e}")
                return SourceMetadata(
                    source_name=None,
                    raw_db_row=None,
                    match_type="none",
                    bias_rating=None,
                    factual_reporting=None
                )
    
    # ========== AGENT 3: ATOMIZER API ==========
    
    def _chunk_text(self, text: str, max_tokens: int) -> List[str]:
        """
        Split text into chunks if it exceeds max_tokens.
        
        Args:
            text: Text to chunk
            max_tokens: Maximum tokens per chunk
            
        Returns:
            List of text chunks
        """
        tokens = estimate_tokens(text)
        if tokens <= max_tokens:
            return [text]
        
        # Split by paragraphs
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = estimate_tokens(para)
            if current_tokens + para_tokens > max_tokens and current_chunk:
                # Flush current chunk
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens
        
        # Add remaining chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        logger.info(f"Split text into {len(chunks)} chunks (max {max_tokens} tokens each)")
        return chunks
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def atomize_text(self, text: str) -> List[AtomicUnit]:
        """
        Agent 3: Atomize text into atomic fact/opinion units using LLM.
        
        Logic:
        1. If text > max_chunk_tokens, split into chunks
        2. Process each chunk with LLM using ATOMIZER_FEW_SHOTS prompt
        3. Aggregate results from all chunks
        4. Return list of AtomicUnit objects
        
        Args:
            text: Text to atomize
            
        Returns:
            List of AtomicUnit objects
        """
        logger.info(f"[Agent 3] atomize_text: {len(text)} chars, ~{estimate_tokens(text)} tokens")
        
        # Chunk text if necessary
        chunks = self._chunk_text(text, self.max_chunk_tokens)
        
        all_units = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            
            # Build prompt
            user_message = f"""{ATOMIZER_FEW_SHOTS}

TEXT TO ANALYZE:
{chunk}

Return a JSON array of atomic units with fields: statement, type, original_sentence."""
            
            messages = [
                HumanMessage(content=user_message)
            ]
            
            try:
                response = self.llm.invoke(messages)
                response_text = response.content
                
                # Parse JSON from response
                json_match = re.search(r'\[[\s\S]*\]', response_text)
                if not json_match:
                    logger.warning(f"No JSON array found in LLM response for chunk {i+1}")
                    continue
                
                units_data = json.loads(json_match.group())
                
                # Create AtomicUnit objects
                for unit_data in units_data:
                    try:
                        unit = AtomicUnit(
                            statement=unit_data.get('statement', ''),
                            type=unit_data.get('type', 'opinion'),
                            original_sentence=unit_data.get('original_sentence'),
                            confidence=unit_data.get('confidence')
                        )
                        all_units.append(unit)
                    except Exception as e:
                        logger.warning(f"Failed to create AtomicUnit: {e}")
                        continue
                
                logger.info(f"Extracted {len(units_data)} units from chunk {i+1}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from LLM response: {e}")
                continue
            except Exception as e:
                logger.error(f"LLM atomization failed for chunk {i+1}: {e}")
                raise
        
        logger.info(f"[Agent 3] Atomized into {len(all_units)} units")
        return all_units
    
    # ========== AGENT 4: BAYESIAN SCORER API ==========
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def calculate_bias(
        self,
        units: List[AtomicUnit],
        metadata: SourceMetadata
    ) -> BiasResult:
        """
        Agent 4: Calculate bias using Bayesian scoring with optional MBFC prior.
        
        Logic:
        1. If metadata exists and match_type != "none"/"disabled":
           - Use metadata as prior probability
           - Prompt: "Update Prior based on Text Evidence"
        2. If metadata is None or match_type == "none"/"disabled":
           - Zero-shot analysis
           - Prompt: "Analyze directly, assume Neutral Prior"
        3. Return BiasResult with probability distribution and reasoning
        
        Args:
            units: List of atomic units to analyze
            metadata: Source metadata (optional prior)
            
        Returns:
            BiasResult with bias distribution and reasoning
        """
        logger.info(f"[Agent 4] calculate_bias: {len(units)} units, metadata={metadata.match_type}")
        
        # Check if metadata should be used
        use_metadata = (
            metadata and
            metadata.match_type not in ("none", "disabled") and
            metadata.bias_rating is not None
        )
        
        # Build prompt based on whether we have metadata
        if use_metadata:
            # With MBFC prior
            system_prompt = f"""{SCORER_FEW_SHOTS}

SOURCE HISTORY (MBFC Prior):
Source: {metadata.source_name}
Bias Rating: {metadata.bias_rating}
Factual Reporting: {metadata.factual_reporting or 'Unknown'}

Use this as your PRIOR probability. Update it based on the text evidence below."""
        else:
            # Without MBFC prior
            system_prompt = f"""{SCORER_FEW_SHOTS}

NO SOURCE HISTORY AVAILABLE.
Analyze the text evidence directly. Assume a Neutral Prior: {{left: 0.33, neutral: 0.34, right: 0.33}}"""
        
        # Format atomic units for analysis
        units_text = "\n".join([
            f"- [{u.type.upper()}] {u.statement}"
            for u in units
        ])
        
        user_message = f"""ATOMIC UNITS TO ANALYZE:
{units_text}

Provide your bias analysis as a JSON object with:
- bias_distribution: {{left: float, right: float, neutral: float}} (must sum to 1.0)
- reasoning: string explaining your analysis step-by-step
- metadata_used: boolean (true if MBFC prior was used)

Return JSON only."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        
        try:
            response = self.llm.invoke(messages)
            response_text = response.content
            
            # Parse JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if not json_match:
                logger.warning("No JSON found in LLM response")
                # Return default neutral distribution
                return BiasResult(
                    bias_distribution=BiasDistribution(left=0.33, neutral=0.34, right=0.33),
                    reasoning="Failed to parse LLM response",
                    metadata_used=use_metadata
                )
            
            result_data = json.loads(json_match.group())
            
            # Extract bias distribution
            bias_dist_data = result_data.get('bias_distribution', {})
            bias_distribution = BiasDistribution(
                left=bias_dist_data.get('left', 0.33),
                right=bias_dist_data.get('right', 0.33),
                neutral=bias_dist_data.get('neutral', 0.34)
            )
            
            reasoning = result_data.get('reasoning', '')
            
            logger.info(f"[Agent 4] Bias calculated: {bias_distribution.dominant_bias}")
            
            return BiasResult(
                bias_distribution=bias_distribution,
                reasoning=reasoning,
                metadata_used=use_metadata
            )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return BiasResult(
                bias_distribution=BiasDistribution(left=0.33, neutral=0.34, right=0.33),
                reasoning=f"JSON parse error: {str(e)}",
                metadata_used=use_metadata
            )
        except Exception as e:
            logger.error(f"LLM bias calculation failed: {e}")
            raise
    
    # ========== PRODUCTION ORCHESTRATOR ==========
    
    def run_pipeline(
        self,
        url: str,
        use_mbfc: bool = True
    ) -> Dict[str, Any]:
        """
        Production orchestrator: Run the complete 4-agent pipeline.
        
        Logic:
        1. article = extract_content(url)
        2. metadata = resolve_metadata(url) IF use_mbfc ELSE disabled metadata
        3. atoms = atomize_text(article.full_text)
        4. result = calculate_bias(atoms, metadata)
        5. Return consolidated JSON
        
        Args:
            url: URL to analyze
            use_mbfc: Whether to use MBFC database (default: True)
            
        Returns:
            Dictionary with complete analysis results
        """
        logger.info(f"[Pipeline] Starting analysis: url={url}, use_mbfc={use_mbfc}")
        
        try:
            # Agent 1: Extract content
            article = self.extract_content(url)
            logger.info(f"[Pipeline] Agent 1 complete: {len(article.full_text)} chars extracted")
            
            # Agent 2: Resolve metadata
            if use_mbfc:
                metadata = self.resolve_metadata(url)
            else:
                metadata = SourceMetadata(
                    source_name=None,
                    raw_db_row=None,
                    match_type="disabled",
                    bias_rating=None,
                    factual_reporting=None
                )
            logger.info(f"[Pipeline] Agent 2 complete: match_type={metadata.match_type}")
            
            # Agent 3: Atomize text
            atoms = self.atomize_text(article.full_text)
            logger.info(f"[Pipeline] Agent 3 complete: {len(atoms)} atomic units")
            
            # Agent 4: Calculate bias
            bias_result = self.calculate_bias(atoms, metadata)
            logger.info(f"[Pipeline] Agent 4 complete: {bias_result.bias_distribution.dominant_bias}")
            
            # Consolidate results
            result = {
                "url": url,
                "article": {
                    "title": article.title,
                    "domain": article.domain,
                    "text_length": len(article.full_text),
                },
                "metadata": {
                    "source_name": metadata.source_name,
                    "match_type": metadata.match_type,
                    "bias_rating": metadata.bias_rating,
                    "factual_reporting": metadata.factual_reporting,
                } if metadata.match_type != "disabled" else None,
                "atomic_units": [
                    {
                        "statement": u.statement,
                        "type": u.type,
                        "original_sentence": u.original_sentence,
                    }
                    for u in atoms
                ],
                "bias_analysis": {
                    "distribution": {
                        "left": bias_result.bias_distribution.left,
                        "right": bias_result.bias_distribution.right,
                        "neutral": bias_result.bias_distribution.neutral,
                    },
                    "dominant_bias": bias_result.bias_distribution.dominant_bias,
                    "reasoning": bias_result.reasoning,
                    "metadata_used": bias_result.metadata_used,
                },
                "pipeline_metadata": {
                    "use_mbfc": use_mbfc,
                    "num_atomic_units": len(atoms),
                    "model": self.llm.model_name,
                }
            }
            
            logger.info(f"[Pipeline] Complete: {bias_result.bias_distribution.dominant_bias} bias detected")
            return result
            
        except Exception as e:
            logger.error(f"[Pipeline] Error: {e}", exc_info=True)
            raise
