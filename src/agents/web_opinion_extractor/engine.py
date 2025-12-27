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
from contextlib import closing, contextmanager
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union
from urllib.parse import urlparse

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_not_exception_type
)
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ...utils.logger import get_logger
from ...config.settings import settings
from ...utils.llm_client import llm_manager
from ...utils.agent_cache import cached
from ...few_shots.web_opinion_extractor.atomizer_shots import ATOMIZER_FEW_SHOTS
from ...few_shots.web_opinion_extractor.scorer_shots import SCORER_FEW_SHOTS
from .models import (
    ArticleContent,
    SourceMetadata,
    AtomicUnit,
    BiasResult,
    BiasDistribution,
    LogicMode,
    PipelineConfig
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

# ========== RETRY CONFIGURATIONS ==========

# Retry configuration for network operations (HTTP requests)
NETWORK_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.RequestError)),
    reraise=True
)

# Retry configuration for LLM operations
# Retry on network/timeout errors only (JSON parsing errors are handled in try/except blocks)
LLM_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.RequestError)),
    reraise=True
)

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

    This engine implements a 4-agent pipeline with configurable options:
    
    Agent 1: extract_content(url) -> ArticleContent
        - Deterministic extraction using trafilatura (no LLM)
        - Falls back to BeautifulSoup if trafilatura fails
        - Returns FULL text (no truncation)
    
    Agent 2: resolve_metadata(url) -> SourceMetadata
        - Optional MBFC database lookup (controlled by use_mbfc parameter)
        - Database path configured in settings.mbfc_db_path
        - Smart domain matching with normalization
        - Returns DB fields when MBFC is enabled and match found
    
    Agent 3: atomize_text(text, use_few_shots, shots) -> List[AtomicUnit]
        - LLM-based atomization with chunking for large texts
        - Few-shot examples: user can enable/disable or provide custom examples
        - Splits compound sentences into atomic units
        - Classifies as fact or opinion
    
    Agent 4: calculate_bias(evidence, metadata, use_few_shots, shots) -> BiasResult
        - Bayesian scoring with optional MBFC prior (when use_mbfc=True)
        - Few-shot examples: user can enable/disable or provide custom examples
        - Returns probability distribution (left, right, neutral)
        - Includes Chain of Thought reasoning
    
    Pure Online: _pure_online_analysis(article, metadata, use_few_shots, shots) -> BiasResult
        - End-to-end LLM analysis without intermediate steps
        - Few-shot examples: user can enable/disable or provide custom examples
        - Returns probability distribution (left, right, neutral)
        - Includes Chain of Thought reasoning
    
    Production: run(url, mode, use_mbfc, use_few_shots, **kwargs) -> dict
        - Orchestrates all agents based on LogicMode
        - Configurable: use_mbfc (MBFC prior), use_few_shots (few-shot optimization for ALL LLM agents)
        - Returns consolidated JSON result with all intermediate data
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.3,
        request_timeout: float = 30.0,
        db_path: Optional[str] = None,
        max_chunk_tokens: int = 15000,
        atomizer_shots_path: Optional[str] = None,
        scorer_shots_path: Optional[str] = None
    ):
        """
        Initialize the WebOpinionEngine.
        
        Args:
            config: Optional PipelineConfig object (if provided, overrides individual params)
            model_name: LLM model name (defaults to settings.openai_model)
            api_key: OpenAI API key or compatible
            api_base: Base URL for the API
            temperature: Temperature for LLM responses
            request_timeout: Timeout for HTTP requests in seconds
            db_path: Path to MBFC SQLite database (optional)
            max_chunk_tokens: Maximum tokens per chunk for text atomization
            atomizer_shots_path: Path to JSON file with atomizer few-shot examples
            scorer_shots_path: Path to JSON file with scorer few-shot examples
        """
        # Use config if provided, otherwise use individual parameters
        if config:
            model_name = config.openai_model
            db_path = config.mbfc_db_path or db_path
            atomizer_shots_path = config.atomizer_shots_path or atomizer_shots_path
            scorer_shots_path = config.scorer_shots_path or scorer_shots_path
        
        model_name = model_name or settings.openai_model
        
        # Use db_path from config, then parameter, then settings
        if not db_path:
            db_path = settings.mbfc_db_path
        
        logger.info(f"Initializing WebOpinionEngine: model={model_name}, db_path={db_path}")
        
        # Check for trafilatura
        if not HAS_TRAFILATURA:
            logger.warning("trafilatura not installed. Will use BeautifulSoup fallback only.")
        
        # Initialize LLM client
        http_client = llm_manager.get_http_client()
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
        
        # Set default paths if not provided
        # Default shots are stored in: src/few_shots/web_opinion_extractor/shots/
        if not atomizer_shots_path:
            default_shots_dir = Path(__file__).parent.parent.parent / "few_shots" / "web_opinion_extractor" / "shots"
            atomizer_shots_path = str(default_shots_dir / "atomizer_shots.json")
        
        if not scorer_shots_path:
            default_shots_dir = Path(__file__).parent.parent.parent / "few_shots" / "web_opinion_extractor" / "shots"
            scorer_shots_path = str(default_shots_dir / "scorer_shots.json")
        
        self.atomizer_shots_path = atomizer_shots_path
        self.scorer_shots_path = scorer_shots_path
        
        logger.debug("WebOpinionEngine initialized successfully")
    
    def _load_shots(self, path: str) -> List[dict]:
        """
        Helper method to safely load few-shot examples from JSON file.
        
        Args:
            path: Path to JSON file containing few-shot examples
            
        Returns:
            List of dictionaries containing few-shot examples
        """
        try:
            file_path = Path(path)
            if not file_path.exists():
                logger.warning(f"Few-shot examples file not found: {path}")
                return []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'examples' in data:
                    return data['examples']
                else:
                    logger.warning(f"Unexpected JSON structure in {path}")
                    return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from {path}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error loading few-shot examples from {path}: {e}")
            return []
    
    def get_default_atomizer_shots(self) -> List[dict]:
        """
        Get default atomizer few-shot examples from the shots folder.
        
        Returns:
            List of dictionaries containing default atomizer few-shot examples
        """
        if self.atomizer_shots_path:
            shots = self._load_shots(self.atomizer_shots_path)
            if shots:
                return shots
        # Return empty list if no default shots available
        logger.warning("No default atomizer shots found")
        return []
    
    def get_default_scorer_shots(self) -> List[dict]:
        """
        Get default scorer few-shot examples from the shots folder.
        
        Returns:
            List of dictionaries containing default scorer few-shot examples
        """
        if self.scorer_shots_path:
            shots = self._load_shots(self.scorer_shots_path)
            if shots:
                return shots
        # Return empty list if no default shots available
        logger.warning("No default scorer shots found")
        return []

    @classmethod
    def set_custom_atomizer_shots(cls, custom_shots: Optional[List[dict]] = None) -> None:
        """
        Set custom atomizer few-shot examples for all instances.

        Args:
            custom_shots: Custom atomizer few-shot examples as list of dicts. If None, clears custom shots.
        """
        from ...config.settings import settings
        from ...storage import get_database

        cls._custom_atomizer_shots = custom_shots

        if settings.enable_persistence:
            database = get_database()
            if database:
                success = database.save_custom_few_shots("web_opinion_extractor_atomizer", custom_shots)
                if success:
                    logger.info(f"Custom atomizer shots saved to database: {custom_shots is not None}")
                else:
                    logger.warning("Failed to save custom atomizer shots to database")
        else:
            logger.info(f"Custom atomizer shots set (no persistence): {custom_shots is not None}")

    @classmethod
    def set_custom_scorer_shots(cls, custom_shots: Optional[List[dict]] = None) -> None:
        """
        Set custom scorer few-shot examples for all instances.

        Args:
            custom_shots: Custom scorer few-shot examples as list of dicts. If None, clears custom shots.
        """
        from ...config.settings import settings
        from ...storage import get_database

        cls._custom_scorer_shots = custom_shots

        if settings.enable_persistence:
            database = get_database()
            if database:
                success = database.save_custom_few_shots("web_opinion_extractor_scorer", custom_shots)
                if success:
                    logger.info(f"Custom scorer shots saved to database: {custom_shots is not None}")
                else:
                    logger.warning("Failed to save custom scorer shots to database")
        else:
            logger.info(f"Custom scorer shots set (no persistence): {custom_shots is not None}")

    @classmethod
    def get_custom_atomizer_shots(cls) -> Optional[List[dict]]:
        """
        Get currently set custom atomizer few-shot examples.

        Returns:
            Custom atomizer shots or None if not set
        """
        from ...config.settings import settings
        from ...storage import get_database

        # If not loaded yet, try to load from database
        if cls._custom_atomizer_shots is None and settings.enable_persistence:
            database = get_database()
            if database:
                stored_shots = database.load_custom_few_shots("web_opinion_extractor_atomizer")
                if isinstance(stored_shots, list):
                    cls._custom_atomizer_shots = stored_shots
                    logger.info("Loaded custom atomizer shots from database")

        return cls._custom_atomizer_shots

    @classmethod
    def get_custom_scorer_shots(cls) -> Optional[List[dict]]:
        """
        Get currently set custom scorer few-shot examples.

        Returns:
            Custom scorer shots or None if not set
        """
        from ...config.settings import settings
        from ...storage import get_database

        # If not loaded yet, try to load from database
        if cls._custom_scorer_shots is None and settings.enable_persistence:
            database = get_database()
            if database:
                stored_shots = database.load_custom_few_shots("web_opinion_extractor_scorer")
                if isinstance(stored_shots, list):
                    cls._custom_scorer_shots = stored_shots
                    logger.info("Loaded custom scorer shots from database")

        return cls._custom_scorer_shots

    def get_effective_atomizer_shots(self) -> List[dict]:
        """
        Get effective atomizer few-shot examples (custom if set, otherwise system default).

        Returns:
            Effective atomizer shots as list of dicts
        """
        return self._custom_atomizer_shots if self._custom_atomizer_shots is not None else self.get_default_atomizer_shots()

    def get_effective_scorer_shots(self) -> List[dict]:
        """
        Get effective scorer few-shot examples (custom if set, otherwise system default).

        Returns:
            Effective scorer shots as list of dicts
        """
        return self._custom_scorer_shots if self._custom_scorer_shots is not None else self.get_default_scorer_shots()

    # ========== AGENT 1: EXTRACTION API ==========
    
    @NETWORK_RETRY
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
        
        # Fetch HTML with proper User-Agent
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; WebOpinionEngine/1.0; +https://github.com/colonel8377/AISearchAgents)'
        }
        try:
            with httpx.Client(timeout=self.request_timeout, trust_env=True) as client:
                response = client.get(url, headers=headers, follow_redirects=True)
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
                # trafilatura will automatically use HTTP_PROXY and HTTPS_PROXY environment variables
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
    
    def _extract_base_domain(self, hostname: str) -> str:
        """
        Extract base domain from hostname, removing subdomains.
        
        Examples:
        - "edition.cnn.com" -> "cnn.com"
        - "www.cnn.com" -> "cnn.com"
        - "subdomain.example.com" -> "example.com"
        - "example.co.uk" -> "example.co.uk" (handles two-part TLDs)
        
        Args:
            hostname: The hostname (e.g., "edition.cnn.com")
            
        Returns:
            Base domain (e.g., "cnn.com")
        """
        hostname = hostname.lower()
        
        # Remove 'www.' prefix if present
        if hostname.startswith('www.'):
            hostname = hostname[4:]
        
        # Split by dots
        parts = hostname.split('.')
        
        # Common two-part TLDs (like .co.uk, .com.au, .net.au, etc.)
        two_part_tlds = ['co.uk', 'com.au', 'net.au', 'org.uk', 'gov.uk', 'ac.uk', 
                        'co.nz', 'com.br', 'co.jp', 'co.za', 'com.mx']
        
        # Check if it's a two-part TLD
        if len(parts) >= 3:
            last_two = '.'.join(parts[-2:])
            if last_two in two_part_tlds:
                # For two-part TLDs, take last 3 parts (domain + TLD)
                return '.'.join(parts[-3:])
        
        # For standard domains, take last 2 parts (domain + TLD)
        if len(parts) >= 2:
            return '.'.join(parts[-2:])
        
        # Fallback: return as-is if less than 2 parts
        return hostname
    
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
        hostname = parsed.netloc.lower()
        
        # Extract base domain (remove subdomains like "edition." from "edition.cnn.com")
        domain = self._extract_base_domain(hostname)
        # Normalize domain (apply aliases, etc.)
        domain = self._normalize_domain(domain)
        logger.info(f"Domain: {domain}")
        # Use instance db_path (which comes from settings)
        effective_db_path = self.db_path
        
        # Check if database is available
        if not effective_db_path:
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
                # Sanitize domain to prevent SQL injection via LIKE wildcards
                sanitized_domain = domain.replace('%', '').replace('_', '')
                query = "SELECT * FROM media_sources WHERE source_url LIKE ?"
                logger.info(f"Query: {query}")
                cursor.execute(query, (f"%{sanitized_domain}%",))
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
                    # Map DB schema fields to SourceMetadata fields:
                    # - 'source'           -> source_name
                    # - 'bias'             -> bias_rating
                    # - 'factual_reporting' stays the same
                    logger.info(f"Exact match found: {row.get('source', 'Unknown')}")
                    return SourceMetadata(
                        source_name=row.get('source'),
                        raw_db_row=row,
                        match_type="exact",
                        bias_rating=row.get('bias'),
                        factual_reporting=row.get('factual_reporting')
                    )
                else:
                    # Multiple matches - use heuristics with proper domain verification
                    logger.info(f"Fuzzy match: {len(rows)} candidates for {domain}")
                    
                    # Heuristic: Check if any row's domain matches or is a subdomain
                    best_match = rows[0]  # Default to first
                    for row in rows:
                        row_url = row['source_url'] if 'source_url' in row.keys() else ''
                        if row_url:
                            # Normalize row URL to domain
                            row_domain = row_url.lower().replace('www.', '').replace('http://', '').replace('https://', '').split('/')[0]
                            # Check if user domain ends with row domain (subdomain check)
                            if domain.endswith(row_domain) or row_domain.endswith(domain):
                                best_match = row
                                logger.info(f"Found domain match: {row_domain}")
                                break
                    
                    row = dict(best_match)
                    return SourceMetadata(
                        source_name=row.get('source'),
                        raw_db_row=row,
                        match_type="fuzzy",
                        bias_rating=row.get('bias'),
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
    
    def _format_atomizer_shots(self, shots: List[dict]) -> str:
        """Format few-shot examples for atomizer prompt."""
        if not shots:
            return ""
        
        examples_text = "EXAMPLES:\n\n"
        for example in shots:
            input_text = example.get('input', '')
            output = example.get('output', [])
            examples_text += f"Input: {input_text}\nOutput:\n"
            examples_text += json.dumps(output, indent=2, ensure_ascii=False)
            examples_text += "\n\n"
        
        return examples_text
    
    @cached()
    @LLM_RETRY
    def atomize_text(
        self,
        text: str,
        use_few_shots: bool = True,
        shots: Optional[List[dict]] = None
    ) -> List[AtomicUnit]:
        """
        Agent 3: Atomize text into atomic fact/opinion units using LLM.
        
        Logic:
        1. If use_few_shots=False: Use Zero-Shot
        2. If use_few_shots=True:
           - Use shots arg if provided (User Override)
           - Else, load from JSON file specified in config.atomizer_shots_path
           - Else, use Zero-Shot
        3. If text > max_chunk_tokens, split into chunks
        4. Process each chunk with LLM
        5. Aggregate results from all chunks
        6. Return list of AtomicUnit objects
        
        Args:
            text: Text to atomize
            use_few_shots: Whether to use few-shot examples (default: True)
            shots: Optional list of few-shot examples (dicts with 'input' and 'output' keys)
                  Only used if use_few_shots=True
            
        Returns:
            List of AtomicUnit objects
        """
        logger.info(f"[Agent 3] atomize_text: {len(text)} chars, ~{estimate_tokens(text)} tokens, use_few_shots={use_few_shots}")
        
        # Determine which few-shots to use
        few_shots_text = ""
        if not use_few_shots:
            logger.info("Few-shot examples disabled by user, using zero-shot")
        elif shots is not None:
            few_shots_text = self._format_atomizer_shots(shots)
            logger.info("Using user-provided few-shot examples for atomization")
        elif self.atomizer_shots_path:
            loaded_shots = self._load_shots(self.atomizer_shots_path)
            if loaded_shots:
                few_shots_text = self._format_atomizer_shots(loaded_shots)
                logger.info(f"Loaded few-shot examples from {self.atomizer_shots_path}")
            else:
                logger.info("No few-shot examples loaded from file, using zero-shot")
        else:
            logger.info("No few-shot examples configured, using zero-shot")
        
        # Chunk text if necessary
        chunks = self._chunk_text(text, self.max_chunk_tokens)
        
        all_units = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            
            # Build prompt
            if few_shots_text:
                user_message = f"""You are an expert text analyzer. Your task is to split complex text into atomic units.

INSTRUCTIONS:
1. Break down compound sentences into separate atomic units
2. Each atomic unit should express ONE and only ONE statement
3. Classify each unit as either "fact" or "opinion"
   - FACT: Verifiable, objective statement (e.g., "The bill was passed on January 5th")
   - OPINION: Subjective viewpoint, belief, or judgment (e.g., "The policy is harmful")
4. For each atomic unit, provide:
   - "original_sentence": the original sentence text this unit came from
   - "confidence": a float between 0.0 and 1.0 indicating how confident you are in this extraction
   - "reasoning": a short explanation of why you classified it as fact/opinion

{few_shots_text}
TEXT TO ANALYZE:
{chunk}

Return a JSON array of atomic units with fields: statement, type, original_sentence, confidence, reasoning."""
            else:
                user_message = f"""You are an expert text analyzer. Split the following text into atomic units.

Each atomic unit should express ONE statement. Classify as "fact" or "opinion".
For each unit also return:
- "original_sentence": the original sentence source
- "confidence": a float between 0.0 and 1.0
- "reasoning": a short explanation of your classification

TEXT TO ANALYZE:
{chunk}

Return a JSON array of atomic units with fields: statement, type, original_sentence, confidence, reasoning."""
            
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
                
                try:
                    units_data = json.loads(json_match.group())
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from LLM response for chunk {i+1}: {e}")
                    logger.debug(f"Response text: {response_text[:200]}...")
                    continue
                
                # Create AtomicUnit objects
                for unit_data in units_data:
                    try:
                        unit = AtomicUnit(
                            statement=unit_data.get('statement', ''),
                            type=unit_data.get('type', 'opinion'),
                            original_sentence=unit_data.get('original_sentence'),
                            confidence=unit_data.get('confidence'),
                            reasoning=unit_data.get('reasoning')
                        )
                        all_units.append(unit)
                    except Exception as e:
                        logger.warning(f"Failed to create AtomicUnit from {unit_data}: {e}")
                        continue
                
                logger.info(f"Extracted {len(units_data)} units from chunk {i+1}")
            except Exception as e:
                # Retry decorator handles network/timeout errors
                # Other errors should be logged and re-raised
                logger.error(f"LLM atomization failed for chunk {i+1}: {e}", exc_info=True)
                raise
        
        logger.info(f"[Agent 3] Atomized into {len(all_units)} units")
        return all_units
    
    # ========== AGENT 4: BAYESIAN SCORER API ==========
    
    def _format_scorer_shots(self, shots: List[dict]) -> str:
        """Format few-shot examples for scorer prompt."""
        if not shots:
            return ""
        
        examples_text = "EXAMPLES:\n\n"
        for example in shots:
            metadata = example.get('metadata')
            atomic_units = example.get('atomic_units', [])
            output = example.get('output', {})
            
            if metadata:
                examples_text += f"Example - WITH MBFC Prior:\n"
                examples_text += f"Source History (MBFC): {metadata.get('source_name', 'Unknown')} - {metadata.get('bias_rating', 'Unknown')} bias, {metadata.get('factual_reporting', 'Unknown')} factual reporting\n"
            else:
                examples_text += f"Example - WITHOUT MBFC Prior:\n"
            
            examples_text += "Atomic Units: "
            units_list = [f'"{u.get("statement", "")}"' for u in atomic_units]
            examples_text += "[" + ", ".join(units_list) + "]\n\n"
            
            bias_dist = output.get('bias_distribution', {})
            reasoning = output.get('reasoning', '')
            examples_text += f"Analysis:\n"
            examples_text += f"- Overall: {json.dumps(bias_dist, ensure_ascii=False)}\n"
            examples_text += f"- Reasoning: {reasoning}\n\n"
        
        return examples_text
    
    @cached()
    @LLM_RETRY
    def calculate_bias(
        self,
        evidence: Union[str, List[AtomicUnit]],
        metadata: SourceMetadata,
        use_few_shots: bool = True,
        shots: Optional[List[dict]] = None
    ) -> BiasResult:
        """
        Agent 4: Calculate bias using Bayesian scoring with optional MBFC prior.
        
        Logic:
        1. Prior: If metadata contains valid DB data, inject it into the System Prompt as the "Prior"
        2. If use_few_shots=False: Use Zero-Shot
        3. If use_few_shots=True:
           - Use shots arg if provided (User Override)
           - Else, load from JSON file specified in config.scorer_shots_path
           - Else, use Zero-Shot
        4. Task: Evaluate bias based on Evidence + Prior. Require Chain-of-Thought in reasoning
        
        Args:
            evidence: Either full text string or list of AtomicUnit objects to analyze
            metadata: Source metadata (optional prior)
            use_few_shots: Whether to use few-shot examples (default: True)
            shots: Optional list of few-shot examples (dicts with 'metadata', 'atomic_units', 'output' keys)
                  Only used if use_few_shots=True
            
        Returns:
            BiasResult with bias distribution and reasoning
        """
        # Determine if evidence is string or list of AtomicUnits
        if isinstance(evidence, str):
            evidence_text = evidence
            units_text = evidence
            logger.info(f"[Agent 4] calculate_bias: text evidence ({len(evidence)} chars), metadata={metadata.match_type}")
        else:
            # Filter to only opinions if mixed (should already be filtered, but double-check)
            opinion_units = [u for u in evidence if u.type == "opinion"]
            if len(opinion_units) < len(evidence):
                logger.info(f"[Agent 4] Filtering: {len(opinion_units)} opinions out of {len(evidence)} total units")
            evidence_text = "\n".join([
                f"- {u.statement}"
                for u in opinion_units
            ])
            units_text = evidence_text
            logger.info(f"[Agent 4] calculate_bias: {len(opinion_units)} opinion units, metadata={metadata.match_type}")
        
        # Determine which few-shots to use
        few_shots_text = ""
        if not use_few_shots:
            logger.info("Few-shot examples disabled by user, using zero-shot")
        elif shots is not None:
            few_shots_text = self._format_scorer_shots(shots)
            logger.info("Using user-provided few-shot examples for bias scoring")
        elif self.scorer_shots_path:
            loaded_shots = self._load_shots(self.scorer_shots_path)
            if loaded_shots:
                few_shots_text = self._format_scorer_shots(loaded_shots)
                logger.info(f"Loaded few-shot examples from {self.scorer_shots_path}")
            else:
                logger.info("No few-shot examples loaded from file, using zero-shot")
        else:
            logger.info("No few-shot examples configured, using zero-shot")
        
        # Check if metadata should be used
        use_metadata = (
            metadata and
            metadata.match_type not in ("none", "disabled") and
            metadata.bias_rating is not None
        )
        
        # Build prompt based on whether we have metadata
        if use_metadata:
            # With MBFC prior (SOFT prior – evidence from opinions must dominate)
            if few_shots_text:
                system_prompt = f"""You are an expert political bias analyzer. Your task is to calculate bias probability distributions.

INSTRUCTIONS:
1. Analyze the provided evidence (atomic units or full text). Focus your bias judgment ONLY on units labeled as "opinion".
2. Determine the overall bias probability distribution based on these OPINIONS:
   - left: Probability of Left/Progressive bias (0.0 to 1.0)
   - right: Probability of Right/Conservative bias (0.0 to 1.0)
   - neutral: Probability of Neutral/Centrist stance (0.0 to 1.0)
   - The three probabilities MUST sum to 1.0
3. Use the Source Metadata (MBFC) only as a SOFT PRIOR:
   - Treat MBFC as an initial guess, not a fixed label.
   - If the OPINIONS in the text clearly disagree with the prior, let the OPINIONS override and dominate the final result.
   - High factual reporting should increase your trust in using the prior, but should not by itself force left/right labels.
4. Provide Chain of Thought reasoning explicitly separating:
   - How the OPINIONS support a particular bias.
   - How (if at all) the MBFC prior nudges your final probabilities.

{few_shots_text}
SOURCE HISTORY (MBFC Prior - soft, overridable):
Source: {metadata.source_name}
Bias Rating: {metadata.bias_rating}
Factual Reporting: {metadata.factual_reporting or 'Unknown'}

Use this as a SOFT PRIOR only. Your final decision MUST primarily reflect the actual OPINIONS in the text evidence below."""
            else:
                system_prompt = f"""You are an expert political bias analyzer. Calculate bias probability distributions.

INSTRUCTIONS:
1. Analyze the provided evidence (atomic units or full text). Focus your bias judgment ONLY on units labeled as "opinion".
2. Determine the overall bias probability distribution based on these OPINIONS:
   - left: Probability of Left/Progressive bias (0.0 to 1.0)
   - right: Probability of Right/Conservative bias (0.0 to 1.0)
   - neutral: Probability of Neutral/Centrist stance (0.0 to 1.0)
   - The three probabilities MUST sum to 1.0
3. Use the Source Metadata (MBFC) only as a SOFT PRIOR:
   - Treat MBFC as an initial guess, not a fixed label.
   - If the OPINIONS in the text clearly disagree with the prior, let the OPINIONS override and dominate the final result.
   - High factual reporting should increase your trust in using the prior, but should not by itself force left/right labels.
4. Provide Chain of Thought reasoning explicitly separating:
   - How the OPINIONS support a particular bias.
   - How (if at all) the MBFC prior nudges your final probabilities.

SOURCE HISTORY (MBFC Prior - soft, overridable):
Source: {metadata.source_name}
Bias Rating: {metadata.bias_rating}
Factual Reporting: {metadata.factual_reporting or 'Unknown'}"""
        else:
            # Without MBFC prior
            if few_shots_text:
                system_prompt = f"""You are an expert political bias analyzer. Your task is to calculate bias probability distributions.

INSTRUCTIONS:
1. Analyze the provided evidence (atomic units or full text)
2. Determine bias probability distribution:
   - left: Probability of Left/Progressive bias (0.0 to 1.0)
   - right: Probability of Right/Conservative bias (0.0 to 1.0)
   - neutral: Probability of Neutral/Centrist stance (0.0 to 1.0)
   - The three probabilities MUST sum to 1.0
3. No source history available - analyze text evidence directly
4. Provide Chain of Thought reasoning explaining your probability assignments

{few_shots_text}
NO SOURCE HISTORY AVAILABLE.
Analyze the text evidence directly. Assume a Neutral Prior: {{left: 0.33, neutral: 0.34, right: 0.33}}"""
            else:
                system_prompt = """You are an expert political bias analyzer. Calculate bias probability distributions.

NO SOURCE HISTORY AVAILABLE.
Analyze the text evidence directly. Assume a Neutral Prior: {left: 0.33, neutral: 0.34, right: 0.33}"""
        
        user_message = f"""OPINIONS TO ANALYZE (facts have been filtered out to save tokens):
{units_text}

Provide your bias analysis as a JSON object with:
- bias_distribution: {{left: float, right: float, neutral: float}} (must sum to 1.0)
- reasoning: string explaining your analysis step-by-step (Chain of Thought required)
- metadata_used: boolean (true if MBFC prior was used)
- mbfc_influence_note: string (only if metadata_used=true) - Brief note describing MBFC prior\'s influence:
  * "strong" - MBFC prior strongly influenced the final result
  * "moderate" - MBFC prior had moderate influence, combined with text evidence
  * "weak" - MBFC prior had minimal influence, text evidence dominated
  * "overridden" - Text opinions clearly contradicted MBFC prior, prior was overridden
  * If metadata_used=false, set to null

Note: Only opinions are included here. Facts are excluded as they don't require bias analysis.

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
                logger.debug(f"Response text: {response_text[:200]}...")
                return BiasResult(
                    bias_distribution=BiasDistribution(left=0.33, neutral=0.34, right=0.33),
                    reasoning="Failed to parse LLM response: No JSON object found",
                    metadata_used=use_metadata
                )
            
            try:
                result_data = json.loads(json_match.group())
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                logger.debug(f"JSON match: {json_match.group()[:200]}...")
                return BiasResult(
                    bias_distribution=BiasDistribution(left=0.33, neutral=0.34, right=0.33),
                    reasoning=f"JSON parse error: {str(e)}",
                    metadata_used=use_metadata
                )
            
            # Extract bias distribution with validation
            try:
                bias_dist_data = result_data.get('bias_distribution', {})
                bias_distribution = BiasDistribution(
                    left=bias_dist_data.get('left', 0.33),
                    right=bias_dist_data.get('right', 0.33),
                    neutral=bias_dist_data.get('neutral', 0.34)
                )
            except Exception as e:
                logger.error(f"Failed to create BiasDistribution: {e}")
                return BiasResult(
                    bias_distribution=BiasDistribution(left=0.33, neutral=0.34, right=0.33),
                    reasoning=f"Invalid bias distribution data: {str(e)}",
                    metadata_used=use_metadata
                )
            
            reasoning = result_data.get('reasoning', '')
            mbfc_influence_note = result_data.get('mbfc_influence_note')
            # If metadata was used but note not provided, try to infer from reasoning
            if use_metadata and not mbfc_influence_note:
                # Simple heuristic: check reasoning for keywords
                reasoning_lower = reasoning.lower()
                if any(word in reasoning_lower for word in ['override', 'contradict', 'disagree']):
                    mbfc_influence_note = "overridden"
                elif any(word in reasoning_lower for word in ['strong', 'heavily', 'primarily']):
                    mbfc_influence_note = "strong"
                elif any(word in reasoning_lower for word in ['moderate', 'somewhat', 'partially']):
                    mbfc_influence_note = "moderate"
                else:
                    mbfc_influence_note = "weak"
            
            logger.info(f"[Agent 4] Bias calculated: {bias_distribution.dominant_bias}, MBFC influence: {mbfc_influence_note}")
            
            return BiasResult(
                bias_distribution=bias_distribution,
                reasoning=reasoning,
                metadata_used=use_metadata,
                mbfc_influence_note=mbfc_influence_note if use_metadata else None
            )
        except Exception as e:
            # Retry decorator handles network/timeout errors
            # Other errors should be logged and re-raised
            logger.error(f"LLM bias calculation failed: {e}", exc_info=True)
            raise
    
    # ========== PURE ONLINE MODE ==========
    
    @LLM_RETRY
    def _pure_online_analysis(
        self,
        article: ArticleContent,
        metadata: SourceMetadata,
        use_few_shots: bool = True,
        shots: Optional[List[dict]] = None
    ) -> BiasResult:
        """
        End-to-end LLM analysis without intermediate atomization step.
        
        Args:
            article: Extracted article content
            metadata: Source metadata (optional prior)
            use_few_shots: Whether to use few-shot examples (default: True)
            shots: Optional list of few-shot examples for bias scoring (only used if use_few_shots=True)
            
        Returns:
            BiasResult with bias distribution and reasoning
        """
        logger.info(f"[Pure Online] Starting end-to-end LLM analysis, use_few_shots={use_few_shots}")
        
        # Determine which few-shots to use
        few_shots_text = ""
        if not use_few_shots:
            logger.info("Few-shot examples disabled by user, using zero-shot")
        elif shots is not None:
            few_shots_text = self._format_scorer_shots(shots)
            logger.info("Using user-provided few-shot examples for pure online analysis")
        elif self.scorer_shots_path:
            loaded_shots = self._load_shots(self.scorer_shots_path)
            if loaded_shots:
                few_shots_text = self._format_scorer_shots(loaded_shots)
                logger.info(f"Loaded few-shot examples from {self.scorer_shots_path}")
            else:
                logger.info("No few-shot examples loaded from file, using zero-shot")
        else:
            logger.info("No few-shot examples configured, using zero-shot")
        
        # Check if metadata should be used
        use_metadata = (
            metadata and
            metadata.match_type not in ("none", "disabled") and
            metadata.bias_rating is not None
        )
        
        # Build prompt
        if use_metadata:
            if few_shots_text:
                system_prompt = f"""You are an expert political bias analyzer. Your task is to analyze articles and calculate bias probability distributions.

INSTRUCTIONS:
1. Analyze the provided article content. Base your judgment primarily on the OPINIONS expressed in the article, not on raw facts.
2. Determine bias probability distribution:
   - left: Probability of Left/Progressive bias (0.0 to 1.0)
   - right: Probability of Right/Conservative bias (0.0 to 1.0)
   - neutral: Probability of Neutral/Centrist stance (0.0 to 1.0)
   - The three probabilities MUST sum to 1.0
3. Use the Source Metadata (MBFC) only as a SOFT PRIOR:
   - Treat MBFC as an initial guess, not a fixed label.
   - If the article’s OPINIONS clearly disagree with the prior, let the OPINIONS override and dominate the final result.
   - High factual reporting should increase your trust in using the prior, but should not by itself force left/right labels.
4. Provide Chain of Thought reasoning explicitly separating:
   - How the article’s OPINIONS support a particular bias.
   - How (if at all) the MBFC prior nudges your final probabilities.

{few_shots_text}
SOURCE HISTORY (MBFC Prior - soft, overridable):
Source: {metadata.source_name}
Bias Rating: {metadata.bias_rating}
Factual Reporting: {metadata.factual_reporting or 'Unknown'}

Use this as a SOFT PRIOR only. Your final decision MUST primarily reflect the actual OPINIONS in the article content below.

Provide your analysis as a JSON object with:
- bias_distribution: {{left: float, right: float, neutral: float}} (must sum to 1.0)
- reasoning: string explaining your analysis step-by-step (Chain of Thought required)
- metadata_used: boolean (true if MBFC prior was used)
- mbfc_influence_note: string (only if metadata_used=true) - Brief note describing MBFC prior\'s influence:
  * "strong" - MBFC prior strongly influenced the final result
  * "moderate" - MBFC prior had moderate influence, combined with text evidence
  * "weak" - MBFC prior had minimal influence, text evidence dominated
  * "overridden" - Text opinions clearly contradicted MBFC prior, prior was overridden
  * If metadata_used=false, set to null"""
            else:
                system_prompt = f"""You are an expert political bias analyzer. Analyze the following article and calculate bias probability distributions.

INSTRUCTIONS:
1. Analyze the provided article content. Base your judgment primarily on the OPINIONS expressed in the article, not on raw facts.
2. Determine bias probability distribution:
   - left: Probability of Left/Progressive bias (0.0 to 1.0)
   - right: Probability of Right/Conservative bias (0.0 to 1.0)
   - neutral: Probability of Neutral/Centrist stance (0.0 to 1.0)
   - The three probabilities MUST sum to 1.0
3. Use the Source Metadata (MBFC) only as a SOFT PRIOR:
   - Treat MBFC as an initial guess, not a fixed label.
   - If the article’s OPINIONS clearly disagree with the prior, let the OPINIONS override and dominate the final result.
   - High factual reporting should increase your trust in using the prior, but should not by itself force left/right labels.
4. Provide Chain of Thought reasoning explicitly separating:
   - How the article’s OPINIONS support a particular bias.
   - How (if at all) the MBFC prior nudges your final probabilities.

SOURCE HISTORY (MBFC Prior - soft, overridable):
Source: {metadata.source_name}
Bias Rating: {metadata.bias_rating}
Factual Reporting: {metadata.factual_reporting or 'Unknown'}

Provide your analysis as a JSON object with:
- bias_distribution: {{left: float, right: float, neutral: float}} (must sum to 1.0)
- reasoning: string explaining your analysis step-by-step (Chain of Thought required)
- metadata_used: boolean (true if MBFC prior was used)
- mbfc_influence_note: string (only if metadata_used=true) - Brief note describing MBFC prior\'s influence:
  * "strong" - MBFC prior strongly influenced the final result
  * "moderate" - MBFC prior had moderate influence, combined with text evidence
  * "weak" - MBFC prior had minimal influence, text evidence dominated
  * "overridden" - Text opinions clearly contradicted MBFC prior, prior was overridden
  * If metadata_used=false, set to null"""
        else:
            if few_shots_text:
                system_prompt = f"""You are an expert political bias analyzer. Your task is to analyze articles and calculate bias probability distributions.

INSTRUCTIONS:
1. Analyze the provided article content
2. Determine bias probability distribution:
   - left: Probability of Left/Progressive bias (0.0 to 1.0)
   - right: Probability of Right/Conservative bias (0.0 to 1.0)
   - neutral: Probability of Neutral/Centrist stance (0.0 to 1.0)
   - The three probabilities MUST sum to 1.0
3. No source history available - analyze article evidence directly
4. Provide Chain of Thought reasoning explaining your probability assignments

{few_shots_text}
NO SOURCE HISTORY AVAILABLE.
Analyze the article directly. Assume a Neutral Prior: {{left: 0.33, neutral: 0.34, right: 0.33}}

Provide your analysis as a JSON object with:
- bias_distribution: {{left: float, right: float, neutral: float}} (must sum to 1.0)
- reasoning: string explaining your analysis step-by-step (Chain of Thought required)
- metadata_used: boolean (false, no prior available)"""
            else:
                system_prompt = """You are an expert political bias analyzer. Analyze the following article and calculate bias probability distributions.

NO SOURCE HISTORY AVAILABLE.
Analyze the article directly. Assume a Neutral Prior: {left: 0.33, neutral: 0.34, right: 0.33}

Provide your analysis as a JSON object with:
- bias_distribution: {left: float, right: float, neutral: float} (must sum to 1.0)
- reasoning: string explaining your analysis step-by-step (Chain of Thought required)
- metadata_used: boolean (false, no prior available)"""
        
        user_message = f"""ARTICLE TO ANALYZE:

Title: {article.title or 'N/A'}

Content:
{article.full_text}

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
                logger.debug(f"Response text: {response_text[:200]}...")
                return BiasResult(
                    bias_distribution=BiasDistribution(left=0.33, neutral=0.34, right=0.33),
                    reasoning="Failed to parse LLM response: No JSON object found",
                    metadata_used=use_metadata,
                    mbfc_influence_note=None
                )
            
            try:
                result_data = json.loads(json_match.group())
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                logger.debug(f"JSON match: {json_match.group()[:200]}...")
                return BiasResult(
                    bias_distribution=BiasDistribution(left=0.33, neutral=0.34, right=0.33),
                    reasoning=f"JSON parse error: {str(e)}",
                    metadata_used=use_metadata,
                    mbfc_influence_note=None
                )
            
            # Extract bias distribution with validation
            try:
                bias_dist_data = result_data.get('bias_distribution', {})
                bias_distribution = BiasDistribution(
                    left=bias_dist_data.get('left', 0.33),
                    right=bias_dist_data.get('right', 0.33),
                    neutral=bias_dist_data.get('neutral', 0.34)
                )
            except Exception as e:
                logger.error(f"Failed to create BiasDistribution: {e}")
                return BiasResult(
                    bias_distribution=BiasDistribution(left=0.33, neutral=0.34, right=0.33),
                    reasoning=f"Invalid bias distribution data: {str(e)}",
                    metadata_used=use_metadata,
                    mbfc_influence_note=None
                )
            
            reasoning = result_data.get('reasoning', '')
            mbfc_influence_note = result_data.get('mbfc_influence_note')
            # If metadata was used but note not provided, try to infer from reasoning
            if use_metadata and not mbfc_influence_note:
                # Simple heuristic: check reasoning for keywords
                reasoning_lower = reasoning.lower()
                if any(word in reasoning_lower for word in ['override', 'contradict', 'disagree']):
                    mbfc_influence_note = "overridden"
                elif any(word in reasoning_lower for word in ['strong', 'heavily', 'primarily']):
                    mbfc_influence_note = "strong"
                elif any(word in reasoning_lower for word in ['moderate', 'somewhat', 'partially']):
                    mbfc_influence_note = "moderate"
                else:
                    mbfc_influence_note = "weak"
            
            logger.info(f"[Pure Online] Bias calculated: {bias_distribution.dominant_bias}, MBFC influence: {mbfc_influence_note}")
            
            return BiasResult(
                bias_distribution=bias_distribution,
                reasoning=reasoning,
                metadata_used=use_metadata,
                mbfc_influence_note=mbfc_influence_note if use_metadata else None
            )
        except Exception as e:
            # Retry decorator handles network/timeout errors
            # Other errors should be logged and re-raised
            logger.error(f"LLM pure online analysis failed: {e}", exc_info=True)
            raise
    
    # ========== PRODUCTION ORCHESTRATOR ==========
    
    @cached()
    def run(
        self,
        url: str,
        mode: LogicMode = LogicMode.LOCAL_CHAIN,
        use_mbfc: bool = True,
        use_few_shots: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Orchestrator: Run the web analysis pipeline based on LogicMode.
        
        Logic:
        1. Extract content
        2. Resolve metadata (only if use_mbfc=True, uses db_path from settings)
        3. Branching:
           - LOCAL_CHAIN: Call atomize_text -> calculate_bias (passing atomic units)
           - NO_CHAIN: Call calculate_bias directly (passing full text string)
           - PURE_ONLINE: Call separate method for end-to-end LLM analysis
        4. Consolidate: Return dictionary with all intermediate results
        
        Args:
            url: URL to analyze
            mode: LogicMode enum (LOCAL_CHAIN, NO_CHAIN, PURE_ONLINE)
            use_mbfc: Whether to use MBFC database for prior probability (default: True)
            use_few_shots: Whether to use few-shot examples (default: True)
            **kwargs: Additional parameters:
                - atomizer_shots: Optional list of few-shot examples for atomization
                - scorer_shots: Optional list of few-shot examples for bias scoring
            
        Returns:
            Dictionary with complete analysis results including:
            - url, article, metadata (with DB fields if use_mbfc=True), atomic_units (if applicable), bias_analysis, pipeline_metadata
        """
        logger.info(f"[Pipeline] Starting analysis: url={url}, mode={mode}, use_mbfc={use_mbfc}, use_few_shots={use_few_shots}")
        
        try:
            # Agent 1: Extract content
            article = self.extract_content(url)
            logger.info(f"[Pipeline] Agent 1 complete: {len(article.full_text)} chars extracted")
            
            # Agent 2: Resolve metadata (only if use_mbfc=True)
            if use_mbfc:
                metadata = self.resolve_metadata(url)
                logger.info(f"[Pipeline] Agent 2 complete: match_type={metadata.match_type}")
            else:
                metadata = SourceMetadata(
                    source_name=None,
                    raw_db_row=None,
                    match_type="disabled",
                    bias_rating=None,
                    factual_reporting=None
                )
                logger.info("[Pipeline] Agent 2 skipped: MBFC disabled by user")
            
            # Extract shots from kwargs
            atomizer_shots = kwargs.get('atomizer_shots', None)
            scorer_shots = kwargs.get('scorer_shots', None)
            
            # Branching based on mode
            atoms = None
            bias_result = None
            
            if mode == LogicMode.LOCAL_CHAIN:
                # Agent 3: Atomize text
                atoms = self.atomize_text(article.full_text, use_few_shots=use_few_shots, shots=atomizer_shots)
                logger.info(f"[Pipeline] Agent 3 complete: {len(atoms)} atomic units")
                
                # Agent 4: Calculate bias - only analyze opinions to save tokens
                # Filter to only opinions before bias calculation
                opinion_units = [u for u in atoms if u.type == "opinion"]
                fact_count = len(atoms) - len(opinion_units)
                logger.info(f"[Pipeline] Filtering: {len(opinion_units)} opinions, {fact_count} facts (facts skipped for bias analysis)")
                
                if opinion_units:
                    bias_result = self.calculate_bias(opinion_units, metadata, use_few_shots=use_few_shots, shots=scorer_shots)
                    logger.info(f"[Pipeline] Agent 4 complete: {bias_result.bias_distribution.dominant_bias}")
                else:
                    # No opinions found, return neutral bias
                    logger.info("[Pipeline] No opinions found, using neutral bias distribution")
                    bias_result = BiasResult(
                        bias_distribution=BiasDistribution(left=0.33, neutral=0.34, right=0.33),
                        reasoning="No opinions found in text - only facts detected",
                        metadata_used=False,
                        mbfc_influence_note=None
                    )
                
            elif mode == LogicMode.NO_CHAIN:
                # Skip atomization, calculate bias directly on full text
                logger.info("[Pipeline] NO_CHAIN mode: Skipping atomization")
                bias_result = self.calculate_bias(article.full_text, metadata, use_few_shots=use_few_shots, shots=scorer_shots)
                logger.info(f"[Pipeline] Agent 4 complete: {bias_result.bias_distribution.dominant_bias}")
                
            elif mode == LogicMode.PURE_ONLINE:
                # End-to-end LLM analysis
                logger.info("[Pipeline] PURE_ONLINE mode: End-to-end analysis")
                bias_result = self._pure_online_analysis(
                    article, 
                    metadata, 
                    use_few_shots=use_few_shots, 
                    shots=scorer_shots
                )
                logger.info(f"[Pipeline] Pure Online complete: {bias_result.bias_distribution.dominant_bias}")
                
            else:
                raise ValueError(f"Unknown LogicMode: {mode}")
            
            # Consolidate results - include full DB row if MBFC was used
            metadata_dict = None
            if use_mbfc and metadata.match_type not in ("none", "disabled"):
                metadata_dict = {
                    "source_name": metadata.source_name,
                    "match_type": metadata.match_type,
                    "bias_rating": metadata.bias_rating,
                    "factual_reporting": metadata.factual_reporting,
                }
                # Include raw DB row fields if available
                if metadata.raw_db_row:
                    metadata_dict["raw_db_row"] = dict(metadata.raw_db_row)
            
            result = {
                "url": url,
                "article": {
                    "title": article.title,
                    "domain": article.domain,
                    "text_length": len(article.full_text),
                },
                "metadata": metadata_dict,
                "atomic_units": [
                    {
                        "statement": u.statement,
                        "type": u.type,
                        "original_sentence": u.original_sentence,
                        "confidence": u.confidence,
                        "reasoning": u.reasoning,
                    }
                    for u in atoms
                ] if atoms else None,
                "bias_analysis": {
                    "distribution": {
                        "left": bias_result.bias_distribution.left,
                        "right": bias_result.bias_distribution.right,
                        "neutral": bias_result.bias_distribution.neutral,
                    },
                    "dominant_bias": bias_result.bias_distribution.dominant_bias,
                    "reasoning": bias_result.reasoning,
                    "metadata_used": bias_result.metadata_used,
                    "mbfc_influence_note": bias_result.mbfc_influence_note,
                },
                "pipeline_metadata": {
                    "mode": mode.value,
                    "use_mbfc": use_mbfc,
                    "use_few_shots": use_few_shots,
                    "num_atomic_units": len(atoms) if atoms else 0,
                    "model": self.llm.model_name,
                }
            }
            
            logger.info(f"[Pipeline] Complete: {bias_result.bias_distribution.dominant_bias} bias detected")
            return result
            
        except Exception as e:
            logger.error(f"[Pipeline] Error: {e}", exc_info=True)
            raise
    
    def run_pipeline(
        self,
        url: str,
        use_mbfc: bool = True,
        use_few_shots: bool = True,
        custom_atomizer_few_shots: Optional[str] = None,
        custom_scorer_few_shots: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Legacy method: Run the complete 4-agent pipeline (backward compatibility).
        
        This method is kept for backward compatibility. New code should use run() instead.
        
        Args:
            url: URL to analyze
            use_mbfc: Whether to use MBFC database (default: True)
            use_few_shots: Whether to include few-shot examples in few_shots (default: True)
            custom_atomizer_few_shots: Optional custom few-shot examples for atomization (deprecated)
            custom_scorer_few_shots: Optional custom few-shot examples for bias scoring (deprecated)
            
        Returns:
            Dictionary with complete analysis results
        """
        logger.warning("run_pipeline() is deprecated. Use run() with LogicMode instead.")
        
        # Convert to new API
        mode = LogicMode.LOCAL_CHAIN
        
        # Note: custom_few_shots parameters are deprecated as we now use List[dict] for shots
        # This method will use default shots from config files
        # db_path is now configured in settings, not passed via API
        
        return self.run(url, mode=mode, use_mbfc=use_mbfc, use_few_shots=use_few_shots)
