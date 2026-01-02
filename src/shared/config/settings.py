"""Configuration management for AI Search Agents Platform."""

from typing import Literal, List, Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

from src.shared.utils import get_logger

BASE_DIR = Path(__file__).resolve().parents[3]
ENV_PATH = BASE_DIR / ".env"


logger = get_logger(__name__)

logger.info(f"Loading environment variables from: {ENV_PATH}")


# Execution mode type for task chains
ExecutionMode = Literal["chain_online", "chain_local", "no_chain"]


class Settings(BaseSettings):
    """Application settings using Pydantic for validation and environment variable loading."""
    
    model_config = SettingsConfigDict(
        env_file=ENV_PATH,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    # API Settings
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    cors_origins: Optional[List[str]] = Field(
        default=None,
        description="CORS allowed origins (comma-separated list or None to disable CORS)"
    )
    
    # Authentication Settings
    api_key_required: bool = Field(default=False, description="Whether API key authentication is required")
    api_keys_str: str = Field(default="", alias="api_keys", description="Comma-separated list of valid API keys (parsed via api_keys property)")
    
    @property
    def api_keys(self) -> List[str]:
        """
        Parse and return API keys as a list.
        
        The raw string is stored in api_keys_str (with alias 'api_keys' for env var),
        and this property provides convenient access as a list of strings.
        """
        if not self.api_keys_str:
            return []
        return [key.strip() for key in self.api_keys_str.split(",") if key.strip()]
    
    # LLM Settings (OpenAI-compatible API for Qwen)
    openai_api_key: str = Field(default="", description="OpenAI API key or Qwen API key")
    openai_api_base: str = Field(default="https://api.openai.com/v1", description="OpenAI API base URL")
    openai_model: str = Field(default="gpt-3.5-turbo", description="Model name (e.g., qwen-turbo)")

    # Embedding Settings
    embedding_provider: str = Field(default="openai", description="Embedding provider: 'openai', 'qwen', 'gemini', 'deepseek', etc.")
    embedding_model: str = Field(default="text-embedding-ada-002", description="Embedding model name")
    embedding_api_key: str = Field(default="", description="API key for embedding provider (uses openai_api_key if empty)")
    embedding_api_base: str = Field(default="", description="API base URL for embedding provider (uses openai_api_base if empty)")
    
    # Vector Store Settings
    vector_store_type: Literal["redis", "postgres", "chroma"] = Field(
        default="chroma",
        description="Type of vector store to use"
    )
    
    # Redis Settings
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_user: str = Field(default='default', description="Redis user")
    redis_password: str = Field(default="", description="Redis password")
    redis_db: int = Field(default=0, description="Redis persistence number")
    
    # PostgreSQL Settings
    postgres_host: str = Field(default="localhost", description="PostgreSQL host")
    postgres_port: int = Field(default=5432, description="PostgreSQL port")
    postgres_user: str = Field(default="postgres", description="PostgreSQL user")
    postgres_password: str = Field(default="", description="PostgreSQL password")
    postgres_db: str = Field(default="vectordb", description="PostgreSQL persistence name")
    
    # Chroma Settings
    chroma_persist_directory: str = Field(default="./chroma_db", description="Chroma persistence directory")
    
    # Agent Settings
    agent_max_turns: int = Field(default=4, description="Maximum number of turns (0-3)")
    agent_temperature: float = Field(default=0.7, description="LLM temperature for agent responses")
    
    # API Retry Settings
    openai_max_retries: int = Field(default=3, description="Maximum number of retries for OpenAI API calls")
    openai_timeout: float = Field(default=60.0, description="Timeout in seconds for OpenAI API requests")
    
    # HTTP Client Optimization Settings
    openai_max_connections: int = Field(default=100, description="Maximum number of HTTP connections in pool")
    openai_max_keepalive_connections: int = Field(default=20, description="Maximum number of keep-alive connections")
    openai_keepalive_expiry: float = Field(default=30.0, description="Keep-alive persistence expiry time in seconds")
    
    # Performance Optimization Settings
    use_optimized_mode: bool = Field(default=True, description="Enable optimized mode with chain caching and persistence pooling")
    use_chain_cache: bool = Field(default=True, description="Enable chain caching (only in optimized mode)")
    use_shared_http_client: bool = Field(default=True, description="Use shared HTTP client with persistence pooling (only in optimized mode)")
    
    # Cache Backend Settings
    cache_backend: Literal["redis", "local"] = Field(
        default="local",
        description="Cache backend type: 'redis' for Redis cache, 'local' for SQLite cache"
    )
    cache_db_path: Optional[str] = Field(
        default=None,
        description="Path to SQLite cache persistence (only used when cache_backend='local'). Defaults to data/agent_cache.db"
    )
    cache_redis_db: int = Field(
        default=1,
        description="Redis persistence number for cache (only used when cache_backend='redis'). Defaults to 1 to separate from vector store"
    )
    use_llm_cache: bool = Field(
        default=True,
        description="Enable caching for LLM API calls to reduce token consumption"
    )
    
    # Task Execution Mode (can be overridden per-request via API)
    default_execution_mode: ExecutionMode = Field(
        default="chain_local",
        description="Default execution mode: 'chain_online' (LLM does chaining), 'chain_local' (we decompose tasks), 'no_chain' (pure prompt)"
    )
    
    # Summarization Settings
    max_conversation_length: int = Field(default=50, description="Maximum number of conversation turns to include in summary")
    max_tokens_per_message: int = Field(default=500, description="Maximum tokens per message in conversation history")
    
    # Chat History Mode Settings
    default_history_mode: bool = Field(
        default=True,
        description="Default history mode: True (include conversation history), False (no history, stateless chat)"
    )
    
    # Web Opinion Extractor Settings
    mbfc_db_path: Optional[str] = Field(
        default=str(BASE_DIR / "data" / "media_bias.db"),
        description="Path to MBFC (Media Bias/Fact Check) SQLite persistence for bias prior probability"
    )
    web_opinion_max_chunk_tokens: int = Field(
        default=15000,
        description="Maximum tokens per chunk for web opinion extraction (default: 15000)"
    )
    web_opinion_network_retry_attempts: int = Field(
        default=3,
        description="Number of retry attempts for network operations (default: 3)"
    )
    web_opinion_network_retry_wait_min: float = Field(
        default=2.0,
        description="Minimum wait time for network retries in seconds (default: 2.0)"
    )
    web_opinion_network_retry_wait_max: float = Field(
        default=10.0,
        description="Maximum wait time for network retries in seconds (default: 10.0)"
    )
    web_opinion_llm_retry_attempts: int = Field(
        default=3,
        description="Number of retry attempts for LLM operations (default: 3)"
    )
    web_opinion_llm_retry_wait_min: float = Field(
        default=2.0,
        description="Minimum wait time for LLM retries in seconds (default: 2.0)"
    )
    web_opinion_llm_retry_wait_max: float = Field(
        default=10.0,
        description="Maximum wait time for LLM retries in seconds (default: 10.0)"
    )
    
    # Debate System Settings
    debate_max_rounds: int = Field(
        default=10,
        description="Maximum number of debate rounds (default: 10)"
    )
    debate_stability_threshold: float = Field(
        default=0.05,
        description="Stability threshold for debate convergence (default: 0.05)"
    )
    debate_consecutive_stable_rounds: int = Field(
        default=2,
        description="Number of consecutive stable rounds required for convergence (default: 2)"
    )
    
    # Logging Settings
    log_level: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    log_file: str = Field(default="logs/ai_search_agents.log", description="Path to log file")
    enable_debug: bool = Field(default=False, description="Enable debug mode with verbose logging")
    
    # Tokenizers Settings (for HuggingFace tokenizers)
    tokenizers_parallelism: str = Field(
        default="true",
        description="Enable HuggingFace tokenizers parallelism ('true' for faster tokenization, 'false' to avoid fork warnings in multi-process environments). Must be set before any fork."
    )
    
    # GLiNER Settings (for privacy detection)
    gliner_max_length: int = Field(
        default=512,
        description="Maximum sequence length for GLiNER model (default: 512). Increase for longer texts, decrease for faster processing."
    )
    gliner_model_name: str = Field(
        default="urchade/gliner_multi-v2.1",
        description="GLiNER model name for privacy detection (default: urchade/gliner_small-v2.1). Available models: gliner_small-v2.1, gliner_medium-v2, gliner_large-v2"
    )
    gliner_threshold: float = Field(
        default=0.3,
        description="GLiNER detection threshold for entity confidence (default: 0.3 for high recall)"
    )
    
    # LLM Verification Settings (for privacy detection)
    enable_llm_verification: bool = Field(
        default=True,
        description="Enable LLM verification for detected entities (default: True)"
    )
    llm_verification_threshold_min: float = Field(
        default=0.3,
        description="Minimum threshold for LLM verification confidence (default: 0.3)"
    )
    llm_verification_threshold_max: float = Field(
        default=0.7,
        description="Maximum threshold for LLM verification confidence (default: 0.7)"
    )

    # Persistence Settings
    enable_persistence: bool = Field(default=True, description="Enable persistent storage for bots, conversations, and sessions")
    storage_db_path: Optional[str] = Field(default=None, description="Path to SQLite persistence for persistent storage. Defaults to data/app_storage.db")

# Global settings instance
settings = Settings()