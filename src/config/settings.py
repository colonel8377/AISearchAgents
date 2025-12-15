"""Configuration management for AI Search Agents Platform."""

from typing import Literal, List

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"


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
    openai_proxy: str = Field(default="", description="HTTP proxy for OpenAI API requests (optional)")
    
    # Vector Store Settings
    vector_store_type: Literal["redis", "postgres", "chroma"] = Field(
        default="chroma",
        description="Type of vector store to use"
    )
    
    # Redis Settings
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_password: str = Field(default="", description="Redis password")
    redis_db: int = Field(default=0, description="Redis database number")
    
    # PostgreSQL Settings
    postgres_host: str = Field(default="localhost", description="PostgreSQL host")
    postgres_port: int = Field(default=5432, description="PostgreSQL port")
    postgres_user: str = Field(default="postgres", description="PostgreSQL user")
    postgres_password: str = Field(default="", description="PostgreSQL password")
    postgres_db: str = Field(default="vectordb", description="PostgreSQL database name")
    
    # Chroma Settings
    chroma_persist_directory: str = Field(default="./chroma_db", description="Chroma persistence directory")
    
    # Agent Settings
    agent_max_turns: int = Field(default=4, description="Maximum number of turns (0-3)")
    agent_temperature: float = Field(default=0.7, description="LLM temperature for agent responses")
    
    # API Retry Settings
    openai_max_retries: int = Field(default=3, description="Maximum number of retries for OpenAI API calls")
    openai_timeout: float = Field(default=60.0, description="Timeout in seconds for OpenAI API requests")
    
    # Summarization Settings
    max_conversation_length: int = Field(default=50, description="Maximum number of conversation turns to include in summary")
    max_tokens_per_message: int = Field(default=500, description="Maximum tokens per message in conversation history")
    
    # Logging Settings
    log_level: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    log_file: str = Field(default="logs/ai_search_agents.log", description="Path to log file")
    enable_debug: bool = Field(default=False, description="Enable debug mode with verbose logging")

# Global settings instance
settings = Settings()